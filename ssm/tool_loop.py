"""
Tool-use inference loop for CodingSSM.

Generates tokens with state caching, intercepts <tool_call> blocks,
dispatches to registered tool handlers, injects <tool_response> and continues.

Usage:
    from ssm.tool_loop import ToolLoop, DEFAULT_REGISTRY
    loop = ToolLoop.from_checkpoint("checkpoints/grpo/best.pt")
    response = loop.chat([{"role": "user", "content": "List files in /tmp"}])
    print(response)
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch


# ── Tool registry ─────────────────────────────────────────────────────────────

class Tool:
    def __init__(self, name: str, description: str, parameters: dict, fn: Callable):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.fn = fn

    def schema(self) -> dict:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}

    def __call__(self, **kwargs) -> str:
        return self.fn(**kwargs)


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, name: str, description: str, parameters: dict):
        def decorator(fn: Callable) -> Callable:
            self._tools[name] = Tool(name, description, parameters, fn)
            return fn
        return decorator

    def add(self, name: str, description: str, parameters: dict, fn: Callable):
        self._tools[name] = Tool(name, description, parameters, fn)

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def names(self) -> List[str]:
        return list(self._tools.keys())

    def schemas(self) -> List[dict]:
        return [t.schema() for t in self._tools.values()]

    def schemas_json(self) -> str:
        return json.dumps(self.schemas(), indent=2)


# ── Built-in tools ────────────────────────────────────────────────────────────

DEFAULT_REGISTRY = ToolRegistry()


@DEFAULT_REGISTRY.register(
    "python_interpreter",
    "Execute Python code and return stdout/stderr. Use for computation, data manipulation, and testing.",
    {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python code to execute"},
            "timeout": {"type": "integer", "description": "Execution timeout in seconds (default 30)"},
        },
        "required": ["code"],
    },
)
def _python(code: str, timeout: int = 30) -> str:
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False, encoding="utf-8") as f:
        f.write(code)
        tmp = f.name
    try:
        r = subprocess.run([sys.executable, tmp], capture_output=True, text=True, timeout=timeout)
        out, err = r.stdout.strip(), r.stderr.strip()
        if err and r.returncode != 0:
            return f"stdout:\n{out}\nstderr:\n{err}" if out else f"Error:\n{err}"
        return (out + ("\n" + err if err else "")).strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return f"Error: timed out after {timeout}s"
    except Exception as e:
        return f"Error: {e}"
    finally:
        Path(tmp).unlink(missing_ok=True)


@DEFAULT_REGISTRY.register(
    "bash",
    "Execute a bash shell command. Use for file operations, git, directory listing, and CLI tools.",
    {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Bash command to execute"},
            "timeout": {"type": "integer", "description": "Timeout in seconds (default 30)"},
        },
        "required": ["command"],
    },
)
def _bash(command: str, timeout: int = 30) -> str:
    try:
        r = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=timeout)
        out, err = r.stdout.strip(), r.stderr.strip()
        if err and r.returncode != 0:
            return f"stdout:\n{out}\nstderr:\n{err}" if out else f"Error (exit {r.returncode}):\n{err}"
        return (out + ("\n" + err if err else "")).strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return f"Error: timed out after {timeout}s"
    except Exception as e:
        return f"Error: {e}"


@DEFAULT_REGISTRY.register(
    "read_file",
    "Read the contents of a file from disk.",
    {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path to read"},
            "max_lines": {"type": "integer", "description": "Max lines to return (default 200)"},
        },
        "required": ["path"],
    },
)
def _read_file(path: str, max_lines: int = 200) -> str:
    try:
        lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
        if len(lines) > max_lines:
            return "\n".join(lines[:max_lines]) + f"\n... ({len(lines) - max_lines} more lines truncated)"
        return "\n".join(lines) or "(empty file)"
    except FileNotFoundError:
        return f"Error: file not found: {path}"
    except Exception as e:
        return f"Error: {e}"


@DEFAULT_REGISTRY.register(
    "write_file",
    "Write content to a file, creating parent directories as needed.",
    {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path to write"},
            "content": {"type": "string", "description": "Content to write"},
            "append": {"type": "boolean", "description": "Append instead of overwrite (default false)"},
        },
        "required": ["path", "content"],
    },
)
def _write_file(path: str, content: str, append: bool = False) -> str:
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        p.open(mode, encoding="utf-8").write(content)
        return f"{'Appended' if append else 'Written'} {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


# ── Tool-call markers ─────────────────────────────────────────────────────────

_TC_OPEN  = "<tool_call>"
_TC_CLOSE = "</tool_call>"
_TR_OPEN  = "<tool_response>"
_TR_CLOSE = "</tool_response>"

_TC_RE = re.compile(
    re.escape(_TC_OPEN) + r"\s*(.*?)\s*" + re.escape(_TC_CLOSE),
    re.DOTALL,
)


# ── Core generation + dispatch loop ──────────────────────────────────────────

class ToolLoop:
    """
    Runs CodingSSM in a loop, intercepting <tool_call> blocks,
    dispatching to registered tools, and injecting <tool_response>.

    Generation uses the model's state cache so each step is O(1) in sequence
    length rather than re-processing the full context.
    """

    def __init__(
        self,
        model,
        tokenizer,
        registry: ToolRegistry = DEFAULT_REGISTRY,
        max_new_tokens: int = 2048,
        max_tool_calls: int = 10,
        temperature: float = 0.7,
        top_p: float = 0.9,
        device: str = "cuda",
        verbose: bool = False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.registry = registry
        self.max_new_tokens = max_new_tokens
        self.max_tool_calls = max_tool_calls
        self.temperature = temperature
        self.top_p = top_p
        self.device = device
        self.verbose = verbose
        self._eos_id = getattr(tokenizer, "eos_token_id", None)

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        device: str = None,
        registry: ToolRegistry = DEFAULT_REGISTRY,
        **kwargs,
    ) -> "ToolLoop":
        from transformers import AutoTokenizer
        from arch import CodingSSM
        from arch.config import ModelConfig700M

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        dev = torch.device(device)

        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-Coder-7B-Instruct", trust_remote_code=True,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model = CodingSSM(ModelConfig700M())
        ckpt = torch.load(path, map_location=dev, weights_only=False)
        state = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(state, strict=False)
        model.to(dev).eval()

        print(
            f"[ToolLoop] {sum(p.numel() for p in model.parameters())/1e9:.2f}B params "
            f"from {path} on {dev}",
            file=sys.stderr,
        )
        return cls(model, tokenizer, registry=registry, device=str(dev), **kwargs)

    # ── Public API ────────────────────────────────────────────────────────────

    def chat(self, messages: List[Dict[str, str]], system: str = "") -> str:
        """
        OpenAI-style chat interface.

        Args:
            messages: [{"role": "user"|"assistant", "content": "..."}, ...]
            system:   Optional system prompt override (uses registry tool schemas by default).

        Returns:
            Full assistant response string (may include <think>, <tool_call>, text).
        """
        if not system:
            system = self._default_system()
        prompt = self._build_chatml(messages, system)
        return self.run(prompt)

    def run(self, prompt: str) -> str:
        """
        Run the full tool loop on a raw prompt string.
        Handles arbitrary numbers of tool call / response cycles.
        """
        return _generate_with_tool_loop(
            self.model,
            self.tokenizer,
            prompt,
            self.registry,
            max_new_tokens=self.max_new_tokens,
            max_tool_calls=self.max_tool_calls,
            temperature=self.temperature,
            top_p=self.top_p,
            device=self.device,
            verbose=self.verbose,
        )

    # ── Internals ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _generate_until_tool_or_eos(
        self,
        states: list,
        max_tokens: int,
    ) -> tuple[str, list, bool, int]:
        """
        Generate tokens one at a time using state cache, stopping at </tool_call> or EOS.

        Returns: (decoded_text, updated_states, hit_tool_call, n_tokens_generated)
        """
        generated_ids = []

        for _ in range(max_tokens):
            # Get last generated token (or a dummy for initial call)
            last_id = generated_ids[-1] if generated_ids else None
            if last_id is None:
                # We have states from prompt — need to sample the first token
                # Re-run with a single dummy step using last prompt token
                # Actually states already include the prompt; we just sample from them
                # by running a zero-length forward... not possible.
                # Instead, track the last logits separately.
                break

            next_id_tensor = torch.tensor([[last_id]], dtype=torch.long, device=self.device)
            logits, _, states = self.model.forward(next_id_tensor, states=states, return_states=True)
            next_token_logits = logits[0, -1, :]

            next_id = int(self.model._sample(
                next_token_logits.unsqueeze(0), self.temperature, self.top_p
            ).item())

            if next_id == self._eos_id:
                break

            generated_ids.append(next_id)
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
            if _TC_CLOSE in text:
                return text, states, True, len(generated_ids)

        text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
        return text, states, False, len(generated_ids)

    def _generate_first_token(self, states: list) -> tuple[int, list, torch.Tensor]:
        """Sample the very first token after priming the state cache."""
        # We can't call forward with no tokens — use a single-step forward on
        # the EOS token as a neutral probe to get the next-token distribution.
        # Actually the states already encode the full prompt context; we just
        # need any single token to step forward. Use a dummy pad token.
        pad_id = getattr(self.tokenizer, "pad_token_id", 0) or 0
        probe = torch.tensor([[pad_id]], dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits, _, new_states = self.model.forward(probe, states=states, return_states=True)
        next_token_logits = logits[0, -1, :]
        next_id = int(self.model._sample(
            next_token_logits.unsqueeze(0), self.temperature, self.top_p
        ).item())
        return next_id, new_states, next_token_logits

    def _dispatch(self, text: str) -> str:
        """Parse a <tool_call>...</tool_call> block and call the registered tool."""
        m = _TC_RE.search(text)
        if not m:
            return "Error: malformed <tool_call> block"

        try:
            call = json.loads(m.group(1))
        except json.JSONDecodeError as e:
            return f"Error: invalid JSON in tool_call: {e}"

        # Support both {"name":..., "arguments":...} and {"function":{"name":...,"arguments":...}}
        name = call.get("name") or (call.get("function") or {}).get("name", "")
        args = call.get("arguments") or (call.get("function") or {}).get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                pass

        tool = self.registry.get(name)
        if tool is None:
            return f"Error: unknown tool '{name}'. Available: {self.registry.names()}"

        try:
            return str(tool(**args))
        except Exception as e:
            return f"Error executing {name}: {e}"

    def _default_system(self) -> str:
        tools_json = self.registry.schemas_json()
        return (
            "You are a helpful coding assistant with access to tools.\n"
            "Think step by step inside <think>...</think> tags before acting.\n"
            "To use a tool, emit:\n"
            f"{_TC_OPEN}\n{{\"name\": \"tool_name\", \"arguments\": {{...}}}}\n{_TC_CLOSE}\n"
            "The tool result will be returned in <tool_response>...</tool_response>.\n\n"
            f"Available tools:\n{tools_json}"
        )

    @staticmethod
    def _build_chatml(messages: List[Dict[str, str]], system: str) -> str:
        parts = []
        if system:
            parts.append(f"<|im_start|>system\n{system}<|im_end|>")
        for msg in messages:
            role = "user" if msg["role"] == "user" else "assistant"
            parts.append(f"<|im_start|>{role}\n{msg['content']}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)


# ── Fixed generation loop (replaces broken _generate_until_tool_or_eos) ───────

def _generate_with_tool_loop(
    model,
    tokenizer,
    prompt: str,
    registry: ToolRegistry,
    max_new_tokens: int = 2048,
    max_tool_calls: int = 10,
    temperature: float = 0.7,
    top_p: float = 0.9,
    device: str = "cuda",
    verbose: bool = False,
) -> str:
    """
    Standalone functional version of the tool loop.
    Uses model.generate() with short budgets and suffix checks to detect tool calls.
    More robust than the stateful version for one-shot use.
    """
    eos_id = getattr(tokenizer, "eos_token_id", None)
    input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(device)

    full_response = ""
    tool_calls_made = 0
    tokens_left = max_new_tokens

    while tokens_left > 0 and tool_calls_made <= max_tool_calls:
        # Generate a chunk (up to 256 tokens or budget, whichever is smaller)
        chunk_budget = min(tokens_left, 256)
        with torch.no_grad():
            new_ids = model.generate(
                input_ids,
                max_new_tokens=chunk_budget,
                temperature=temperature,
                top_p=top_p,
                eos_token_id=eos_id,
            )  # (1, new_tokens) — only new tokens

        chunk_text = tokenizer.decode(new_ids[0].tolist(), skip_special_tokens=False)
        tokens_left -= new_ids.shape[1]

        # Append new tokens to input for next iteration
        input_ids = torch.cat([input_ids, new_ids], dim=-1)

        if _TC_CLOSE in chunk_text:
            # Find the tool call and trim to its end
            end = chunk_text.index(_TC_CLOSE) + len(_TC_CLOSE)
            before_tool = chunk_text[:end]
            after_tool = chunk_text[end:]

            full_response += before_tool

            # Dispatch
            tool_result = _dispatch(before_tool, registry)
            tool_calls_made += 1

            if verbose:
                print(f"[tool_loop] call #{tool_calls_made}: {tool_result[:120]}", file=sys.stderr)

            response_block = f"\n{_TR_OPEN}\n{tool_result}\n{_TR_CLOSE}\n"
            full_response += response_block

            # Re-encode everything so far for the next round
            continuation = prompt + full_response
            input_ids = tokenizer.encode(
                continuation, add_special_tokens=False, return_tensors="pt"
            ).to(device)
        else:
            full_response += chunk_text
            # Check for EOS
            if new_ids.shape[1] < chunk_budget:
                break

    return full_response


def _dispatch(text: str, registry: ToolRegistry) -> str:
    m = _TC_RE.search(text)
    if not m:
        return "Error: malformed <tool_call> block"
    try:
        call = json.loads(m.group(1))
    except json.JSONDecodeError as e:
        return f"Error: invalid JSON: {e}"
    name = call.get("name") or (call.get("function") or {}).get("name", "")
    args = call.get("arguments") or (call.get("function") or {}).get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            pass
    tool = registry.get(name)
    if tool is None:
        return f"Error: unknown tool '{name}'. Available: {registry.names()}"
    try:
        return str(tool(**args))
    except Exception as e:
        return f"Error executing {name}: {e}"


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    import argparse

    p = argparse.ArgumentParser(description="CodingSSM tool-use inference")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--prompt", default=None)
    p.add_argument("--system", default="")
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--max-tool-calls", type=int, default=10)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--device", default=None)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    loop = ToolLoop.from_checkpoint(
        args.checkpoint,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        max_tool_calls=args.max_tool_calls,
        temperature=args.temperature,
        verbose=args.verbose,
    )

    prompt = args.prompt or sys.stdin.read().strip()
    response = loop.chat(
        [{"role": "user", "content": prompt}],
        system=args.system or None,
    )
    print(response)


if __name__ == "__main__":
    main()
