"""
CodingSSM inference engine — loads the SFT-trained Mamba-2 model.

Usage:
    from ssm.inference_sft import CodingSSMInference

    model = CodingSSMInference()          # loads checkpoints/sft/sft_best.pt
    response = model.ask("implement binary search")
    print(response)
"""

import re
import sys
from pathlib import Path
from typing import Optional

import torch

_ROOT = Path(__file__).parent.parent


class CodingSSMInference:
    """
    Loads the SFT checkpoint and provides ask() / complete() / stream() interfaces.
    Integrates EpiChatRAG for knowledge-grounded context at inference time.
    """

    DEFAULT_CHECKPOINT = _ROOT / "checkpoints" / "sft" / "sft_best.pt"

    def __init__(
        self,
        checkpoint: str = None,
        device: str = "cpu",
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_new_tokens: int = 512,
        epichat_dir: str = "/home/me/EpiChat",
    ):
        self.device = torch.device(device)
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens

        ckpt_path = Path(checkpoint) if checkpoint else self.DEFAULT_CHECKPOINT
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt_path}\n"
                "Run: bash scripts/train_sft_reasoning.sh"
            )

        print(f"[SSM] Loading model from {ckpt_path} ...", file=sys.stderr)
        self._model, self._tokenizer = self._load(ckpt_path)
        self._model.eval()
        print(f"[SSM] Ready — {self._model.num_parameters()/1e9:.2f}B params", file=sys.stderr)

        # EpiChat RAG for knowledge-grounded context
        self._rag = None
        try:
            from ssm.epichat_rag import EpiChatRAG
            self._rag = EpiChatRAG(epichat_dir)
            print("[SSM] EpiChatRAG enabled", file=sys.stderr)
        except Exception as e:
            print(f"[SSM] EpiChatRAG disabled: {e}", file=sys.stderr)

    def _load(self, ckpt_path: Path):
        from arch.config import ModelConfig700M
        from arch.model import CodingSSM

        # Tokenizer — Qwen2.5 (same vocab the model was trained with)
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(
                "Qwen/Qwen2.5-0.5B",
                trust_remote_code=True,
                cache_dir=str(Path.home() / ".ssm" / "tokenizer_cache"),
            )
        except Exception:
            # Fallback: tiktoken with cl100k if HF isn't available offline
            tok = _FallbackTokenizer()

        cfg = ModelConfig700M()
        model = CodingSSM(cfg)

        state = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        # checkpoint may be wrapped: {"model": ..., "step": ...} or raw state dict
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        model.load_state_dict(state)
        model.to(self.device)
        return model, tok

    # ── Public API ────────────────────────────────────────────────────────────

    def ask(self, question: str, show_thinking: bool = False) -> str:
        """
        Answer a coding question. Returns the assistant response.
        If show_thinking=True, includes the <think>...</think> block.
        """
        prompt = self._build_prompt(question)
        raw = self._generate(prompt)
        return self._format_output(raw, show_thinking)

    def complete(self, code_prefix: str) -> str:
        """Complete a code snippet (no system prompt, raw continuation)."""
        input_ids = self._encode(code_prefix)
        out_ids = self._model.generate(
            input_ids,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            eos_token_id=self._eos_id(),
        )
        return self._decode(out_ids[0])

    def stream(self, question: str):
        """Generator that yields tokens one at a time (for CLI streaming)."""
        prompt = self._build_prompt(question)
        input_ids = self._encode(prompt)
        states = [None] * self._model.config.n_layers

        # Process prompt in one shot
        with torch.no_grad():
            logits, _, states = self._model.forward(input_ids, states=states, return_states=True)

        next_logits = logits[:, -1, :]
        eos_id = self._eos_id()
        in_think = False
        buf = ""

        for _ in range(self.max_new_tokens):
            token_id = self._model._sample(next_logits, self.temperature, self.top_p)
            if token_id.item() == eos_id:
                break

            token_str = self._tokenizer.decode([token_id.item()], skip_special_tokens=False)
            buf += token_str

            # Skip <think>...</think> content in stream output
            if "<think>" in buf:
                in_think = True
            if in_think and "</think>" in buf:
                in_think = False
                buf = buf[buf.index("</think>") + len("</think>"):]

            if not in_think and buf:
                yield buf
                buf = ""

            with torch.no_grad():
                logits, _, states = self._model.forward(
                    token_id.unsqueeze(0).unsqueeze(0),
                    states=states,
                    return_states=True,
                )
            next_logits = logits[:, -1, :]

        if buf and not in_think:
            yield buf

    # ── Internals ─────────────────────────────────────────────────────────────

    SYSTEM_PROMPT = (
        "You are an expert Python programmer and software engineer. "
        "Think carefully step by step.\n"
        "Use <think> tags to show your reasoning, then provide a clear explanation and code.\n"
        "Format:\n<think>\n[your reasoning]\n</think>\n[explanation and code]"
    )

    def _build_prompt(self, question: str) -> str:
        context = ""
        if self._rag:
            context = self._rag.get_context(question)

        user_msg = question
        if context:
            user_msg = f"{context}\n\n{question}"

        return (
            f"<|im_start|>system\n{self.SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{user_msg}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def _generate(self, prompt: str) -> str:
        input_ids = self._encode(prompt)
        out_ids = self._model.generate(
            input_ids,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            eos_token_id=self._eos_id(),
        )
        return self._decode(out_ids[0])

    def _encode(self, text: str) -> torch.Tensor:
        ids = self._tokenizer.encode(text, add_special_tokens=False)
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    def _decode(self, token_ids: torch.Tensor) -> str:
        return self._tokenizer.decode(token_ids.tolist(), skip_special_tokens=True)

    def _eos_id(self) -> Optional[int]:
        try:
            return self._tokenizer.eos_token_id
        except Exception:
            return None

    def _format_output(self, raw: str, show_thinking: bool) -> str:
        if show_thinking:
            return raw.strip()
        # Strip <think>...</think> block for clean output
        clean = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
        return clean.strip()


class _FallbackTokenizer:
    """Minimal tiktoken-based fallback when transformers is unavailable offline."""
    eos_token_id = 151645  # Qwen2.5 <|im_end|>

    def __init__(self):
        import tiktoken
        self._enc = tiktoken.get_encoding("cl100k_base")

    def encode(self, text, **kwargs):
        return self._enc.encode(text)

    def decode(self, ids, **kwargs):
        return self._enc.decode(ids)
