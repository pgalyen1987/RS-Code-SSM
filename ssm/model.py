"""LLM inference backend using llama-cpp-python.

Supports any GGUF model: Qwen2.5-Coder, DeepSeek-R1, Llama, etc.
Runs entirely on CPU — no GPU required.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Iterator, Optional


class LLM:
    """Wrapper around llama-cpp-python for GGUF model inference."""

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 16384,
        n_threads: int = -1,
        n_batch: int = 512,
        verbose: bool = False,
    ):
        from llama_cpp import Llama

        if n_threads == -1:
            n_threads = os.cpu_count() or 4

        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_batch=n_batch,
            verbose=verbose,
        )
        self.n_ctx = n_ctx

    def _inject_context(self, messages: list[dict], context: Optional[str]) -> list[dict]:
        """Prepend RAG context into the system message."""
        if not context:
            return messages

        messages = list(messages)
        sys_idx = next((i for i, m in enumerate(messages) if m["role"] == "system"), None)

        if sys_idx is not None:
            messages[sys_idx] = {
                "role": "system",
                "content": messages[sys_idx]["content"]
                + f"\n\nRelevant code from the codebase:\n{context}",
            }
        else:
            messages.insert(0, {
                "role": "system",
                "content": f"Relevant code from the codebase:\n{context}",
            })

        return messages

    def generate(
        self,
        messages: list[dict],
        context: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.8,
        top_p: float = 0.9,
        callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        messages = self._inject_context(messages, context)

        stream = self.llm.create_chat_completion(
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stream=True,
        )

        full: list[str] = []
        for chunk in stream:
            text = chunk["choices"][0]["delta"].get("content", "")
            if text:
                full.append(text)
                if callback:
                    callback(text)

        return "".join(full)

    def generate_stream(
        self,
        messages: list[dict],
        context: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.8,
        top_p: float = 0.9,
    ) -> Iterator[str]:
        messages = self._inject_context(messages, context)

        stream = self.llm.create_chat_completion(
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stream=True,
        )

        for chunk in stream:
            text = chunk["choices"][0]["delta"].get("content", "")
            if text:
                yield text
