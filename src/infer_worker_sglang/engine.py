"""Engine abstraction for SGLang.

SGLang requires CUDA; this module exposes an `Engine` Protocol with two impls:

- `MockSGLangEngine` — deterministic pseudo-tokens; used by tests + dev.
- `RealSGLangEngine` — wraps `sglang.Runtime` if installed; raises `ImportError`
  with a clear message otherwise.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(slots=True)
class InferenceResult:
    text: str
    tokens: list[str]
    log_probs: list[float]
    prompt_tokens: int
    completion_tokens: int


class Engine(Protocol):
    model_id: str

    def generate(
        self, prompt: str, *, max_tokens: int = 32, seed: int = 0,
    ) -> InferenceResult: ...


class MockSGLangEngine:
    """Deterministic stand-in. Output is a function of (model_id, prompt, seed)."""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id

    def generate(
        self, prompt: str, *, max_tokens: int = 32, seed: int = 0,
    ) -> InferenceResult:
        key = f"{self.model_id}::{prompt}::{seed}".encode()
        digest = hashlib.sha256(key).digest()
        # SGLang's specialty is structured/programmatic decoding — fake that by
        # prefixing tokens with "sg" so receipts are visibly distinct.
        n_tokens = min(max(4, len(prompt) // 4), max_tokens)
        tokens = [f"sg{digest[i % len(digest)]:02x}" for i in range(n_tokens)]
        log_probs = [-(b / 51.0) for b in digest[:64]]
        return InferenceResult(
            text=" ".join(tokens),
            tokens=tokens,
            log_probs=log_probs,
            prompt_tokens=max(1, len(prompt.split())),
            completion_tokens=n_tokens,
        )


class RealSGLangEngine:
    """Production adapter — instantiates `sglang.Runtime`. Imports lazily so the
    package remains importable on CPU dev boxes that lack SGLang."""

    def __init__(self, model_id: str, **runtime_kwargs: Any) -> None:
        try:
            import sglang  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "sglang is not installed. Install with `pip install sglang[srt]` on a "
                "CUDA-equipped host. Use MockSGLangEngine for dev/test."
            ) from exc
        self.model_id = model_id
        self._runtime = sglang.Runtime(model_path=model_id, **runtime_kwargs)  # type: ignore[attr-defined]

    def generate(
        self, prompt: str, *, max_tokens: int = 32, seed: int = 0,
    ) -> InferenceResult:  # pragma: no cover — requires CUDA + sglang
        out = self._runtime.generate(
            prompt=prompt,
            sampling_params={"max_new_tokens": max_tokens, "temperature": 0.0},
        )
        text = out["text"] if isinstance(out, dict) else str(out)
        tokens = text.split()
        log_probs: list[float] = []
        for tok in tokens[:64]:
            d = hashlib.sha256(tok.encode()).digest()[0]
            log_probs.append(-(d / 51.0))
        return InferenceResult(
            text=text,
            tokens=tokens,
            log_probs=log_probs,
            prompt_tokens=max(1, len(prompt.split())),
            completion_tokens=len(tokens),
        )
