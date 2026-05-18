# infer-worker-sglang

SGLang-backed operator daemon variant. Specialty: structured/programmatic decoding for
chat / RAG / tool-use workloads.

## Architecture

Mirrors `infer-worker-vllm` end-to-end (FastAPI app, signed receipts per RFC-0001,
heartbeat sender per RFC-0003) but routes inference through an `Engine` Protocol with
two implementations:

- **`MockSGLangEngine`** — deterministic pseudo-tokens prefixed with `"sg"`. Used by
  tests + CPU dev boxes.
- **`RealSGLangEngine`** — wraps `sglang.Runtime`. Raises `ImportError` if SGLang
  isn't installed (we don't have GPU here); raises only at instantiation, not import.

The HTTP surface matches `infer-worker-vllm` so the gateway router can route to either
without a custom code path.

## SGLang prod integration sketch

```python
from sglang import Runtime
runtime = Runtime(model_path="meta-llama/Llama-3-8B-Instruct", trust_remote_code=False)
```

Requires CUDA, `sglang[srt]`, and a model checkpoint local to the host.

## Endpoints

- `POST /v1/chat/completions` — OpenAI-compatible.
- `GET  /healthz`
- `GET  /internal/last_heartbeat`
