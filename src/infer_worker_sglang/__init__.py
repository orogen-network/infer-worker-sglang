"""SGLang-backed operator daemon variant (chat/RAG specialty)."""

from infer_worker_sglang.app import build_app
from infer_worker_sglang.config import WorkerConfig
from infer_worker_sglang.engine import (
    Engine,
    InferenceResult,
    MockSGLangEngine,
    RealSGLangEngine,
)

__version__ = "0.1.0"

__all__ = [
    "Engine",
    "InferenceResult",
    "MockSGLangEngine",
    "RealSGLangEngine",
    "WorkerConfig",
    "build_app",
]
