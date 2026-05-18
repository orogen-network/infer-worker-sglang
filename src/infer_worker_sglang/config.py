"""Worker configuration. Injected by harness/tests."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class WorkerConfig:
    operator_id: str
    operator_private_key_hex: str
    gateway_id: str
    attestation_report_hash: str
    model_id: str = "mock-sglang-7b"
    model_weight_hash: str = "0x" + "ab" * 32
    kernel_pack_hash: str = "0x" + "cd" * 32
    heartbeat_interval_s: float = 12.0
    base_url: str = ""
    capabilities: list[str] = field(default_factory=lambda: ["mock-sglang-7b"])
    deterministic_mode: bool = True
    # On-disk path to the weights file or HF-style directory. If unset, the
    # weight-hash verification step at startup is skipped (Mock engines have
    # no weights to verify). See `weights.verify_weights`.
    model_path: str | None = None
