"""Instrument specification for LLM Blind-ID runs."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class InstrumentSpec:
    """測定器仕様 I。予算写像は「B = 課金 fetch 回数」で固定(環境が強制)。"""

    provider: str
    model: str
    temperature: float = 0.0
    max_output_tokens: int = 300
    prompt_version: str = "blind-id-v1"
    protocol_version: str = "1"
    max_turns: int = 60
    max_protocol_errors: int = 5
    requested_seed: int | None = None

    def __post_init__(self) -> None:
        if self.provider not in {"anthropic", "openai", "offline"}:
            raise ValueError("provider must be anthropic, openai, or offline")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if self.max_turns < 1:
            raise ValueError("max_turns must be at least 1")
        if self.max_protocol_errors < 1:
            raise ValueError("max_protocol_errors must be at least 1")

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "InstrumentSpec":
        return InstrumentSpec(**d)
