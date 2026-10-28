"""One configuration for D, set by E before starting an experiment."""

from dataclasses import asdict, dataclass
import math

from .errors import ConfigurationError


PROMPT_VERSIONS = {
    "verify_draft": "d-draft-v1",
    "verify_claims": "d-claims-v1",
    "revise_answer": "d-revision-v1",
}


@dataclass(frozen=True)
class Settings:
    model: str
    temperature: float = 0.0
    verification_max_tokens: int = 2048
    revision_max_tokens: int = 2048


_settings: Settings | None = None


def configure(
    model: str,
    temperature: float = 0.0,
    verification_max_tokens: int = 2048,
    revision_max_tokens: int = 2048,
) -> None:
    """Set once per run. M3/M4/M5 must not reconfigure between conditions."""
    if not isinstance(model, str) or not model.strip():
        raise ConfigurationError("A team-selected model name is required.")
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(temperature)
        or temperature < 0
    ):
        raise ConfigurationError("temperature must be a finite non-negative number.")
    for name, value in (
        ("verification_max_tokens", verification_max_tokens),
        ("revision_max_tokens", revision_max_tokens),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ConfigurationError(f"{name} must be a positive integer.")
    global _settings
    _settings = Settings(model, float(temperature), verification_max_tokens, revision_max_tokens)


def settings() -> Settings:
    if _settings is None:
        raise ConfigurationError("Call src.verification.configure(model=...) before LLM use.")
    return _settings


def runtime_metadata() -> dict:
    """E can include this snapshot in its ExperimentRecords."""
    return {"settings": asdict(settings()), "prompt_versions": dict(PROMPT_VERSIONS)}
