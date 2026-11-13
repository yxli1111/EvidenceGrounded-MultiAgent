"""D's evidence verification public API."""

from .config import configure, runtime_metadata
from .verifier import verify_claims, verify_draft

__all__ = ["configure", "runtime_metadata", "verify_draft", "verify_claims"]
