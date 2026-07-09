"""Typing surface for the parts of ``llm_sdk.Small_LLM_Model`` we depend on.

The provided SDK ships without type stubs, so importing it directly leaves
``mypy`` seeing ``Any`` everywhere.  We instead program against this small
:class:`typing.Protocol`.  Any object exposing these four methods (the real
``Small_LLM_Model`` *or* a test double) satisfies it, which keeps the rest of
the package fully typed and makes the engine trivially unit-testable.
"""

from __future__ import annotations

from typing import Any, List, Protocol


class LLM(Protocol):
    """Structural type for the frozen language model we drive.

    Only the methods this project actually calls are listed.  The signatures
    follow the *real* SDK source (``llm_sdk/llm_sdk/__init__.py``), which
    differs from the subject's description of the API.
    """

    def encode(self, text: str) -> Any:
        """Return a 2-D ``input_ids`` tensor for *text* (no special tokens)."""
        ...

    def decode(self, ids: Any) -> str:
        """Turn token ids back into text (used only as a vocab fallback)."""
        ...

    def get_logits_from_input_ids(self, input_ids: List[int]) -> List[float]:
        """Return next-token logits over the whole vocabulary (no softmax)."""
        ...

    def get_path_to_vocab_file(self) -> str:
        """Return the local path to the downloaded ``vocab.json``."""
        ...
