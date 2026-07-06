"""Typing surface for the parts of llm_sdk.Small_LLM_Model we depend on.

The provided SDK ships without type stubs, so importing it directly leaves mypy
seeing Any everywhere. Instead we program against a small typing Protocol.
Any object exposing these four methods, the real Small_LLM_Model or a test
double, satisfies it. That keeps the rest of the package fully typed and makes
the decoding engine easy to unit test.
"""

from __future__ import annotations

from typing import Any, List, Protocol


class LLM(Protocol):
    """Structural type for the frozen language model we drive.

    Only the methods this project actually calls are listed. The real SDK
    exposes get_path_to_vocab_file (not the get_path_to_vocabulary_json named
    in the subject), so the signatures follow the SDK, not the subject text.
    """

    def encode(self, text: str) -> Any:
        """Return a 2-D input_ids tensor for text, without special tokens."""
        ...

    def decode(self, ids: Any) -> str:
        """Turn token ids back into text (used only as a vocab fallback)."""
        ...

    def get_logits_from_input_ids(self, input_ids: List[int]) -> List[float]:
        """Return next-token logits over the whole vocabulary (no softmax)."""
        ...

    def get_path_to_vocab_file(self) -> str:
        """Return the local path to the downloaded vocab.json."""
        ...
