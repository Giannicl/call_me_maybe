"""Typing surface for the parts of `llm_sdk.Small_LLM_Model` we depend on.

The provided SDK ships without type stubs, so importing it directly leaves
`mypy` seeing `Any` everywhere. We program against this small `typing.Protocol`
instead. Any object exposing these four methods satisfies it: the real
`Small_LLM_Model` or a test double. That keeps the rest of the package fully
typed and lets the engine be unit-tested without a model.
"""

from __future__ import annotations

from typing import Any, List, Protocol


class LLM(Protocol):
    """Structural type for the frozen language model we drive.

    Only the methods this project calls are listed. The signatures follow the
    real SDK source (`llm_sdk/llm_sdk/__init__.py`), which differs from the
    subject's description of the API.
    """

    def encode(self, text: str) -> Any:
        """Return a 2-D `input_ids` tensor for the text, with no special tokens.

        Args:
            text: The text to tokenize.

        Returns:
            The `input_ids` tensor.
        """
        ...

    def decode(self, ids: Any) -> str:
        """Turn token ids back into text. Used only as a vocab fallback.

        Args:
            ids: The token ids to decode.

        Returns:
            The decoded text.
        """
        ...

    def get_logits_from_input_ids(self, input_ids: List[int]) -> List[float]:
        """Return next-token logits over the whole vocabulary, without softmax.

        Args:
            input_ids: The primed context token ids.

        Returns:
            One raw logit per vocabulary token.
        """
        ...

    def get_path_to_vocab_file(self) -> str:
        """Return the local path to the downloaded `vocab.json`.

        Returns:
            The path to `vocab.json`.
        """
        ...
