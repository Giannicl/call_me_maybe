"""Token tables: map every token id to its text and group ids by type.

Constrained decoding needs to know, for any token id, *what characters it spells*
so it can decide whether that token is legal right now.  We build that map once
(a few seconds) and precompute the id-sets the masks reuse every step:

* ``digit_ids`` / ``minus_ids`` / ``dot_ids`` — the pieces of a JSON number,
* ``string_or_quote`` — every token safe inside a JSON string, plus the closing
  quote that ends one (drives the free-generation fallback for values that are
  not substrings of the prompt, e.g. an inferred regex),
* ``name_candidates`` — tokens made only of function-name characters, used by the
  function-name state machine,
* ``text_to_ids`` / ``max_token_len`` — the reverse map (token text -> ids), used
  by the span-copy string decoder to probe substrings of the prompt without ever
  scanning the whole vocabulary.

Token text comes from ``vocab.json`` (path via ``get_path_to_vocab_file``).  Qwen
uses *byte-level BPE*: each token string is written in a reversible alphabet where,
for example, a leading space is the character ``Ġ``.  :func:`bytes_to_unicode`
gives the byte→character map; we invert it to recover the real bytes, then UTF-8
decode.  Doing it this way (instead of calling ``decode`` 150k times) is both
faster and satisfies the optional "recode the tokenizer" bonus.
"""

from __future__ import annotations

import json
from typing import Dict, List, Tuple

import numpy as np
from pydantic import BaseModel, ConfigDict

from .protocols import LLM

# Characters that may appear in a function name (Qwen names: fn_add_numbers, ...).
NAME_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
)
_DIGITS = frozenset("0123456789")


def bytes_to_unicode() -> Dict[int, str]:
    """Return the GPT-2/Qwen byte-level BPE map: byte value -> printable char.

    Every one of the 256 byte values is mapped to a unique unicode character so a
    token string is always printable and reversible.  Bytes that are already
    printable map to themselves; the rest (spaces, controls, high bytes) map to
    code points starting at 256 — this is why a space shows up as ``Ġ``.
    """
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    mapping = printable[:]
    spare = 0
    for byte in range(256):
        if byte not in printable:
            printable.append(byte)
            mapping.append(256 + spare)
            spare += 1
    return {byte: chr(code) for byte, code in zip(printable, mapping)}


def _decode_token(token: str, byte_decoder: Dict[str, int]) -> str:
    """Decode one byte-level token string back to its real text."""
    try:
        raw = bytes(byte_decoder[ch] for ch in token)
    except KeyError:
        # Token contains a character outside the byte alphabet (rare added
        # tokens); fall back to the literal string.
        return token
    return raw.decode("utf-8", errors="replace")


def _build_id_to_text(model: LLM, vocab_size: int) -> List[str]:
    """Build the id -> text table, sized to the model's logits vector.

    Falls back to the SDK's ``decode`` if ``vocab.json`` cannot be read, so the
    tool still works rather than crashing.
    """
    id_to_text: List[str] = [""] * vocab_size
    byte_decoder = {ch: byte for byte, ch in bytes_to_unicode().items()}
    try:
        with open(model.get_path_to_vocab_file(), "r", encoding="utf-8") as handle:
            vocab: Dict[str, int] = json.load(handle)
        for token, token_id in vocab.items():
            if 0 <= token_id < vocab_size:
                id_to_text[token_id] = _decode_token(token, byte_decoder)
    except (OSError, json.JSONDecodeError, ValueError):
        for token_id in range(vocab_size):
            id_to_text[token_id] = model.decode([token_id])
    return id_to_text


def _has_control(text: str) -> bool:
    """True if *text* contains a character illegal inside a raw JSON string."""
    return any(ord(ch) < 0x20 for ch in text)


class TokenTables(BaseModel):
    """Immutable lookup tables shared by every decoding step.

    ``arbitrary_types_allowed`` is enabled so we can store the precomputed numpy
    index arrays the masks use (numpy fancy-indexing is what makes per-step
    masking fast).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    id_to_text: List[str]
    vocab_size: int
    digit_ids: np.ndarray
    minus_ids: np.ndarray
    dot_ids: np.ndarray
    string_or_quote: np.ndarray
    quote_id: int
    digit_set: frozenset[int]
    minus_set: frozenset[int]
    dot_set: frozenset[int]
    name_candidates: List[Tuple[int, str, bool]]
    text_to_ids: Dict[str, List[int]]
    max_token_len: int


def tables_from_id_to_text(id_to_text: List[str]) -> TokenTables:
    """Compute every typed id-set from a finished id -> text table.

    Split out from :func:`build_token_tables` so unit tests can build tables from
    a tiny hand-written vocabulary without a real model.
    """
    digit_ids: List[int] = []
    minus_ids: List[int] = []
    dot_ids: List[int] = []
    string_ids: List[int] = []
    name_candidates: List[Tuple[int, str, bool]] = []
    text_to_ids: Dict[str, List[int]] = {}
    max_token_len = 0
    quote_id = -1

    for token_id, text in enumerate(id_to_text):
        if text == "":
            continue
        text_to_ids.setdefault(text, []).append(token_id)
        max_token_len = max(max_token_len, len(text))
        if all(ch in _DIGITS for ch in text):
            digit_ids.append(token_id)
        if text == "-":
            minus_ids.append(token_id)
        if text == ".":
            dot_ids.append(token_id)
        if text == '"':
            quote_id = token_id
        # A token is string-safe if it contains no quote, backslash-free control
        # char, or raw newline/tab — json.dump handles any remaining escaping.
        if '"' not in text and not _has_control(text):
            string_ids.append(token_id)
        # Name-state-machine candidates: tokens that are pure name characters,
        # optionally with one leading space (the Ġ marker on the first token).
        stripped = text[1:] if text.startswith(" ") else text
        if stripped and all(ch in NAME_CHARS for ch in stripped):
            name_candidates.append((token_id, stripped, text.startswith(" ")))

    string_or_quote_ids = string_ids + ([quote_id] if quote_id >= 0 else [])
    return TokenTables(
        id_to_text=id_to_text,
        vocab_size=len(id_to_text),
        digit_ids=np.asarray(digit_ids, dtype=np.int64),
        minus_ids=np.asarray(minus_ids, dtype=np.int64),
        dot_ids=np.asarray(dot_ids, dtype=np.int64),
        string_or_quote=np.asarray(string_or_quote_ids, dtype=np.int64),
        quote_id=quote_id,
        digit_set=frozenset(digit_ids),
        minus_set=frozenset(minus_ids),
        dot_set=frozenset(dot_ids),
        name_candidates=name_candidates,
        text_to_ids=text_to_ids,
        max_token_len=max_token_len,
    )


def build_token_tables(model: LLM) -> TokenTables:
    """Build the token tables for a loaded model (one-time setup)."""
    # The logits vector length is the authoritative vocabulary size to mask over.
    probe_ids: List[int] = list(model.encode("hi")[0].tolist())
    vocab_size = len(model.get_logits_from_input_ids(probe_ids))
    id_to_text = _build_id_to_text(model, vocab_size)
    return tables_from_id_to_text(id_to_text)
