"""The constrained-decoding engine.

This is the core idea of the project: at each generation step we take the
model's logits, blank out every token that would break the rules by setting it
to -inf, then pick the best survivor. -inf can never be the maximum, so an
illegal token is impossible to choose. Validity comes from the structure, not
from hoping the prompt behaves.

Three primitives live here:

    masked_argmax: the masking operation itself.
    generate_number and generate_string: typed value emitters.
    decode_one_of: a state machine that forces the output to be exactly one
        string from a fixed set (function-name selection and booleans).
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from .protocols import LLM
from .vocab import TokenTables

# hard caps so a misbehaving model can never loop forever (never hang).
_MAX_NUMBER_CHARS = 24
_MAX_STRING_CHARS = 128
_MAX_NAME_CHARS = 64


def masked_argmax(logits: np.ndarray, allowed: np.ndarray) -> int:
    """Return the highest-logit token id among the allowed ids only.

    Args:
        logits: Raw next-token logits over the whole vocabulary.
        allowed: The token ids permitted at this step.

    Returns:
        The id of the best legal token. This is constrained decoding in a
        single operation: illegal ids are set to -inf and can never win.
    """
    masked = np.full(logits.shape, -np.inf, dtype=np.float64)
    masked[allowed] = logits[allowed]
    return int(np.argmax(masked))


def _logits(model: LLM, ids: List[int]) -> np.ndarray:
    """Fetch next-token logits for ids as a float64 numpy array."""
    return np.asarray(model.get_logits_from_input_ids(ids), dtype=np.float64)


def generate_number(model: LLM, ids: List[int], tables: TokenTables) -> str:
    """Constrained-decode a JSON number, appending its tokens to ids.

    The grammar is enforced by masking: an optional leading "-", then one or
    more digits, then at most one "." followed by more digits. Generation stops
    when the model's own unconstrained favourite is no longer a legal
    continuation, which is the model signalling the number is finished.

    Args:
        ids: The primed context; extended in place as tokens are chosen.
        tables: The precomputed token tables.

    Returns:
        The decoded number text, e.g. "42" or "-3.5", or "" if nothing valid
        was produced.
    """
    out = ""
    has_digit = False
    has_dot = False
    while len(out) < _MAX_NUMBER_CHARS:
        logits = _logits(model, ids)
        if out == "":
            allowed = np.concatenate([tables.digit_ids, tables.minus_ids])
            allowed_set = tables.digit_set | tables.minus_set
        elif has_digit and not has_dot:
            allowed = np.concatenate([tables.digit_ids, tables.dot_ids])
            allowed_set = tables.digit_set | tables.dot_set
        else:
            allowed = tables.digit_ids
            allowed_set = tables.digit_set
        # once we have a digit, let the model end the number naturally.
        if has_digit and int(np.argmax(logits)) not in allowed_set:
            break
        if allowed.size == 0:
            break
        token_id = masked_argmax(logits, allowed)
        piece = tables.id_to_text[token_id]
        out += piece
        ids.append(token_id)
        if any(ch.isdigit() for ch in piece):
            has_digit = True
        if "." in piece:
            has_dot = True
    return out


def _looks_repetitive(text: str) -> bool:
    """True if text ends with one short block repeated three times.

    Greedy decoding with a tiny model can fall into a loop (for example a
    digits-then-spaces unit over and over). Spotting an immediate 3x repeat of
    a 1-8 char unit lets us stop such a runaway early: it cleans up the output
    and saves forward passes. A triple repeat is degenerate for real content,
    so false positives are unlikely.
    """
    length = len(text)
    for unit in range(1, 9):
        if length >= unit * 3:
            block = text[-unit:]
            prev = text[-2 * unit:-unit]
            prev2 = text[-3 * unit:-2 * unit]
            if prev == block and prev2 == block:
                return True
    return False


def generate_string(model: LLM, ids: List[int], tables: TokenTables) -> str:
    """Constrained-decode a JSON string body, appending its tokens to ids.

    Only string-safe tokens or the closing quote are ever legal, so the model
    cannot emit a bare quote, newline or control char inside the body. When it
    chooses the closing quote the string is done. The opening and closing
    quotes are written by the caller, so this returns only the body. A
    repetition guard breaks degenerate greedy loops (see _looks_repetitive).

    Args:
        ids: The primed context; extended in place as tokens are chosen.
        tables: The precomputed token tables.

    Returns:
        The decoded string content. Escaping is applied later by json.dump.
    """
    if tables.quote_id < 0:
        return ""
    out = ""
    while len(out) < _MAX_STRING_CHARS:
        logits = _logits(model, ids)
        token_id = masked_argmax(logits, tables.string_or_quote)
        if token_id == tables.quote_id:
            break
        out += tables.id_to_text[token_id]
        ids.append(token_id)
        if _looks_repetitive(out):
            break
    return out


def _span_continuations(
    source: str, prefix: str, tables: TokenTables
) -> set[int]:
    """Token ids whose text keeps the prefix a substring of the source.

    Finds every place the prefix already occurs in the source and collects the
    tokens that could come next without leaving that span. When the prefix is
    empty this returns every token that starts a substring anywhere in the
    source, so every legal place the value could begin.
    """
    allowed: set[int] = set()
    idx = source.find(prefix)
    while idx != -1:
        suffix = source[idx + len(prefix):]
        limit = min(len(suffix), tables.max_token_len)
        for length in range(1, limit + 1):
            token_id = tables.text_to_id.get(suffix[:length])
            if token_id is not None:
                allowed.add(token_id)
        idx = source.find(prefix, idx + 1)
    return allowed


def generate_extractive_string(
    model: LLM,
    ids: List[int],
    tables: TokenTables,
    source: str,
) -> str:
    """Constrained-decode a string that is a contiguous span of the source.

    Most string arguments are copied straight from the request: a file path, a
    SQL query, a template. Restricting the value to a real span of the source
    means the model reproduces genuine text instead of drifting, doubling
    escapes or running on. The model still chooses which span and where it
    stops. At each step the legal tokens are those that keep the value a
    substring of the source, and the value ends when the model would rather
    emit the closing quote than any in-span continuation. A value that is
    quoted in the request is bounded at its closing quote, and every value is
    extended back to the start of its run so leading punctuation (a path's
    slash) is kept.

    Args:
        ids: The primed context; extended in place as tokens are chosen.
        tables: The precomputed token tables.
        source: The text the value must be copied from (the user request).

    Returns:
        The decoded span, or "" if nothing could be matched.
    """
    out = ""
    window = source
    while len(out) < _MAX_STRING_CHARS:
        allowed = _span_continuations(window, out, tables)
        if not allowed:
            break
        logits = _logits(model, ids)
        allowed_arr = np.asarray(sorted(allowed), dtype=np.int64)
        best = float(np.max(logits[allowed_arr]))
        # the model ends the value when it prefers closing over continuing, but
        # only when the closing quote would leave the span (an interior quote
        # that is part of the source stays a legal continuation).
        if out and tables.quote_id >= 0 and tables.quote_id not in allowed:
            if float(logits[tables.quote_id]) >= best:
                break
        token_id = masked_argmax(logits, allowed_arr)
        out += tables.id_to_text[token_id]
        ids.append(token_id)
        # once the value's first token lands right after a quote, it is a
        # quoted value: cut the window at the matching closing quote.
        if window is source and out:
            start = source.find(out)
            if start > 0 and source[start - 1] in "'\"":
                close = source.find(source[start - 1], start)
                if close != -1:
                    window = source[:close]
        if _looks_repetitive(out):
            break
    # a value includes the whole run it sits in: pull in leading punctuation
    # (for example the "/" of a path) back to a space or a quote.
    start = source.find(out)
    while start > 0 and source[start - 1] not in " \t\r\n'\"":
        start -= 1
        out = source[start] + out
    return out


def decode_one_of(
    model: LLM,
    ids: List[int],
    options: Sequence[str],
    tables: TokenTables,
) -> str:
    """Force the model to emit exactly one whole string from options.

    A prefix state machine: prefix is what has been spelled so far. At each
    step the only legal tokens are those that extend prefix toward some option
    without overshooting its end. The model still chooses which option through
    its own preferences at every branch point. The subject's rule holds: the
    LLM decides, the mask only guarantees the answer is a real, correctly
    spelled option.

    Args:
        ids: The primed context; extended in place as tokens are chosen.
        options: The allowed complete strings, e.g. the function names.
        tables: The precomputed token tables.

    Returns:
        The chosen option.
    """
    if not options:
        return ""
    prefix = ""
    while True:
        candidates = [opt for opt in options if opt.startswith(prefix)]
        if not candidates:
            return options[0]
        # done: prefix is a complete option and nothing longer extends it.
        longer_exists = any(len(opt) > len(prefix) for opt in candidates)
        if prefix in options and not longer_exists:
            return prefix
        if len(prefix) > _MAX_NAME_CHARS:
            return candidates[0]

        logits = _logits(model, ids)
        first = prefix == ""
        allowed: List[int] = []
        for token_id, piece, had_space in tables.name_candidates:
            # after the first token a name has no spaces, so a leading-space
            # token can never be a valid continuation.
            if had_space and not first:
                continue
            for candidate in candidates:
                if candidate[len(prefix):].startswith(piece):
                    allowed.append(token_id)
                    break
        if not allowed:
            return candidates[0]

        token_id = masked_argmax(logits, np.asarray(allowed, dtype=np.int64))
        text = tables.id_to_text[token_id]
        piece = text[1:] if (first and text.startswith(" ")) else text
        ids.append(token_id)
        prefix += piece
