"""The constrained-decoding engine.

This is the project's core idea (subject V.3). At each generation step we take
the model's logits, blank out every token that would break the rules by setting
it to `-inf`, then pick the best survivor. `-inf` can never be the maximum, so
an illegal token is impossible to choose. Validity is guaranteed by structure,
not hoped for from the prompt.

Three primitives live here:

* `masked_argmax`: the masking operation itself.
* `generate_number` / `generate_string`: typed value emitters.
* `decode_one_of`: a state machine that forces the output to be exactly one
  string from a fixed set, used for function-name selection and booleans.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from .protocols import LLM
from .vocab import TokenTables

_MAX_NUMBER_CHARS = 24
_MAX_STRING_CHARS = 128
_MAX_NAME_CHARS = 64

_VALUE_BOUNDARY_CHARS = "'\"`([{:=,"


def masked_argmax(logits: np.ndarray, allowed: np.ndarray) -> int:
    """Return the highest-logit token id among the allowed ids only.

    Args:
        logits: Raw next-token logits over the whole vocabulary.
        allowed: Token ids permitted at this step.

    Returns:
        The id of the best legal token. This is constrained decoding in one
        operation.
    """
    masked = np.full(logits.shape, -np.inf, dtype=np.float64)
    masked[allowed] = logits[allowed]
    return int(np.argmax(masked))


def _logits(model: LLM, ids: List[int]) -> np.ndarray:
    """Fetch next-token logits for the ids as a float64 numpy array.

    Args:
        model: The language model to query.
        ids: The primed context token ids.

    Returns:
        The next-token logits.
    """
    return np.asarray(model.get_logits_from_input_ids(ids), dtype=np.float64)


def generate_number(
    model: LLM,
    ids: List[int],
    tables: TokenTables,
    allow_dot: bool = True,
) -> str:
    """Constrained-decode a JSON number, appending its tokens to `ids`.

    The mask enforces the grammar: an optional leading `-`, then one or more
    digit tokens, then at most one `.` followed by more digits. Generation stops
    when the model's own unconstrained favourite is no longer a legal
    continuation, which is the model signalling the number is finished.

    Args:
        model: The language model to drive.
        ids: The primed context, extended in place as tokens are chosen.
        tables: The precomputed token tables.
        allow_dot: Whether a decimal point is part of the grammar. Pass `False`
            for an `integer` parameter, so a stray `.` can never be generated in
            the first place.

    Returns:
        The decoded number text (e.g. `"42"` or `"-3.5"`), or `""` if nothing
        valid was produced.
    """
    out = ""
    has_digit = False
    has_dot = False
    while len(out) < _MAX_NUMBER_CHARS:
        logits = _logits(model, ids)
        if out == "":
            allowed = np.concatenate([tables.digit_ids, tables.minus_ids])
            allowed_set = tables.digit_set | tables.minus_set
        elif has_digit and not has_dot and allow_dot:
            allowed = np.concatenate([tables.digit_ids, tables.dot_ids])
            allowed_set = tables.digit_set | tables.dot_set
        else:
            allowed = tables.digit_ids
            allowed_set = tables.digit_set
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


def generate_string(
    model: LLM,
    ids: List[int],
    tables: TokenTables,
    source: str,
) -> str:
    """Constrained-decode a string value by copying a span of `source`.

    Every expected string argument (a path, a query, a template) appears
    verbatim in the user request, so the grammar here is "a contiguous substring
    of `source`". At each step the only legal tokens are those whose text
    extends the copied prefix so it still matches `source` somewhere. We track
    the set of `(start, end)` offset pairs where the copied text currently
    lives, so both boundaries of every surviving occurrence are known exactly. A
    token stays legal only if it continues the text at the end of one of them.
    The seed contains only non-whitespace start positions, so a value can never
    begin with a space. The model still chooses where the span starts and ends
    by its own preferences: the LLM decides, the mask only guarantees the value
    is real. It is physically barred from hallucinating text that is not in the
    request.

    Stopping is decided by quote-as-stop. Once at least one character is copied,
    the closing quote joins the candidate set as an explicit STOP option, and a
    masked argmax runs over the legal continuations plus the quote. If the quote
    wins and is not itself a legal continuation, the model is closing the
    string, so stop. If the quote wins and is a legal continuation (the next
    source character really is a `"`), the known span starts disambiguate. When
    a surviving span was opened by that same quote (the character just before
    its start is `"`), the model is closing its delimiter, so stop without
    copying it. Otherwise it is an embedded quote and is copied like any other
    token. The caller writes the opening and closing quotes of the JSON value
    and `json.dump` applies escaping, so embedded quotes and backslashes copy
    through untouched.

    One deterministic post-step normalises the result. A copied value is
    extended left to its word boundary using the real tracked start, never a
    `find`-style first-occurrence guess, so a span the model started in the
    middle of a word (e.g. dropping a leading `/`) is completed. See
    `_snap_left`.

    Args:
        model: The language model to drive.
        ids: The primed context, extended in place as tokens are chosen.
        tables: The precomputed token tables.
        source: The request text the value is copied out of.

    Returns:
        The copied span text, or `""` if `source` offers no copyable token.
    """
    active: set[tuple[int, int]] = {
        (i, i) for i in range(len(source)) if not source[i].isspace()
    }
    out = ""
    while len(out) < _MAX_STRING_CHARS:
        allowed_set: set[int] = set()
        for end in {e for _, e in active}:
            limit = min(tables.max_token_len, len(source) - end)
            for length in range(1, limit + 1):
                token_ids = tables.text_to_ids.get(source[end:end + length])
                if token_ids:
                    allowed_set.update(token_ids)
        if not allowed_set:
            break
        candidate_set = set(allowed_set)
        if out and tables.quote_id >= 0:
            candidate_set.add(tables.quote_id)
        logits = _logits(model, ids)
        token_id = masked_argmax(
            logits, np.asarray(sorted(candidate_set), dtype=np.int64)
        )
        if token_id == tables.quote_id and out:
            if token_id not in allowed_set:
                break
            if any(s > 0 and source[s - 1] == '"' for s, _ in active):
                break
        piece = tables.id_to_text[token_id]
        ids.append(token_id)
        out += piece
        active = {
            (s, e + len(piece))
            for s, e in active
            if source[e:e + len(piece)] == piece
        }
    return _snap_left(out, source, {s for s, _ in active})


def _snap_left(out: str, source: str, starts: set[int]) -> str:
    """Extend a copied value left to the start of the word it began inside.

    `starts` are the real start offsets of the surviving occurrences of `out` in
    `source`, tracked during generation, never recovered with a first-occurrence
    search, so a repeated substring can never snap against the wrong occurrence.
    If any surviving occurrence already begins at a boundary (position 0, after
    whitespace, or after a value delimiter in `_VALUE_BOUNDARY_CHARS`), the value
    is returned unchanged. Otherwise every occurrence began mid-word: walk left
    from a real start to the nearest boundary and prepend the skipped prefix.

    Args:
        out: The copied value so far.
        source: The request text the value was copied out of.
        starts: The real start offsets of the surviving occurrences of `out`.

    Returns:
        The value, extended left to its word boundary where needed.
    """
    if not out or not starts:
        return out
    for start in starts:
        prev = source[start - 1] if start > 0 else ""
        if start == 0 or prev.isspace() or prev in _VALUE_BOUNDARY_CHARS:
            return out
    start = min(starts)
    j = start
    while (
        j > 0
        and not source[j - 1].isspace()
        and source[j - 1] not in _VALUE_BOUNDARY_CHARS
    ):
        j -= 1
    return source[j:start] + out


def generate_string_free(
    model: LLM,
    ids: List[int],
    tables: TokenTables,
    stop_on_space: bool = False,
) -> str:
    """Constrained-decode a free (non-extractive) string value.

    Fallback for values that are not substrings of the request and therefore
    unreachable by span-copy, typically a value the model must infer, such as a
    regex pattern. Generation is constrained to string-safe tokens (no raw
    quote, no control characters) so the value always fits inside a JSON string,
    with the closing quote as the explicit STOP option once at least one
    character exists (the same quote-as-stop idiom as span-copy). A first
    token's leading space is stripped, and a light repetition guard (four
    identical consecutive tokens) breaks the runaway loops a small model is
    prone to on regex-like output.

    Args:
        model: The language model to drive.
        ids: The primed context, extended in place as tokens are chosen.
        tables: The precomputed token tables.
        stop_on_space: When `True`, a whitespace-bearing token also ends the
            value once at least one character exists. This is a tighter grammar
            for values that are compact by nature, like a regex pattern. A small
            model that fails to emit the closing quote drifts into rambling
            precisely at a whitespace boundary, so whitespace acts as a second
            stop symbol there.

    Returns:
        The generated text, stripped of surrounding whitespace, or `""` if
        nothing was produced.
    """
    if tables.string_or_quote.size == 0:
        return ""
    no_quote = tables.string_or_quote[tables.string_or_quote != tables.quote_id]
    out = ""
    last_piece = ""
    repeats = 0
    for _ in range(_MAX_STRING_CHARS):
        if len(out) >= _MAX_STRING_CHARS:
            break
        allowed = tables.string_or_quote if out else no_quote
        if allowed.size == 0:
            break
        logits = _logits(model, ids)
        token_id = masked_argmax(logits, allowed)
        if out and token_id == tables.quote_id:
            break
        piece = tables.id_to_text[token_id]
        if stop_on_space and out and any(ch.isspace() for ch in piece):
            break
        if piece == last_piece:
            repeats += 1
        else:
            last_piece, repeats = piece, 1
        if repeats > 3:
            break
        ids.append(token_id)
        if not out and piece.startswith(" "):
            piece = piece[1:]
        out += piece
    return out.strip()


def decode_one_of(
    model: LLM,
    ids: List[int],
    options: Sequence[str],
    tables: TokenTables,
) -> str:
    """Force the model to emit exactly one whole string from `options`.

    A prefix state machine, where `prefix` is what has been spelled so far. At
    each step the only legal tokens are those that extend `prefix` toward some
    option without overshooting its end. The model still chooses which option by
    its own preferences at every branch point (the subject's rule: the LLM
    decides, the mask only guarantees the answer is a real, correctly-spelled
    option).

    When one option is a strict prefix of another (`fn_reverse` vs
    `fn_reverse_words`), the shorter one stays selectable through a natural stop,
    mirroring `generate_number`. Once `prefix` is itself a complete option,
    generation ends and returns it as soon as the model's own unconstrained
    favourite is not a token extending toward a strictly longer candidate.

    Args:
        model: The language model to drive.
        ids: The primed context, extended in place as tokens are chosen.
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
        complete = prefix in options
        if complete and not any(len(opt) > len(prefix) for opt in candidates):
            return prefix
        if len(prefix) > _MAX_NAME_CHARS:
            return candidates[0]

        logits = _logits(model, ids)
        first = prefix == ""
        allowed: List[int] = []
        for token_id, piece, had_space in tables.name_candidates:
            if had_space and not first:
                continue
            for candidate in candidates:
                if candidate[len(prefix):].startswith(piece):
                    allowed.append(token_id)
                    break
        if not allowed:
            return prefix if complete else candidates[0]
        if complete and int(np.argmax(logits)) not in set(allowed):
            return prefix

        token_id = masked_argmax(logits, np.asarray(allowed, dtype=np.int64))
        text = tables.id_to_text[token_id]
        piece = text[1:] if (first and text.startswith(" ")) else text
        ids.append(token_id)
        prefix += piece
