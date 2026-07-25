"""Stage B: fill the chosen function's typed arguments under constraint.

The key design decision answers the subject's warning "do not rely on the model
spontaneously producing JSON" directly: we do not ask the model for the JSON
skeleton. The schema fully determines the braces, the keys and the punctuation,
so there is nothing for the model to decide there. We write them. The model is
invoked only for the unknown parts, the typed values.

So the structural layer is deterministic (always valid) and the schema layer is
constrained generation (right keys, right value types). We assemble a plain
Python `dict` of parameters, and `json.dump` later turns it into
guaranteed-valid, properly-escaped JSON.

One string sub-case gets extra care: a parameter whose schema name declares it
to be a regular expression (`regex`, `pattern`). A regex is often not a
substring of the request ("Replace all numbers ..." never spells `[0-9]+`), so
span-copy alone cannot reach it. For those parameters only, and only after every
parameter is decoded so the sibling corpus never depends on schema order, the
extractive copy is validated as a regex against the call's other string
arguments, and a second constrained generation may infer a pattern in a
dedicated few-shot context. The inferred candidate replaces the extractive copy
only on strict improvement (it demonstrably matches more), so a failed inference
can never make the result worse. See `_resolve_pattern_value`.

Candidate patterns are untrusted input (a prompt span or raw model output), so
every `re` call on one is defended: a catastrophic-backtracking shape filter,
corpus slicing, and a hard `SIGALRM` time cap. A bad or hostile pattern degrades
to the extractive fallback. It can never hang or crash the run.
"""

from __future__ import annotations

import re
import signal
from contextlib import contextmanager
from types import FrameType
from typing import Any, Dict, Iterator, List, Optional

from .decoder import (
    decode_one_of,
    generate_number,
    generate_string,
    generate_string_free,
)
from .models import FunctionDefinition
from .protocols import LLM
from .selector import encode_to_ids
from .vocab import TokenTables


def _emit(model: LLM, ids: List[int], text: str) -> None:
    """Append a known structural fragment to the running context.

    Args:
        model: The language model whose tokenizer encodes the fragment.
        ids: The running context, extended in place.
        text: The structural fragment to append (a brace, key or separator).
    """
    ids.extend(encode_to_ids(model, text))


_PATTERN_HINTS = ("regex", "pattern")


def _is_pattern_parameter(name: str) -> bool:
    """True when the schema declares this parameter to be a regex.

    The decision reads only the parameter name from the function definition
    (`regex`, `search_pattern`, ...), never the user prompt. This is schema
    awareness, not keyword matching on the request.

    Args:
        name: The parameter name from the function definition.

    Returns:
        True when the name marks a regex parameter.
    """
    lowered = name.lower()
    return any(hint in lowered for hint in _PATTERN_HINTS)


_REGEX_TIMEOUT_SECONDS = 1

_VALIDATION_SLICE = 64

_PATHOLOGICAL = re.compile(r"\([^)]*[+*][^)]*\)\s*[+*{]")


class _RegexTimeout(Exception):
    """Raised when a candidate pattern exceeds its validation time budget."""


def _raise_regex_timeout(signum: int, frame: Optional[FrameType]) -> None:
    """Handle `SIGALRM`: abort a regex validation that ran too long.

    Args:
        signum: The signal number.
        frame: The interrupted stack frame.

    Raises:
        _RegexTimeout: Always, to unwind the timed-out validation.
    """
    raise _RegexTimeout()


@contextmanager
def _time_limit(seconds: int) -> Iterator[None]:
    """Hard-cap the enclosed block. Degrade to a no-op where alarms can't work.

    `signal.SIGALRM` interrupts even C-level regex backtracking, but it is
    POSIX-only and usable only on the main thread (`signal.signal` raises
    `ValueError` elsewhere). In those environments the alarm is skipped, never
    crashing, and the block relies on the other defences: the pathological-shape
    filter, corpus slicing, `re.error` handling. A `finally` always restores the
    previous handler and clears the alarm.

    Args:
        seconds: The time budget for the enclosed block.

    Yields:
        None. The block runs inside the time cap.
    """
    armed = False
    previous: Any = None
    if hasattr(signal, "SIGALRM"):
        try:
            previous = signal.signal(signal.SIGALRM, _raise_regex_timeout)
            signal.alarm(seconds)
            armed = True
        except ValueError:
            armed = False
    try:
        yield
    finally:
        if armed:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)


def _compiles(candidate: str) -> bool:
    """True when the candidate is a non-empty regex that compiles.

    Compilation alone never executes a match, so the backtracking smell filter
    does not apply here. The time cap still does, as a belt against pathological
    compile costs, and `re.error` covers malformed output.

    Args:
        candidate: The pattern text to test.

    Returns:
        True when the candidate is non-empty and compiles within the time cap.
    """
    if not candidate:
        return False
    try:
        with _time_limit(_REGEX_TIMEOUT_SECONDS):
            re.compile(candidate)
    except (re.error, _RegexTimeout):
        return False
    return True


def _regex_hits(candidate: str, corpus: List[str]) -> int:
    """Count how often a candidate regex matches the call's own arguments.

    Pure output validation with the stdlib `re` module: compile the candidate
    and count its non-overlapping matches across the corpus (the call's other
    string arguments, e.g. the `source_string` a substitution will run over). A
    candidate that does not compile, or is empty, scores `0`. It would do
    nothing (or anything) at call time. The count matters, not just a boolean: a
    value that matches repeatedly demonstrates pattern behaviour, while a single
    hit may simply be the literal text the value was copied out of.

    The candidate is untrusted (a prompt span or raw model output), so the match
    run is defended in depth. An obviously pathological shape (a quantified
    group itself quantified) scores `0` outright, every corpus string is sliced
    to `_VALIDATION_SLICE` characters, and the whole run sits under a hard
    `SIGALRM` cap. A timeout also scores `0`: a pattern too expensive to
    validate is treated exactly like one that does not compile, and the caller
    falls back to the extractive value.

    Args:
        candidate: The pattern text to score.
        corpus: The call's other string arguments to match against.

    Returns:
        The number of non-overlapping matches, or `0` if the candidate is
        empty, pathological, non-compiling or times out.
    """
    if not candidate or _PATHOLOGICAL.search(candidate):
        return 0
    try:
        with _time_limit(_REGEX_TIMEOUT_SECONDS):
            compiled = re.compile(candidate)
            return sum(
                len(tuple(compiled.finditer(text[:_VALIDATION_SLICE])))
                for text in corpus
            )
    except (re.error, _RegexTimeout):
        return 0


def _to_number(raw: str, param_type: str) -> Any:
    """Parse decoded number text according to the schema `param_type`.

    The schema type is authoritative, not the surface form. A `number` is always
    a `float` (`"3"` becomes `3.0`, so it serialises as a JSON float) and an
    `integer` is always an `int`. An integer with no decimal point is parsed
    with `int` directly, never through `float`, which silently rounds past 15-16
    significant digits.

    Args:
        raw: The decoded number text.
        param_type: The schema type, `number` or `integer`.

    Returns:
        The parsed value as the matching Python type, or a typed `0` fallback
        when the text is empty or unparseable.
    """
    fallback: Any = 0.0 if param_type == "number" else 0
    if not raw or raw in {"-", ".", "-."}:
        return fallback
    try:
        if param_type == "integer":
            return int(raw) if "." not in raw else int(float(raw))
        return float(raw)
    except ValueError:
        return fallback


def build_arguments_prompt(prompt: str, function: FunctionDefinition) -> str:
    """Build the priming text that frames the value-extraction task.

    Two framing devices steer the model's start/stop choices inside the mask.
    First, an instruction demanding a verbatim, character-for-character copy.
    Second, a single fictional worked example that demonstrates both boundary
    behaviours at once: a quote-delimited value copied without its quotes (stop
    at the closing delimiter) and a path copied with its leading `/` (start at
    the true first character). The example is generic on purpose. It names no
    real function or value, and the constraint mask guarantees the generated
    value is a substring of the real request, so the example can bias boundaries
    but can never leak into the output.

    Args:
        prompt: The natural-language user request.
        function: The chosen function, whose name and description frame the task.

    Returns:
        The priming text, ending on `Arguments as JSON: `.
    """
    return (
        f"Extract the arguments for the function `{function.name}` "
        f"from the user request.\n"
        f"Copy every argument value verbatim from the request, character for "
        f"character, from its very first character to its last. Include any "
        f"leading symbols such as '/', '\\' or '.'. When a value is wrapped "
        f"in quotes in the request, copy only the text between the quotes.\n"
        f"Example request: Save the note 'buy milk' to /var/notes/list.txt\n"
        f'Example arguments as JSON: {{"text": "buy milk", '
        f'"location": "/var/notes/list.txt"}}\n'
        f"Function description: {function.description}\n"
        f"User request: {prompt}\n"
        f"Arguments as JSON: "
    )


def build_pattern_prompt(prompt: str) -> str:
    """Build the few-shot priming text for inferring a regex value.

    The examples teach the model general regex idioms only: bracket character
    classes and plain literals, backslash-free on purpose, because a small model
    reproduces escape sequences in doubled (re-escaped) form that no longer
    matches anything. None of the examples name a real function or a real test
    input. The priming ends on an opening quote so the closing quote is the
    natural stop for `generate_string_free`.

    Args:
        prompt: The natural-language user request.

    Returns:
        The few-shot priming text, ending on an opening quote.
    """
    return (
        "Write one regular expression pattern for the request.\n"
        "Prefer a character class in square brackets. Never use a "
        "backslash. Keep the pattern short.\n"
        "Request: replace all numbers in a text\n"
        'Pattern: "[0-9]+"\n'
        "Request: replace all lowercase letters in a text\n"
        'Pattern: "[a-z]"\n'
        "Request: replace the word apple in a text\n"
        'Pattern: "apple"\n'
        f"Request: {prompt}\n"
        'Pattern: "'
    )


def _infer_pattern(model: LLM, prompt: str, tables: TokenTables) -> str:
    """Constrained-generate a regex candidate in its own few-shot context.

    Runs in a fresh context (not the running JSON argument context), so the
    inference can be primed with regex examples without disturbing the token
    stream the remaining argument values are decoded against. The pattern
    grammar is compact by design (`stop_on_space`): whitespace ends the value,
    because a small model that misses the closing quote starts rambling exactly
    at a whitespace boundary, and the few-shot patterns are all space-free.

    Args:
        model: The language model to drive.
        prompt: The natural-language user request.
        tables: The precomputed token tables.

    Returns:
        The inferred regex candidate, possibly `""`.
    """
    ids = encode_to_ids(model, build_pattern_prompt(prompt))
    return generate_string_free(model, ids, tables, stop_on_space=True)


def _resolve_pattern_value(
    model: LLM,
    prompt: str,
    tables: TokenTables,
    extractive: str,
    corpus: List[str],
) -> str:
    """Pick the value for a pattern parameter: extractive unless strictly beaten.

    The extractive span-copy is the conservative baseline. It is real text from
    the request, exactly what the plain string branch would have produced, so
    the invariant holds: a failed inference can never make the result worse. An
    inferred candidate replaces the copy only on strict, demonstrated
    improvement.

    With a corpus (this call's other string arguments, typically the text a
    substitution will run over), both values are scored by `_regex_hits`
    (time-bounded; a non-compiling, pathological or timed-out value scores `0`)
    and the inferred candidate wins only when it matches strictly more often
    than the extractive copy. So a literal like `cat` copied from a request
    about a text with two `cat`s stays (a tie is not an improvement), while
    `[0-9]+` beats a copied `34 I'm 233` that only ever finds itself once. A
    doubled-escape artefact (`\\\\d+`), a hallucinated literal, or a glob like
    `*.txt` that does not compile at all scores `0` and can never win.

    Without a corpus (the pattern is the function's only string parameter),
    semantic validation is impossible, so the extractive copy is kept whenever
    it compiles. A verbatim `[0-9]+` spelled in the prompt must survive.
    Inference runs only when the copy does not even compile, and its result is
    accepted only if it compiles itself.

    Args:
        model: The language model to drive.
        prompt: The natural-language user request.
        tables: The precomputed token tables.
        extractive: The value copied out of the request by span-copy.
        corpus: The call's other string arguments, for scoring candidates.

    Returns:
        The chosen pattern value: the inferred candidate on strict improvement,
        otherwise the extractive copy.
    """
    if not corpus:
        if _compiles(extractive):
            return extractive
        candidate = _infer_pattern(model, prompt, tables)
        if _compiles(candidate):
            return candidate
        return extractive
    extractive_hits = _regex_hits(extractive, corpus)
    candidate = _infer_pattern(model, prompt, tables)
    if _regex_hits(candidate, corpus) > extractive_hits:
        return candidate
    return extractive


def extract_arguments(
    model: LLM,
    prompt: str,
    function: FunctionDefinition,
    tables: TokenTables,
) -> Dict[str, Any]:
    """Constrained-decode the argument object for the function.

    Args:
        model: The language model to drive.
        prompt: The user request, which gives the model context for the values.
        function: The chosen function. Its schema drives the keys and types.
        tables: The precomputed token tables.

    Returns:
        A parameters dict with every declared key present and correctly typed.
    """
    ids = encode_to_ids(model, build_arguments_prompt(prompt, function))
    params: Dict[str, Any] = {}
    pattern_params: List[str] = []
    items = list(function.parameters.items())

    _emit(model, ids, "{")
    for index, (name, spec) in enumerate(items):
        _emit(model, ids, f'"{name}": ')
        if spec.type in ("number", "integer"):
            raw = generate_number(
                model, ids, tables, allow_dot=spec.type != "integer"
            )
            params[name] = _to_number(raw, spec.type)
        elif spec.type == "boolean":
            choice = decode_one_of(model, ids, ["true", "false"], tables)
            params[name] = choice == "true"
        else:
            _emit(model, ids, '"')
            value = generate_string(model, ids, tables, prompt)
            if value == "":
                value = generate_string_free(model, ids, tables)
            _emit(model, ids, '"')
            if _is_pattern_parameter(name):
                pattern_params.append(name)
            params[name] = value
        if index < len(items) - 1:
            _emit(model, ids, ", ")
    _emit(model, ids, "}")

    if pattern_params:
        corpus = [
            text
            for key, text in params.items()
            if isinstance(text, str) and not _is_pattern_parameter(key)
        ]
        for name in pattern_params:
            params[name] = _resolve_pattern_value(
                model, prompt, tables, params[name], corpus
            )
    return params
