"""Unit tests for the constrained-decoding engine.

These prove the *logic* without downloading the real model: a ``FakeModel``
returns scripted logits over a tiny hand-written vocabulary, so each generator's
behaviour is fully deterministic and checkable offline.  This is the backbone of
the project's testing strategy — the masking math is what must be correct; the
real model only supplies preferences on top of it.
"""

from __future__ import annotations

import threading
import time
from typing import Dict, List

import numpy as np

from src.arguments import (
    _is_pattern_parameter,
    _regex_hits,
    _resolve_pattern_value,
    _to_number,
    extract_arguments,
)
from src.decoder import (
    decode_one_of,
    generate_number,
    generate_string,
    generate_string_free,
    masked_argmax,
)
from src.models import FunctionDefinition, ParameterSpec
from src.vocab import tables_from_id_to_text

# A minimal vocabulary covering digits, a quote, and a few name/letter tokens.
VOCAB: List[str] = [
    "", '"', "-", ".",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "fn", "_", "g", "a", "reet", "add", "greet", "h", "i", "x", " ",
    "/", "b",
    "o", "ring", "rev", "erse", "words", " h",
]


class FakeModel:
    """A scripted stand-in for ``Small_LLM_Model`` (satisfies the ``LLM`` Protocol)."""

    def __init__(self, vocab: List[str], wants: List[str], stop: str = "x") -> None:
        self._vocab = vocab
        self._first_id: Dict[str, int] = {}
        for token_id, text in enumerate(vocab):
            self._first_id.setdefault(text, token_id)
        self._wants = wants
        self._stop = stop
        self._step = 0

    def get_logits_from_input_ids(self, input_ids: List[int]) -> List[float]:
        """Return logits that strongly favour the next scripted token."""
        logits = [0.0] * len(self._vocab)
        target = self._wants[self._step] if self._step < len(self._wants) else self._stop
        logits[self._first_id[target]] = 10.0
        self._step += 1
        return logits

    def encode(self, text: str) -> np.ndarray:
        """Stub encoder — the scripted logits ignore context."""
        return np.array([[0]])

    def decode(self, ids: List[int]) -> str:
        """Decode ids back to text (unused by these tests, kept for the Protocol)."""
        return "".join(self._vocab[i] for i in ids)

    def get_path_to_vocab_file(self) -> str:
        """No vocab file in tests."""
        raise OSError("no vocab file in tests")


class ScriptedLogitsModel(FakeModel):
    """A ``FakeModel`` whose *full* logit row is scripted per step.

    Where ``FakeModel`` boosts a single token, this variant ranks several at
    once — needed to test quote-as-stop, where the outcome depends on how an
    illegal favourite, a legal continuation and the closing quote are ordered
    against each other.
    """

    def __init__(self, vocab: List[str], rows: List[Dict[str, float]]) -> None:
        super().__init__(vocab, wants=[])
        self._rows = rows

    def get_logits_from_input_ids(self, input_ids: List[int]) -> List[float]:
        """Return the scripted logit row for the current step."""
        logits = [0.0] * len(self._vocab)
        row = self._rows[self._step] if self._step < len(self._rows) else {}
        for text, value in row.items():
            logits[self._first_id[text]] = value
        self._step += 1
        return logits


def test_masked_argmax_ignores_forbidden_tokens() -> None:
    """The masked argmax must pick the best *allowed* id, never a higher illegal one."""
    logits = np.array([5.0, 9.0, 1.0, 7.0])
    allowed = np.array([0, 2, 3])  # index 1 (the global max) is forbidden
    assert masked_argmax(logits, allowed) == 3


def test_generate_number_decodes_integer_and_stops() -> None:
    """Digits are forced; when the model's free choice leaves digits, it stops."""
    tables = tables_from_id_to_text(VOCAB)
    model = FakeModel(VOCAB, wants=["2", "3"], stop="x")
    assert generate_number(model, [], tables) == "23"


def test_generate_string_copies_a_span_and_stops_on_quote() -> None:
    """The string decoder copies a contiguous span of the source, then stops
    when the model prefers the closing quote over every legal continuation."""
    tables = tables_from_id_to_text(VOCAB)
    model = FakeModel(VOCAB, wants=["h", "i"], stop='"')
    assert generate_string(model, [], tables, "say hi now") == "hi"


def test_generate_string_masks_non_substring_tokens() -> None:
    """A token that is not a span continuation is impossible, however preferred."""
    tables = tables_from_id_to_text(VOCAB)
    # The model "wants" `greet`, but the source never contains it, so the mask
    # forces a real substring instead; choosing the closing quote next then
    # ends the value.
    model = FakeModel(VOCAB, wants=["greet"], stop='"')
    assert generate_string(model, [], tables, "hi") == "h"


def test_generate_string_copies_an_embedded_quote() -> None:
    """A quote inside the source span is copyable — it is not a terminator."""
    tables = tables_from_id_to_text(VOCAB)
    model = FakeModel(VOCAB, wants=['"', "h", "i", '"'], stop='"')
    assert generate_string(model, [], tables, 'say "hi" now') == '"hi"'


def test_generate_string_continues_past_an_illegal_favourite() -> None:
    """Mid-span, an illegal raw favourite no longer ends the value.

    The old heuristic broke as soon as the unconstrained argmax left the legal
    set.  Under quote-as-stop the choice is between the legal continuations and
    the closing quote; here the continuation outranks the quote, so copying
    carries on, and the span ends only when the quote itself wins.
    """
    tables = tables_from_id_to_text(VOCAB)
    rows = [
        {"h": 10.0},
        {"x": 10.0, "i": 5.0, '"': 1.0},  # raw favourite illegal — keep going
        {'"': 10.0, "a": 1.0},  # quote beats the continuation — genuine stop
    ]
    model = ScriptedLogitsModel(VOCAB, rows)
    assert generate_string(model, [], tables, "hia") == "hi"


def test_generate_string_quote_stop_requires_one_copied_char() -> None:
    """The stop option is never offered before the first char is copied."""
    tables = tables_from_id_to_text(VOCAB)
    rows = [
        {'"': 10.0, "h": 1.0, "i": 0.5},  # no quote in source: must copy "h"
        {'"': 10.0},  # now the quote may end the span
    ]
    model = ScriptedLogitsModel(VOCAB, rows)
    assert generate_string(model, [], tables, "hi") == "h"


def test_generate_string_embedded_quote_continues_not_stops() -> None:
    """A winning quote that IS a legal continuation is copied, not a stop."""
    tables = tables_from_id_to_text(VOCAB)
    rows = [
        {"h": 10.0},
        {'"': 10.0},  # the next source char is a quote: copy it, do not stop
        {"i": 10.0},
        {'"': 10.0, "a": 1.0},  # no longer a continuation: genuine stop
    ]
    model = ScriptedLogitsModel(VOCAB, rows)
    assert generate_string(model, [], tables, 'h"ia') == 'h"i'


def test_generate_string_snaps_left_to_the_word_boundary() -> None:
    """A span started mid-word is completed leftward to its word boundary.

    The model starts copying at the inner ``a`` of ``/a/b`` (preceded by
    ``/``, not a boundary) and copies ``a/b``.  The snap-left post-step sees
    the mid-word start and prepends the skipped prefix up to the nearest
    boundary (the space before ``/``), restoring the full value ``/a/b``.
    """
    tables = tables_from_id_to_text(VOCAB)
    rows = [
        {"a": 10.0, "/": 5.0},  # start mid-word, inside "/a/b"
        {"/": 10.0},
        {"b": 10.0},
        {'"': 10.0, "h": 1.0},  # genuine stop
    ]
    model = ScriptedLogitsModel(VOCAB, rows)
    assert generate_string(model, [], tables, "read /a/b here") == "/a/b"


def test_generate_string_no_snap_when_already_at_a_boundary() -> None:
    """A value that already starts right after a boundary is untouched.

    Here the copy starts at ``h`` of ``hi``, preceded by a space — a genuine
    value boundary — so the snap-left post-step changes nothing.
    """
    tables = tables_from_id_to_text(VOCAB)
    rows = [
        {"h": 10.0},
        {'"': 10.0},  # genuine stop
    ]
    model = ScriptedLogitsModel(VOCAB, rows)
    assert generate_string(model, [], tables, "say hi") == "h"


def test_generate_string_returns_empty_when_nothing_is_copyable() -> None:
    """A source with no vocabulary match yields an empty value, never a crash."""
    tables = tables_from_id_to_text(VOCAB)
    model = FakeModel(VOCAB, wants=[], stop="x")
    assert generate_string(model, [], tables, "") == ""


def test_decode_one_of_walks_to_the_chosen_option() -> None:
    """The name state machine yields exactly one real option, chosen at each branch."""
    tables = tables_from_id_to_text(VOCAB)
    options = ["fn_add_numbers", "fn_greet"]
    model = FakeModel(VOCAB, wants=["fn", "_", "g", "reet"])
    assert decode_one_of(model, [], options, tables) == "fn_greet"


def test_tables_group_token_ids_by_type() -> None:
    """The token tables sort ids into the right typed buckets."""
    tables = tables_from_id_to_text(VOCAB)
    assert tables.quote_id == VOCAB.index('"')
    assert VOCAB.index("5") in tables.digit_set
    assert VOCAB.index("fn") not in tables.digit_set


def test_snap_left_uses_the_real_span_start_not_first_occurrence() -> None:
    """Snap-left must never guess the span with a first-occurrence search.

    ``ring`` occurs twice in ``Reverse the string 'ring'``: inside ``string``
    (mid-word) and as the quoted value (after ``'``, a boundary).  The tracked
    start after the quote wins, so the value stays ``ring`` — a ``find``-based
    snap would match inside ``string`` first and corrupt it to ``string``.
    """
    tables = tables_from_id_to_text(VOCAB)
    model = FakeModel(VOCAB, wants=["ring"], stop='"')
    assert generate_string(model, [], tables, "Reverse the string 'ring'") == "ring"


def test_generate_string_stops_at_the_closing_delimiter_quote() -> None:
    """A quote closing a quote-delimited span is a stop, never absorbed.

    In ``Greet "bob" now`` the copied span starts right after an opening
    ``"``; when the model then picks the quote (which is *also* a legal
    continuation), it is closing that delimiter — the value is ``bob``,
    not ``bob"``.
    """
    tables = tables_from_id_to_text(VOCAB)
    model = FakeModel(VOCAB, wants=["b", "o", "b", '"'], stop='"')
    assert generate_string(model, [], tables, 'Greet "bob" now') == "bob"


def test_generate_string_copies_internal_quotes_of_an_outer_value() -> None:
    """A span NOT opened by a quote keeps its internal quotes.

    The span starts after a space (no opening ``"``), so a winning quote that
    is a legal continuation is embedded content and is copied through.
    """
    tables = tables_from_id_to_text(VOCAB)
    model = FakeModel(VOCAB, wants=["h", '"', "i", '"', '"'], stop='"')
    assert generate_string(model, [], tables, 'say h"i" x') == 'h"i"'


def test_generate_string_never_starts_on_whitespace() -> None:
    """A first token beginning with whitespace can never seed a span.

    Only non-whitespace source positions are seeded, so the model's leading-
    space favourite (`` h``) is masked out and the value never starts with a
    space; the copy lands elsewhere and snaps to its real word start.
    """
    tables = tables_from_id_to_text(VOCAB)
    model = FakeModel(VOCAB, wants=[" h"], stop='"')
    result = generate_string(model, [], tables, "say hi")
    assert not result.startswith(" ")
    assert result == "sa"


def test_generate_string_free_stops_on_quote_and_strips_leading_space() -> None:
    """Free generation infers a value, ends on the quote, drops the Ġ-space."""
    tables = tables_from_id_to_text(VOCAB)
    model = FakeModel(VOCAB, wants=[" h", "i"], stop='"')
    assert generate_string_free(model, [], tables) == "hi"


def test_generate_string_free_stop_on_space_ends_a_compact_value() -> None:
    """With ``stop_on_space`` a whitespace token is a stop, not content.

    Default behaviour keeps the space (free text may contain spaces); the
    compact-value grammar ends at it instead — the anti-rambling stop used
    for regex inference.
    """
    tables = tables_from_id_to_text(VOCAB)
    model = FakeModel(VOCAB, wants=["h", " h", "i"], stop='"')
    assert generate_string_free(model, [], tables) == "h hi"
    model = FakeModel(VOCAB, wants=["h", " h", "i"], stop='"')
    assert generate_string_free(model, [], tables, stop_on_space=True) == "h"


def test_generate_string_free_breaks_a_repetition_runaway() -> None:
    """Four identical consecutive tokens end free generation (runaway guard)."""
    tables = tables_from_id_to_text(VOCAB)
    model = FakeModel(VOCAB, wants=["a"] * 20, stop="a")
    assert generate_string_free(model, [], tables) == "aaa"


def test_extract_arguments_falls_back_to_free_generation() -> None:
    """An empty span-copy result triggers the non-extractive fallback.

    With an empty prompt nothing is copyable, so the string value comes from
    free constrained generation instead of staying empty forever.
    """
    tables = tables_from_id_to_text(VOCAB)
    function = FunctionDefinition(
        name="fn_x",
        description="",
        parameters={"note": ParameterSpec(type="string")},
    )
    model = FakeModel(VOCAB, wants=["add"], stop='"')
    params = extract_arguments(model, "", function, tables)
    assert params == {"note": "add"}


def test_extract_arguments_routes_unknown_type_to_string() -> None:
    """An unrecognised parameter type must not crash — it becomes a string."""
    tables = tables_from_id_to_text(VOCAB)
    function = FunctionDefinition(
        name="fn_x",
        description="",
        parameters={"items": ParameterSpec(type="array")},
    )
    model = FakeModel(VOCAB, wants=["h", "i"], stop='"')
    params = extract_arguments(model, "say hi", function, tables)
    assert params == {"items": "hi"}


def test_decode_one_of_can_select_a_prefix_option() -> None:
    """A shorter option that prefixes a longer one stays selectable.

    Once the prefix equals ``fn_reverse`` and the model's free favourite does
    not extend toward ``fn_reverse_words``, the natural stop returns the
    shorter name instead of forcing the longer one.
    """
    tables = tables_from_id_to_text(VOCAB)
    options = ["fn_reverse", "fn_reverse_words"]
    model = FakeModel(VOCAB, wants=["fn", "_", "rev", "erse"], stop="x")
    assert decode_one_of(model, [], options, tables) == "fn_reverse"


def test_decode_one_of_can_still_walk_past_the_prefix() -> None:
    """When the model keeps extending, the longer option is still reachable."""
    tables = tables_from_id_to_text(VOCAB)
    options = ["fn_reverse", "fn_reverse_words"]
    model = FakeModel(VOCAB, wants=["fn", "_", "rev", "erse", "_", "words"])
    assert decode_one_of(model, [], options, tables) == "fn_reverse_words"


def test_generate_number_integer_never_takes_a_dot() -> None:
    """With ``allow_dot=False`` the decimal point is out of the grammar."""
    tables = tables_from_id_to_text(VOCAB)
    model = FakeModel(VOCAB, wants=["2", ".", "5"], stop="x")
    assert generate_number(model, [], tables, allow_dot=False) == "2"
    model = FakeModel(VOCAB, wants=["2", ".", "5"], stop="x")
    assert generate_number(model, [], tables, allow_dot=True) == "2.5"


def test_is_pattern_parameter_reads_only_the_schema_name() -> None:
    """Pattern detection is schema-based: the parameter *name*, nothing else."""
    assert _is_pattern_parameter("regex")
    assert _is_pattern_parameter("Pattern")
    assert _is_pattern_parameter("search_regex")
    assert not _is_pattern_parameter("source_string")
    assert not _is_pattern_parameter("replacement")


def test_regex_hits_counts_matches_and_rejects_broken_patterns() -> None:
    """Validation counts real matches in the call's own string arguments."""
    assert _regex_hits("[0-9]+", ["Hello 34 and 233"]) == 2
    assert _regex_hits("cat", ["The cat sat by a cat"]) == 2
    assert _regex_hits("34 I'm 233", ["Hello 34 I'm 233 years old"]) == 1
    assert _regex_hits("(", ["Hello"]) == 0  # does not compile
    assert _regex_hits("dog", ["The cat sat"]) == 0  # matches nothing
    assert _regex_hits("", ["Hello"]) == 0  # empty is never usable
    # The doubled-escape artefact a small model produces (`\\d+` re-escaped)
    # compiles but matches nothing, so validation rejects it.
    assert _regex_hits("\\\\d+", ["Hello 34"]) == 0


def test_regex_guard_rejects_a_pathological_shape_instantly() -> None:
    """A nested-quantifier candidate is refused before ``re`` ever runs it.

    ``(a+)+b`` against a run of ``a``s is the classic catastrophic-
    backtracking bomb; the shape filter scores it 0 without matching, so the
    call returns immediately instead of burning seconds (or hanging).
    """
    start = time.monotonic()
    assert _regex_hits("(a+)+b", ["a" * 200]) == 0
    assert _regex_hits("(x*y)*z", ["x" * 200]) == 0
    assert time.monotonic() - start < 0.5


def test_regex_guard_time_caps_catastrophic_backtracking() -> None:
    """A bomb that slips past the shape filter is cut off by the alarm.

    ``(a|aa)+$`` carries no quantifier *inside* the group, so the smell
    filter passes it — but it backtracks exponentially on a near-miss input.
    The SIGALRM cap must abort the match within its 1-second budget and score
    the candidate 0 (treated as invalid), never hang the run.
    """
    start = time.monotonic()
    assert _regex_hits("(a|aa)+$", ["a" * 63 + "b"]) == 0
    assert time.monotonic() - start < 5.0


def test_regex_guard_degrades_gracefully_off_the_main_thread() -> None:
    """Off the main thread the alarm cannot arm; validation still works.

    ``signal.signal`` raises ``ValueError`` outside the main thread; the
    guard must swallow that (not crash) and still count matches for a benign
    pattern.
    """
    results: List[int] = []
    worker = threading.Thread(
        target=lambda: results.append(_regex_hits("[0-9]+", ["a 12 b 34"]))
    )
    worker.start()
    worker.join()
    assert results == [2]


def test_regex_hits_slices_the_corpus_for_validation() -> None:
    """Matching runs on a bounded prefix of each corpus string.

    Defence in depth: backtracking cost grows with input length, so hits are
    counted on the first ``_VALIDATION_SLICE`` characters only.  A match
    beyond the slice is invisible — an accepted, deliberate trade: both
    candidates are scored on the same slice, so the comparison stays fair.
    """
    assert _regex_hits("b", ["a" * 100 + "b"]) == 0
    assert _regex_hits("a", ["a" + "c" * 100]) == 1


def test_pattern_parameter_uses_a_validated_free_candidate() -> None:
    """A pattern param whose span-copy fails validation takes the inferred one.

    ``src`` span-copies ``hi``.  For ``regex`` the span-copy lands on ``sa``
    (snapped from the ``a`` of ``say``), which matches nothing in ``hi`` —
    no pattern behaviour.  The few-shot inference then produces ``h``, which
    compiles and matches the corpus, so it is accepted.
    """
    tables = tables_from_id_to_text(VOCAB)
    function = FunctionDefinition(
        name="fn_sub",
        description="",
        parameters={
            "src": ParameterSpec(type="string"),
            "regex": ParameterSpec(type="string"),
        },
    )
    model = FakeModel(VOCAB, wants=["h", "i", "a", "h"], stop='"')
    params = extract_arguments(model, "say hi", function, tables)
    assert params == {"src": "hi", "regex": "h"}


def test_pattern_parameter_falls_back_to_span_copy_on_garbage() -> None:
    """An inferred candidate that matches nothing is discarded.

    The span-copy for ``regex`` is ``sa`` (invalid against ``hi``), and the
    inferred candidate ``b`` also matches nothing — so the extractive value is
    kept, exactly the behaviour the branch would have had without pattern
    awareness.
    """
    tables = tables_from_id_to_text(VOCAB)
    function = FunctionDefinition(
        name="fn_sub",
        description="",
        parameters={
            "src": ParameterSpec(type="string"),
            "regex": ParameterSpec(type="string"),
        },
    )
    model = FakeModel(VOCAB, wants=["h", "i", "a", "b"], stop='"')
    params = extract_arguments(model, "say hi", function, tables)
    assert params == {"src": "hi", "regex": "sa"}


def test_pattern_parameter_keeps_span_copy_on_a_tied_candidate() -> None:
    """Ties go to the extractive copy — inference must strictly improve.

    ``src`` span-copies ``hi hi``; the ``regex`` span-copy ``h`` matches that
    corpus twice.  The inferred candidate ``i`` also matches twice — equal,
    not better — so the extractive value is kept.  This is the test-11 shape:
    a literal that already demonstrates pattern behaviour survives any
    candidate that is not a strict improvement.
    """
    tables = tables_from_id_to_text(VOCAB)
    function = FunctionDefinition(
        name="fn_sub",
        description="",
        parameters={
            "src": ParameterSpec(type="string"),
            "regex": ParameterSpec(type="string"),
        },
    )
    model = FakeModel(
        VOCAB, wants=["h", "i", " h", "i", "h", '"', "i"], stop='"'
    )
    params = extract_arguments(model, "say hi hi", function, tables)
    assert params == {"src": "hi hi", "regex": "h"}


def test_pattern_parameter_uses_a_strictly_better_candidate() -> None:
    """A candidate that matches strictly more often replaces the copy.

    The extractive ``hia`` finds only itself (1 hit); the inferred ``hi``
    matches three times — a strict improvement, so it wins.  This is the
    test-9 shape (``[0-9]+`` beating a copied ``34 I'm 233``).
    """
    tables = tables_from_id_to_text(VOCAB)
    model = FakeModel(VOCAB, wants=["h", "i"], stop='"')
    value = _resolve_pattern_value(
        model, "replace things", tables, "hia", ["hia hi hi"]
    )
    assert value == "hi"


def test_pattern_parameter_sees_a_source_declared_after_it() -> None:
    """Corpus assembly is schema-order independent.

    The schema declares ``regex`` BEFORE ``src``; resolution is deferred
    until every parameter is decoded, so the copy ``h`` is still validated
    against ``hi`` (1 hit) and survives the worthless candidate ``b``
    (0 hits).  Before the deferral this case saw an empty corpus and let any
    compiling candidate clobber the copy.
    """
    tables = tables_from_id_to_text(VOCAB)
    function = FunctionDefinition(
        name="fn_sub",
        description="",
        parameters={
            "regex": ParameterSpec(type="string"),
            "src": ParameterSpec(type="string"),
        },
    )
    model = FakeModel(VOCAB, wants=["h", '"', "h", "i", "b"], stop='"')
    params = extract_arguments(model, "say hi", function, tables)
    assert params == {"regex": "h", "src": "hi"}


def test_pattern_parameter_single_hit_copy_survives_a_bad_candidate() -> None:
    """A once-matching span-copy triggers inference but is never made worse.

    The copied ``h`` matches the corpus value ``hi`` only once — not proof of
    a pattern — so a candidate is inferred.  That candidate (``b``) matches
    nothing and is rejected, and the extractive value stays: exactly the
    result this branch would have produced without pattern awareness.
    """
    tables = tables_from_id_to_text(VOCAB)
    function = FunctionDefinition(
        name="fn_sub",
        description="",
        parameters={
            "src": ParameterSpec(type="string"),
            "regex": ParameterSpec(type="string"),
        },
    )
    model = FakeModel(VOCAB, wants=["h", "i", "h", '"', "b"], stop='"')
    params = extract_arguments(model, "say hi", function, tables)
    assert params == {"src": "hi", "regex": "h"}


def test_pattern_parameter_without_corpus_keeps_a_compiling_extractive() -> None:
    """With no sibling string argument, a compiling span-copy is kept as-is.

    The pattern is the function's only string parameter, so semantic
    validation has nothing to match against.  The copy ``h`` compiles, so it
    survives and inference never runs (the step counter proves it: two steps
    for the copy, none after).  A verbatim ``[0-9]+`` spelled in the prompt
    must never be replaced by an unverifiable guess.
    """
    tables = tables_from_id_to_text(VOCAB)
    function = FunctionDefinition(
        name="fn_sub",
        description="",
        parameters={"regex": ParameterSpec(type="string")},
    )
    model = FakeModel(VOCAB, wants=["h", '"', "a"], stop='"')
    params = extract_arguments(model, "say hi", function, tables)
    assert params == {"regex": "h"}
    assert model._step == 2


def test_pattern_parameter_without_corpus_infers_for_a_broken_copy() -> None:
    """Empty corpus: inference runs only when the copy does not compile.

    The extractive ``(`` is not a valid regex, so a compiling inferred
    candidate (``hi``) may replace it — the one empty-corpus case where
    inference can help at all.
    """
    tables = tables_from_id_to_text(VOCAB)
    model = FakeModel(VOCAB, wants=["h", "i"], stop='"')
    value = _resolve_pattern_value(model, "replace things", tables, "(", [])
    assert value == "hi"


def test_non_pattern_string_parameters_are_unaffected() -> None:
    """An ordinary string param never routes through pattern resolution.

    Same script as the working-span-copy test, but with a non-pattern name:
    the value is the raw span-copy and the step count is identical, proving
    the pattern machinery never fired.
    """
    tables = tables_from_id_to_text(VOCAB)
    function = FunctionDefinition(
        name="fn_x",
        description="",
        parameters={
            "src": ParameterSpec(type="string"),
            "note": ParameterSpec(type="string"),
        },
    )
    model = FakeModel(VOCAB, wants=["h", "i", "a"], stop='"')
    params = extract_arguments(model, "say hi", function, tables)
    assert params == {"src": "hi", "note": "sa"}
    assert model._step == 3


def test_to_number_preserves_big_integer_precision() -> None:
    """A long integer must not round-trip through float (which rounds it)."""
    assert _to_number("12345678901234567890", "integer") == 12345678901234567890
    assert _to_number("7.0", "integer") == 7
    assert _to_number("3", "number") == 3.0
    assert _to_number("", "integer") == 0
    assert _to_number("-.", "number") == 0.0
