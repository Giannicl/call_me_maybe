*This project has been created as part of the 42 curriculum by glieuw-a.*

# call me maybe

Turn natural-language prompts into **structured, schema-valid function calls** using
a small local LLM (`Qwen/Qwen3-0.6B`), made reliable with **constrained decoding**.

Given `"What is the sum of 40 and 2?"` the tool does **not** answer `42`. It emits the
order ticket a program could actually execute:

```json
{ "prompt": "What is the sum of 40 and 2?", "name": "fn_add_numbers", "parameters": { "a": 40.0, "b": 2.0 } }
```

---

## Description

The program is a *waiter, not a cook*: it translates a fuzzy request into a precise
`{name, parameters}` ticket instead of answering it. A 0.6-billion-parameter model is
clever enough to understand the request but clumsy at producing perfectly formatted
JSON. Rather than *hoping* the prompt yields valid JSON, we **guarantee** it: at every
generation step we mask out (set to `-inf`) every token that would break valid,
schema-correct JSON, so only legal tokens can be chosen. The model supplies the
intelligence (which function, what values); the mask supplies the correctness.

## Instructions

Requires Python ≥ 3.10 and [`uv`](https://docs.astral.sh/uv/). The `llm_sdk/` package
sits next to `src/`; `uv sync` installs everything (including the SDK's heavy
dependencies — `torch`, `transformers`, `huggingface-hub`).

```bash
make install        # uv sync
make run            # uv run python -m src   (downloads the model on first run)
make lint           # flake8 + mypy (required flags)
make lint-strict    # flake8 + mypy --strict
make test           # pytest (offline engine tests, no model needed)
```

Run directly with explicit paths:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

All three flags default to the `data/` paths above, so `uv run python -m src` is enough.
The first run downloads the model weights (needs internet); later runs are offline.

## Resources

- The project subject (`docs/en.subject.pdf`) and the byte-level BPE convention used
  by GPT-2 / Qwen tokenizers (`bytes_to_unicode`).
- The provided `llm_sdk` (`Small_LLM_Model`), used only through its public methods.

**How AI was used.** AI assistance was used as a teaching aid: to explain the theory of constrained decoding and the constrained-decoding engine, as well as for to review the code. The AI assistant was instructed to not give the solution but guide using the socratic method.

## Algorithm explanation

For each prompt the pipeline runs **two constrained generations**:

**Stage A — function selection (`selector.py`, `decoder.decode_one_of`).** The model is
shown the menu of functions and their descriptions and asked to name the best one. The
output is constrained to the *exact set of real function names* by a prefix state
machine: at each step only tokens that extend the name-so-far toward a real name without
overshooting are legal. The model still chooses *which* name at every branch point — the
selection is the LLM's, not a keyword heuristic — but the result is always a real,
correctly-spelled name.

**Stage B — argument extraction (`arguments.py`, `decoder.generate_number/string`).** The
JSON *skeleton* (braces, keys, quotes, commas) is fully determined by the chosen
function's schema, so we write it deterministically. The model is invoked only for the
genuinely unknown parts — the typed values:

- **numbers**: only `-`, digits and a single `.` are ever legal (no `.` at all for an
  `integer` parameter); generation ends when the model's own next-token preference
  leaves the number grammar.
- **strings**: decoded by *span-copy* — the only legal tokens are those that extend a
  contiguous substring of the user request, tracked as exact `(start, end)` offset
  pairs, so the value can never be hallucinated, never starts on whitespace, and a
  span begun mid-word is snapped left to its real word boundary. The closing quote
  is the explicit stop option; when it is also a legal continuation, the tracked
  start disambiguates a closing delimiter (stop) from an embedded quote (copy). If
  span-copy finds nothing to extract, a *free* constrained fallback generates over
  string-safe tokens instead, so non-extractive values (e.g. an inferred regex)
  stay reachable.
- **patterns** (a parameter the *schema* names `regex`/`pattern`): the value may be a
  pattern that is nowhere in the request, so span-copy alone cannot produce it. The
  code generates a candidate by free constrained generation (primed with general,
  backslash-free regex examples), then **validates** it: it is kept only if it
  compiles and produces *strictly more* real matches against the request's other
  string arguments than the copied span does — otherwise the copied span stays. Every
  `re` call on a candidate runs under a hard `SIGALRM` time cap, so a pathological
  pattern can never hang the run. This is generate → validate → fall back: a bad
  inference can never make the result worse than the copy.
- **unknown types** (`array`, `object`, anything a review-time definition invents)
  route through the string path — an unrecognised type never fails the run.

The values are collected into a plain Python `dict`; `json.dump` serialises it, which
guarantees valid, correctly-escaped JSON by construction.

**The one masking primitive** (`decoder.masked_argmax`) is the whole trick: copy the
logits, set every non-allowed id to `-inf`, take the argmax. `-inf` can never win, so an
illegal token is impossible — not merely discouraged.

The token tables (`vocab.py`) map every token id to the text it spells (decoded from
`vocab.json` via the reversible byte-level alphabet) and pre-sort ids into typed buckets
(digits, quote, string-safe, name characters) so per-step masking is fast.

## Design decisions

- **Write structure, generate values.** Not asking the model for the JSON skeleton is
  what makes output *always* valid — it directly answers the subject's "do not rely on
  the model spontaneously producing JSON."
- **Program against a `Protocol`, not the SDK directly** (`protocols.py`). This keeps the
  code fully typed (`mypy --strict`-clean) despite the un-stubbed SDK and makes the engine
  unit-testable with a scripted fake model.
- **Decode the vocabulary ourselves** from `vocab.json` (byte-level BPE) instead of
  calling `decode` 150k times — faster, and it satisfies the "recode the tokenizer" bonus.
- **Never crash.** Every file/JSON operation maps to one clear `InputError`; each prompt
  is processed independently with safe fallbacks, so one bad prompt can't sink the run.
- **All data classes are pydantic** for boundary validation.

## Performance analysis

Measured on the 11 provided prompts (CPU, `Qwen/Qwen3-0.6B`):

- **Reliability:** **11/11 valid, schema-compliant JSON** — by construction, the masks make
  malformed output unreachable, independent of model quality.
- **Function selection:** **11/11 correct** — every prompt routed to the right function.
- **Argument extraction:** **10/11 fully correct** on the provided prompts. The single miss
  is one regex-substitution prompt whose *replacement* must be inferred — the word
  "asterisks" maps to the symbol `*`, which appears nowhere in the request; its function,
  `source_string` and `regex` are all correct. (Against the moulinette's hidden grading set
  the tool scores a full 11/11.)
- **Speed:** **~2–3 min** wall-clock with the model cached (one forward pass per *generated*
  token; only values are generated, the JSON skeleton is free). The first run additionally
  downloads ~1.2 GB of weights (~8.5 min total) — a one-time cost. Comfortably inside the
  5-minute budget once cached; faster again on MPS/CUDA.

**The hardest case, honestly.** A regex-substitution prompt asks the model to *infer* a
pattern (`\d+`, `[aeiou]`) rather than copy a literal — text that is nowhere in the request,
so span-copy alone cannot reach it. The pattern path handles this by generating a candidate
and *keeping it only when it provably matches better than the copied span* (see Algorithm),
which is enough to get the numbers and word-boundary cases right. What remains is a prompt
that needs *two* inferences at once — both the pattern and a symbol replacement (`*` from
"asterisks"). Rather than hard-code that, it is left as a known limit: the constraint
guarantees the JSON is always valid and never worse than a verbatim copy, but it cannot
supply reasoning a 0.6B model lacks.

## Challenges faced

- **The SDK docs are wrong.** The subject's described API does not match the real
  `Small_LLM_Model` (list-in/list-out logits, `get_path_to_vocab_file`, tensor `encode`).
  Resolved by reading the SDK source (`llm_sdk/llm_sdk/__init__.py`) and building
  against reality.
- **Byte-level tokens.** A leading space is baked into a token as `Ġ`; numbers may be one
  token or several. Handled by decoding tokens through the reversible byte alphabet and by
  treating the name state machine's first token specially.
- **Knowing when a value ends.** Numbers end when the model's free choice leaves the digit
  grammar; strings end when the model elects the closing quote — both expressed as masks.

## Testing strategy

- **Offline engine tests** (`tests/test_decoding.py`, `make test`): a `FakeModel` returns
  scripted logits over a tiny vocabulary, making `masked_argmax`, `generate_number`,
  `generate_string` and `decode_one_of` fully deterministic — the masking *math* is proven
  without any model download. The suite (41 tests) also covers the span-copy edge cases
  (embedded quotes, snap-left, repeated substrings), the pattern validate-and-fallback
  branches, and the `SIGALRM` guard against a pathological regex hanging the run.
- **Static gates:** `flake8`-clean and `mypy`-clean (including `--strict`).
- **End-to-end:** running on the 11 provided prompts and validating that every output
  object parses and matches its function's schema.

## Example usage

```bash
$ uv run python -m src
wrote 11 results to data/output/function_calling_results.json
```

```json
[
  { "prompt": "What is the sum of 2 and 3?", "name": "fn_add_numbers", "parameters": { "a": 2.0, "b": 3.0 } },
  { "prompt": "Greet shrek", "name": "fn_greet", "parameters": { "name": "shrek" } }
]
```
