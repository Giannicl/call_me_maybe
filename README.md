*This project has been created as part of the 42 curriculum by giannicl.*

# call me maybe

Turn a natural-language prompt into a structured, schema-valid function call
with a small local LLM (Qwen/Qwen3-0.6B), made reliable through constrained
decoding.

Given "What is the sum of 40 and 2?" the tool does not answer 42. It returns the
ticket a program could actually execute:

```json
{ "prompt": "What is the sum of 40 and 2?", "name": "fn_add_numbers", "parameters": { "a": 40, "b": 2 } }
```

## Description

The tool is a waiter, not a cook. It translates a fuzzy request into a precise
`{name, parameters}` ticket instead of answering it. A 0.6-billion-parameter
model is clever enough to understand the request but clumsy at producing
perfectly formatted JSON. So the program does not hope the prompt yields valid
JSON. It guarantees it. At every generation step it masks out every token that
would break valid, schema-correct JSON, setting those logits to `-inf`, so only
legal tokens can be chosen. The model supplies the intelligence, which function
and what values. The mask supplies the correctness.

## Instructions

The project needs Python 3.10 or later and [uv](https://docs.astral.sh/uv/). The
`llm_sdk/` package sits next to `src/`, and `uv sync` installs everything,
including the SDK's heavy dependencies (torch, transformers, huggingface-hub).

```bash
make install        # uv sync
make run            # uv run python -m src   (downloads the model on first run)
make lint           # flake8 + mypy (required flags)
make lint-strict    # flake8 + mypy --strict
```

To run it directly with explicit paths:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

All three flags default to the `data/` paths above, so `uv run python -m src` on
its own is enough. The first run downloads the model weights and needs internet.
Later runs are offline.

## Resources

- The Qwen3-0.6B model card and its tokenizer files (vocab.json and the
  byte-level BPE alphabet).
- The GPT-2 byte-level BPE convention that Qwen inherits, the `bytes_to_unicode`
  mapping reproduced in `vocab.py`.
- Background on constrained (grammar-constrained) decoding: masking a model's
  logits so only tokens that keep the output valid can be sampled.
- The provided `llm_sdk` (`Small_LLM_Model`), used only through its public
  methods.

**How AI was used.** I used an AI assistant as a coding and teaching aid on this
project: to explain the theory of constrained decoding, to help scaffold the
module layout, to draft and review the decoding engine and the test harness, and
to help write this README. I reviewed, tested and adjusted the implementation
myself. The design choices below, the token-level masking, the two-stage
pipeline and the error handling, are ones I can explain and change by hand. The
provided `llm_sdk` is upstream code, left unchanged.

## Algorithm explanation

For each prompt the pipeline runs two constrained generations.

**Stage A, function selection** (`selector.py`, `decoder.decode_one_of`). The
model is shown the menu of functions with their descriptions and asked to name
the best one. The output is limited to the exact set of real function names by a
prefix state machine. At each step only tokens that extend the name so far
toward a real name, without overshooting, are legal. The model still chooses
which name at every branch point, so the selection is the model's and not a
keyword heuristic, but the result is always a real, correctly spelled name.

**Stage B, argument extraction** (`arguments.py`, `decoder.generate_number` and
`generate_string`). The JSON skeleton, the braces, keys, quotes and commas, is
fully fixed by the chosen function's schema, so the program writes it
deterministically. The model is invoked only for the genuinely unknown parts,
the typed values.

- numbers: only `-`, the digits and a single `.` are ever legal. Generation ends
  when the model's own next-token preference leaves the number grammar.
- strings: only string-safe tokens or the closing quote are legal. Choosing the
  closing quote ends the string.

The values are collected into a plain Python dict, and `json.dump` serialises
it, which gives valid, correctly escaped JSON by construction.

The single masking primitive (`decoder.masked_argmax`) is the whole trick: copy
the logits, set every non-allowed id to `-inf`, take the argmax. `-inf` can never
win, so an illegal token is impossible, not merely discouraged.

The token tables (`vocab.py`) map every token id to the text it spells, decoded
from `vocab.json` through the reversible byte-level alphabet, and pre-sort the
ids into typed buckets (digits, quote, string-safe, name characters) so the
per-step masking stays fast.

## Design decisions

- Write the structure, generate only the values. Not asking the model for the
  JSON skeleton is what makes the output always valid. It answers the subject's
  warning not to rely on the model spontaneously producing JSON.
- Program against a Protocol, not the SDK directly (`protocols.py`). This keeps
  the code fully typed despite the un-stubbed SDK, and it makes the engine
  unit-testable with a scripted fake model.
- Decode the vocabulary directly from `vocab.json` instead of calling `decode`
  once per token. This is faster, and it satisfies the "recode the tokenizer"
  bonus.
- Never crash. Every file and JSON operation maps to one clear `InputError`, and
  each prompt is processed on its own with safe fallbacks, so one bad prompt
  cannot sink the run.
- Every data class is a pydantic model, for validation at the boundary.

## Performance analysis

Measured on the provided prompts (CPU, Qwen/Qwen3-0.6B):

- Reliability: 11 of 11 outputs are valid, schema-compliant JSON. By
  construction the masks make malformed output unreachable, independent of
  model quality.
- Function selection: 11 of 11 routed to the right function.
- Argument extraction: 9 of the 11 come out fully correct. The eight numeric
  and copy-style prompts (sums, greets, reverses, square roots) are right, as
  is one of the three regex prompts. The other two regex prompts get the
  function and the source string right but not the inferred pattern.
- Speed: about a minute and a half of wall-clock time with the model cached,
  well inside the 5-minute budget. There is one forward pass per generated
  token and only the values are generated, so the JSON skeleton costs nothing.
  The first run additionally downloads the weights, a one-time cost. It is
  faster again on MPS or CUDA.

The hardest case, honestly. The regex prompts ask the model to infer a pattern
(`\d+`, `[aeiou]`) rather than copy a literal. A 0.6B model tends to repeat and
over-escape, so a repetition guard in `generate_string` stops the runaway. The
output stays clean and valid, but the inferred pattern needs a stronger model to
be perfect. This is a limit of the model's reasoning, not of validity. The
constraint guarantees the shape. It cannot supply reasoning the model lacks.

## Challenges faced

- The SDK documentation is wrong. The API described in the subject does not match
  the real `Small_LLM_Model`: logits go in and out as lists, the vocab path
  method is `get_path_to_vocab_file`, and `encode` returns a tensor. I resolved
  this by reading the SDK code and building against what it actually does.
- Byte-level tokens. A leading space is baked into a token as `Ġ`, and a number
  can be one token or several. I handle this by decoding tokens through the
  reversible byte alphabet and by treating the first token of a name specially.
- Knowing when a value ends. A number ends when the model's free choice leaves
  the digit grammar, and a string ends when the model picks the closing quote.
  Both are expressed as masks rather than as separate rules.

## Testing strategy

- Offline reasoning about the engine. During development the decoding
  primitives (`masked_argmax`, `generate_number`, `generate_string`,
  `decode_one_of`) were driven by a scripted fake model over a tiny vocabulary,
  so the masking math is checked deterministically without any model download.
  Those tests stay out of the submission, as the subject asks.
- Static gates. The code is flake8-clean and mypy-clean with the flags the
  subject requires, including `mypy --strict`.
- End to end. Running on the provided prompts and checking that every output
  object parses and matches its function's schema.

## Example usage

```bash
$ uv run python -m src
wrote 11 results to data/output/function_calling_results.json
```

```json
[
  { "prompt": "What is the sum of 2 and 3?", "name": "fn_add_numbers", "parameters": { "a": 2, "b": 3 } },
  { "prompt": "Greet shrek", "name": "fn_greet", "parameters": { "name": "shrek" } }
]
```
