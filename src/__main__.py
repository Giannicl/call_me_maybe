"""Command-line entry point: python -m src.

Loads the function definitions and test prompts, loads the model once, builds
the token tables, processes every prompt with constrained decoding, and writes
the results array. Any expected failure (bad input, a model-load error, an
unwritable output) is reported as one clear line, never a stack trace.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from .io import InputError, load_functions, load_prompts, write_results
from .pipeline import process_all
from .vocab import build_token_tables

_DEFAULT_FUNCTIONS = "data/input/functions_definition.json"
_DEFAULT_INPUT = "data/input/function_calling_tests.json"
_DEFAULT_OUTPUT = "data/output/function_calling_results.json"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="python -m src",
        description="Turn prompts into structured function calls.",
    )
    parser.add_argument(
        "--functions_definition",
        default=_DEFAULT_FUNCTIONS,
        help=f"function definitions JSON (default: {_DEFAULT_FUNCTIONS})",
    )
    parser.add_argument(
        "--input",
        default=_DEFAULT_INPUT,
        help=f"test prompts JSON (default: {_DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        default=_DEFAULT_OUTPUT,
        help=f"results JSON to write (default: {_DEFAULT_OUTPUT})",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Run the tool end to end. Returns a process exit code."""
    args = parse_args(argv)

    try:
        functions = load_functions(args.functions_definition)
        prompts = load_prompts(args.input)
    except InputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not functions:
        print("error: no functions defined", file=sys.stderr)
        return 1

    try:
        from llm_sdk import Small_LLM_Model
        model = Small_LLM_Model()
    except Exception as exc:  # any model-load failure becomes a message
        print(f"error: could not load model: {exc}", file=sys.stderr)
        return 1

    tables = build_token_tables(model)
    prompt_texts = [p.prompt for p in prompts]
    results = process_all(model, prompt_texts, functions, tables)

    try:
        write_results(args.output, results)
    except InputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"wrote {len(results)} results to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
