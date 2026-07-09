"""Reading, validating and writing the JSON files — defensively.

The subject demands the tool *never crash* on bad input.  Every filesystem and
JSON operation here funnels its failure into a single :class:`InputError` with a
clear message, which ``__main__`` turns into a friendly one-line error instead of
a stack trace.  All file access uses context managers.
"""

from __future__ import annotations

import json
import os
from typing import Any, List

from pydantic import ValidationError

from .models import CallResult, FunctionDefinition, TestPrompt


class InputError(Exception):
    """Raised when an input file is missing, unreadable or malformed."""


def _read_json(path: str) -> Any:
    """Load and parse a JSON file, mapping every failure to :class:`InputError`."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        raise InputError(f"input file not found: {path}")
    except json.JSONDecodeError as exc:
        raise InputError(f"invalid JSON in {path}: {exc}")
    except OSError as exc:
        raise InputError(f"could not read {path}: {exc}")


def load_functions(path: str) -> List[FunctionDefinition]:
    """Load and validate the function definitions array.

    Args:
        path: Path to ``functions_definition.json``.

    Returns:
        The validated function definitions.

    Raises:
        InputError: If the file is missing, not a JSON array, or any entry fails
            schema validation.
    """
    raw = _read_json(path)
    if not isinstance(raw, list):
        raise InputError(f"{path}: expected a JSON array of functions")
    try:
        return [FunctionDefinition.model_validate(item) for item in raw]
    except ValidationError as exc:
        raise InputError(f"{path}: invalid function definition: {exc}")


def load_prompts(path: str) -> List[TestPrompt]:
    """Load and validate the test prompts array.

    Args:
        path: Path to ``function_calling_tests.json``.

    Returns:
        The validated prompts.

    Raises:
        InputError: If the file is missing, not a JSON array, or any entry fails
            schema validation.
    """
    raw = _read_json(path)
    if not isinstance(raw, list):
        raise InputError(f"{path}: expected a JSON array of prompts")
    try:
        return [TestPrompt.model_validate(item) for item in raw]
    except ValidationError as exc:
        raise InputError(f"{path}: invalid prompt entry: {exc}")


def write_results(path: str, results: List[CallResult]) -> None:
    """Write results as a JSON array, creating the output directory if needed.

    Args:
        path: Destination file (e.g. ``data/output/function_calling_results.json``).
        results: The call results to serialise.

    Raises:
        InputError: If the file cannot be written.
    """
    payload = [result.model_dump() for result in results]
    directory = os.path.dirname(os.path.abspath(path))
    try:
        os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
    except OSError as exc:
        raise InputError(f"could not write {path}: {exc}")
