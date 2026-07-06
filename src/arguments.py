"""Stage B: fill the chosen function's typed arguments under constraint.

This is the direct answer to the subject's warning not to rely on the model
spontaneously producing JSON. We do not ask the model for the JSON skeleton at
all. The braces, the keys and the punctuation are fully fixed by the schema, so
there is nothing for the model to decide there. We write them ourselves. The
model is invoked only for the genuinely unknown parts: the typed values.

So the structural layer is deterministic and always valid, and the schema layer
is constrained generation with the right keys and value types. We build a plain
Python dict of parameters, and json.dump later turns it into valid, properly
escaped JSON.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .decoder import (
    decode_one_of,
    generate_extractive_string,
    generate_number,
    generate_string,
)
from .models import FunctionDefinition
from .protocols import LLM
from .selector import encode_to_ids
from .vocab import TokenTables


def _emit(model: LLM, ids: List[int], text: str) -> None:
    """Append a known structural fragment to the running context."""
    ids.extend(encode_to_ids(model, text))


def _to_number(raw: str) -> Any:
    """Parse decoded number text into an int (if integral) or a float."""
    if not raw or raw in {"-", ".", "-."}:
        return 0
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        return 0


def build_arguments_prompt(prompt: str, function: FunctionDefinition) -> str:
    """Build the priming text that frames the value-extraction task."""
    return (
        f"Extract the arguments for the function `{function.name}` "
        f"from the user request.\n"
        f"Function description: {function.description}\n"
        f"User request: {prompt}\n"
        f"Arguments as JSON: "
    )


def extract_arguments(
    model: LLM,
    prompt: str,
    function: FunctionDefinition,
    tables: TokenTables,
) -> Dict[str, Any]:
    """Constrained-decode the argument object for the function.

    Args:
        prompt: The user request (gives the model context for the values).
        function: The chosen function, whose schema drives the keys and types.
        tables: The precomputed token tables.

    Returns:
        A parameters dict with every declared key present and correctly typed.
    """
    ids = encode_to_ids(model, build_arguments_prompt(prompt, function))
    params: Dict[str, Any] = {}
    items = list(function.parameters.items())

    _emit(model, ids, "{")
    for index, (name, spec) in enumerate(items):
        _emit(model, ids, f'"{name}": ')  # the key is known, we write it
        if spec.type in ("number", "integer"):
            number = _to_number(generate_number(model, ids, tables))
            if spec.type == "integer":
                params[name] = int(number)
            else:
                params[name] = float(number)
        elif spec.type == "boolean":
            choice = decode_one_of(model, ids, ["true", "false"], tables)
            params[name] = choice == "true"
        else:  # "string"
            _emit(model, ids, '"')
            value = generate_extractive_string(model, ids, tables, prompt)
            if not value:
                value = generate_string(model, ids, tables)
            params[name] = value
            _emit(model, ids, '"')
        if index < len(items) - 1:
            _emit(model, ids, ", ")
    _emit(model, ids, "}")
    return params
