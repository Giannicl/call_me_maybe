"""Stage A: let the LLM choose which function to call.

The subject forbids picking the function with keyword heuristics (no
'if "sum" in prompt'). Instead we show the model the menu of functions with
their descriptions and let it decide, then constrain generation with
decode_one_of so the output can only ever be one real, correctly spelled
function name. The intelligence is the model's. The spelling guarantee is ours.
"""

from __future__ import annotations

from typing import List

from .decoder import decode_one_of
from .models import FunctionDefinition
from .protocols import LLM
from .vocab import TokenTables


def encode_to_ids(model: LLM, text: str) -> List[int]:
    """Encode text into a flat list of token ids."""
    return list(model.encode(text)[0].tolist())


def build_selection_prompt(
    prompt: str, functions: List[FunctionDefinition]
) -> str:
    """Build the priming text that asks the model for the best function."""
    lines = [
        "You are a function router. Read the user request and reply with "
        "the name of the single best matching function.",
        "",
        "Functions:",
    ]
    for function in functions:
        lines.append(f"- {function.name}: {function.description}")
    lines.append("")
    lines.append(f"User request: {prompt}")
    lines.append("Best function name: ")
    return "\n".join(lines)


def select_function(
    model: LLM,
    prompt: str,
    functions: List[FunctionDefinition],
    tables: TokenTables,
) -> str:
    """Return the model's chosen function name, limited to the real names.

    Args:
        prompt: The natural-language user request.
        functions: The available function definitions.
        tables: The precomputed token tables.

    Returns:
        One of the function names, guaranteed to exist and be spelled right.
    """
    names = [function.name for function in functions]
    ids = encode_to_ids(model, build_selection_prompt(prompt, functions))
    return decode_one_of(model, ids, names, tables)
