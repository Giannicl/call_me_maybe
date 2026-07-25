"""Stage A: let the LLM choose which function to call.

The subject forbids picking the function with keyword heuristics (no
`if "sum" in prompt`). We show the model the menu of functions with their
descriptions and let it decide, but constrain generation with `decode_one_of`
so the output can only ever be one real, correctly-spelled function name. The
intelligence is the model's. The spelling guarantee is ours.
"""

from __future__ import annotations

from typing import List

from .decoder import decode_one_of
from .models import FunctionDefinition
from .protocols import LLM
from .vocab import TokenTables


def encode_to_ids(model: LLM, text: str) -> List[int]:
    """Encode the text into a flat list of token ids.

    Args:
        model: The language model whose tokenizer is used.
        text: The text to encode.

    Returns:
        The token ids.
    """
    return list(model.encode(text)[0].tolist())


def build_selection_prompt(prompt: str, functions: List[FunctionDefinition]) -> str:
    """Build the priming text that asks the model to name the best function.

    Args:
        prompt: The natural-language user request.
        functions: The available function definitions, listed as the menu.

    Returns:
        The priming text, ending on `Best function name: ` so the model
        continues straight into the name.
    """
    lines = [
        "You are a function router. Read the user request and reply with the "
        "name of the single best matching function.",
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
    """Return the model's chosen function name, constrained to the real names.

    Args:
        model: The language model to drive.
        prompt: The natural-language user request.
        functions: The available function definitions.
        tables: The precomputed token tables.

    Returns:
        One of the function names, guaranteed to exist and be spelled correctly.
    """
    names = [function.name for function in functions]
    ids = encode_to_ids(model, build_selection_prompt(prompt, functions))
    return decode_one_of(model, ids, names, tables)
