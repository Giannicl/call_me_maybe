"""Per-prompt orchestration: select a function, fill its arguments, wrap up.

Each prompt is processed independently and defensively. A failure on one prompt
must never abort the whole run, so selection and extraction are guarded and fall
back to safe defaults that still satisfy the output contract: every declared
argument present and correctly typed.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .arguments import extract_arguments
from .models import CallResult, FunctionDefinition
from .protocols import LLM
from .selector import select_function
from .vocab import TokenTables


def _default_value(param_type: str) -> Any:
    """Return a schema-valid placeholder for a parameter type.

    `string` and every unrecognised type (`array`, `object`, anything a
    review-time definition invents) fall through to `""`. An unknown type must
    never fail the run.

    Args:
        param_type: The declared parameter type.

    Returns:
        A placeholder value of the matching Python type.
    """
    if param_type == "number":
        return 0.0
    if param_type == "integer":
        return 0
    if param_type == "boolean":
        return False
    return ""


def _coerce(params: Dict[str, Any], function: FunctionDefinition) -> Dict[str, Any]:
    """Ensure every declared parameter is present with a value of its type.

    Args:
        params: The parameters decoded from the model.
        function: The chosen function, whose schema declares the full key set.

    Returns:
        A dict with every declared parameter present, filling any that are
        missing with a typed default.
    """
    coerced: Dict[str, Any] = {}
    for name, spec in function.parameters.items():
        coerced[name] = params.get(name, _default_value(spec.type))
    return coerced


def process_prompt(
    model: LLM,
    prompt: str,
    functions: List[FunctionDefinition],
    tables: TokenTables,
) -> CallResult:
    """Run Stage A then Stage B for one prompt and return its output ticket.

    Selection and extraction are each guarded. If selection fails the first
    function is used; if extraction fails the parameters start empty. Either way
    `_coerce` fills the result so the output contract always holds.

    Args:
        model: The language model to drive.
        prompt: The natural-language user request.
        functions: The available function definitions.
        tables: The precomputed token tables.

    Returns:
        The output ticket for this prompt.
    """
    fallback = functions[0]
    try:
        name = select_function(model, prompt, functions, tables)
    except Exception:
        name = fallback.name
    function = next((f for f in functions if f.name == name), fallback)

    try:
        params = extract_arguments(model, prompt, function, tables)
    except Exception:
        params = {}
    params = _coerce(params, function)

    return CallResult(prompt=prompt, name=function.name, parameters=params)


def process_all(
    model: LLM,
    prompts: List[str],
    functions: List[FunctionDefinition],
    tables: TokenTables,
) -> List[CallResult]:
    """Process every prompt, isolating per-prompt failures.

    Args:
        model: The language model to drive.
        prompts: The user requests, in order.
        functions: The available function definitions.
        tables: The precomputed token tables.

    Returns:
        One output ticket per prompt, in the same order.
    """
    results: List[CallResult] = []
    for prompt in prompts:
        results.append(process_prompt(model, prompt, functions, tables))
    return results
