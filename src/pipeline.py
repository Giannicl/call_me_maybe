"""Per-prompt orchestration: select a function, fill its arguments, wrap up.

Each prompt is handled on its own. A failure on one prompt must never abort the
whole run, so selection and extraction are guarded and fall back to safe
defaults that still satisfy the output contract: every declared argument is
present with the correct type.
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

    Args:
        param_type: The declared type ("number", "integer", "boolean" or
            "string").

    Returns:
        0 for numbers, False for booleans, "" otherwise.
    """
    if param_type == "number":
        return 0.0
    if param_type == "integer":
        return 0
    if param_type == "boolean":
        return False
    return ""


def _coerce(
    params: Dict[str, Any], function: FunctionDefinition
) -> Dict[str, Any]:
    """Ensure every declared parameter is present with a value of its type.

    Args:
        params: The parameters extracted for this call (may be incomplete).
        function: The chosen function, whose schema lists the required keys.

    Returns:
        A dict with every declared key present, filling gaps with defaults.
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
    """Run stage A then stage B for one prompt and return its result.

    Args:
        prompt: The natural-language request.
        functions: The available function definitions.
        tables: The precomputed token tables.

    Returns:
        The call result for this prompt, always schema-valid.
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
        prompts: The natural-language requests.
        functions: The available function definitions.
        tables: The precomputed token tables.

    Returns:
        One call result per prompt, in the same order.
    """
    results: List[CallResult] = []
    for prompt in prompts:
        results.append(process_prompt(model, prompt, functions, tables))
    return results
