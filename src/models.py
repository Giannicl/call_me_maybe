"""Pydantic data models for the input and output files.

The subject requires every class to be a pydantic model, so validation happens
at the boundary and the rest of the code can assume well-shaped data. These
models mirror the JSON files under data/input and the result objects written to
data/output.
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

# Argument types a function definition may declare. The provided data uses
# only "number" and "string", but "integer" and "boolean" are supported as
# well, so the tool still works if a review-time definition adds them.
ParamType = Literal["number", "integer", "string", "boolean"]


class ParameterSpec(BaseModel):
    """Schema of a single function parameter, e.g. {"type": "number"}."""

    model_config = ConfigDict(extra="ignore")

    type: ParamType


class FunctionDefinition(BaseModel):
    """One callable function, as described in functions_definition.json."""

    model_config = ConfigDict(extra="ignore")

    name: str
    description: str = ""
    parameters: Dict[str, ParameterSpec] = Field(default_factory=dict)
    returns: Optional[ParameterSpec] = None


class TestPrompt(BaseModel):
    """A single natural-language request from function_calling_tests.json."""

    model_config = ConfigDict(extra="ignore")

    prompt: str


class CallResult(BaseModel):
    """One output record: exactly prompt, name and parameters.

    Dumping this model with model_dump yields precisely the three keys the
    output contract requires, with no extras.
    """

    prompt: str
    name: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
