"""Pydantic data models for inputs and outputs.

Every class in the project is a ``pydantic`` model (subject requirement IV.1):
validation happens at the boundary, so the rest of the code can assume the data
is already well-shaped.  These mirror the JSON files under ``data/input`` and the
result objects written to ``data/output``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class ParameterSpec(BaseModel):
    """The schema of a single function parameter (e.g. ``{"type": "number"}``).

    ``type`` is deliberately a plain ``str``, not a closed enum: review-time
    definitions may declare types this tool does not special-case (``array``,
    ``object``, anything else), and an unrecognised type must degrade
    gracefully — it is routed through the string path downstream — rather
    than fail validation and kill the whole run.  A missing ``type`` defaults
    to ``"string"`` for the same reason.
    """

    model_config = ConfigDict(extra="ignore")

    type: str = "string"


class FunctionDefinition(BaseModel):
    """One callable function as described in ``functions_definition.json``."""

    model_config = ConfigDict(extra="ignore")

    name: str
    description: str = ""
    parameters: Dict[str, ParameterSpec] = Field(default_factory=dict)
    returns: Optional[ParameterSpec] = None


class TestPrompt(BaseModel):
    """A single natural-language request from ``function_calling_tests.json``."""

    model_config = ConfigDict(extra="ignore")

    prompt: str


class CallResult(BaseModel):
    """One output ticket: exactly ``prompt``, ``name`` and ``parameters``.

    Serialising this model with :py:meth:`model_dump` yields precisely the three
    keys the output contract (subject V.4) requires — no extras.
    """

    prompt: str
    name: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
