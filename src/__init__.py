"""call me maybe — natural language to structured function calls.

A small local LLM is turned into a reliable function-caller via *constrained
decoding*: at every generation step illegal tokens are masked to ``-inf`` so the
output is always valid JSON that matches the requested function's schema.  Run it
with ``python -m src``.
"""
