"""call me maybe: natural language to structured function calls.

A small local LLM becomes a reliable function caller through constrained
decoding. At each generation step the illegal tokens are masked to -inf, so
the output is always valid JSON matching the requested function's schema. Run
the tool with "python -m src".
"""
