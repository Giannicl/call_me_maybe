"""call me maybe: natural language to structured function calls.

A small local LLM becomes a reliable function-caller through constrained
decoding. At every generation step the illegal tokens are masked to `-inf`, so
the output is always valid JSON that matches the requested function's schema.
Run it with `python -m src`.
"""
