.PHONY: install run debug clean lint lint-strict test

install:
	uv sync

run:
	uv run python -m src

debug:
	uv run python -m pdb -m src

test:
	uv run pytest -q

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports \
		--disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 .
	uv run mypy . --strict

clean:
	rm -rf __pycache__ */__pycache__ */*/__pycache__ .mypy_cache .pytest_cache
	find . -type d -name '__pycache__' -prune -exec rm -rf {} +
