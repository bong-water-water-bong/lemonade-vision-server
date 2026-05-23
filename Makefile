.PHONY: install lint type test test-integration all

install:
	uv venv
	uv pip install -e ".[dev]"

lint:
	uv run ruff check src/ tests/

type:
	uv run pyright src/

test:
	uv run pytest tests/ -v -k "not integration" || [ $$? -eq 5 ]

test-integration:
	VISION_INTEGRATION=1 uv run pytest tests/ -v

all: lint type test
