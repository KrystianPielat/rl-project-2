.PHONY: lint format check install-hooks

lint:
	uv run pre-commit run --all-files

format:
	uv run ruff check --fix .
	uv run ruff format .

check: lint

install-hooks:
	uv run pre-commit install
