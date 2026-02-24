.PHONY: run test

INIT_SEG_SIZE ?= 1000000

run:
	@set -e; \
	for f in examples/*.py; do \
		echo "Running $$f"; \
		INIT_SEG_SIZE=$(INIT_SEG_SIZE) uv run python "$$f"; \
	done

test:
	uv run --extra dev python -m pytest -q
