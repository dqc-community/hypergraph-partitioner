.PHONY: run test

run:
	@set -e; \
	for f in examples/*.py; do \
		echo "Running $$f"; \
		uv run python "$$f"; \
	done

test:
	uv run --extra dev pytest -q
