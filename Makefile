.PHONY: build debug-local-deps publish run test

### PyPi config
DIST_DIR := dist
REPOSITORY ?= testpypi

ifeq ($(REPOSITORY),testpypi)
PUBLISH_URL := https://test.pypi.org/legacy/
else ifeq ($(REPOSITORY),pypi)
PUBLISH_URL := https://upload.pypi.org/legacy/
else
$(error Unsupported REPOSITORY '$(REPOSITORY)'; use REPOSITORY=testpypi or REPOSITORY=pypi)
endif
###

INIT_SEG_SIZE ?= 1000000

run:
	@set -e; \
	for f in examples/*.py; do \
		echo "Running $$f"; \
		INIT_SEG_SIZE=$(INIT_SEG_SIZE) uv run python "$$f"; \
	done

PYTEST_ARGS ?=

test:
	uv run --extra dev python -m pytest -q $(PYTEST_ARGS)

debug-local-deps:
	uv pip install --no-deps \
		-e ../dqcomp/packages/bosonic-model \
		-e ../dqcomp/packages/bosonic-converters

build:
	rm -rf $(DIST_DIR)
	uvx --from build pyproject-build --outdir $(DIST_DIR)
	uvx twine check $(DIST_DIR)/*

publish: build
	uvx twine upload --repository-url $(PUBLISH_URL) $(DIST_DIR)/*
