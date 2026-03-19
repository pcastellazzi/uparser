MAKEFLAGS += --warn-undefined-variables
MAKEFLAGS += --no-builtin-rules

PYTHON_CODE = src/ tests/ examples/
PYTHON_VERSIONS = 3.12 3.13 3.14

PYTEST_COVERAGE = --cov=src/ --cov-context=test --cov-report=term-missing
PYTEST_SETTINGS ?=


.PHONY: all
all: install check coverage integration docs


.PHONY: clean
clean:
	git clean -fdx


.PHONY: install
install:
	uv sync --all-groups


.PHONY: check
check:
	prek run --all-files


.PHONY: coverage
coverage:
	uv run -- pytest $(PYTEST_COVERAGE) $(PYTEST_SETTINGS) $(PYTHON_CODE)


.PHONY: test
test:
	uv run -- pytest $(PYTEST_SETTINGS) $(PYTHON_CODE)


.PHONY: update-dependencies
update-dependencies:
	uv sync --all-groups --upgrade
	prek autoupdate


.PHONY: integration
integration: $(PYTHON_VERSIONS)


.PHONY: $(PYTHON_VERSIONS)
$(PYTHON_VERSIONS):
	$(MAKE) UV_PROJECT_ENVIRONMENT=.uv/$@ UV_PYTHON=$@ install test
