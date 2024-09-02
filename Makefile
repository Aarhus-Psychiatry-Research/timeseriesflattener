# Install environments
install:
	uv sync --all-extras

install-tests:
	uv sync --no-dev --extra test

install-tutorials:
	uv sync --no-dev --extra tutorials

install-docs:
	uv sync --no-dev --extra docs --extra tutorials

# Tests
test-tutorials:
	make install-tutorials
	find docs/tutorials -name '*.ipynb' | grep -v 'nbconvert' | xargs uv run jupyter nbconvert --to notebook --execute

test:
	make install-tests
	uv run pytest src -n auto -rfE --failed-first --benchmark-disable

benchmark:
	make install
	uv run pytest src --codspeed

qtest:
	uv run pytest src -n auto -rfE --failed-first --benchmark-disable --testmon

# Lint and types
types:
	make install
	uv run pyright src

lint:
	make install
	uv run pre-commit run --all-files

docs:
	make install-docs
	uv run sphinx-build docs docs/_build

