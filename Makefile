test-tutorials:
	find docs/tutorials -name '*.ipynb' | grep -v 'nbconvert' | xargs jupyter nbconvert --to notebook --execute

types:
	uv run pyright src

install:
	uv sync --all-extras

lint:
	uv run pre-commit run --all-files

test:
	uv run pytest src -n auto -rfE --failed-first --benchmark-disable

qtest:
	uv run pytest src -n auto -rfE --failed-first --benchmark-disable --testmon