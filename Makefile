.PHONY: install smoke test clean-smoke

install:
	python3 -m pip install -e .

smoke:
	uninet smoke --output-dir data/smoke

test:
	python3 -m unittest discover -s tests -v

clean-smoke:
	python3 -c 'from pathlib import Path; import shutil; shutil.rmtree(Path("data/smoke"), ignore_errors=True)'

