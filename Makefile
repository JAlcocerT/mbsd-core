.PHONY: help sync test examples corpus qualify check clean

PYTHON ?= python
UV ?= uv

help:
	@echo "Targets:"
	@echo "  make sync      Install package and dev tools with uv"
	@echo "  make test      Run pytest"
	@echo "  make examples  Run bundled examples"
	@echo "  make corpus    Run the validation corpus against its record"
	@echo "  make qualify   Release qualification: clean build, every Python, deps floor/ceiling"
	@echo "  make check     Run tests and examples"
	@echo "  make clean     Remove local caches/build outputs"

sync:
	$(UV) sync --extra dev

test:
	$(UV) run pytest -q

examples:
	$(UV) run python examples/planar_driven_slider.py
	$(UV) run python examples/planar_mass_spring.py
	$(UV) run python examples/planar_slider_crank_analysis.py
	$(UV) run python examples/planar_slider_crank_reactions.py

corpus:
	$(UV) run python corpus/run.py

qualify:
	scripts/qualify_release.sh

check: test examples

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist htmlcov *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
