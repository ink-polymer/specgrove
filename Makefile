PYTHON ?= python3

.PHONY: check test prepare-data

check:
	$(PYTHON) scripts/check_repository.py

test:
	$(PYTHON) -m unittest discover -s tests -v
	PYTHONPATH=code/natural/src:code/natural/scripts $(PYTHON) -m pytest -q -p no:cacheprovider code/natural/tests/gbv_paper/test_natural_paper_plan.py code/natural/tests/gbv_paper/test_once_protocol.py

prepare-data:
	$(PYTHON) scripts/prepare_benchmarks.py --output data/prepared
