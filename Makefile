.PHONY: doctor test test-unit test-geo test-matching test-training test-api format lint

doctor:
	python scripts/doctor.py

test: test-unit

test-unit:
	python -m pytest

test-geo:
	python -m pytest tests/test_lunar_crs.py tests/test_pixel_world.py tests/test_catalog.py

test-matching:
	python -m pytest tests/test_roma_coordinates.py

test-training:
	python -m pytest tests/test_training_losses.py

test-api:
	python -m pytest tests/test_api.py

format:
	python -m ruff format .

lint:
	python -m ruff check .
