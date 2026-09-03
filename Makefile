.PHONY: setup lint test run-ui run-pipeline clean

setup:
	pip install -e .[dev]

lint:
	ruff check .
	mypy src/

test:
	pytest tests/

run-ui:
	streamlit run ui/frontend/streamlit_app.py

run-pipeline:
	python scripts/run_pipeline.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
