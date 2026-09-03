#!/usr/bin/env bash
set -e

echo "Setting up MASCV environment..."
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev,docs]"
echo "Environment setup complete. Activate with 'source venv/bin/activate'."
