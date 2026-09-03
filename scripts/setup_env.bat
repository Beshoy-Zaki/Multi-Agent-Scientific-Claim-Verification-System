@echo off
echo Setting up MASCV environment...
python -m venv venv
call venv\Scripts\activate.bat
pip install --upgrade pip
pip install -e ".[dev,docs]"
echo Environment setup complete. Activate with 'venv\Scripts\activate.bat'.
