@echo off
REM Installs the packages, runs the tests, runs the notebook and opens the dashboard.
cd /d "%~dp0"
echo [1/4] Installing packages...
python -m pip install -r requirements.txt
echo [2/4] Running unit tests...
python -m pytest -q tests
echo [3/4] Running the notebook (takes about 10 minutes)...
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebook\Notebook.ipynb
if errorlevel 1 (
  echo The notebook stopped with an error - open it in Jupyter or VS Code to see which cell.
  pause
  exit /b 1
)
echo [4/4] Starting the dashboard...
python -m streamlit run app\app.py
