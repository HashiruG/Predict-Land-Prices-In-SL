@echo off
REM Opens the dashboard only (the notebook must have been run once, e.g. with run_all.bat).
cd /d "%~dp0"
python -m streamlit run app\app.py
