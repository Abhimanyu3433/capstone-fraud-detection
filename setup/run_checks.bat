@echo off
REM Installs the API/test packages, pins CI versions to this PC, validates the hand-off,
REM then runs ruff and pytest. Everything is logged to logs\checks.txt.
cd /d "%~dp0\.."
set "PY=%USERPROFILE%\anaconda3\python.exe"
if not exist "%PY%" set "PY=python"
if not exist logs mkdir logs
set "LOG=logs\checks.txt"
echo Running checks with %PY% ... (1-3 minutes)
echo ===== python ===== > "%LOG%"
"%PY%" --version >> "%LOG%" 2>&1
echo ===== pip install ===== >> "%LOG%"
"%PY%" -m pip install -q fastapi "uvicorn[standard]" httpx pytest ruff >> "%LOG%" 2>&1
echo ===== install_config ===== >> "%LOG%"
"%PY%" setup\install_config.py >> "%LOG%" 2>&1
echo ===== pin CI versions ===== >> "%LOG%"
"%PY%" -m src.config --pin >> "%LOG%" 2>&1
echo ===== validate hand-off ===== >> "%LOG%"
"%PY%" -m src.config >> "%LOG%" 2>&1
echo ===== ruff ===== >> "%LOG%"
"%PY%" -m ruff check . >> "%LOG%" 2>&1
echo ===== pytest ===== >> "%LOG%"
"%PY%" -m pytest -v -p no:cacheprovider >> "%LOG%" 2>&1
echo ===== done ===== >> "%LOG%"
echo.
echo Finished. Results are in %LOG%
pause
