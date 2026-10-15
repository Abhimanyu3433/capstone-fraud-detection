@echo off
REM Starts the FastAPI service on http://127.0.0.1:8000  (Swagger UI: /docs). Close this window to stop it.
cd /d "%~dp0\.."
set "PY=%USERPROFILE%\anaconda3\python.exe"
if not exist "%PY%" set "PY=python"
if not exist logs mkdir logs
echo Starting API at http://127.0.0.1:8000/docs  (log: logs\api.txt)
"%PY%" -m uvicorn api.main:app --host 127.0.0.1 --port 8000 > logs\api.txt 2>&1
