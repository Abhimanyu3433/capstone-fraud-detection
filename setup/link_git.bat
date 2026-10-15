@echo off
REM Connects this folder to https://github.com/Abhimanyu3433/capstone-fraud-detection
REM without touching your files (data, logs and the full model stay local; they are git-ignored).
cd /d "%~dp0\.."
if exist .git (
  echo This folder is already a git repository.
  git status -sb
  pause
  exit /b
)
git init -b main
git remote add origin https://github.com/Abhimanyu3433/capstone-fraud-detection.git
git fetch origin
git reset --mixed origin/main
git branch --set-upstream-to=origin/main main
echo.
git status -sb
echo.
echo Linked. Files listed above as modified/untracked differ from GitHub main.
pause
