@echo off
title Push Sri Lankan Legal AI to GitHub
cd /d "%~dp0"
echo ====================================================
echo Pushing Sri Lankan Legal AI to GitHub (Pabadi127)...
echo Working Directory: %CD%
echo ====================================================
echo.
git push -u origin main
echo.
echo ====================================================
echo Done! If successful, visit https://github.com/Pabadi127/sri-lankan-legal-ai
echo ====================================================
pause
