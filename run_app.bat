@echo off
title Sri Lankan Legal Case Law Recommender
cd /d "%~dp0"
echo Starting Sri Lankan Legal Case Law Recommendation System...
echo Project Folder: %CD%
echo.
streamlit run app.py
pause
