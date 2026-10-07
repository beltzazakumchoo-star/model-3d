@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0app\scripts\open-studio.ps1" -InstallRoot "%~dp0."
if errorlevel 1 pause

