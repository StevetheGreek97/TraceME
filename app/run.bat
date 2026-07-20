@echo off
setlocal EnableExtensions

set "VENV=%VENV%"
if "%VENV%"=="" set "VENV=.venv"

rem Prefer local portable FFmpeg if present
set "FFMPEG_DIR=%FFMPEG_DIR%"
if "%FFMPEG_DIR%"=="" set "FFMPEG_DIR=%~dp0.ffmpeg"
set "FFMPEG_BIN="
if exist "%FFMPEG_DIR%\bin\ffmpeg.exe" set "FFMPEG_BIN=%FFMPEG_DIR%\bin"
if "%FFMPEG_BIN%"=="" (
  for /r "%FFMPEG_DIR%" %%F in (ffmpeg.exe) do if not defined FFMPEG_BIN set "FFMPEG_BIN=%%~dpF"
)
if not "%FFMPEG_BIN%"=="" set "PATH=%FFMPEG_BIN%;%PATH%"

set "PY=%VENV%\Scripts\python.exe"

if not exist "%PY%" (
  echo Error: Python not found at %PY%
  echo Tip: run install.bat first.
  exit /b 1
)

"%PY%" -m src.tracewave
