@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "VENV=%VENV%"
if "%VENV%"=="" set "VENV=.venv"

set "PYTHON=%PYTHON%"

if "%PYTHON%"=="" (
  where py >nul 2>nul && set "PYTHON=py -3"
)
if "%PYTHON%"=="" (
  where python >nul 2>nul && set "PYTHON=python"
)
if "%PYTHON%"=="" (
  where python3 >nul 2>nul && set "PYTHON=python3"
)

if "%PYTHON%"=="" (
  echo Error: Python not found. Install Python 3.9+ and ensure it is on PATH.
  exit /b 1
)

rem ---- Ensure FFmpeg / FFprobe are available (required for video import) ----
set "FFMPEG_DIR=%FFMPEG_DIR%"
if "%FFMPEG_DIR%"=="" set "FFMPEG_DIR=%~dp0.ffmpeg"
set "FFMPEG_URL=%FFMPEG_URL%"
if "%FFMPEG_URL%"=="" (
  set "FFMPEG_URL_1=https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
  set "FFMPEG_URL_2=https://github.com/BtbN/FFmpeg-Builds/releases/latest/download/ffmpeg-master-latest-win64-gpl.zip"
) else (
  set "FFMPEG_URL_1=%FFMPEG_URL%"
  set "FFMPEG_URL_2="
)

where ffmpeg >nul 2>nul
set "HAS_FFMPEG=%errorlevel%"
where ffprobe >nul 2>nul
set "HAS_FFPROBE=%errorlevel%"

if not "%HAS_FFMPEG%"=="0" (
  rem Try local portable FFmpeg first
  set "FFMPEG_BIN="
  if exist "%FFMPEG_DIR%\bin\ffmpeg.exe" set "FFMPEG_BIN=%FFMPEG_DIR%\bin"
  if "%FFMPEG_BIN%"=="" (
    for /r "%FFMPEG_DIR%" %%F in (ffmpeg.exe) do if not defined FFMPEG_BIN set "FFMPEG_BIN=%%~dpF"
  )
  if "%FFMPEG_BIN%"=="" (
    echo FFmpeg not found. Downloading portable build...
    if not exist "%FFMPEG_DIR%" mkdir "%FFMPEG_DIR%"
    set "FFMPEG_ZIP=%FFMPEG_DIR%\ffmpeg.zip"
    for %%U in ("%FFMPEG_URL_1%" "%FFMPEG_URL_2%") do (
      if not "%%~U"=="" (
        call :download_ffmpeg "%%~U" "%FFMPEG_ZIP%"
        if not errorlevel 1 goto :ffmpeg_download_ok
      )
    )
    echo Error: Failed to download FFmpeg.
    echo You can retry, or set FFMPEG_URL to a direct .zip, or install manually.
    exit /b 1
    :ffmpeg_download_ok
    powershell -NoProfile -Command "$ErrorActionPreference='Stop'; Expand-Archive -Force -Path '%FFMPEG_ZIP%' -DestinationPath '%FFMPEG_DIR%'"
    if errorlevel 1 (
      echo Error: Failed to extract FFmpeg.
      exit /b 1
    )
    del /q "%FFMPEG_ZIP%" >nul 2>nul
    for /r "%FFMPEG_DIR%" %%F in (ffmpeg.exe) do if not defined FFMPEG_BIN set "FFMPEG_BIN=%%~dpF"
  )
  if "%FFMPEG_BIN%"=="" (
    echo Error: ffmpeg.exe not found after extraction.
    exit /b 1
  )
  if not exist "%FFMPEG_BIN%\ffprobe.exe" (
    echo Error: ffprobe.exe not found after extraction.
    exit /b 1
  )
  set "PATH=%FFMPEG_BIN%;%PATH%"
)

where ffmpeg >nul 2>nul
if not "%errorlevel%"=="0" (
  echo Error: FFmpeg not found on PATH.
  echo Please install FFmpeg manually and ensure ffmpeg/ffprobe are on PATH.
  exit /b 1
)

where ffprobe >nul 2>nul
if not "%errorlevel%"=="0" (
  echo Error: FFprobe not found on PATH.
  echo Please ensure FFmpeg is installed correctly and ffprobe is on PATH.
  exit /b 1
)

if not exist "%VENV%\" (
  %PYTHON% -m venv "%VENV%"
  if errorlevel 1 exit /b 1
)

set "PY=%VENV%\Scripts\python.exe"
set "PIP=%VENV%\Scripts\pip.exe"

if not exist "%PY%" (
  echo Error: Python not found at %PY%
  echo Tip: delete "%VENV%" and re-run this script.
  exit /b 1
)

"%PIP%" install --upgrade pip
if errorlevel 1 exit /b 1

"%PIP%" install -e ".[sam2,yaml]"
if errorlevel 1 exit /b 1

set "SAM2_URL=https://huggingface.co/facebook/sam2-hiera-tiny/resolve/main/sam2_hiera_tiny.pt"
set "SAM2_WEIGHTS=src\sam2_configs\sam2_hiera_tiny.pt"

if not exist "%SAM2_WEIGHTS%" (
  powershell -NoProfile -Command "$u='%SAM2_URL%'; $o='%SAM2_WEIGHTS%'; New-Item -ItemType Directory -Force -Path (Split-Path $o) | Out-Null; Invoke-WebRequest -Uri $u -OutFile $o"
  if errorlevel 1 (
    if exist "%SAM2_WEIGHTS%" (
      rem ok
    ) else (
      echo Error: Failed to download SAM2 weights.
      exit /b 1
    )
  )
)

echo Done.
echo Activate with: %VENV%\Scripts\activate.bat
echo Run with: %VENV%\Scripts\python.exe -m src.tracewave
exit /b 0

:download_ffmpeg
set "DL_URL=%~1"
set "DL_OUT=%~2"
echo Downloading: %DL_URL%
powershell -NoProfile -Command "$ProgressPreference='SilentlyContinue'; $ErrorActionPreference='Stop'; [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%DL_URL%' -OutFile '%DL_OUT%'"
if errorlevel 1 (
  where curl >nul 2>nul
  if "%errorlevel%"=="0" (
    curl -L -o "%DL_OUT%" "%DL_URL%"
  ) else (
    where bitsadmin >nul 2>nul
    if "%errorlevel%"=="0" (
      bitsadmin /transfer ffmpegdownload /download /priority normal "%DL_URL%" "%DL_OUT%"
    )
  )
)
if not exist "%DL_OUT%" exit /b 1
exit /b 0
