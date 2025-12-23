@echo off
setlocal enabledelayedexpansion

echo ========================================
echo Watermark Remover AI - Right Click Menu
echo ========================================
echo.

REM Change directory to the script's location to ensure model paths are correct.
echo Changing to script directory...
cd /d "E:\Github\WatermarkRemover-AI"
if %errorlevel% neq 0 (
    echo ERROR: Cannot change to script directory.
    echo Current directory: %CD%
    echo Expected directory: E:\Github\WatermarkRemover-AI
    pause
    exit /b 1
)
echo Current directory: %CD%

REM Check if input file was provided
if "%~1"=="" (
    echo ERROR: No input file provided.
    echo Usage: run_remover.bat "file_path"
    pause
    exit /b 1
)

echo Input file: "%~1"

REM Check if input file exists
if not exist "%~1" (
    echo ERROR: Input file does not exist: "%~1"
    pause
    exit /b 1
)

REM Use Python from miniconda
echo.
echo Using Python from miniconda...
set "PYTHON_CMD=E:\miniconda\python.exe"
if not exist "%PYTHON_CMD%" (
    echo ERROR: Python not found at %PYTHON_CMD%
    echo Please check if miniconda is installed at E:\miniconda
    pause
    exit /b 1
)
echo Found Python at: %PYTHON_CMD%

REM Check Python version
echo Checking Python version...
%PYTHON_CMD% --version
if %errorlevel% neq 0 (
    echo ERROR: Failed to get Python version.
    pause
    exit /b 1
)

REM Check if main.py exists
echo.
echo Checking main.py...
if not exist "main.py" (
    echo ERROR: main.py not found in current directory.
    echo Current directory: %CD%
    echo Please ensure you are running this from the correct directory.
    pause
    exit /b 1
)

REM Output to the same directory as the input file
set OUTPUT_DIR=%~dp1
echo Output directory: "%OUTPUT_DIR%"

REM Check if required models exist
echo.
echo Checking required models...
if not exist "models\yolo.pt" (
    echo WARNING: YOLO model not found at models\yolo.pt
    echo Please download the YOLO model and place it in the models folder.
)
if not exist "models\big-lama.pt" (
    echo WARNING: LaMa model not found at models\big-lama.pt
    echo Please download the LaMa model and place it in the models folder.
)

REM Run the main Python script.
echo.
echo Starting watermark removal...
echo Command: %PYTHON_CMD% main.py --input "%~1" --output %OUTPUT_DIR%
echo.
%PYTHON_CMD% main.py --input "%~1" --output %OUTPUT_DIR%

set PYTHON_EXIT_CODE=%errorlevel%
echo.
echo ========================================
if %PYTHON_EXIT_CODE% neq 0 (
    echo ERROR: Python script failed with exit code %PYTHON_EXIT_CODE%
    echo Please check the error messages above.
) else (
    echo SUCCESS: Processing completed.
    echo Output saved to: "%OUTPUT_DIR%"
)
echo ========================================

echo.
echo Press any key to exit...
pause >nul
exit /b %PYTHON_EXIT_CODE%
