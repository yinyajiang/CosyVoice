@echo off
REM Windows batch wrapper for docker_build.sh

REM Check if Git Bash is available
where bash >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Error: bash not found. Please install Git for Windows or use WSL.
    echo.
    echo Options:
    echo   1. Install Git for Windows: https://git-scm.com/download/win
    echo   2. Use WSL: wsl bash docker_build.sh %*
    exit /b 1
)

REM Execute the bash script with all arguments
bash docker_build.sh %*

