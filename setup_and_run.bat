@echo off
echo Starting RAG Expert Q&A Tool Setup...
setlocal

REM Check for Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not found in your system PATH.
    echo Please install Python 3.9, 3.10, or 3.11 from python.org and ensure it's added to PATH.
    pause
    exit /b 1
)
echo Found Python.

REM Define virtual environment name
set VENV_NAME=rag_env

REM Create virtual environment if it doesn't exist
if not exist %VENV_NAME%\Scripts\activate.bat (
    echo Creating Python virtual environment (%VENV_NAME%)...
    python -m venv %VENV_NAME%
    if %errorlevel% neq 0 (
        echo ERROR: Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo Virtual environment created.
) else (
    echo Virtual environment (%VENV_NAME%) already exists.
)

REM Activate virtual environment
echo Activating virtual environment...
call %VENV_NAME%\Scripts\activate.bat

REM Install packages from local wheels directory (offline install)
echo Installing required packages (this may take a while on first run, especially for llama-cpp-python)...
pip install --no-index --find-links=./wheels -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install packages.
    echo If llama-cpp-python failed, you might need Microsoft C++ Build Tools.
    echo See README.txt for more details.
    pause
    exit /b 1
)
echo Packages installed successfully.

REM Run the Streamlit application
echo Starting the RAG Q&A Tool...
REM Disable file watcher for Streamlit for better compatibility with some libraries
streamlit run app.py --server.fileWatcherType none --server.port 8501

echo Application has been launched. Check your web browser or open http://localhost:8501
pause
endlocal