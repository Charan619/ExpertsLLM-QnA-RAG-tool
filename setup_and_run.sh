#!/bin/bash
echo "Starting RAG Expert Q&A Tool Setup..."

# Check for Python
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "ERROR: Python is not installed or not found in your system PATH."
    echo "Please install Python 3.9, 3.10, or 3.11 and ensure it's available as 'python3' or 'python'."
    exit 1
fi
# Prefer python3 if available
PYTHON_CMD=python3
if ! command -v python3 &> /dev/null; then
    PYTHON_CMD=python
fi
echo "Found Python using '$PYTHON_CMD'."

# Define virtual environment name
VENV_NAME="rag_env"

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_NAME/bin/activate" ]; then
    echo "Creating Python virtual environment ($VENV_NAME)..."
    $PYTHON_CMD -m venv $VENV_NAME
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to create virtual environment."
        exit 1
    fi
    echo "Virtual environment created."
else
    echo "Virtual environment ($VENV_NAME) already exists."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_NAME/bin/activate"

# Install packages from local wheels directory (offline install)
echo "Installing required packages (this may take a while on first run, especially for llama-cpp-python)..."
# The --no-build-isolation flag can sometimes help with llama-cpp-python if pre-built wheels are tricky
pip install --no-index --find-links=./wheels -r requirements.txt
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install packages."
    echo "If llama-cpp-python failed, ensure you have a C++ compiler (like GCC or Clang)."
    echo "See README.txt for more details."
    exit 1
fi
echo "Packages installed successfully."

# Run the Streamlit application
echo "Starting the RAG Q&A Tool..."
# Disable file watcher for Streamlit
streamlit run app.py --server.fileWatcherType none --server.port 8501

echo "Application has been launched. Check your web browser or open http://localhost:8501"
# Keep terminal open if run directly, or user can close it
# read -p "Press Enter to close this window..."