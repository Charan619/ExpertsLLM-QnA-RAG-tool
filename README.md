# RAG-QA-tool README

## RAG Expert Q&A Tool Setup Instructions

### Prerequisites:
1.  **Python 3.9, 3.10, or 3.11 installed.**
    *   Download Python from [python.org](https://www.python.org/).
    *   During installation, **make sure to check the box "Add Python to PATH"**.
2.  **(For Windows, if `llama-cpp-python` compilation fails): Microsoft C++ Build Tools might be needed.**
    *   Search for "Visual Studio Build Tools" and install the "Desktop development with C++" workload.

### Setup:
1.  Extract the `RAG_Expert_QA_Tool.zip` file to a location on your computer (e.g., your Desktop).
2.  Open the extracted `RAG_Expert_QA_Tool` folder.

### Running the Application:
*   **On Windows:**
    *   Double-click the `setup_and_run.bat` file.
*   **On Linux or macOS (Tested):**
    1.  Open a **Terminal**.
    2.  Navigate to this `RAG_Expert_QA_Tool` folder (e.g., `cd Desktop/RAG_Expert_QA_Tool`).
    3.  Make the script executable: `chmod +x setup_and_run.sh`
    4.  Run the script: `./setup_and_run.sh`

### First Run:
*   The script will first try to set up a local Python environment and install necessary packages.
    *   This might take **several minutes** and will show a lot of text in the command window. This is normal.
*   If `llama-cpp-python` needs to compile, it can take extra time.

### Subsequent Runs:
*   Simply run the same `.bat` or `.sh` script again. It should start much faster.

### Accessing the Tool:
*   After the script finishes, a web browser should open automatically to the application.
*   If not, the script will print a URL (e.g., `http://localhost:8501`). Open this URL in your web browser.

### Troubleshooting:
*   **If `llama-cpp-python` fails to install/compile:**
    *   Ensure you have a C++ compiler installed:
        *   **Windows:** Microsoft C++ Build Tools (see Prerequisites).
        *   **Linux:** GCC (usually installed via `sudo apt install build-essential` or similar).
        *   **macOS:** Clang (usually installed via XCode Command Line Tools: `xcode-select --install`).
    *   For **GPU support (NVIDIA)**, the CUDA toolkit must be installed and configured correctly. This is an advanced setup not covered by the basic installation. This package is configured for CPU by default (or minimal GPU offload if `llama-cpp-python` detects a compatible GPU and your `N_GPU_LAYERS` setting in `app.py` is > 0).
*   Ensure your PDF transcripts are placed in the `expert_call_transcripts` folder within the main `RAG_Expert_QA_Tool` directory.