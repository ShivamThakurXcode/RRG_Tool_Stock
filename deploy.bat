@echo off
echo 🚀 RRG Tool Deployment Script for Windows
echo ==========================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo ✅ Python found
python --version

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo ⬆️ Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo 📦 Installing required packages...
pip install -r rrg_tool\requirements.txt

REM Check if installation was successful
if errorlevel 1 (
    echo ❌ Package installation failed. Please check the requirements.txt file.
    pause
    exit /b 1
)

echo ✅ All packages installed successfully!
echo.
echo 🚀 Starting RRG Tool...
echo Access the tool at: http://localhost:8501
echo.
echo Press Ctrl+C to stop the application
echo.

REM Start the Streamlit app
streamlit run rrg_tool\rrg_tool.py --server.port 8501

pause 