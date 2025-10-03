@echo off
REM Quick setup script for the forecasting application (Windows)

echo 🚀 Setting up Stock Forecasting Application...
echo ================================================

REM Check if we're in the right directory
if not exist "requirements.txt" (
    echo ❌ Error: requirements.txt not found. Make sure you're in the project directory.
    pause
    exit /b 1
)

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo 📚 Installing dependencies...
pip install -r requirements.txt

REM Create necessary directories
echo 📁 Creating directories...
if not exist "logs" mkdir logs
if not exist "data\raw" mkdir data\raw
if not exist "data\processed" mkdir data\processed

REM Initialize Git if not already done
if not exist ".git" (
    echo 🔄 Initializing Git repository...
    git init
    git add .
    git commit -m "Initial project setup"
)

echo.
echo ✅ Setup complete!
echo.
echo 🏁 To start the application:
echo    1. Make sure MongoDB is running
echo    2. Run: python app.py
echo    3. Visit: http://localhost:5000
echo.
echo 📋 Next steps:
echo    1. Check README.md for detailed phase tracking
echo    2. Start with Phase 1 tasks
echo    3. Implement data collection first
echo.
pause