#!/bin/bash
# Quick setup script for the forecasting application

echo "🚀 Setting up Stock Forecasting Application..."
echo "================================================"

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: requirements.txt not found. Make sure you're in the project directory."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment (Windows)
echo "🔧 Activating virtual environment..."
source venv/Scripts/activate 2>/dev/null || source venv/bin/activate

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p logs
mkdir -p data/raw
mkdir -p data/processed

# Initialize Git if not already done
if [ ! -d ".git" ]; then
    echo "🔄 Initializing Git repository..."
    git init
    git add .
    git commit -m "Initial project setup"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "🏁 To start the application:"
echo "   1. Make sure MongoDB is running"
echo "   2. Run: python app.py"
echo "   3. Visit: http://localhost:5000"
echo ""
echo "📋 Next steps:"
echo "   1. Check README.md for detailed phase tracking"
echo "   2. Start with Phase 1 tasks"
echo "   3. Implement data collection first"
echo ""