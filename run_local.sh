#!/bin/bash

# Declutter App Local Runner
# This script helps you run the application locally

echo "🚀 Starting Declutter App Locally..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install/update dependencies
echo "📥 Installing dependencies (this may take a few minutes on first run)..."
cd backend
pip install --upgrade pip
pip install -r requirements.txt
pip install openai

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "⚠️  No .env file found. Creating template..."
    echo "OPENAI_API_KEY=your-api-key-here" > .env
    echo "📝 Please edit backend/.env with your OpenAI API key (optional)"
fi

# Run the server
echo ""
echo "✅ Starting server at http://localhost:8000"
echo "Press CTRL+C to stop"
echo ""
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
