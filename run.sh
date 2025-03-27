#!/bin/bash

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Check if OPENAI_API_KEY is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "WARNING: OPENAI_API_KEY environment variable is not set."
    echo "The application will run with simulated responses."
    echo "To set the API key, run: export OPENAI_API_KEY='your-api-key'"
fi

# Create necessary directories if they don't exist
mkdir -p data/raw
mkdir -p data/processed
mkdir -p outputs
mkdir -p templates
mkdir -p static

# Run the application
echo "Starting the application..."
python app.py
