#!/bin/bash

# Navigate to the project root directory
cd "$(dirname "$0")/.."

echo "Setting up virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r debate-analyzer/requirements.txt

echo "Launching application..."
python3 debate-analyzer/debate_unit_extractor.py
python3 debate-analyzer/idea_extractor.py
python3 debate-analyzer/graph_mindmap.py