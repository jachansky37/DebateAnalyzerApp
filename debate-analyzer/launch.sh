#!/bin/bash

echo "Setting up virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Launching application..."
python3 debate-analyzer/graph_mindmap.py