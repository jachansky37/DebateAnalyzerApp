#!/bin/bash

# Navigate to the project root directory
cd "$(dirname "$0")/.."

# Install Homebrew and Graphviz if not present
if ! command -v brew &> /dev/null; then
    echo "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

if ! command -v dot &> /dev/null; then
    echo "Installing Graphviz system binary via Homebrew..."
    brew install graphviz
fi

# Set up Python virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

# Set locale (avoids matplotlib font issues on some systems)
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8

echo "Upgrading pip and installing dependencies..."
pip install --upgrade pip
pip install -r debate-analyzer/requirements.txt

# Install pygraphviz after virtual environment is active
GRAPHVIZ_PREFIX=$(brew --prefix graphviz)
pip install --no-cache-dir \
    --config-settings="--global-option=build_ext" \
    --config-settings="--global-option=-I${GRAPHVIZ_PREFIX}/include/" \
    --config-settings="--global-option=-L${GRAPHVIZ_PREFIX}/lib/" \
    pygraphviz

echo "Launching application..."
python3 debate-analyzer/debate_unit_extractor.py
python3 debate-analyzer/idea_extractor.py
python3 debate-analyzer/semantic_graph_builder.py
python3 debate-analyzer/graph_mindmap_v3.py