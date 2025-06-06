#!/bin/bash

# Open Terminal window if not already
cd "$(dirname "$0")"

# Prompt for admin access
sudo -v

echo "➡️  Checking for Homebrew..."
if ! command -v brew &>/dev/null; then
    echo "🍺 Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
else
    echo "✅ Homebrew already installed."
fi

echo "➡️  Installing Git and Python (if missing)..."
brew install git python

# Move into project subfolder if launched from root
cd "$(dirname "$0")/debate-analyzer"

echo "🐍 Setting up Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Install pygraphviz with system headers
GRAPHVIZ_PREFIX=$(brew --prefix graphviz)
pip install --no-cache-dir \
    --config-settings="--global-option=build_ext" \
    --config-settings="--global-option=-I${GRAPHVIZ_PREFIX}/include/" \
    --config-settings="--global-option=-L${GRAPHVIZ_PREFIX}/lib/" \
    pygraphviz

echo "🚀 Launching app..."
bash launch.sh