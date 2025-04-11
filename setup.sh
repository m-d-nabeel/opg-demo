#!/usr/bin/env bash
# This script sets up the environment for the project.
# It installs the required packages and sets up the virtual environment.

# setup the environment
echo "Setting up the environment..."
# Check if Python 3 is installed
if ! command -v python3 &> /dev/null
then
    echo "Python 3 is not installed. Please install Python 3 and try again."
    exit
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null
then
    echo "pip is not installed. Please install pip and try again."
    exit
fi

# setup the virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
else
    echo "Virtual environment already exists."
fi

# Make the script work with both bash and zsh
if [ -n "$BASH_VERSION" ] || [ -n "$ZSH_VERSION" ]; then
    SCRIPT_PATH="${BASH_SOURCE[0]:-$0}"
    SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
    # Use POSIX-compatible approach for sourcing virtual environment
    . "$SCRIPT_DIR/venv/bin/activate"
else
    # For other shells
    source venv/bin/activate
fi

echo "Installing required packages..."
pip install -r requirements.txt
echo "Environment setup complete."
echo "To activate the virtual environment, run: source venv/bin/activate"
echo "To deactivate the virtual environment, run: deactivate"
echo "To run the project, use: python run.py"