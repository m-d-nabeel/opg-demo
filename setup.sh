#!/bin/env bash
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
source venv/bin/activate
echo "Installing required packages..."
pip install -r requirements.txt
echo "Environment setup complete."
echo "To activate the virtual environment, run 'source venv/bin/activate'."
echo "To deactivate the virtual environment, run 'deactivate'."
echo "To run the project, use 'python run.py'."