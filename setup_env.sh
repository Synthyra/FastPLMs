#!/bin/bash

# chmod +x setup_protify.sh
# ./setup_protify.sh

# Set up error handling
set -e  # Exit immediately if a command exits with a non-zero status

echo "Setting up Python virtual environment..."

# Create virtual environment
python3 -m venv ~/env

# Activate virtual environment
source ~/env/bin/activate

# Update pip and setuptools
echo "Upgrading pip and setuptools..."
pip install pip setuptools -U

# Install requirements with force reinstall
echo "Installing requirements"
# Install torch and torchvision
echo "Installing torch and torchvision..."
git clone https://github.com/huggingface/transformers.git
cd transformers
pip install -e .
cd ..
git clone https://github.com/Profluent-AI/E1.git
cd E1
pip install -e .
cd ..
pip install esm
pip install -r requirements.txt -U
pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128 -U
pip install --force-reinstall numpy==1.26.4

# List installed packages for verification
echo -e "\nInstalled packages:"
pip list

# Instructions for future use
echo -e "\n======================="
echo "Setup complete!"
echo "======================="
echo "To activate this environment in the future, run:"
echo "    source ~/env/bin/activate"
echo ""
echo "To deactivate the environment, simply run:"
echo "    deactivate"
echo ""
echo "Your virtual environment is located at: ~/env"
echo "======================="

