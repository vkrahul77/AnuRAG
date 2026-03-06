#!/bin/bash
# AnuRAG Setup Script for Linux/Mac

echo "============================================================"
echo "AnuRAG Environment Setup"
echo "============================================================"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: Conda is not installed or not in PATH"
    echo "Please install Anaconda or Miniconda first"
    exit 1
fi

# Create conda environment
echo ""
echo "Creating conda environment 'anurag'..."
conda create -n anurag python=3.11 -y

# Activate environment
echo ""
echo "Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate anurag

# Install PyTorch
echo ""
echo "Installing PyTorch..."
pip install torch torchvision

# Install main requirements
echo ""
echo "Installing requirements..."
pip install -r requirements.txt

# Install additional dependencies
echo ""
echo "Installing additional dependencies..."
pip install google-generativeai
pip install nltk
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

echo ""
echo "============================================================"
echo "Setup Complete!"
echo "============================================================"
echo ""
echo "Next steps:"
echo "1. Create a .env file with your GOOGLE_API_KEY"
echo "2. Run: conda activate anurag"
echo "3. cd gemini/tools"
echo "4. python main.py --help"
