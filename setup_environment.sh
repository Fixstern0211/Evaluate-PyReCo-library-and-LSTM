#!/bin/bash
set -e

echo "🚀 Setting up PyReCo vs LSTM Evaluation Environment"
echo "=================================================="

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda is not installed. Please install Anaconda or Miniconda first."
    echo "   Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Create conda environment from environment.yml
echo "📦 Creating conda environment from environment.yml..."
conda env create -f environment.yml

echo ""
echo "✅ Environment setup complete!"
echo ""
echo "🔧 To activate the environment, run:"
echo "   conda activate pyreco-lstm-evaluation"
echo ""
echo "📋 To verify the installation, run:"
echo "   ./validate_environment.sh"
echo ""
echo "🧪 To run a quick test, try:"
echo "   python test_new_datasets.py"