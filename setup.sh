#!/bin/bash

# Traffic Sign Recognition System Setup Script

echo "Setting up Traffic Sign Recognition System..."

# Create virtual environment
echo "Creating virtual environment..."
python -m venv traffic_sign_env

# Activate virtual environment
echo "Activating virtual environment..."
source traffic_sign_env/bin/activate  # Linux/Mac
# traffic_sign_env\Scripts\activate  # Windows

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p data
mkdir -p outputs
mkdir -p models
mkdir -p evaluation_results

# Download GTSRB dataset (optional)
echo "To download GTSRB dataset, run:"
echo "python train.py --download_data"

echo "Setup complete!"
echo ""
echo "To get started:"
echo "1. Activate the environment: source traffic_sign_env/bin/activate"
echo "2. Download data: python train.py --download_data"
echo "3. Train model: python train.py --architecture custom --epochs 50"
echo "4. Evaluate model: python evaluate.py --model_path outputs/final_model_custom.h5"
echo "5. Real-time detection: python realtime_detection.py --model_path outputs/final_model_custom.h5"
