#!/bin/bash

# Create directory structure for the project
echo "Creating directory structure for Traffic Sign Recognition System..."

# Core directories
mkdir -p data
mkdir -p outputs  
mkdir -p models
mkdir -p evaluation_results

# Web app assets
mkdir -p static
mkdir -p templates  
mkdir -p samples

# Documentation
mkdir -p docs
mkdir -p examples

# Tests (for future development)
mkdir -p tests

echo "Directory structure created successfully!"
echo ""
echo "Created directories:"
echo "├── data/                    # Dataset storage"
echo "├── outputs/                 # Trained models and results"
echo "├── models/                  # Model architectures"  
echo "├── evaluation_results/      # Evaluation outputs"
echo "├── static/                  # Web app static files"
echo "├── templates/               # Web app templates"
echo "├── samples/                 # Sample images for testing"
echo "├── docs/                    # Additional documentation"
echo "├── examples/                # Usage examples"
echo "└── tests/                   # Unit tests"
echo ""
echo "Ready to use! Run 'python test_installation.py' to verify setup."
