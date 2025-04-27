#!/bin/bash

echo "Creating Conda environment from environment.yml..."

# Create environment
conda env create -f environment.yml

echo "Environment created!"

# Activate environment (cannot activate inside script easily)
echo "To activate the environment, run:"
echo "  conda activate spline-interpolation-enhancement"

echo "Setup completed!"
