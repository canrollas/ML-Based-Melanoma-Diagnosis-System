#!/bin/bash

echo "1. Downloading ISIC images..."

isic image download --search 'benign_malignant:"malignant"' cli_images/malignant --limit 4500
isic image download --search 'benign_malignant:"benign"' cli_images/benign --limit 4500

echo -e "\n2. Processing images and preparing for neural network..."
# Run Python script
python cli_data_prepare.py

echo -e "\nProcess completed! Data is ready in neural_network_data directory." 