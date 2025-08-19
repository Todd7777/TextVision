#!/bin/bash
set -e  # Exit on error

# Install system dependencies
apt-get update
apt-get install -y --no-install-recommends \
    poppler-utils \
    tesseract-ocr \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1-mesa-glx

# Clean up to reduce image size
apt-get clean
rm -rf /var/lib/apt/lists/*
