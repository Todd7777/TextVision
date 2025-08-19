#!/bin/bash
set -e  # Exit on error

# Install system dependencies
apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    tesseract-ocr \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1-mesa-glx \
    libjpeg-dev \
    zlib1g-dev \
    libtiff5-dev \
    libfreetype6-dev \
    liblcms2-dev \
    libwebp-dev \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages with --no-cache-dir to prevent caching
pip3 install --no-cache-dir -r requirements.txt
