#!/bin/bash

# LTX-2 Model Download Script
# Wrapper for the Python downloader which handles authentication and large files better

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}LTX-2 Model Downloader${NC}"
echo "======================"

# Check if python is available
if ! command -v python &> /dev/null; then
    echo "Error: python not found"
    exit 1
fi

# Check if huggingface_hub is installed
if ! python -c "import huggingface_hub" &> /dev/null; then
    echo -e "${YELLOW}Installing huggingface_hub...${NC}"
    pip install huggingface_hub
fi

# Run the python downloader
python download_models.py
