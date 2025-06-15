#!/bin/bash

# LeRobot Setup Script
# This script automates the setup of the lerobot project

set -e  # Exit on any error

echo "🤖 Starting LeRobot setup..."

# Clean up any existing lerobot directory
if [ -d "lerobot" ]; then
    echo "📁 Removing existing lerobot directory..."
    rm -rf lerobot/
fi

# Clean up any existing virtual environment
if [ -d "lerobot_env" ]; then
    echo "🗑️  Removing existing lerobot_env..."
    rm -rf lerobot_env/
fi

# Clone the repository
echo "📥 Cloning lerobot repository..."
git clone https://github.com/huggingface/lerobot.git

# Enter the directory
cd lerobot

# Checkout specific commit
echo "🔄 Checking out commit b536f47..."
git checkout b536f47

# Go back to parent directory to create virtual environment
cd ..

# Create Python virtual environment
echo "🐍 Creating Python virtual environment..."
python3 -m venv lerobot_env

# Activate virtual environment
echo "✅ Activating virtual environment..."
source lerobot_env/bin/activate

# Enter lerobot directory
cd lerobot

# Install lerobot with pi0 extras
echo "📦 Installing lerobot with pi0 extras..."
pip install -e ".[pi0]"

# Uninstall and reinstall specific transformers version
echo "🔄 Installing transformers==4.51.3..."
pip uninstall -y transformers
pip install transformers==4.51.3

# Install additional packages
echo "📦 Installing additional packages..."
pip install wandb pytest scipy
sudo apt install -y ffmpeg libavformat-dev libavcodec-dev libavutil-dev libswscale-dev libavfilter-dev

echo "🎉 Installation complete!"
echo ""
echo "To use lerobot:"
echo "1. Activate the environment: source lerobot_env/bin/activate"
echo "2. Login to wandb: wandb login"
echo "3. Login to huggingface: huggingface-cli login"
echo ""
echo "Note: You'll need to run the login commands manually for security."
