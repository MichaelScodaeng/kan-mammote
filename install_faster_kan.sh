#!/bin/bash
# Install Faster-KAN dependency for KAN-MAMMOTE

echo "Installing Faster-KAN for KAN-MAMMOTE..."

# Check if git is available
if ! command -v git &> /dev/null; then
    echo "Error: git is required but not installed."
    exit 1
fi

# Clone and install Faster-KAN
if [ ! -d "faster-kan" ]; then
    echo "Cloning Faster-KAN repository..."
    git clone https://github.com/AthanasiosDelis/faster-kan.git
else
    echo "Faster-KAN repository already exists, pulling latest changes..."
    cd faster-kan
    git pull
    cd ..
fi

# Install Faster-KAN
echo "Installing Faster-KAN..."
cd faster-kan
pip install -e .
cd ..

echo "Faster-KAN installation completed!"
echo ""
echo "You can now run the KAN-MAMMOTE tests with Faster-KAN support:"
echo "python test_c_mamba_embeddings.py"
