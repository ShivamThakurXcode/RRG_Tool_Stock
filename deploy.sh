#!/bin/bash

# RRG Tool Deployment Script
# This script helps deploy the RRG tool to your server

echo "🚀 RRG Tool Deployment Script"
echo "=============================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

echo "✅ Python found: $(python3 --version)"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📦 Installing required packages..."
pip install -r rrg_tool/requirements.txt

# Check if installation was successful
if [ $? -eq 0 ]; then
    echo "✅ All packages installed successfully!"
else
    echo "❌ Package installation failed. Please check the requirements.txt file."
    exit 1
fi

# Create systemd service file for auto-start
echo "🔧 Creating systemd service..."
cat > rrg_tool.service << EOF
[Unit]
Description=RRG Tool Streamlit App
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
Environment=PATH=$(pwd)/venv/bin
ExecStart=$(pwd)/venv/bin/streamlit run rrg_tool/rrg_tool.py --server.port 8501 --server.headless true --server.enableCORS false
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

echo "✅ Deployment completed!"
echo ""
echo "📋 Next steps:"
echo "1. Copy the files to your server"
echo "2. Run: sudo cp rrg_tool.service /etc/systemd/system/"
echo "3. Run: sudo systemctl enable rrg_tool.service"
echo "4. Run: sudo systemctl start rrg_tool.service"
echo "5. Access at: http://your-domain.com/rrg_tool.php"
echo ""
echo "🔧 Manual start command:"
echo "source venv/bin/activate && streamlit run rrg_tool/rrg_tool.py --server.port 8501" 