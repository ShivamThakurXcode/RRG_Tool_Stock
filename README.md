# RRG Tool - Sector Rotation Analysis

A powerful Relative Rotation Graph (RRG) analysis tool for sector rotation analysis using Python Streamlit.

## 🚀 Features

- **Real-time Market Data**: Fetches data from Yahoo Finance
- **Interactive RRG Charts**: Visualize sector performance vs benchmark
- **Quadrant Analysis**: Leading, Weakening, Lagging, and Improving sectors
- **Customizable Parameters**: Adjustable analysis periods and tail lengths
- **Beautiful UI**: Modern, responsive interface

## 📋 Requirements

- Python 3.8+
- pip (Python package manager)
- Internet connection for market data

## 🛠️ Installation

### Option 1: Quick Start (Windows)
```bash
# Run the deployment script
deploy.bat
```

### Option 2: Manual Installation
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate.bat
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r rrg_tool/requirements.txt

# Run the application
streamlit run rrg_tool/rrg_tool.py --server.port 8501
```

## 🌐 Server Deployment

### Step 1: Upload Files
Upload all project files to your server at `dandeli.com/rrg_tool/`

### Step 2: Install Dependencies
```bash
cd /path/to/rrg_tool
python3 -m venv venv
source venv/bin/activate
pip install -r rrg_tool/requirements.txt
```

### Step 3: Set Up System Service
```bash
# Copy service file
sudo cp rrg_tool.service /etc/systemd/system/

# Enable and start service
sudo systemctl enable rrg_tool.service
sudo systemctl start rrg_tool.service

# Check status
sudo systemctl status rrg_tool.service
```

### Step 4: Configure Web Server
Add to your Apache/Nginx configuration:
```apache
# Apache
ProxyPass /rrg_tool http://localhost:8501
ProxyPassReverse /rrg_tool http://localhost:8501
```

```nginx
# Nginx
location /rrg_tool {
    proxy_pass http://localhost:8501;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
}
```

## 📊 Usage

1. **Access the Tool**: Visit `dandeli.com/rrg_tool.php`
2. **Enter Benchmark**: Default is `^NSEI` (Nifty 50)
3. **Add Sectors**: Enter sector symbols (one per line)
4. **Configure Parameters**:
   - Analysis Period: 65-365 days
   - Tail Length: 2-25 days
5. **Run Analysis**: Click "Run Analysis"
6. **Interpret Results**: View quadrants and trajectories

## 🎯 Understanding RRG Quadrants

- **Leading (Top-Right)**: High strength, positive momentum
- **Weakening (Bottom-Right)**: High strength, negative momentum  
- **Lagging (Bottom-Left)**: Low strength, negative momentum
- **Improving (Top-Left)**: Low strength, positive momentum

## 📁 Project Structure

```
RRG_TOOL/
├── rrg_tool/
│   ├── rrg_tool.py          # Main Streamlit application
│   └── requirements.txt     # Python dependencies
├── rrg_tool.php            # PHP wrapper for web interface
├── deploy.sh               # Linux deployment script
├── deploy.bat              # Windows deployment script
├── rrg_tool.service        # Systemd service file
└── README.md              # This file
```

## 🔧 Troubleshooting

### Common Issues

1. **Port 8501 already in use**:
   ```bash
   # Find and kill process
   lsof -ti:8501 | xargs kill -9
   ```

2. **Package installation fails**:
   ```bash
   # Upgrade pip and setuptools
   pip install --upgrade pip setuptools wheel
   ```

3. **Streamlit not starting**:
   ```bash
   # Check Python version
   python --version
   # Should be 3.8+
   ```

### Service Management
```bash
# Start service
sudo systemctl start rrg_tool.service

# Stop service
sudo systemctl stop rrg_tool.service

# Restart service
sudo systemctl restart rrg_tool.service

# View logs
sudo journalctl -u rrg_tool.service -f
```

## 📈 Default Sectors

The tool comes pre-configured with popular Indian sector indices:
- `^CNXAUTO` - Auto
- `^CNXPHARMA` - Pharma
- `^CNXMETAL` - Metal
- `^CNXIT` - IT
- `^CNXENERGY` - Energy
- `^CNXREALTY` - Realty
- `^CNXPSUBANK` - PSU Bank
- `^CNXMEDIA` - Media
- `^CNXINFRA` - Infrastructure
- `^CNXPSE` - PSE
- `RELIANCE.NS` - Reliance Industries
- `INFY.NS` - Infosys

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🆘 Support

For issues and questions:
- Check the troubleshooting section above
- Review the logs: `sudo journalctl -u rrg_tool.service`
- Ensure all dependencies are installed correctly
