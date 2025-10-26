# 📈 Invesight - Advanced Stock Prediction & Analysis System

**Invesight** is a comprehensive stock market analysis and prediction system built with Python, featuring machine learning models, technical analysis, and a beautiful web interface powered by Alpha Vantage API.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org)
[![API](https://img.shields.io/badge/API-Alpha%20Vantage-red.svg)](https://www.alphavantage.co)

## 🌐 Live Demo

**Try the live application:** [https://moh109.github.io/invesight/](https://moh109.github.io/invesight/)

## ✨ Features

### 🎯 **Core Functionality**
- **Real-time Stock Data**: Integration with Alpha Vantage API and Yahoo Finance
- **Machine Learning Predictions**: Multiple ML models (Random Forest, Linear Regression, Gradient Boosting, Neural Networks)
- **Technical Analysis**: 20+ technical indicators (RSI, MACD, Bollinger Bands, Moving Averages)
- **Performance Metrics**: Sharpe ratio, volatility, max drawdown, returns analysis
- **Interactive Visualizations**: Beautiful charts with Plotly

### 🎨 **User Interface**
- **Modern Web Dashboard**: Responsive Flask web interface
- **Dark Blue Theme**: Professional dark blue gradient backgrounds with glassmorphism effects
- **Animated Elements**: Hover effects, transitions, and smooth animations
- **Real-time Updates**: Live data fetching and analysis
- **Mobile Friendly**: Responsive design for all devices

### 📊 **Data Sources**
- **Primary**: Alpha Vantage API (professional financial data)
- **Fallback**: Yahoo Finance (yfinance library)
- **Sample Data**: Realistic approximations when APIs are unavailable

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Moh109/invesight.git
cd invesight
```

2. **Install dependencies**
```bash
pip install -r requirements_minimal.txt
```

3. **Run the application**
```bash
# Web Dashboard (Recommended)
python comprehensive_web_fixed.py

# Command Line Interface
python main.py

# Quick Start Script
python start.py
```

4. **Open your browser**
```
http://127.0.0.1:5000
```

## 📁 Project Structure

```
invesight/
├── 📊 Data Collection & Processing
│   ├── data_collector.py          # Alpha Vantage & Yahoo Finance integration
│   ├── feature_engineering.py     # Technical indicators & feature creation
│   └── utils.py                   # Data validation & performance calculations
│
├── 🤖 Machine Learning
│   ├── models.py                  # ML model training & prediction
│   └── config.py                  # Model configuration & parameters
│
├── 🎨 User Interface
│   ├── comprehensive_web_fixed.py # Main web dashboard (Flask)
│   ├── index.html                 # Static version for GitHub Pages
│   ├── visualization.py           # Chart generation with Plotly
│   └── start.py                   # Application launcher
│
├── 📈 Analysis & Testing
│   └── main.py                    # CLI interface
│
└── 📚 Documentation
    ├── README.md                  # This file
    ├── requirements_minimal.txt   # Python dependencies
    └── LICENSE                    # MIT License
```

## 🎯 Supported Stocks

### Major Tech Stocks
- **AAPL** - Apple Inc.
- **MSFT** - Microsoft Corporation
- **GOOGL** - Alphabet Inc.
- **AMZN** - Amazon.com Inc.
- **META** - Meta Platforms Inc.
- **NVDA** - NVIDIA Corporation
- **TSLA** - Tesla Inc.

### Other Popular Stocks
- **NFLX** - Netflix Inc.
- **AMD** - Advanced Micro Devices
- **INTC** - Intel Corporation
- **BABA** - Alibaba Group
- **JPM** - JPMorgan Chase
- **WMT** - Walmart Inc.

*And many more! The system supports any valid stock symbol.*

## 🔧 Configuration

### API Keys (Optional)
For enhanced data access, you can add your Alpha Vantage API key:

1. Get a free API key from [Alpha Vantage](https://www.alphavantage.co/support/#api-key)
2. Create a `.env` file in the project root:
```env
ALPHA_VANTAGE_API_KEY=your_api_key_here
```

### Customization
- **Time Periods**: 1 month, 3 months, 6 months, 1 year
- **Technical Indicators**: RSI, MACD, Bollinger Bands, Moving Averages
- **ML Models**: Random Forest, Linear Regression, Gradient Boosting, Neural Networks

## 📊 Technical Indicators

The system calculates 20+ technical indicators:

- **Trend Indicators**: SMA, EMA, MACD, ADX
- **Momentum Indicators**: RSI, Stochastic, Williams %R
- **Volatility Indicators**: Bollinger Bands, ATR
- **Volume Indicators**: OBV, Volume SMA, Volume Ratio
- **Custom Features**: Price ratios, lag features, rolling statistics

## 🤖 Machine Learning Models

### Model Types
1. **Random Forest**: Ensemble method for robust predictions
2. **Linear Regression**: Baseline trend analysis
3. **Gradient Boosting**: Advanced ensemble learning
4. **Neural Networks**: Deep learning for complex patterns

### Performance Metrics
- **Accuracy**: Model prediction accuracy
- **Sharpe Ratio**: Risk-adjusted returns
- **Volatility**: Price movement variability
- **Max Drawdown**: Maximum loss from peak
- **Returns**: Percentage price changes

## 🎨 Web Interface Features

### Dashboard Components
- **Stock Input**: Symbol and time period selection
- **Real-time Analysis**: Live data processing
- **Interactive Charts**: Candlestick, volume, and technical analysis
- **Performance Metrics**: Key financial indicators
- **Responsive Design**: Works on desktop and mobile

### Visual Elements
- **Gradient Backgrounds**: Modern purple-blue gradients
- **Animated Elements**: Hover effects and transitions
- **Color-coded Metrics**: Green/red for gains/losses
- **Professional Styling**: Clean, modern interface

## 📈 Sample Output

### Web Dashboard
- Interactive stock charts with technical indicators
- Real-time performance metrics
- Beautiful gradient-based UI
- Responsive design for all devices

### Command Line
```
📊 Stock Analysis Results for AAPL (1y)
✅ Analysis completed successfully!

Current Price: $175.23
Price Change: +12.45%
Volatility: 2.34%
Sharpe Ratio: 1.67
Max Drawdown: -8.23%
Average Volume: 45,234,567

🤖 ML Predictions (Next 5 Days):
Day 1: $176.45 (+0.70%)
Day 2: $177.89 (+1.52%)
Day 3: $179.12 (+2.22%)
Day 4: $178.67 (+1.96%)
Day 5: $180.34 (+2.92%)
```

## 🔄 Data Flow

```
1. User Input (Stock Symbol + Time Period)
   ↓
2. Data Collection (Alpha Vantage → Yahoo Finance → Sample Data)
   ↓
3. Feature Engineering (Technical Indicators + Custom Features)
   ↓
4. Machine Learning (Model Training + Predictions)
   ↓
5. Performance Analysis (Metrics + Risk Assessment)
   ↓
6. Visualization (Interactive Charts + Results Display)
```

## 🛠️ Dependencies

### Core Libraries
- **Flask**: Web framework
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning
- **Plotly**: Interactive visualizations

### Data Sources
- **yfinance**: Yahoo Finance data
- **requests**: HTTP requests for Alpha Vantage
- **Alpha Vantage API**: Professional financial data

See `requirements_minimal.txt` for complete list

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Alpha Vantage** for providing professional financial data API
- **Yahoo Finance** for backup data source
- **Plotly** for beautiful interactive visualizations
- **Scikit-learn** for machine learning capabilities

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/Moh109/invesight/issues) page
2. Create a new issue with detailed description
3. Include error messages and system information

## 🎯 Future Enhancements

- [ ] Real-time portfolio tracking
- [ ] Options analysis
- [ ] Cryptocurrency support
- [ ] Advanced ML models (LSTM, Transformer)
- [ ] Mobile app development
- [ ] Social sentiment analysis
- [ ] Automated trading signals

---

**⭐ Star this repository if you found it helpful!**

*Built with ❤️ for the financial analysis community*