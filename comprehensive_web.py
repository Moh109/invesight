"""
Comprehensive web interface with multiple technical analysis charts.
"""
from flask import Flask, render_template_string, request, send_from_directory
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
# yfinance, requests, json imports removed - using StockDataCollector instead
from feature_engineering import FeatureEngineer
from utils import PerformanceCalculator
from data_collector import StockDataCollector

# Alpha Vantage configuration removed - using StockDataCollector instead

app = Flask(__name__)

# Add route to serve static files
@app.route('/plots/<path:filename>')
def serve_plot(filename):
    return send_from_directory('plots', filename)

# Alpha Vantage data fetching function removed - using StockDataCollector class instead

def fetch_real_data(symbol="AAPL", period="1y"):
    """Fetch real stock data using StockDataCollector."""
    try:
        print(f"📊 Fetching data for {symbol}...")
        data_collector = StockDataCollector()
        data = data_collector.get_yahoo_data(symbol, period)
        
        if data is not None and not data.empty:
            print(f"✅ Successfully fetched {len(data)} days of data for {symbol}")
            return data
        else:
            print(f"❌ No data found for {symbol}")
            return None
            
    except Exception as e:
        print(f"❌ Error fetching data for {symbol}: {e}")
        return None

def create_sample_data(symbol="AAPL", days=180):
    """Create realistic sample stock data as fallback."""
    print(f"⚠️  Using sample data for {symbol}")
    dates = pd.date_range(start=datetime.now() - timedelta(days=days), end=datetime.now(), freq='D')
    
    # Different base prices and characteristics for different stocks (updated with current realistic prices)
    stock_profiles = {
        'AAPL': {'base_price': 180, 'volatility': 0.02, 'trend': 0.0005},
        'GOOGL': {'base_price': 140, 'volatility': 0.025, 'trend': 0.0003},
        'MSFT': {'base_price': 380, 'volatility': 0.018, 'trend': 0.0004},
        'TSLA': {'base_price': 250, 'volatility': 0.04, 'trend': 0.0008},
        'AMZN': {'base_price': 228, 'volatility': 0.022, 'trend': 0.0002},
        'NVDA': {'base_price': 850, 'volatility': 0.035, 'trend': 0.001},
        'META': {'base_price': 320, 'volatility': 0.03, 'trend': 0.0006},
        'NFLX': {'base_price': 450, 'volatility': 0.028, 'trend': 0.0003},
        'AMD': {'base_price': 140, 'volatility': 0.032, 'trend': 0.0007},
        'INTC': {'base_price': 35, 'volatility': 0.025, 'trend': 0.0001},
        'AMC': {'base_price': 5, 'volatility': 0.08, 'trend': 0.0001},
        'PLTR': {'base_price': 15, 'volatility': 0.05, 'trend': 0.0003},
        'BABA': {'base_price': 159, 'volatility': 0.03, 'trend': 0.0002},
        'ADBE': {'base_price': 580, 'volatility': 0.025, 'trend': 0.0003},
        'IBM': {'base_price': 180, 'volatility': 0.02, 'trend': 0.0001},
        'KO': {'base_price': 60, 'volatility': 0.015, 'trend': 0.0001},
        'JPM': {'base_price': 180, 'volatility': 0.02, 'trend': 0.0002},
        'WMT': {'base_price': 160, 'volatility': 0.015, 'trend': 0.0001},
        'JNJ': {'base_price': 160, 'volatility': 0.015, 'trend': 0.0001},
        'PG': {'base_price': 150, 'volatility': 0.015, 'trend': 0.0001},
        'XOM': {'base_price': 120, 'volatility': 0.025, 'trend': 0.0001}
    }
    
    # Get stock profile or use default based on symbol characteristics
    profile = stock_profiles.get(symbol.upper(), None)
    
    if profile is None:
        # Create realistic default based on symbol characteristics
        symbol_upper = symbol.upper()
        
        # Tech stocks typically higher prices
        if any(tech in symbol_upper for tech in ['GOOGL', 'GOOG', 'META', 'FB', 'NFLX', 'ADBE', 'CRM', 'ORCL', 'BABA']):
            profile = {'base_price': 200, 'volatility': 0.025, 'trend': 0.0003}
        # Semiconductor stocks
        elif any(semi in symbol_upper for semi in ['NVDA', 'AMD', 'INTC', 'QCOM', 'AVGO', 'TXN', 'MRVL']):
            profile = {'base_price': 300, 'volatility': 0.035, 'trend': 0.0007}
        # Financial stocks
        elif any(fin in symbol_upper for fin in ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP']):
            profile = {'base_price': 80, 'volatility': 0.02, 'trend': 0.0002}
        # Healthcare/Biotech
        elif any(health in symbol_upper for health in ['JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO', 'ABT']):
            profile = {'base_price': 120, 'volatility': 0.018, 'trend': 0.0001}
        # Energy stocks
        elif any(energy in symbol_upper for energy in ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'OXY']):
            profile = {'base_price': 60, 'volatility': 0.03, 'trend': 0.0001}
        # Retail/Consumer
        elif any(retail in symbol_upper for retail in ['WMT', 'TGT', 'HD', 'LOW', 'COST', 'SBUX']):
            profile = {'base_price': 150, 'volatility': 0.02, 'trend': 0.0002}
        # Crypto-related stocks
        elif any(crypto in symbol_upper for crypto in ['RIOT', 'MARA', 'COIN', 'HOOD', 'SQ', 'PYPL']):
            profile = {'base_price': 15, 'volatility': 0.08, 'trend': 0.001}
        # Meme stocks
        elif any(meme in symbol_upper for meme in ['AMC', 'GME', 'BB', 'NOK', 'CLOV', 'WISH']):
            profile = {'base_price': 8, 'volatility': 0.1, 'trend': 0.0001}
        # Default for unknown stocks
        else:
            profile = {'base_price': 50, 'volatility': 0.025, 'trend': 0.0003}
    
    # Use symbol hash for consistent but different random seed
    np.random.seed(hash(symbol) % 2**32)
    
    base_price = profile['base_price']
    volatility = profile['volatility']
    trend = profile['trend']
    
    # Create price series with trend and volatility
    price_changes = np.random.randn(len(dates)) * volatility + trend
    prices = [base_price]
    
    for change in price_changes[1:]:
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 1))
    
    # Create OHLC data
    data = pd.DataFrame({
        'open': [p * (1 + np.random.randn() * 0.005) for p in prices],
        'high': [p * (1 + abs(np.random.randn()) * 0.01) for p in prices],
        'low': [p * (1 - abs(np.random.randn()) * 0.01) for p in prices],
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, len(dates))
    }, index=dates)
    
    # Ensure high >= low and close is within range
    data['high'] = np.maximum(data['high'], data['close'])
    data['low'] = np.minimum(data['low'], data['close'])
    data['high'] = np.maximum(data['high'], data['open'])
    data['low'] = np.minimum(data['low'], data['open'])
    
    return data

# Technical indicators function removed - using FeatureEngineer class instead

# Metrics calculation function removed - using PerformanceCalculator class instead

def create_charts(data, symbol):
    """Create multiple comprehensive charts."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Chart 1: Price with Moving Averages
        fig1 = go.Figure()
        
        # Candlestick
        fig1.add_trace(go.Candlestick(
            x=data.index,
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            name='Price',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        ))
        
        # Moving averages
        fig1.add_trace(go.Scatter(x=data.index, y=data['SMA_20'], name='SMA 20', line=dict(color='blue')))
        fig1.add_trace(go.Scatter(x=data.index, y=data['EMA_20'], name='EMA 20', line=dict(color='red')))
        
        fig1.update_layout(
            title=f'{symbol} Price Chart with Moving Averages',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            height=400,
            template='plotly_white'
        )
        
        # Chart 2: Technical Indicators
        fig2 = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('RSI', 'MACD', 'Bollinger Bands'),
            row_heights=[0.3, 0.3, 0.4]
        )
        
        # RSI
        fig2.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='purple')), row=1, col=1)
        fig2.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
        fig2.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)
        
        # MACD
        fig2.add_trace(go.Scatter(x=data.index, y=data['MACD'], name='MACD', line=dict(color='blue')), row=2, col=1)
        fig2.add_trace(go.Scatter(x=data.index, y=data['MACD_signal'], name='MACD Signal', line=dict(color='red')), row=2, col=1)
        fig2.add_trace(go.Bar(x=data.index, y=data['MACD_histogram'], name='MACD Histogram'), row=2, col=1)
        
        # Bollinger Bands
        fig2.add_trace(go.Scatter(x=data.index, y=data['close'], name='Close', line=dict(color='black')), row=3, col=1)
        fig2.add_trace(go.Scatter(x=data.index, y=data['BB_upper'], name='BB Upper', line=dict(color='gray', dash='dash')), row=3, col=1)
        fig2.add_trace(go.Scatter(x=data.index, y=data['BB_middle'], name='BB Middle', line=dict(color='blue')), row=3, col=1)
        fig2.add_trace(go.Scatter(x=data.index, y=data['BB_lower'], name='BB Lower', line=dict(color='gray', dash='dash')), row=3, col=1)
        
        fig2.update_layout(
            title=f'{symbol} Technical Analysis',
            height=600,
            template='plotly_white'
        )
        
        # Chart 3: Volume Analysis
        fig3 = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=(f'{symbol} Price', 'Volume Analysis'),
            row_heights=[0.7, 0.3]
        )
        
        # Price
        fig3.add_trace(go.Scatter(x=data.index, y=data['close'], name='Close Price', line=dict(color='black')), row=1, col=1)
        
        # Volume
        fig3.add_trace(go.Bar(x=data.index, y=data['volume'], name='Volume', marker_color='rgba(158,202,225,0.8)'), row=2, col=1)
        fig3.add_trace(go.Scatter(x=data.index, y=data['Volume_SMA'], name='Volume SMA', line=dict(color='red')), row=2, col=1)
        
        fig3.update_layout(
            title=f'{symbol} Volume Analysis',
            height=500,
            template='plotly_white'
        )
        
        # Save charts
        os.makedirs("plots", exist_ok=True)
        
        chart1_file = f"plots/{symbol}_price_chart.html"
        chart2_file = f"plots/{symbol}_technical_analysis.html"
        chart3_file = f"plots/{symbol}_volume_analysis.html"
        
        fig1.write_html(chart1_file)
        fig2.write_html(chart2_file)
        fig3.write_html(chart3_file)
        
        return [chart1_file, chart2_file, chart3_file]
        
    except Exception as e:
        print(f"Chart creation failed: {e}")
        return []

# HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Comprehensive Stock Analysis Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f0f0f0; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #333; text-align: center; }
        .form-group { margin-bottom: 20px; }
        label { display: block; margin-bottom: 5px; font-weight: bold; }
        input, select { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 5px; font-size: 16px; }
        button { background-color: #007bff; color: white; padding: 12px 30px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; width: 100%; }
        button:hover { background-color: #0056b3; }
        .results { margin-top: 30px; padding: 20px; background-color: #f8f9fa; border-radius: 5px; }
        .metric { display: inline-block; margin: 10px; padding: 15px; background: white; border-radius: 5px; min-width: 120px; text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        .metric-value { font-size: 24px; font-weight: bold; color: #007bff; }
        .metric-label { font-size: 14px; color: #666; margin-top: 5px; }
        .success { color: green; background: #d4edda; padding: 15px; border-radius: 5px; margin: 15px 0; }
        .warning { color: #856404; background: #fff3cd; padding: 15px; border-radius: 5px; margin: 15px 0; }
        .chart-section { margin: 30px 0; padding: 20px; background: #f8f9fa; border-radius: 5px; }
        .chart-container { margin: 20px 0; }
        iframe { width: 100%; height: 500px; border: none; border-radius: 5px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>📈 Comprehensive Stock Analysis Dashboard</h1>
        
        <form method="POST">
            <div class="form-group">
                <label for="symbol">Stock Symbol:</label>
                <input type="text" id="symbol" name="symbol" placeholder="Enter stock symbol (e.g., AAPL, GOOGL, MSFT)" value="{{ symbol or 'AAPL' }}" required>
            </div>
            
            <div class="form-group">
                <label for="period">Time Period:</label>
                <select id="period" name="period">
                    <option value="1mo" {{ 'selected' if period == '1mo' else '' }}>1 Month</option>
                    <option value="3mo" {{ 'selected' if period == '3mo' else '' }}>3 Months</option>
                    <option value="6mo" {{ 'selected' if period == '6mo' else '' }}>6 Months</option>
                    <option value="1y" {{ 'selected' if period == '1y' else '' }}>1 Year</option>
                </select>
            </div>
            
            <button type="submit">🔍 Analyze Stock</button>
        </form>
        
        {% if results %}
        <div class="results">
            <h2>📊 Analysis Results for {{ symbol.upper() }}</h2>
            
            {% if error %}
            <div class="error">{{ error }}</div>
            {% else %}
            <div class="success">✅ Comprehensive analysis completed successfully!</div>
            
            {% if data_source %}
            <div class="warning">
                <strong>📊 Data Source:</strong> {{ data_source }}
            </div>
            {% endif %}
            
            <div style="margin-top: 20px; text-align: center;">
                <div class="metric">
                    <div class="metric-value">${{ "%.2f"|format(results.current_price) }}</div>
                    <div class="metric-label">Current Price</div>
                </div>
                
                <div class="metric">
                    <div class="metric-value" style="color: {{ 'green' if results.price_change >= 0 else 'red' }};">
                        {{ "%.2f"|format(results.price_change) }}%
                    </div>
                    <div class="metric-label">Price Change</div>
                </div>
                
                <div class="metric">
                    <div class="metric-value">{{ "%.2f"|format(results.volatility) }}%</div>
                    <div class="metric-label">Volatility</div>
                </div>
                
                <div class="metric">
                    <div class="metric-value">{{ "%.2f"|format(results.sharpe_ratio) }}</div>
                    <div class="metric-label">Sharpe Ratio</div>
                </div>
                
                <div class="metric">
                    <div class="metric-value" style="color: red;">{{ "%.2f"|format(results.max_drawdown) }}%</div>
                    <div class="metric-label">Max Drawdown</div>
                </div>
                
                <div class="metric">
                    <div class="metric-value">{{ "{:,.0f}".format(results.volume_avg) }}</div>
                    <div class="metric-label">Avg Volume</div>
                </div>
            </div>
            
            {% if charts %}
            <div class="chart-section">
                <h3>📈 Interactive Charts</h3>
                
                <div class="chart-container">
                    <h4>1. Price Chart with Moving Averages</h4>
                    <iframe src="{{ charts[0] }}"></iframe>
                </div>
                
                <div class="chart-container">
                    <h4>2. Technical Analysis (RSI, MACD, Bollinger Bands)</h4>
                    <iframe src="{{ charts[1] }}"></iframe>
                </div>
                
                <div class="chart-container">
                    <h4>3. Volume Analysis</h4>
                    <iframe src="{{ charts[2] }}"></iframe>
                </div>
            </div>
            {% endif %}
            {% endif %}
        </div>
        {% endif %}
        
        <div style="margin-top: 40px; text-align: center; color: #666;">
            <p>💡 <strong>Try:</strong> AAPL, GOOGL, MSFT, TSLA, NVDA, AMZN, META, NFLX</p>
        </div>
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        symbol = request.form.get('symbol', 'AAPL').upper().strip()
        period = request.form.get('period', '1y')
        
        try:
            # Try to fetch real data first
            data = fetch_real_data(symbol, period)
            data_source = "Real-time data from Yahoo Finance"
            
            # If real data fails, use sample data
            if data is None or data.empty:
                print(f"⚠️  Real data failed for {symbol}, using sample data")
                data = create_sample_data(symbol, days=180)
                data_source = "Sample data (real-time data unavailable)"
            
            # Add technical indicators using FeatureEngineer
            feature_engineer = FeatureEngineer()
            data_with_indicators = feature_engineer.add_technical_indicators(data)
            
            # Calculate metrics using PerformanceCalculator
            performance_calc = PerformanceCalculator()
            returns = performance_calc.calculate_returns(data['close'])
            volatility = performance_calc.calculate_volatility(returns)
            sharpe_ratio = performance_calc.calculate_sharpe_ratio(returns)
            max_drawdown = performance_calc.calculate_max_drawdown(data['close'])
            
            metrics = {
                'current_price': data['close'].iloc[-1],
                'price_change': data['close'].pct_change().iloc[-1] * 100,
                'volatility': volatility.iloc[-1] if not volatility.empty else 0,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'volume_avg': data['volume'].mean()
            }
            
            # Create charts
            charts = create_charts(data_with_indicators, symbol)
            
            return render_template_string(HTML_TEMPLATE, 
                symbol=symbol, period=period, results=metrics, error=None, charts=charts, data_source=data_source)
                
        except Exception as e:
            return render_template_string(HTML_TEMPLATE, 
                symbol=symbol, period=period, results=None, error=f"Error: {str(e)}", charts=None)
    
    return render_template_string(HTML_TEMPLATE, 
        symbol='AAPL', period='1y', results=None, error=None, charts=None)

if __name__ == '__main__':
    print("🚀 Starting Comprehensive Stock Analysis Dashboard...")
    print("📱 Open your browser and go to: http://127.0.0.1:5000")
    print("⏹️  Press Ctrl+C to stop the server")
    app.run(debug=True, host='127.0.0.1', port=5000)

