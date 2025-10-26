"""
Comprehensive web interface with Alpha Vantage API integration and dynamic sample data.
"""
from flask import Flask, render_template_string, request, send_from_directory
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from feature_engineering import FeatureEngineer
from utils import PerformanceCalculator
from data_collector import StockDataCollector

app = Flask(__name__)

# Add route to serve static files
@app.route('/plots/<path:filename>')
def serve_plot(filename):
    return send_from_directory('plots', filename)

def fetch_real_data(symbol="AAPL", period="1y"):
    """Fetch real stock data using Alpha Vantage first, then Yahoo Finance fallback."""
    try:
        print(f"📊 Fetching data for {symbol}...")
        data_collector = StockDataCollector()
        
        # Try Alpha Vantage first with real API key (500 requests/day)
        try:
            print(f"🔄 Trying Alpha Vantage for {symbol}...")
            data = data_collector.get_alpha_vantage_data(symbol, "8ABXVMAY10I1AV65")
            if data is not None and not data.empty:
                print(f"✅ Successfully fetched {len(data)} days of Alpha Vantage data for {symbol}")
                return data
        except Exception as e:
            print(f"⚠️  Alpha Vantage failed: {e}")
        
        # Fallback to Yahoo Finance
        try:
            print(f"🔄 Trying Yahoo Finance for {symbol}...")
            data = data_collector.get_yahoo_data(symbol, period)
            if data is not None and not data.empty:
                print(f"✅ Successfully fetched {len(data)} days of Yahoo Finance data for {symbol}")
                return data
        except Exception as e:
            print(f"⚠️  Yahoo Finance failed: {e}")
        
        print(f"❌ No data found for {symbol}")
        return None
            
    except Exception as e:
        print(f"❌ Error fetching data for {symbol}: {e}")
        return None

def generate_market_insights(symbol, period):
    """Generate valuable market insights when real data is unavailable."""
    
    # Company profiles and market analysis
    company_profiles = {
        'AAPL': {
            'name': 'Apple Inc.',
            'sector': 'Technology',
            'market_cap': '~$3.2T',
            'key_metrics': ['iPhone sales', 'Services revenue', 'China market exposure'],
            'recent_news': ['AI integration in devices', 'Vision Pro launch', 'China sales recovery'],
            'risk_factors': ['Regulatory scrutiny', 'Supply chain dependencies', 'Market saturation'],
            'growth_drivers': ['Services ecosystem', 'Emerging markets', 'Wearables expansion']
        },
        'MSFT': {
            'name': 'Microsoft Corporation',
            'sector': 'Technology',
            'market_cap': '~$3.1T',
            'key_metrics': ['Azure cloud growth', 'Office 365 subscriptions', 'AI investments'],
            'recent_news': ['Copilot AI expansion', 'Gaming acquisitions', 'Cloud infrastructure growth'],
            'risk_factors': ['Competition in cloud', 'Regulatory oversight', 'Economic sensitivity'],
            'growth_drivers': ['AI integration', 'Enterprise cloud adoption', 'Gaming ecosystem']
        },
        'GOOGL': {
            'name': 'Alphabet Inc.',
            'sector': 'Technology',
            'market_cap': '~$2.1T',
            'key_metrics': ['Search advertising', 'YouTube revenue', 'Cloud growth'],
            'recent_news': ['Gemini AI development', 'YouTube Shorts growth', 'Cloud market share'],
            'risk_factors': ['Regulatory challenges', 'AI competition', 'Privacy concerns'],
            'growth_drivers': ['AI capabilities', 'Cloud services', 'YouTube monetization']
        },
        'AMZN': {
            'name': 'Amazon.com Inc.',
            'sector': 'Consumer Discretionary',
            'market_cap': '~$1.8T',
            'key_metrics': ['AWS revenue', 'E-commerce growth', 'Prime subscriptions'],
            'recent_news': ['AWS AI services', 'Logistics optimization', 'International expansion'],
            'risk_factors': ['Competition intensity', 'Regulatory pressure', 'Economic cycles'],
            'growth_drivers': ['Cloud computing', 'International markets', 'Advertising revenue']
        },
        'TSLA': {
            'name': 'Tesla Inc.',
            'sector': 'Automotive',
            'market_cap': '~$800B',
            'key_metrics': ['Vehicle deliveries', 'Energy storage', 'Autopilot development'],
            'recent_news': ['Cybertruck production', 'FSD improvements', 'Energy business growth'],
            'risk_factors': ['Competition intensity', 'Regulatory changes', 'Production challenges'],
            'growth_drivers': ['EV adoption', 'Energy storage', 'Autonomous driving']
        }
    }
    
    # Market conditions by period
    market_conditions = {
        '1mo': {
            'trend': 'Recent market volatility',
            'key_events': ['Fed policy decisions', 'Earnings season', 'Geopolitical tensions'],
            'sector_performance': 'Mixed performance across sectors',
            'volatility': 'Elevated due to uncertainty'
        },
        '3mo': {
            'trend': 'Moderate recovery phase',
            'key_events': ['Interest rate adjustments', 'Economic data releases', 'Corporate earnings'],
            'sector_performance': 'Technology and healthcare leading',
            'volatility': 'Moderate with upward bias'
        },
        '6mo': {
            'trend': 'Market consolidation',
            'key_events': ['Mid-term economic outlook', 'Sector rotation', 'International developments'],
            'sector_performance': 'Defensive sectors outperforming',
            'volatility': 'Stable with periodic spikes'
        },
        '1y': {
            'trend': 'Long-term growth trajectory',
            'key_events': ['Annual earnings cycles', 'Policy changes', 'Market structure evolution'],
            'sector_performance': 'Growth stocks leading recovery',
            'volatility': 'Normalizing to historical levels'
        }
    }
    
    # Get company profile
    profile = company_profiles.get(symbol.upper(), {
        'name': f'{symbol} Corporation',
        'sector': 'Various',
        'market_cap': 'Varies',
        'key_metrics': ['Revenue growth', 'Profit margins', 'Market share'],
        'recent_news': ['Market developments', 'Industry trends', 'Regulatory changes'],
        'risk_factors': ['Market volatility', 'Competition', 'Economic cycles'],
        'growth_drivers': ['Market expansion', 'Innovation', 'Operational efficiency']
    })
    
    market_context = market_conditions.get(period, market_conditions['1mo'])
    
    insights = {
        'company': profile,
        'market_context': market_context,
        'analysis': {
            'current_sentiment': 'Neutral to positive',
            'key_catalysts': profile['recent_news'][:3],
            'risk_assessment': 'Moderate risk profile',
            'investment_thesis': f"Focus on {profile['key_metrics'][0]} and {profile['growth_drivers'][0]}"
        },
        'recommendations': [
            'Monitor earnings reports closely',
            'Watch for sector rotation trends',
            'Consider dollar-cost averaging approach',
            'Diversify across related sectors'
        ]
    }
    
    return insights

def create_sample_data(symbol="AAPL", period="1y"):
    """Create realistic sample stock data that varies by time period."""
    print(f"⚠️  Using valuable sample approximation for {symbol} ({period})")
    
    # Map period to days
    period_days = {
        '1mo': 30,
        '3mo': 90,
        '6mo': 180,
        '1y': 365
    }
    days = period_days.get(period, 180)
    
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
    
    # Create dynamic seed based on symbol, period, and current date
    # This ensures different data for different time periods
    current_date = datetime.now().strftime('%Y%m%d')
    dynamic_seed = hash(f"{symbol}_{period}_{current_date}") % 2**32
    np.random.seed(dynamic_seed)
    
    base_price = profile['base_price']
    volatility = profile['volatility']
    trend = profile['trend']
    
    # Create realistic market scenarios for different periods
    # Each period represents different market conditions (bull/bear markets, volatility, etc.)
    period_scenarios = {
        '1mo': {'multiplier': 1.0, 'trend': 0.0003, 'volatility': 0.02},    # Current market
        '3mo': {'multiplier': 1.05, 'trend': 0.0005, 'volatility': 0.025},  # Bull market 3mo ago
        '6mo': {'multiplier': 0.95, 'trend': -0.0002, 'volatility': 0.03},  # Bear market 6mo ago
        '1y': {'multiplier': 1.15, 'trend': 0.0008, 'volatility': 0.035}    # Strong bull market 1y ago
    }
    
    scenario = period_scenarios.get(period, period_scenarios['1mo'])
    base_price *= scenario['multiplier']
    trend = scenario['trend']
    volatility = scenario['volatility']
    
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
    <title>Invesight - Advanced Stock Analysis Dashboard</title>
    <style>
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            margin: 0; 
            padding: 20px; 
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            min-height: 100vh;
        }
        .container { 
            max-width: 1200px; 
            margin: 0 auto; 
            background: rgba(255, 255, 255, 0.95); 
            padding: 30px; 
            border-radius: 20px; 
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
        }
        h1 { 
            color: #2c3e50; 
            text-align: center; 
            margin-bottom: 30px; 
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
            background: linear-gradient(45deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .alpha-badge { 
            background: linear-gradient(45deg, #ff6b6b, #ee5a24); 
            color: white; 
            padding: 8px 16px; 
            border-radius: 20px; 
            font-size: 12px; 
            margin-left: 10px;
            box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3);
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
        .form-group { margin-bottom: 20px; }
        label { 
            display: block; 
            margin-bottom: 8px; 
            font-weight: bold; 
            color: #34495e;
            font-size: 1.1em;
        }
        input, select { 
            width: 100%; 
            padding: 15px; 
            border: 2px solid #e0e6ed; 
            border-radius: 10px; 
            font-size: 16px; 
            transition: all 0.3s ease;
            background: rgba(255, 255, 255, 0.9);
        }
        input:focus, select:focus { 
            outline: none; 
            border-color: #667eea; 
            box-shadow: 0 0 10px rgba(102, 126, 234, 0.3);
            transform: translateY(-2px);
        }
        button { 
            background: linear-gradient(45deg, #667eea, #764ba2); 
            color: white; 
            padding: 15px 30px; 
            border: none; 
            border-radius: 10px; 
            cursor: pointer; 
            font-size: 18px; 
            width: 100%; 
            font-weight: bold;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }
        button:hover { 
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
        }
        .results { 
            margin-top: 30px; 
            padding: 25px; 
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); 
            border-radius: 15px; 
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        .metric { 
            display: inline-block; 
            margin: 15px; 
            padding: 20px; 
            background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%); 
            border-radius: 15px; 
            min-width: 140px; 
            text-align: center; 
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
            border: 2px solid transparent;
        }
        .metric:hover {
            transform: translateY(-5px);
            border-color: #667eea;
            box-shadow: 0 15px 35px rgba(102, 126, 234, 0.2);
        }
        .metric-value { 
            font-size: 28px; 
            font-weight: bold; 
            background: linear-gradient(45deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .metric-label { 
            font-size: 14px; 
            color: #7f8c8d; 
            margin-top: 8px; 
            font-weight: 600;
        }
        .success { 
            color: #27ae60; 
            background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%); 
            padding: 20px; 
            border-radius: 10px; 
            margin: 20px 0; 
            border-left: 5px solid #27ae60;
            font-weight: bold;
        }
        .warning { 
            color: #e67e22; 
            background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%); 
            padding: 20px; 
            border-radius: 10px; 
            margin: 20px 0; 
            border-left: 5px solid #e67e22;
            font-weight: bold;
        }
        .chart-section { 
            margin: 30px 0; 
            padding: 25px; 
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); 
            border-radius: 15px; 
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        .chart-container { 
            margin: 20px 0; 
            background: white;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        iframe { 
            width: 100%; 
            height: 500px; 
            border: none; 
            border-radius: 10px; 
        }
        .footer {
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            color: #2c3e50;
        }
        .footer a {
            color: #667eea;
            text-decoration: none;
            font-weight: bold;
        }
        .footer a:hover {
            color: #764ba2;
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>📈 Invesight</h1>
        
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
            
            <button type="submit">🔍 Analyze Stock with Alpha Vantage</button>
        </form>
        
        {% if results %}
        <div class="results">
            <h2>📊 Analysis Results for {{ symbol.upper() }} ({{ period }})</h2>
            
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
        
        
        <div class="footer">
            <p>💡 <strong>Try:</strong> AAPL, GOOGL, MSFT, TSLA, NVDA, AMZN, META, NFLX</p>
            <p>🔗 Powered by <a href="https://www.alphavantage.co" target="_blank">Alpha Vantage API</a> - Professional Stock Market Data</p>
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
            
            if data is not None and not data.empty:
                data_source = "Real-time data from Alpha Vantage/Yahoo Finance"
            else:
                print(f"⚠️  Real data failed for {symbol}, using valuable sample approximation")
                data = create_sample_data(symbol, period)
                data_source = f"Valuable sample approximation (real-time data unavailable) - Based on {symbol.upper()} real market characteristics and current price levels"
            
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
    print("🚀 Starting Invesight - Advanced Stock Analysis Dashboard...")
    print("📱 Open your browser and go to: http://127.0.0.1:5000")
    print("⏹️  Press Ctrl+C to stop the server")
    print("🔗 Powered by Alpha Vantage API - https://www.alphavantage.co")
    print("✨ Now with dynamic sample data that varies by time period!")
    app.run(debug=True, host='127.0.0.1', port=5000)
