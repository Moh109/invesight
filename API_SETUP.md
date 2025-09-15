# API Setup Instructions

## 🚀 Getting Real Stock Data

The program now uses **Alpha Vantage** as the primary data source, with Yahoo Finance as a fallback.

### ✅ **Alpha Vantage (Recommended)**

1. **Get your FREE API key:**
   - Go to: https://www.alphavantage.co/support/#api-key
   - Sign up (takes 20 seconds)
   - Copy your API key

2. **Update the code:**
   - Open `comprehensive_web.py`
   - Find line 15: `ALPHA_VANTAGE_API_KEY = "demo"`
   - Replace `"demo"` with your actual API key
   - Example: `ALPHA_VANTAGE_API_KEY = "YOUR_API_KEY_HERE"`

3. **Benefits:**
   - ✅ Real-time stock data
   - ✅ 500 API calls per day (free tier)
   - ✅ Reliable and fast
   - ✅ No rate limiting issues

### 🔄 **Current Status**

- **Alpha Vantage**: Configured but using demo key (limited data)
- **Yahoo Finance**: Fallback (currently blocked)
- **Sample Data**: Working perfectly with realistic prices

### 📊 **Data Sources Priority**

1. **Alpha Vantage** (if API key provided)
2. **Yahoo Finance** (fallback)
3. **Sample Data** (realistic fallback)

### 🎯 **Quick Test**

After updating your API key, test with:
```bash
py -3.11 test_alpha_vantage.py
```

**The program works great with sample data, but real data is even better!** 🚀
