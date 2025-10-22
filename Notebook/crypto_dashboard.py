# crypto_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from dotenv import load_dotenv
import os

# ==========================
# 🔐 Load API key
# ==========================
load_dotenv()
api_key = os.getenv("COINGECKO_API_KEY")

if not api_key:
    st.error("⚠️ Missing CoinGecko API key. Please set COINGECKO_API_KEY in your .env file.")
    st.stop()
def fetch_coingecko_data(coin_id="bitcoin", days=30, interval="hourly"):
    base_urls = [
        "https://pro-api.coingecko.com/api/v3/coins",
        "https://api.coingecko.com/api/v3/coins"  # fallback
    ]
    
    headers = {"x-cg-pro-api-key": api_key}
    params = {"vs_currency": "usd", "days": days, "interval": interval}
    
    for base in base_urls:
        url = f"{base}/{coin_id}/market_chart"
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                break
        except requests.exceptions.RequestException as e:
            st.warning(f"⚠️ Could not connect to {base}. Trying fallback...")
            continue
    else:
        st.error("🚫 Unable to connect to both CoinGecko endpoints.")
        return pd.DataFrame()
    

    data = response.json()

    # Convert JSON to DataFrame
    prices = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
    market_caps = pd.DataFrame(data["market_caps"], columns=["timestamp", "market_cap"])
    volumes = pd.DataFrame(data["total_volumes"], columns=["timestamp", "volume"])

    df = prices.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    df["market_cap"] = market_caps["market_cap"].values
    df["volume"] = volumes["volume"].values

    # Approximate OHLC structure (since CoinGecko gives only prices)
    df["Open"] = df["price"].shift(1)
    df["High"] = df["price"].rolling(window=3).max()
    df["Low"] = df["price"].rolling(window=3).min()
    df["Close"] = df["price"]
    df.dropna(inplace=True)
    return df


# ==========================
# 📊 Indicators & Signals
# ==========================
def calculate_indicators(df):
    bb = BollingerBands(close=df["Close"], window=20, window_dev=2)
    df["BB_upper"] = bb.bollinger_hband()
    df["BB_lower"] = bb.bollinger_lband()
    df["BB_mid"] = bb.bollinger_mavg()

    rsi = RSIIndicator(close=df["Close"], window=14)
    df["RSI"] = rsi.rsi()

    df["Signal"] = 0
    for i in range(1, len(df)):
        if (df["Close"].iloc[i] <= df["BB_lower"].iloc[i]) and (df["RSI"].iloc[i] < 30):
            df.loc[df.index[i], "Signal"] = 1
        elif (df["Close"].iloc[i] >= df["BB_upper"].iloc[i]) and (df["RSI"].iloc[i] > 70):
            df.loc[df.index[i], "Signal"] = -1
    return df


# ==========================
# 🧩 Support & Resistance
# ==========================
def find_support_resistance(df, window=20):
    supports, resistances = [], []
    for i in range(window, len(df) - window):
        if df["Low"].iloc[i] == df["Low"].iloc[i - window:i + window].min():
            supports.append((df.index[i], df["Low"].iloc[i]))
        if df["High"].iloc[i] == df["High"].iloc[i - window:i + window].max():
            resistances.append((df.index[i], df["High"].iloc[i]))
    return supports, resistances


# ==========================
# 📈 Plotly Visualization
# ==========================
def create_plot(df, coin_id, supports, resistances):
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.1, row_heights=[0.7, 0.3],
        subplot_titles=(f"{coin_id.upper()} Price Chart with Bollinger Bands", "RSI")
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="Price"
    ), row=1, col=1)

    # Bollinger Bands
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_upper"], line=dict(color="purple"), name="Upper BB"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_mid"], line=dict(color="blue"), name="Middle BB"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_lower"], line=dict(color="purple"), name="Lower BB"), row=1, col=1)

    # Buy/Sell signals
    buys = df[df["Signal"] == 1]
    sells = df[df["Signal"] == -1]
    fig.add_trace(go.Scatter(x=buys.index, y=buys["Close"], mode="markers",
                             name="Buy", marker=dict(color="green", size=8, symbol="triangle-up")), row=1, col=1)
    fig.add_trace(go.Scatter(x=sells.index, y=sells["Close"], mode="markers",
                             name="Sell", marker=dict(color="red", size=8, symbol="triangle-down")), row=1, col=1)

    # Support & Resistance
    for s in supports:
        fig.add_hline(y=s[1], line_dash="dot", line_color="green", annotation_text="Support")
    for r in resistances:
        fig.add_hline(y=r[1], line_dash="dot", line_color="red", annotation_text="Resistance")

    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], line=dict(color="orange"), name="RSI"), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    fig.update_layout(
        height=900,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        showlegend=True,
    )
    return fig


# ==========================
# 🖥️ Streamlit App
# ==========================
st.set_page_config(page_title="Crypto Technical Dashboard", layout="wide")

st.title("📊 Crypto Futures Trading Dashboard (CoinGecko Data)")
st.markdown("Analyze token price action, volatility, RSI, and Bollinger signals directly from on-chain data sources.")

# Sidebar inputs
coin_id = st.sidebar.text_input("Enter Coin ID (as per CoinGecko):", value="bitcoin")
days = st.sidebar.slider("Number of Days", 7, 180, 30)
interval = st.sidebar.selectbox("Interval", ["hourly", "daily"], index=0)

# Fetch and compute
with st.spinner(f"Fetching data for {coin_id}..."):
    df = fetch_coingecko_data(coin_id, days=days, interval=interval)
    if df.empty:
        st.stop()

    df = calculate_indicators(df)
    supports, resistances = find_support_resistance(df)
    fig = create_plot(df, coin_id, supports, resistances)

# Display outputs
st.plotly_chart(fig, use_container_width=True)

# Token info summary
st.subheader(f"📈 {coin_id.upper()} Market Overview")
current = df["Close"].iloc[-1]
high_24h = df["High"].iloc[-24:].max()
low_24h = df["Low"].iloc[-24:].min()
st.metric("Current Price (USD)", f"${current:,.2f}")
st.metric("24h High", f"${high_24h:,.2f}")
st.metric("24h Low", f"${low_24h:,.2f}")

st.markdown("---")
st.caption("Data powered by [CoinGecko Pro API](https://www.coingecko.com/en/api) | Built with ❤️ using Streamlit + Plotly + TA")
