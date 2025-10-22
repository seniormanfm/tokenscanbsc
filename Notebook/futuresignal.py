# %%
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from datetime import datetime, timedelta

# %%
# Function to fetch token data
def fetch_token_data(symbol, period="1mo", interval="1h"):
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval)
    return df, ticker.info

# %%
# Function to calculate indicators and trade signals
def calculate_indicators(df):
    # Bollinger Bands
    bb_indicator = BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BB_upper'] = bb_indicator.bollinger_hband()
    df['BB_lower'] = bb_indicator.bollinger_lband()
    df['BB_mid'] = bb_indicator.bollinger_mavg()
       # RSI
    rsi_indicator = RSIIndicator(close=df['Close'], window=14)
    df['RSI'] = rsi_indicator.rsi()

    # Trade signals
    df['Signal'] = 0
    df['Stop_Loss'] = np.nan
    df['Take_Profit'] = np.nan
    for i in range(1, len(df)):
        # Buy signal: Price touches lower Bollinger Band and RSI < 30
        if (df['Close'].iloc[i] <= df['BB_lower'].iloc[i]) and (df['RSI'].iloc[i] < 30):
            df['Signal'].iloc[i] = 1
            df['Stop_Loss'].iloc[i] = df['Close'].iloc[i] * 0.98  # 2% below entry
            df['Take_Profit'].iloc[i] = df['Close'].iloc[i] * 1.04  # 4% above entry
        # Sell signal: Price touches upper Bollinger Band and RSI > 70
        elif (df['Close'].iloc[i] >= df['BB_upper'].iloc[i]) and (df['RSI'].iloc[i] > 70):
            df['Signal'].iloc[i] = -1
            df['Stop_Loss'].iloc[i] = df['Close'].iloc[i] * 1.02  # 2% above entry
            df['Take_Profit'].iloc[i] = df['Close'].iloc[i] * 0.96  # 4% below entry

    return df

# %%
# Function to identify support and resistance levels
def find_support_resistance(df, window=20):
    supports = []
    resistances = []
    for i in range(window, len(df) - window):
        # Support: local minimum
        if df['Low'].iloc[i] == df['Low'].iloc[i-window:i+window].min():
            supports.append((df.index[i], df['Low'].iloc[i]))
        # Resistance: local maximum
        if df['High'].iloc[i] == df['High'].iloc[i-window:i+window].max():
            resistances.append((df.index[i], df['High'].iloc[i]))
    return supports, resistances

# %%
# Function to create the plot
def create_plot(df, symbol, supports, resistances):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.1, subplot_titles=(f"{symbol} Price with Bollinger Bands", "RSI"),
                        row_heights=[0.7, 0.3])

    # Candlestick chart
    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df['Open'],
                                 high=df['High'],
                                 low=df['Low'],
                                 close=df['Close'],
                                 name='Price'), row=1, col=1)

    # Bollinger Bands
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_upper'], name='Upper BB', line=dict(color='purple'), opacity=0.5), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_mid'], name='Middle BB', line=dict(color='blue'), opacity=0.5), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_lower'], name='Lower BB', line=dict(color='purple'), opacity=0.5), row=1, col=1)
    # Buy/Sell signals
    buy_signals = df[df['Signal'] == 1]
    sell_signals = df[df['Signal'] == -1]
    fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Close'], mode='markers', name='Buy Signal',
                             marker=dict(symbol='triangle-up', size=10, color='green')), row=1, col=1)
    fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Close'], mode='markers', name='Sell Signal',
                             marker=dict(symbol='triangle-down', size=10, color='red')), row=1, col=1)

    # Support and Resistance levels
    for s in supports:
        fig.add_hline(y=s[1], line_dash="dash", line_color="green", annotation_text="Support", row=1, col=1)
    for r in resistances:
        fig.add_hline(y=r[1], line_dash="dash", line_color="red", annotation_text="Resistance", row=1, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='orange')), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    fig.update_layout(title=f"{symbol} Futures Trading Dashboard",
                      yaxis_title="Price (USD)",
                      xaxis_rangeslider_visible=False,
                      height=800)
    return fig

# Initialize Dash app
app = Dash(__name__)

# Layout
app.layout = html.Div([
    html.H1("Token Futures Trading Dashboard"),
    html.Label("Enter Token Symbol (e.g., BTC-USD):"),
    dcc.Input(id='token-symbol', value='BTC-USD', type='text'),
    dcc.Graph(id='trading-chart'),
    html.Div(id='token-info')
])

# Callback to update the dashboard
@app.callback(
    [Output('trading-chart', 'figure'),
     Output('token-info', 'children')],
    [Input('token-symbol', 'value')]
)
def update_dashboard(symbol):
    try:
        # Fetch data
        df, info = fetch_token_data(symbol)
        if df.empty:
            return go.Figure(), f"No data available for {symbol}"

        # Calculate indicators and signals
        df = calculate_indicators(df)

        # Find support and resistance
        supports, resistances = find_support_resistance(df)

        # Create plot
        fig = create_plot(df, symbol, supports, resistances)

        # Get token information
        ath = df['High'].max()
        atl = df['Low'].min()
        last_24h = df.last('24H')
        high_24h = last_24h['High'].max()
        low_24h = last_24h['Low'].min()
        current_price = df['Close'].iloc[-1]

        token_info = html.Div([
            html.H3(f"{symbol} Information"),
            html.P(f"Current Price: ${current_price:.2f}"),
            html.P(f"All-Time High: ${ath:.2f}"),
            html.P(f"All-Time Low: ${atl:.2f}"),
            html.P(f"24-Hour High: ${high_24h:.2f}"),
            html.P(f"24-Hour Low: ${low_24h:.2f}")
        ])

        return fig, token_info
    except Exception as e:
        return go.Figure(), f"Error: {str(e)}"

# Run the app
if __name__ == '__main__':
    try:
        host = '127.0.0.1'
        port = 8050
        print(f"Starting Dash app on http://{host}:{port} ...")
        # Dash 3.x uses app.run
        # disable the reloader so logs appear only once in this process
        app.run(debug=True, host=host, port=port, use_reloader=False)
    except Exception as e:
        print('Error starting Dash app:', e)

# %%
