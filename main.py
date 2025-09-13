"""
📌 Multi-Coin Auto Strategy (15m interval)
- Runs continuously, checks every 15 minutes.
- Uses EMA200, RSI, and Volume Spike filter.
- Works for BTC, ETH, XRP, SOL, LINK, etc.
- Skips coins that are not available on Yahoo Finance.
"""

import time
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# ---------------- Parameters ----------------
COINS = {
    "BTC": "BTC-USD",        # Bitcoin (since 2009) – the original & most stable store of value
    "ETH": "ETH-USD",        # Ethereum (2015) – smart contracts & DeFi ecosystem
    "XRP": "XRP-USD",        # Ripple (2012) – cross-border payments
    "SOL": "SOL-USD",        # Solana (2020) – high throughput, low fees
    "LINK": "LINK-USD",      # Chainlink (2017) – oracle network
    "LTC": "LTC-USD",        # Litecoin (2011) – “silver to Bitcoin’s gold”
    "ADA": "ADA-USD",        # Cardano (2017) – research-driven blockchain
    "BNB": "BNB-USD",        # Binance Coin (2017) – exchange utility & Binance Smart Chain
    "DOT": "DOT-USD",        # Polkadot (2020) – interoperability & parachains
}


PERIOD = "60d"
INTERVAL = "15m"
EMA_LONG = 200
RSI_PERIOD = 14
VOLUME_MA_PERIOD = 20
VOLUME_SPIKE_MULTIPLIER = 1.5

# ---------------- Data Fetching ----------------
def fetch_ohlcv(symbol):
    try:
        df = yf.Ticker(symbol).history(period=PERIOD, interval=INTERVAL)
        if df.empty:
            return None
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        df.index = pd.to_datetime(df.index)
        return df
    except Exception:
        return None

# ---------------- Indicators ----------------
def ema(series, period):
    return series.ewm(span=period, adjust=False, min_periods=period).mean()

def rsi(series, period=RSI_PERIOD):
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def volume_spike(volume, period=VOLUME_MA_PERIOD, multiplier=VOLUME_SPIKE_MULTIPLIER):
    vol_ma = volume.rolling(window=period, min_periods=period).mean()
    return volume > (vol_ma * multiplier)

# ---------------- Strategy ----------------
def combined_strategy(df):
    close = df['Close']
    df['EMA_long'] = ema(close, EMA_LONG)
    df['RSI'] = rsi(close)
    df['Vol_Spike'] = volume_spike(df['Volume'])

    df['Signal'] = 0
    buy_cond = (close > df['EMA_long']) & (df['RSI'] < 30) & (df['Vol_Spike'])
    sell_cond = (close < df['EMA_long']) | (df['RSI'] > 70)

    df.loc[buy_cond, 'Signal'] = 1
    df.loc[sell_cond, 'Signal'] = -1
    return df

# ---------------- Main Loop ----------------
def main_loop():
    while True:
        print("\n==============================")
        print("⏰ Checking strategy update for all coins...", datetime.now())

        for name, symbol in COINS.items():
            df = fetch_ohlcv(symbol)
            if df is None:
                print(f"⚠️ {name} ({symbol}) → No data (not available on Yahoo Finance).")
                continue

            df = combined_strategy(df)
            last_row = df.iloc[-1]

            close_price = last_row['Close']
            signal = last_row['Signal']

            if signal == 1:
                print(f"✅ {name}: BUY @ ${close_price:.2f}")
            elif signal == -1:
                print(f"❌ {name}: SELL @ ${close_price:.2f}")
            else:
                print(f"🤝 {name}: HOLD (Price: ${close_price:.2f})")

        print("💤 Waiting 15 minutes before next check...\n")
        time.sleep(900)  # 15 minutes

if __name__ == "__main__":
    main_loop()
