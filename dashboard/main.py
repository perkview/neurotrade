"""
📌 Multi-Coin Auto Strategy (15m interval)
- Runs continuously, checks every 15 minutes.
- Uses EMA200, RSI, and Volume Spike filter.
- Works for BTC, ETH, XRP, SOL, LINK, etc.
- Skips coins that are not available on Yahoo Finance.
- Safely logs predictions to Django CoinPrediction model.
"""

import time
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from django.utils import timezone
from dashboard.models import CoinPrediction

# ---------------- Parameters ----------------
COINS = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "XRP": "XRP-USD",
    "SOL": "SOL-USD",
    "LINK": "LINK-USD",
    "LTC": "LTC-USD",
    "ADA": "ADA-USD",
    "BNB": "BNB-USD",
    "DOT": "DOT-USD",
    "SYRUP": "SYRUP-USD",
}

PERIOD = "60d"
INTERVAL = "15m"
EMA_LONG = 200
RSI_PERIOD = 14
VOLUME_MA_PERIOD = 20
VOLUME_SPIKE_MULTIPLIER = 1.5
CHECK_INTERVAL_SECONDS = 15 * 60  # 15 minutes

bot_running = False
latest_results = []  # store last cycle results

# ---------------- Data Fetching ----------------
def fetch_ohlcv(symbol):
    """
    Fetch OHLCV data for a coin from Yahoo Finance.
    """
    try:
        df = yf.Ticker(symbol).history(period=PERIOD, interval=INTERVAL)
        if df.empty:
            return None
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None

# ---------------- Indicators ----------------
def ema(series, period):
    return series.ewm(span=period, adjust=False, min_periods=period).mean()

def rsi(series, period=RSI_PERIOD):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def volume_spike(volume, period=VOLUME_MA_PERIOD, multiplier=VOLUME_SPIKE_MULTIPLIER):
    vol_ma = volume.rolling(window=period, min_periods=period).mean()
    return volume > (vol_ma * multiplier)

# ---------------- Strategy ----------------
def combined_strategy(df):
    """
    Generate BUY/SELL/HOLD signals based on EMA200, RSI, and volume spike.
    """
    close = df['Close']
    df['EMA_long'] = ema(close, EMA_LONG)
    df['RSI'] = rsi(close)
    df['Vol_Spike'] = volume_spike(df['Volume'])

    # Initialize signal
    df['Signal'] = 0

    # BUY: Uptrend + oversold + volume spike
    buy_cond = (close > df['EMA_long']) & (df['RSI'] < 40) & (df['Vol_Spike'])

    # SELL: Downtrend OR overbought
    sell_cond = (close < df['EMA_long']) | (df['RSI'] > 70)

    df.loc[buy_cond, 'Signal'] = 1
    df.loc[sell_cond, 'Signal'] = -1

    return df

# ---------------- Logging ----------------
def log_prediction(coin, signal_str, close_price, ema_val, rsi_val, vol_spike):
    """
    Save prediction to Django CoinPrediction model.
    """
    log_obj, _ = CoinPrediction.objects.get_or_create(pk=1)
    log_obj.append_prediction(
        coin=coin,
        signal=signal_str,
        price=close_price,
        ema200=ema_val,
        rsi=rsi_val,
        volume_spike=vol_spike,
        next_check=timezone.now() + timezone.timedelta(minutes=15)
    )

# ---------------- Main Loop ----------------
def main_loop():
    """
    Continuous bot loop that checks all coins every 15 minutes.
    """
    global bot_running, latest_results
    bot_running = True

    while bot_running:
        latest_results = []
        print("\n==============================")
        print("⏰ Checking strategy update for all coins...", timezone.now())

        for name, symbol in COINS.items():
            df = fetch_ohlcv(symbol)
            if df is None:
                msg = f"⚠️ {name} ({symbol}) → No data"
                print(msg)
                latest_results.append(msg)
                continue

            df = combined_strategy(df)
            last_row = df.iloc[-1]

            close_price = last_row["Close"]
            signal = last_row["Signal"]
            ema_val = last_row["EMA_long"]
            rsi_val = last_row["RSI"]
            vol_spike = last_row["Vol_Spike"]

            # Decide signal
            if signal == 1:
                msg = f"✅ {name}: BUY @ ${close_price:.2f}"
                signal_str = "BUY"
            elif signal == -1:
                msg = f"❌ {name}: SELL @ ${close_price:.2f}"
                signal_str = "SELL"
            else:
                msg = f"🤝 {name}: HOLD (Price: ${close_price:.2f})"
                signal_str = "HOLD"

            print(msg)
            latest_results.append(msg)

            # Save to DB log
            log_prediction(
                coin=name,
                signal=signal_str,
                price=close_price,
                ema200=ema_val,
                rsi=rsi_val,
                volume_spike=vol_spike,
                next_check=timezone.now() + timezone.timedelta(seconds=CHECK_INTERVAL_SECONDS),
            )

        # Wait until next run (safe interruption)
        for _ in range(CHECK_INTERVAL_SECONDS):
            if not bot_running:
                break
            time.sleep(1)


# ---------------- Run Bot ----------------
if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        print("\n⏹ Bot stopped by user.")
        bot_running = False
