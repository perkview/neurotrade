"""
📌 Multi-Coin Auto Strategy (15m interval, improved)
- Runs continuously, checks every 15 minutes.
- Uses EMA, RSI, and Volume Spike filter with improved thresholds.
- Works for BTC, ETH, XRP, SOL, LINK, etc.
- Skips coins that are not available on Yahoo Finance.
- Safely logs predictions to Django CoinPrediction model (one row per prediction).
"""
import time
import yfinance as yf
import pandas as pd
from django.utils import timezone
from dashboard.models import CoinPrediction, StrategySettings


# ---------------- Fetch Strategy Settings ----------------
def get_strategy_settings():
    try:
        settings = StrategySettings.objects.latest("created_at")
    except StrategySettings.DoesNotExist:
        return None

    return {
        "coins": settings.coins,
        "period": settings.period,
        "interval": settings.interval,
        "ema_period": settings.ema_period,
        "rsi_period": settings.rsi_period,
        "volume_ma_period": settings.volume_ma_period,
        "volume_spike_multiplier": settings.volume_spike_multiplier,
        "buy_rsi_low": settings.buy_rsi_low,
        "sell_rsi_low": settings.sell_rsi_low,
        "buy_rsi_high": settings.buy_rsi_high,
        "sell_rsi_high": settings.sell_rsi_high,
        "use_volume": settings.use_volume,
        "ema_buy_buffer": settings.ema_buffer_buy,
        "ema_sell_buffer": settings.ema_buffer_sell,
        "check_interval_seconds": settings.check_interval_seconds,
        "active": settings.active,
        "low_vol_coins": settings.low_vol_coins,
        "high_vol_coins": settings.high_vol_coins,
        "mild_rsi_buffer_low": settings.mild_rsi_buffer_low,
        "mild_rsi_buffer_high": settings.mild_rsi_buffer_high,
        "confidence_threshold": getattr(settings, "confidence_threshold", 0.6),  # default fallback
    }


# ---------------- Data Fetching ----------------
def fetch_ohlcv(symbol, period, interval):
    """Fetch OHLCV data for a coin from Yahoo Finance."""
    try:
        df = yf.Ticker(symbol).history(period=period, interval=interval)
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
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def rsi(series, period):
    """Relative Strength Index."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def volume_spike(volume, period, multiplier, rsi=None, confirm_with_rsi=True):
    """
    Detect volume spikes.
    Args:
      - volume: pd.Series
      - period: int (rolling window for volume MA)
      - multiplier: float (e.g., 1.4 means 1.4x avg volume)
      - rsi: optional pd.Series to confirm direction
      - confirm_with_rsi: bool (if True, require RSI to be rising for buy spikes)
    Returns:
      - pd.Series of 1.0/0.0 floats
    Notes:
      - Uses only multiplier (removed the redundant 1.25 condition).
      - Safe to call with NaNs; returns 0/1 floats.
    """
    # use min_periods=1 to avoid all-NaN early returns
    vol_ma = volume.rolling(window=period, min_periods=1).mean()
    spike = (volume > vol_ma * multiplier)

    if confirm_with_rsi and rsi is not None:
        # require a rising RSI for buy-confirmation (rsi.diff() > 0)
        spike = spike & (rsi.diff() > 0)

    return spike.astype(float)


def adaptive_thresholds(base_buy, base_sell, mild_buffer, close, window=20,
                        vol_type="low", recent_accuracy=0.7, confidence_override=None):
    """
    Adjust RSI thresholds and confidence based on volatility and accuracy.
    Optionally allow a confidence_override to come from strategy settings.
    """
    # Rolling volatility as percent
    volatility = close.pct_change().rolling(window).std().iloc[-1] * 100

    # Adjust thresholds based on volatility
    if vol_type == "low":
        buy_rsi = base_buy - (volatility * 0.3)
        sell_rsi = base_sell + (volatility * 0.3)
    else:  # high volatility
        buy_rsi = base_buy - (volatility * 0.5)
        sell_rsi = base_sell + (volatility * 0.5)

    # Clamp thresholds to sane bounds
    buy_rsi = round(max(25, min(buy_rsi, 60)), 2)
    sell_rsi = round(max(40, min(sell_rsi, 80)), 2)

    # Mild buffer reduces with volatility
    mild_rsi_buffer = round(max(2.0, mild_buffer - (volatility * 0.2)), 2)

    # Confidence threshold
    if confidence_override is not None:
        confidence_threshold = confidence_override
    else:
        confidence_threshold = round(0.65 + (recent_accuracy * 0.1), 2)  # default = 0.72

    return {
        "buy_rsi": buy_rsi,
        "sell_rsi": sell_rsi,
        "mild_rsi_buffer": mild_rsi_buffer,
        "confidence_threshold": confidence_threshold
    }


# ---------------- Strategy ----------------
def combined_strategy(df, params, coin_name):
    """
    Multi-Coin Strategy (15m):
    - Uses EMA, RSI, Volume Spike
    - Includes Mild Buy/Sell logic
    - Confidence-based downgrade of BUY to MILD_BUY
    """
    df = df.copy()
    close = df['Close']

    # ---------------- Indicators ----------------
    df['EMA'] = ema(close, params["ema_period"]).ffill()
    df['RSI'] = rsi(close, params["rsi_period"]).ffill()

    # Volume spike: pass RSI so confirm_with_rsi can use it
    if params.get("use_volume", True):
        df['Vol_Spike'] = volume_spike(
            volume=df['Volume'],
            period=params["volume_ma_period"],
            multiplier=params["volume_spike_multiplier"],
            rsi=df['RSI'],
            confirm_with_rsi=params.get("confirm_volume_with_rsi", True)
        ).ffill().fillna(0.0)
    else:
        df['Vol_Spike'] = 1.0

    # ---------------- Buffers ----------------
    # Keep EMA buffers as absolute multipliers (e.g., 0.002 = 0.2%)
    ema_buy_level = df['EMA'] * (1 + params["ema_buy_buffer"])
    ema_sell_level = df['EMA'] * (1 - params["ema_sell_buffer"])

    # ---------------- Conditions ----------------
    # Loosened buy condition: EMA breakout OR RSI oversold, with optional volume confirmation
    strict_buy = ((close > ema_buy_level) | (df['RSI'] <= params["buy_rsi_threshold"])) & (df['Vol_Spike'] == 1.0)
    strict_sell = ((close < ema_sell_level) | (df['RSI'] >= params["sell_rsi_threshold"]))

    # Mild signals: proximity-based, less strict
    mild_buffer = params.get("mild_rsi_buffer", 2.0)
    mild_buy = ((close > df['EMA']) & (df['RSI'] < (params["buy_rsi_threshold"] + mild_buffer))) | \
               ((df['RSI'] < params["buy_rsi_threshold"]) & (df['Vol_Spike'] == 1.0))

    mild_sell = ((close < df['EMA']) & (df['RSI'] > (params["sell_rsi_threshold"] - mild_buffer))) | \
                (df['RSI'] > params["sell_rsi_threshold"])

    # ---------------- Signal Encoding ----------------
    # Use consistent numeric codes (same semantics as your main loop expects)
    df['Signal'] = 0.0  # 1=BUY, 0.5=MILD_BUY, 0=HOLD, -0.5=MILD_SELL, -1=SELL (we'll use similar labels)
    df.loc[strict_buy, 'Signal'] = 1.0
    df.loc[strict_sell, 'Signal'] = -1.0
    # Mild only if no strict signal
    df.loc[mild_buy & (df['Signal'] == 0.0), 'Signal'] = 0.5
    df.loc[mild_sell & (df['Signal'] == 0.0), 'Signal'] = -0.5

    # ---------------- Confidence Scoring ----------------
    # Score is based on closeness to favorable conditions (smaller distance -> higher score)
    # RSI score: closeness to buy threshold (for buys). For sells, the logic still gives a meaningful score.
    buy_thresh = params["buy_rsi_threshold"]
    # Use normalized absolute diff / 100 -> convert to closeness
    df['RSI_score'] = (1.0 - (abs(df['RSI'] - buy_thresh) / 100.0)).clip(0, 1)
    df['EMA_score'] = (1.0 - (abs(close - df['EMA']) / close)).clip(0, 1)
    df['Vol_score'] = df['Vol_Spike'].clip(0, 1)

    df['Confidence'] = (0.5 * df['RSI_score'] +
                        0.3 * df['EMA_score'] +
                        0.2 * df['Vol_score'])

    # Downgrade weak strict buys to mild buys based on confidence threshold
    confidence_threshold = params.get("confidence_threshold", 0.6)
    df.loc[(df['Signal'] == 1.0) & (df['Confidence'] < confidence_threshold), 'Signal'] = 0.5

    # Relaxed suppression floor (don't wipe out everything)
    df.loc[df['Confidence'] < 0.1, 'Signal'] = 0.0

    # ---------------- Label ----------------
    df['Signal_Label'] = df['Signal'].map({
        1.0: 'BUY',
        0.5: 'MILD_BUY',
        0.0: 'HOLD',
        -0.5: 'MILD_SELL',
        -1.0: 'SELL'
    })

    return df


# ---------------- Logging ----------------
def log_prediction(coin, signal_str, close_price, ema_val, rsi_val, vol_spike):
    """Save prediction to Django CoinPrediction model."""
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
    """Main loop for running the multi-coin strategy every N seconds."""
    global bot_running, latest_results
    bot_running = True

    while bot_running:
        latest_results = []

        # ---------------- Load strategy settings ----------------
        settings = get_strategy_settings()
        if not settings or not settings["active"]:
            print("⚠️ No active strategy settings found. Bot paused.")
            break

        coins = settings["coins"]
        period = settings["period"]
        interval = settings["interval"]
        check_interval_seconds = settings["check_interval_seconds"]

        # Volatility category settings
        low_vol_coins = settings.get("low_vol_coins", [])
        high_vol_coins = settings.get("high_vol_coins", [])

        # Shared parameters
        common_params = {
            "ema_period": settings["ema_period"],
            "rsi_period": settings["rsi_period"],
            "use_volume": settings["use_volume"],
            "volume_ma_period": settings["volume_ma_period"],
            "volume_spike_multiplier": settings["volume_spike_multiplier"],
            "ema_buy_buffer": round(settings.get("ema_buffer_buy", 0.003), 6),
            "ema_sell_buffer": round(settings.get("ema_buffer_sell", 0.003), 6),
            "confirm_volume_with_rsi": True,  # you can toggle this if needed
        }

        print("\n==============================")
        print(f"⏰ Running strategy at {timezone.now()} for {len(coins)} coins")

        for name, symbol in coins.items():
            df = fetch_ohlcv(symbol, period, interval)
            if df is None or df.empty:
                msg = f"⚠️ {name} ({symbol}) → No data available."
                print(msg)
                latest_results.append(msg)
                continue

            # ---------------- Adaptive RSI thresholds ----------------
            if name in low_vol_coins:
                thresholds = adaptive_thresholds(
                    base_buy=settings.get("buy_rsi_low", 30),
                    base_sell=settings.get("sell_rsi_low", 70),
                    mild_buffer=settings.get("mild_rsi_buffer_low", 2.0),
                    close=df['Close'],
                    vol_type="low",
                    recent_accuracy=0.7,
                    confidence_override=settings.get("confidence_threshold")
                )
            elif name in high_vol_coins:
                thresholds = adaptive_thresholds(
                    base_buy=settings.get("buy_rsi_high", 30),
                    base_sell=settings.get("sell_rsi_high", 70),
                    mild_buffer=settings.get("mild_rsi_buffer_high", 3.0),
                    close=df['Close'],
                    vol_type="high",
                    recent_accuracy=0.7,
                    confidence_override=settings.get("confidence_threshold")
                )
            else:
                thresholds = adaptive_thresholds(
                    base_buy=settings.get("buy_rsi_low", 30),
                    base_sell=settings.get("sell_rsi_low", 70),
                    mild_buffer=settings.get("mild_rsi_buffer_low", 2.0),
                    close=df['Close'],
                    vol_type="low",
                    recent_accuracy=0.7,
                    confidence_override=settings.get("confidence_threshold")
                )

            # Merge dynamic thresholds into strategy parameters
            params = {
                **common_params,
                "buy_rsi_threshold": thresholds["buy_rsi"],
                "sell_rsi_threshold": thresholds["sell_rsi"],
                "mild_rsi_buffer": thresholds["mild_rsi_buffer"],
                "confidence_threshold": thresholds["confidence_threshold"]
            }

            # ---------------- Run strategy ----------------
            df = combined_strategy(df, params, coin_name=name)
            last_row = df.iloc[-1]

            close_price = last_row['Close']
            signal_val = last_row['Signal']
            ema_val = last_row['EMA']
            rsi_val = last_row['RSI']
            vol_spike = last_row['Vol_Spike']
            signal_label = last_row['Signal_Label']
            confidence = last_row['Confidence']

            # ---------------- Print & Log ----------------
            icon_map = {
                "BUY": "✅", "MILD_BUY": "🔹",
                "HOLD": "🤝", "MILD_SELL": "🔸", "SELL": "❌"
            }
            icon = icon_map.get(signal_label, "🤝")
            msg = f"{icon} {name}: {signal_label} @ ${close_price:.2f} | RSI={rsi_val:.2f} | EMA={ema_val:.2f} | Conf={confidence:.2f}"
            print(msg)
            latest_results.append(msg)

            # Save to DB (one row per prediction)
            log_prediction(
                coin=name,
                signal_str=signal_label,
                close_price=close_price,
                ema_val=ema_val,
                rsi_val=rsi_val,
                vol_spike=vol_spike
            )

        # ---------------- Wait before next check ----------------
        for _ in range(check_interval_seconds):
            if not bot_running:
                break
            time.sleep(1)

