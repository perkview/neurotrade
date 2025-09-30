#!/usr/bin/env python3
"""
Merged trading bot for MEXC Spot
- Uses quoteOrderQty for buys (reliable when buying with USDT)
- Uses quantity-based sells, adjusted for LOT_SIZE step
- Combines strategy/indicators + working order placement & signing
- Logs trades to Django TradeLog (append/create functions kept)
"""

import os
import time
import math
import hmac
import hashlib
import urllib.parse
import requests
import yfinance as yf
import pandas as pd
from datetime import timedelta
from django.utils import timezone
from dashboard.models import StrategySettings, TradeLog

# ---------------- CONFIGURATION ----------------
API_KEY = os.getenv("MEXC_API_KEY") or "mx0vglZPIwylJNCEnD"
API_SECRET_RAW = os.getenv("MEXC_API_SECRET") or "a45a612799654559b497adcdc34e1cca"
# Use bytes for HMAC
API_SECRET = API_SECRET_RAW.encode() if isinstance(API_SECRET_RAW, str) else API_SECRET_RAW
BASE_URL = "https://api.mexc.com"
API_V3 = BASE_URL + "/api/v3"

# Defaults (tweak as needed)
BASE_QUOTE = "USDT"
FALLBACK_SLEEP = 30
EXCHANGE_INFO_CACHE_TTL = 300
base_asset = "USDT"

# Percent usage for buys / mild buys (can be overridden via StrategySettings)
BUY_PERCENT = 3.0
MILD_BUY_PERCENT = 1.5

# Symbol map (yfinance ticker, exchange symbol)
SYMBOL_MAP = {
    "BTC": ("BTC-USD", "BTC" + BASE_QUOTE),
    "ETH": ("ETH-USD", "ETH" + BASE_QUOTE),
    "BNB": ("BNB-USD", "BNB" + BASE_QUOTE),
    "ADA": ("ADA-USD", "ADA" + BASE_QUOTE),
    "SOL": ("SOL-USD", "SOL" + BASE_QUOTE),
    "XRP": ("XRP-USD", "XRP" + BASE_QUOTE),
    "AVAX": ("AVAX-USD", "AVAX" + BASE_QUOTE),
    "DOT": ("DOT-USD", "DOT" + BASE_QUOTE),
    "LINK": ("LINK-USD", "LINK" + BASE_QUOTE),
    "LTC": ("LTC-USD", "LTC" + BASE_QUOTE),
    "DOGE": ("DOGE-USD", "DOGE" + BASE_QUOTE),
    "MANA": ("MANA-USD", "MANA" + BASE_QUOTE),
    "SAND": ("SAND-USD", "SAND" + BASE_QUOTE),
    "CAKE": ("CAKE-USD", "CAKE" + BASE_QUOTE),
    "FIL": ("FIL-USD", "FIL" + BASE_QUOTE),
}

# Exchange info cache
_exchange_info_cache = {"timestamp": 0, "data": None}

# ---------------- Utilities ----------------
def _now_ts_ms():
    return int(time.time() * 1000)


def _format_num(val: float, precision: int = 8) -> str:
    """Format numbers for MEXC (strip trailing zeros)."""
    return f"{val:.{precision}f}".rstrip("0").rstrip(".")


def _sign_request(params: dict) -> dict:
    """
    Canonicalize params (sorted), append signature using HMAC-SHA256,
    return a new dict containing original params + 'signature'.
    """
    # Filter out None / empty
    items = [(k, str(v)) for k, v in params.items() if v is not None and str(v) != ""]
    ordered_pairs = sorted(items, key=lambda x: x[0])
    query_string = urllib.parse.urlencode(ordered_pairs, quote_via=urllib.parse.quote)
    signature = hmac.new(API_SECRET, query_string.encode("utf-8"), hashlib.sha256).hexdigest()
    signed = dict(ordered_pairs)
    signed["signature"] = signature
    return signed


# ---------------- Exchange helpers ----------------
def get_exchange_info(force_refresh=False):
    now_ts = time.time()
    if not force_refresh and _exchange_info_cache["data"] and (now_ts - _exchange_info_cache["timestamp"] < EXCHANGE_INFO_CACHE_TTL):
        return _exchange_info_cache["data"]
    try:
        url = f"{API_V3}/exchangeInfo"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        _exchange_info_cache["data"] = data
        _exchange_info_cache["timestamp"] = now_ts
        return data
    except Exception as exc:
        print(f"Warning: failed to fetch exchange info: {exc}")
        return None


def _adjust_qty_to_step(symbol, qty, exchange_info):
    """Round qty down to allowed stepSize for symbol (LOT_SIZE)."""
    try:
        symbols = exchange_info.get("symbols", [])
        info = next((s for s in symbols if s.get("symbol") == symbol), None)
        if not info:
            return round(qty, 8)
        for f in info.get("filters", []):
            if f.get("filterType") == "LOT_SIZE":
                step = float(f.get("stepSize", 0) or 0)
                min_qty = float(f.get("minQty", 0) or 0)
                if step and step > 0:
                    decimals = max(0, int(round(-math.log10(step)))) if step < 1 else 0
                    adj = math.floor(qty / step) * step
                    if adj <= 0:
                        adj = 0.0
                    if min_qty and adj < min_qty:
                        # Not enough base balance to meet min_qty
                        return 0.0
                    return float(round(adj, decimals))
        return float(round(qty, 8))
    except Exception:
        return float(round(qty, 8))


# ---------------- Account & Balance ----------------
def get_account_balance(asset):
    """
    Get free balance for an asset using signed /account endpoint.
    """
    if not API_KEY or not API_SECRET:
        print("⚠️ API_KEY/API_SECRET not set")
        return 0.0

    path = "/api/v3/account"
    url = API_V3 + "/account"
    timestamp = _now_ts_ms()
    params = {"timestamp": timestamp, "recvWindow": 5000}
    signed = _sign_request(params)
    headers = {"X-MEXC-APIKEY": API_KEY}
    try:
        resp = requests.get(url, headers=headers, params=signed, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        for bal in data.get("balances", []):
            if bal.get("asset", "").upper() == asset.upper():
                return float(bal.get("free", 0.0))
        return 0.0
    except Exception as e:
        print(f"⚠️ HTTP error while fetching balance for {asset}: {e}")
        return 0.0


# ---------------- Orders (combined logic) ----------------
def place_market_buy_with_quote(symbol: str, quote_order_qty: float):
    """
    Place MARKET BUY using quoteOrderQty (spend USDT) -- recommended for BUYs.
    Returns API response dict or error dict.
    """
    if not API_KEY or not API_SECRET:
        return {"error": "API keys not set"}

    url = API_V3 + "/order"
    params = {
        "symbol": symbol.upper(),
        "side": "BUY",
        "type": "MARKET",
        "quoteOrderQty": _format_num(quote_order_qty, precision=8),
        "timestamp": _now_ts_ms(),
        "recvWindow": 5000,
    }
    signed = _sign_request(params)
    headers = {"X-MEXC-APIKEY": API_KEY}
    try:
        # using params as query parameters (MEXC accepts this for POST)
        r = requests.post(url, headers=headers, params=signed, timeout=15)
        return r.json()
    except requests.Timeout:
        return {"error": "Timeout while placing order"}
    except requests.RequestException as exc:
        return {"error": f"Request failed: {exc}"}
    except Exception as exc:
        return {"error": f"Unexpected error: {exc}"}


def place_market_sell_by_qty(symbol: str, quantity: float):
    """
    Place MARKET SELL specifying quantity (base asset).
    """
    if not API_KEY or not API_SECRET:
        return {"error": "API keys not set"}

    url = API_V3 + "/order"
    params = {
        "symbol": symbol.upper(),
        "side": "SELL",
        "type": "MARKET",
        "quantity": _format_num(quantity, precision=8),
        "timestamp": _now_ts_ms(),
        "recvWindow": 5000,
    }
    signed = _sign_request(params)
    headers = {"X-MEXC-APIKEY": API_KEY}
    try:
        r = requests.post(url, headers=headers, params=signed, timeout=15)
        return r.json()
    except requests.Timeout:
        return {"error": "Timeout while placing sell order"}
    except requests.RequestException as exc:
        return {"error": f"Request failed: {exc}"}
    except Exception as exc:
        return {"error": f"Unexpected error: {exc}"}


# ---------------- Data fetch & indicators (from first code) ----------------
def fetch_ohlcv(coin_key: str, period: str, interval: str):
    mapping = SYMBOL_MAP.get(coin_key.upper())
    if not mapping:
        print(f"⚠️ No symbol mapping for {coin_key}; skipping")
        return None
    yahoo_ticker = mapping[0]
    try:
        df = yf.Ticker(yahoo_ticker).history(period=period, interval=interval)
        if df is None or df.empty:
            print(f"${yahoo_ticker}: no price data found (period={period})")
            return None
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as exc:
        print(f"Error fetching {yahoo_ticker} for {coin_key}: {exc}")
        return None


def ema(series, period):
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def rsi(series, period):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def volume_spike(volume, period, multiplier, rsi=None, confirm_with_rsi=True):
    vol_ma = volume.rolling(window=period, min_periods=1).mean()
    spike = (volume > vol_ma * multiplier)
    if confirm_with_rsi and rsi is not None:
        spike = spike & (rsi.diff() > 0)
    return spike.astype(float)


def combined_strategy(df, params, coin_name):
    df = df.copy()
    close = df["Close"]
    df["EMA"] = ema(close, params["ema_period"]).ffill()
    df["RSI"] = rsi(close, params["rsi_period"]).ffill()

    if params.get("use_volume", True):
        df["Vol_Spike"] = volume_spike(
            volume=df["Volume"],
            period=params["volume_ma_period"],
            multiplier=params["volume_spike_multiplier"],
            rsi=df["RSI"],
            confirm_with_rsi=params.get("confirm_volume_with_rsi", True),
        ).ffill().fillna(0.0)
    else:
        df["Vol_Spike"] = 1.0

    ema_buy_level = df["EMA"] * (1 + params["ema_buy_buffer"])
    ema_sell_level = df["EMA"] * (1 - params["ema_sell_buffer"])

    strict_buy = ((close > ema_buy_level) | (df["RSI"] <= params["buy_rsi_threshold"])) & (df["Vol_Spike"] == 1.0)
    strict_sell = ((close < ema_sell_level) | (df["RSI"] >= params["sell_rsi_threshold"]))

    mild_buffer = params.get("mild_rsi_buffer", 2.0)
    mild_buy = ((close > df["EMA"]) & (df["RSI"] < (params["buy_rsi_threshold"] + mild_buffer))) | (
        (df["RSI"] < params["buy_rsi_threshold"]) & (df["Vol_Spike"] == 1.0)
    )
    mild_sell = ((close < df["EMA"]) & (df["RSI"] > (params["sell_rsi_threshold"] - mild_buffer))) | (df["RSI"] > params["sell_rsi_threshold"])

    df["Signal"] = 0.0
    df.loc[strict_buy, "Signal"] = 1.0
    df.loc[strict_sell, "Signal"] = -1.0
    df.loc[mild_buy & (df["Signal"] == 0.0), "Signal"] = 0.5
    df.loc[mild_sell & (df["Signal"] == 0.0), "Signal"] = -0.5

    buy_thresh = params["buy_rsi_threshold"]
    df["RSI_score"] = (1.0 - (abs(df["RSI"] - buy_thresh) / 100.0)).clip(0, 1)
    df["EMA_score"] = (1.0 - (abs(close - df["EMA"]) / close)).clip(0, 1)
    df["Vol_score"] = df["Vol_Spike"].clip(0, 1)

    df["Confidence"] = (0.5 * df["RSI_score"] + 0.3 * df["EMA_score"] + 0.2 * df["Vol_score"])
    confidence_threshold = params.get("confidence_threshold", 0.6)
    df.loc[(df["Signal"] == 1.0) & (df["Confidence"] < confidence_threshold), "Signal"] = 0.5
    df.loc[df["Confidence"] < 0.1, "Signal"] = 0.0

    df["Signal_Label"] = df["Signal"].map({1.0: "BUY", 0.5: "MILD_BUY", 0.0: "HOLD", -0.5: "MILD_SELL", -1.0: "SELL"})
    return df


# ---------------- Strategy Settings ----------------
def get_strategy_settings():
    """
    Reads latest StrategySettings from Django DB (assumes model fields exist).
    Returns None if missing.
    """
    try:
        settings = StrategySettings.objects.latest("created_at")
    except Exception:
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
        "confidence_threshold": getattr(settings, "confidence_threshold", 0.6),
    }


# ---------------- Signal helper (integrates combined_strategy) ----------------
def check_signal(coin):
    params = get_strategy_settings()
    if not params:
        return "HOLD", 0.0, 0.0

    df = fetch_ohlcv(coin, params["period"], params["interval"])
    if df is None or df.empty:
        mapping = SYMBOL_MAP.get(coin)
        if mapping:
            mexc_symbol = mapping[1]
            price = get_symbol_price(mexc_symbol)
            return "HOLD", 0.0, price
        return "HOLD", 0.0, 0.0

    df_res = combined_strategy(
        df,
        {
            "ema_period": params["ema_period"],
            "rsi_period": params["rsi_period"],
            "volume_ma_period": params["volume_ma_period"],
            "volume_spike_multiplier": params["volume_spike_multiplier"],
            "use_volume": params["use_volume"],
            "ema_buy_buffer": params["ema_buy_buffer"],
            "ema_sell_buffer": params["ema_sell_buffer"],
            "buy_rsi_threshold": params["buy_rsi_low"],
            "sell_rsi_threshold": params["sell_rsi_high"],
            "mild_rsi_buffer": params["mild_rsi_buffer_low"],
            "confidence_threshold": params["confidence_threshold"],
        },
        coin,
    )

    last = df_res.iloc[-1]
    action = last["Signal_Label"]
    mapping = SYMBOL_MAP.get(coin)
    mexc_symbol = mapping[1] if mapping else None
    live_price = get_symbol_price(mexc_symbol) if mexc_symbol else 0.0
    price = float(last["Close"]) if float(last["Close"]) > 0 else live_price
    if price <= 0:
        price = live_price
    confidence = float(last.get("Confidence", 0.0))
    return action, confidence, price


def _symbols_from_settings_or_default(params):
    coins_raw = params.get("coins") if params else None
    if isinstance(coins_raw, dict):
        return [k.upper() for k in coins_raw.keys()]
    if isinstance(coins_raw, str):
        return [c.strip().upper() for c in coins_raw.split(",") if c.strip()]
    if isinstance(coins_raw, list):
        return [c.strip().upper() for c in coins_raw]
    return ["BTC", "ETH", "SOL"]


def get_symbol_price(mexc_symbol: str):
    try:
        url = f"{API_V3}/ticker/price"
        resp = requests.get(url, params={"symbol": mexc_symbol}, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        return float(data.get("price", 0.0))
    except Exception as exc:
        print(f"⚠️ Failed to fetch ticker price for {mexc_symbol}: {exc}")
        return 0.0


# ---------------- Logging and DB helpers ----------------
def append_trade_safe(**kwargs):
    """
    Add trade record safely. Try to use TradeLog.create; fallback to update/get_or_create logic.
    """
    try:
        TradeLog.objects.create(**kwargs)
    except Exception as exc:
        try:
            # fallback to append method if exists on singleton
            trade_log, _ = TradeLog.objects.get_or_create(id=1)
            if hasattr(trade_log, "append_trade"):
                trade_log.append_trade(**kwargs)
        except Exception as exc2:
            print(f"Warning: failed to append trade to DB: {exc2}")


def has_open_position(coin):
    try:
        return TradeLog.objects.filter(coin=coin, status="OPEN").exists()
    except Exception:
        return False


def get_open_positions(coin):
    try:
        return TradeLog.objects.filter(coin=coin, status="OPEN").order_by("id")
    except Exception:
        return []


def close_open_positions(coin, exit_price, qty_sold, order_response):
    """
    Close positions by iterating open TradeLog entries and marking closed until qty_sold is exhausted.
    """
    try:
        remaining = float(qty_sold)
        open_positions = get_open_positions(coin)
        for pos in open_positions:
            pos_qty = float(getattr(pos, "quantity", 0) or 0)
            if pos_qty <= 0:
                append_trade_safe(
                    coin=coin,
                    action="SELL",
                    exit_price=exit_price,
                    status="CLOSED",
                    order_symbol=getattr(pos, "order_symbol", None),
                    order_price=exit_price,
                    timestamp_ms=_now_ts_ms(),
                    actions=order_response,
                )
                continue

            if remaining >= pos_qty:
                append_trade_safe(
                    coin=coin,
                    action="SELL",
                    exit_price=exit_price,
                    quantity=pos_qty,
                    status="CLOSED",
                    order_symbol=getattr(pos, "order_symbol", None),
                    order_price=exit_price,
                    timestamp_ms=_now_ts_ms(),
                    actions=order_response,
                )
                remaining -= pos_qty
            else:
                append_trade_safe(
                    coin=coin,
                    action="SELL",
                    exit_price=exit_price,
                    quantity=remaining,
                    status="CLOSED",
                    order_symbol=getattr(pos, "order_symbol", None),
                    order_price=exit_price,
                    timestamp_ms=_now_ts_ms(),
                    actions=order_response,
                )
                remaining = 0

            if remaining <= 0:
                break

        if remaining > 0:
            append_trade_safe(
                coin=coin,
                action="SELL",
                exit_price=exit_price,
                quantity=remaining,
                status="CLOSED",
                order_symbol=coin + BASE_QUOTE,
                order_price=exit_price,
                timestamp_ms=_now_ts_ms(),
                actions=order_response,
            )

    except Exception as exc:
        print(f"Warning: failed to update/close open positions for {coin}: {exc}")


# ---------------- Qty calculation helper ----------------
def calculate_buy_quote_amount(percent_of_usdt: float):
    """
    Return USDT amount to spend given percent_of_usdt of available USDT balance.
    """
    usdt_balance = get_account_balance(base_asset)
    if usdt_balance <= 0:
        return 0.0
    return usdt_balance * (percent_of_usdt / 100.0)


def calculate_sell_qty_from_balance(symbol, exchange_info, fraction=1.0):
    """
    Get base asset balance and adjust to LOT_SIZE for selling.
    fraction: fraction of base asset balance to sell (1.0 = all).
    """
    base_asset_sym = symbol.upper().replace(BASE_QUOTE, "")
    base_balance = get_account_balance(base_asset_sym)
    if base_balance <= 0:
        return 0.0
    qty = base_balance * fraction
    return _adjust_qty_to_step(symbol, qty, exchange_info)


# ---------------- Main Loop ----------------
def main_loop():
    print("🚀 Starting Merged Trading Bot...\n")
    while True:
        try:
            params = get_strategy_settings()
            if not params or not params.get("active", False):
                print("⚠️ No active strategy. Sleeping...")
                time.sleep(FALLBACK_SLEEP)
                continue

            symbols = _symbols_from_settings_or_default(params)
            exchange_info = get_exchange_info()
            sleep_for = int(params.get("check_interval_seconds", 900))

            for coin in symbols:
                try:
                    print(f"\n🔎 Checking {coin} ...")
                    action, confidence, price = check_signal(coin)

                    if price <= 0:
                        print(f"⚠️ {coin}: invalid price ({price}) — skipping trading")
                        continue

                    mapping = SYMBOL_MAP.get(coin)
                    if not mapping:
                        print(f"⚠️ No symbol mapping for {coin}; skipping trading")
                        continue
                    mexc_symbol = mapping[1]

                    # ---------------- BUY ----------------
                    if action == "BUY":
                        usdt_to_spend = calculate_buy_quote_amount(BUY_PERCENT)
                        usdt_to_spend = max(0.0, usdt_to_spend)
                        if usdt_to_spend <= 0:
                            print(f"⚠️ {coin}: no USDT available to buy.")
                            continue
                        resp = place_market_buy_with_quote(mexc_symbol, usdt_to_spend)
                        print(f"✅ {coin}: BUY spend={usdt_to_spend} USDT | Resp={resp} | Conf={confidence:.2f}")
                        

                    # ---------------- MILD BUY ----------------
                    elif action == "MILD_BUY":
                        usdt_to_spend = calculate_buy_quote_amount(MILD_BUY_PERCENT)
                        if usdt_to_spend <= 0:
                            print(f"⚠️ {coin}: no USDT available for mild buy.")
                            continue
                        resp = place_market_buy_with_quote(mexc_symbol, usdt_to_spend)
                        print(f"🤏 {coin}: MILD_BUY spend={usdt_to_spend} USDT | Resp={resp} | Conf={confidence:.2f}")
                        

                    # ---------------- SELL ----------------
                    elif action == "SELL":
                        # Sell a reasonable amount (default: sell all available base asset)
                        qty = calculate_sell_qty_from_balance(mexc_symbol, exchange_info, fraction=1.0)
                        if qty <= 0:
                            print(f"⚠️ {coin}: nothing to sell (qty={qty})")
                            continue
                        resp = place_market_sell_by_qty(mexc_symbol, qty)
                        print(f"✅ {coin}: SELL qty={qty} @ {price} | Resp={resp} | Conf={confidence:.2f}")
                        close_open_positions(coin, exit_price=price, qty_sold=qty, order_response=resp)
                        

                    # ---------------- MILD SELL ----------------
                    elif action == "MILD_SELL":
                        qty = calculate_sell_qty_from_balance(mexc_symbol, exchange_info, fraction=0.5)
                        if qty <= 0:
                            print(f"⚠️ {coin}: nothing to mild-sell (qty={qty})")
                            continue
                        resp = place_market_sell_by_qty(mexc_symbol, qty)
                        print(f"🤏 {coin}: MILD_SELL qty={qty} @ {price} | Resp={resp} | Conf={confidence:.2f}")
                        close_open_positions(coin, exit_price=price, qty_sold=qty, order_response=resp)
                        

                    # ---------------- HOLD ----------------
                    else:
                        print(f"⏸️ {coin}: HOLD @ {price} | Conf={confidence:.2f}")

                except Exception as exc_coin:
                    print(f"⚠️ Error while processing {coin}: {exc_coin}")
                    continue

            print(f"\n⏳ Sleeping {sleep_for}s...\n")
            time.sleep(sleep_for)

        except Exception as e:
            print(f"❌ Error in main_loop: {e}")
            time.sleep(FALLBACK_SLEEP)


if __name__ == "__main__":
    main_loop()
