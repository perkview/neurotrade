from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from . import main as strategy
from . import trade as code
from .models import CoinPrediction, BotStatus, StrategySettings, TradeStatus, TradeLog
import threading
from django.utils import timezone
from datetime import datetime
import json
import pandas as pd

# Create your views here.
    
# Global variable to hold the bot thread
bot_thread = None
trade_thread = None

# Login page
def login_view(request):
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")
        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            return redirect("dashboard")
        else:
            messages.error(request, "Invalid username or password")

    return render(request, "login.html")

# Logout
def logout_view(request):
    logout(request)
    return redirect("login")

# Protected landing page (controller)
@login_required
def dashboard(request):
    # ---------------- Strategy Settings ----------------
    strategy = StrategySettings.objects.last()  # assuming one strategy row; adjust as needed

    # ---------------- Bot / Trade Status ----------------
    bot_status = TradeStatus.objects.first()  # assuming one row only

    # ---------------- All Trades ----------------
    all_trades = TradeLog.objects.all().order_by('-updated_at')

    # ---------------- Open Trades (parse latest logs) ----------------
    open_trades = []
    for trade in all_trades:
        lines = trade.log.strip().split("\n")[-10:]  # get last 10 entries
        for line in lines:
            parts = line.split('|')
            data = {}
            for part in parts:
                if ':' in part:
                    k, v = part.split(':', 1)
                    data[k.strip()] = v.strip()
            open_trades.append({
                'id': trade.id,
                'coin': data.get('coin', 'N/A'),
                'action': data.get('action', 'N/A'),
                'quantity': data.get('Quantity', 'N/A'),
                'pnl': data.get('PnL', 'N/A'),
                'status': data.get('Status', 'N/A')
            })

    open_trades = sorted(open_trades, key=lambda x: x['id'], reverse=True)[:10]

    # ---------------- Last Trade ----------------
    last_trade = None
    if all_trades.exists():
        last_line = all_trades.first().log.strip().split('\n')[-1]
        parts = last_line.split('|')
        data = {}
        for part in parts:
            if ':' in part:
                k, v = part.split(':', 1)
                data[k.strip()] = v.strip()
        last_trade = {
            'coin': data.get('coin', 'N/A'),
            'action': data.get('action', 'N/A')
        }

    # ---------------- Daily PnL ----------------
    today = timezone.now().date()
    pnl_today = 0.0
    for trade in all_trades:
        for line in trade.log.strip().split("\n"):
            timestamp_str = line.split(']')[0][1:]
            trade_date = pd.to_datetime(timestamp_str).date()
            if trade_date == today and 'PnL:' in line:
                pnl_str = line.split('PnL:')[1].split(',')[0].replace('$','').replace('+','')
                try:
                    pnl_today += float(pnl_str)
                except:
                    continue

    # ---------------- Balance / Equity ----------------
    balance = 12540  # placeholder: replace with real wallet/balance calculation

    # ---------------- Equity Curve & Monthly PnL ----------------
    equity_curve = [10000, 10200, 10800, 11200, 10750, 11500, 12000, 12600, 12870]  # example
    equity_labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep']

    monthly_pnl = [500, 300, -200, 700, -150, 1200, 900, 1500]  # example
    pnl_labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug']

    context = {
        'strategy': strategy,
        'bot_status': bot_status,
        'last_trade': last_trade,
        'open_trades': open_trades,
        'pnl_today': pnl_today,
        'balance': balance,
        'equity_curve': equity_curve,
        'equity_labels': equity_labels,
        'monthly_pnl': monthly_pnl,
        'pnl_labels': pnl_labels,
    }

    return render(request, 'dashboard.html', context)


@login_required
def trades(request):
    # Get current bot status
    trade_status = TradeStatus.objects.order_by('-updated_at').first()

    # Get latest trade log
    trade_log_obj = TradeLog.objects.order_by('-updated_at').first()
    log_entries = []
    if trade_log_obj and trade_log_obj.log:
        # Split the log by newlines to display in the template
        log_entries = trade_log_obj.log.strip().split('\n')
        log_entries.reverse()  # latest first

    context = {
        "trade_status": trade_status,
        "log_entries": log_entries
    }
    return render(request, "trades.html", context)




@login_required
def strategies(request):
    # Ensure a single settings row exists (id=1)
    settings, created = StrategySettings.objects.get_or_create(id=1)

    if request.method == "POST":
        try:
            # ---------------- Coins JSON ----------------
            coins_raw = request.POST.get("coins", "{}").strip()
            try:
                coins_data = json.loads(coins_raw)
                if not isinstance(coins_data, dict):
                    raise ValueError("Coins must be a JSON object (dictionary).")
                settings.coins = coins_data
            except Exception as e:
                messages.error(request, f"⚠️ Invalid JSON format for Coins: {e}")

            # ---------------- Volatility Coin Lists ----------------
            low_vol_raw = request.POST.get("low_vol_coins", "[]").strip()
            try:
                low_vol_data = json.loads(low_vol_raw)
                if not isinstance(low_vol_data, list):
                    raise ValueError("Low volatility coins must be a JSON list.")
                settings.low_vol_coins = low_vol_data
            except Exception as e:
                messages.error(request, f"⚠️ Invalid JSON format for Low Volatility coins: {e}")

            high_vol_raw = request.POST.get("high_vol_coins", "[]").strip()
            try:
                high_vol_data = json.loads(high_vol_raw)
                if not isinstance(high_vol_data, list):
                    raise ValueError("High volatility coins must be a JSON list.")
                settings.high_vol_coins = high_vol_data
            except Exception as e:
                messages.error(request, f"⚠️ Invalid JSON format for High Volatility coins: {e}")

            # ---------------- General Settings ----------------
            settings.period = request.POST.get("period", settings.period)
            settings.interval = request.POST.get("interval", settings.interval)

            # ---------------- Indicators ----------------
            settings.ema_period = int(request.POST.get("ema_period", settings.ema_period))
            settings.rsi_period = int(request.POST.get("rsi_period", settings.rsi_period))
            settings.volume_ma_period = int(request.POST.get("volume_ma_period", settings.volume_ma_period))
            settings.volume_spike_multiplier = float(
                request.POST.get("volume_spike_multiplier", settings.volume_spike_multiplier)
            )

            # ---------------- RSI Thresholds ----------------
            settings.buy_rsi_low = int(request.POST.get("buy_rsi_low", settings.buy_rsi_low))
            settings.sell_rsi_low = int(request.POST.get("sell_rsi_low", settings.sell_rsi_low))
            settings.buy_rsi_high = int(request.POST.get("buy_rsi_high", settings.buy_rsi_high))
            settings.sell_rsi_high = int(request.POST.get("sell_rsi_high", settings.sell_rsi_high))

            # ---------------- EMA Buffers ----------------
            settings.ema_buffer_buy = float(request.POST.get("ema_buffer_buy", settings.ema_buffer_buy))
            settings.ema_buffer_sell = float(request.POST.get("ema_buffer_sell", settings.ema_buffer_sell))

            # ---------------- Mild RSI Buffers ----------------
            settings.mild_rsi_buffer_low = float(
                request.POST.get("mild_rsi_buffer_low", settings.mild_rsi_buffer_low)
            )
            settings.mild_rsi_buffer_high = float(
                request.POST.get("mild_rsi_buffer_high", settings.mild_rsi_buffer_high)
            )

            # ---------------- Confidence Threshold ----------------
            settings.confidence_threshold = float(
                request.POST.get("confidence_threshold", settings.confidence_threshold or 0.6)
            )

            # ---------------- Volume & Bot ----------------
            settings.use_volume = "use_volume" in request.POST
            settings.check_interval_seconds = int(
                request.POST.get("check_interval_seconds", settings.check_interval_seconds)
            )
            settings.active = "active" in request.POST

            # ---------------- Save ----------------
            settings.save()
            messages.success(request, "✅ Strategy settings updated successfully!")
            return redirect("strategies")

        except Exception as e:
            messages.error(request, f"❌ Error updating settings: {e}")

    return render(request, "strategies.html", {"settings": settings})



@login_required
def reports(request):
    logs = TradeLog.objects.order_by('updated_at')

    trades_data = []
    for log_obj in logs:
        for line in log_obj.log.strip().split("\n"):
            try:
                timestamp = line.split("]")[0].strip("[")
                rest = line.split("]")[1].strip()
                coin_action, *fields = rest.split(" | ")

                coin, action = coin_action.split(" - ")
                field_dict = {}
                for f in fields:
                    if ":" in f:
                        key, value = f.split(":", 1)
                        field_dict[key.strip()] = value.strip()

                # Handle PnL safely (convert 'N/A' to None)
                pnl_str = field_dict.get("PnL", "0")
                try:
                    pnl = float(pnl_str)
                except (ValueError, TypeError):
                    pnl = None

                trades_data.append({
                    "coin": coin.strip(),
                    "action": action.strip(),
                    "quantity": float(field_dict.get("Quantity", 0)),
                    "entry_price": float(field_dict.get("Entry Price", 0)),
                    "exit_price": float(field_dict.get("Exit Price", 0)) if field_dict.get("Exit Price") not in [None, 'N/A'] else None,
                    "pnl": pnl,
                    "status": field_dict.get("Status", "OPEN"),
                    "timestamp": pd.to_datetime(timestamp),
                })
            except Exception:
                continue  # skip badly formatted lines

    df = pd.DataFrame(trades_data)

    # Ensure 'pnl' column exists even if df is empty
    if 'pnl' not in df.columns:
        df['pnl'] = pd.Series(dtype=float)
    else:
        df['pnl'] = df['pnl'].fillna(0)

    total_trades = len(df)
    if total_trades > 0:
        winning_trades = df[df['pnl'] > 0]
        losing_trades = df[df['pnl'] < 0]
        break_even_trades = df[df['pnl'] == 0]

        win_rate = round(len(winning_trades)/total_trades*100, 2)
        loss_rate = round(len(losing_trades)/total_trades*100, 2)
        break_even_rate = round(len(break_even_trades)/total_trades*100, 2)

        gross_profit = winning_trades['pnl'].sum()
        gross_loss = abs(losing_trades['pnl'].sum())
        profit_factor = round(gross_profit/gross_loss, 2) if gross_loss != 0 else 0
        expectancy = round(df['pnl'].mean(), 4)
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) else 0
        avg_loss = abs(losing_trades['pnl'].mean()) if len(losing_trades) else 0
        payoff_ratio = round(avg_win/avg_loss, 2) if avg_loss != 0 else 0
        edge_ratio = round((win_rate/100 * avg_win) / (loss_rate/100 * avg_loss), 2) if avg_loss != 0 else 0
        net_profit = round(df['pnl'].sum(), 2)
        equity_curve = df['pnl'].cumsum()
        max_drawdown = round((equity_curve.min()/equity_curve.max()*100), 2) if not equity_curve.empty else 0
    else:
        # Defaults if no trades exist
        winning_trades = losing_trades = break_even_trades = pd.DataFrame()
        win_rate = loss_rate = break_even_rate = 0
        profit_factor = expectancy = payoff_ratio = edge_ratio = net_profit = 0
        equity_curve = pd.Series([], dtype=float)
        max_drawdown = 0
        avg_win = avg_loss = 0

    context = {
        "total_trades": total_trades,
        "win_rate": win_rate,
        "loss_rate": loss_rate,
        "break_even_rate": break_even_rate,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "payoff_ratio": payoff_ratio,
        "edge_ratio": edge_ratio,
        "net_profit": net_profit,
        "max_drawdown": max_drawdown,
        "equity_curve": list(equity_curve.values) if not equity_curve.empty else [],
        "trade_dates": [x.strftime("%d-%b") for x in df['timestamp']] if not df.empty else [],
        "winning_count": len(winning_trades),
        "losing_count": len(losing_trades),
        "trades": trades_data,  # send trades to template
    }

    return render(request, "reports.html", context)

@login_required
def bot(request):
    # Ensure bot status row exists
    bot_status, _ = BotStatus.objects.get_or_create(id=1)

    # Fetch prediction log object
    log_obj = CoinPrediction.objects.first()
    log_lines = log_obj.log.strip().split('\n') if log_obj and log_obj.log else []

    predictions = []
    for line in log_lines:
        try:
            # Example format:
            # [2025-09-14 15:30:00] BTC - BUY @ 26100.5 | EMA200: 25800, RSI: 35.4, Volume Spike: Yes, Next Check: 2025-09-14 16:30:00

            # --- Timestamp ---
            timestamp_part, rest = line.split('] ', 1)
            timestamp_str = timestamp_part.strip('[')
            timestamp_dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
            timestamp_dt = timezone.make_aware(timestamp_dt, timezone=timezone.get_current_timezone())

            # --- Coin, signal, price ---
            coin_signal, indicators = rest.split('|', 1)
            coin, signal_price = coin_signal.strip().split(' - ')
            signal, price = signal_price.strip().split(' @ ')
            price = float(price)

            # --- Indicators ---
            parts = indicators.split(',')
            ema200 = parts[0].split(':', 1)[1].strip()
            rsi = parts[1].split(':', 1)[1].strip()
            volume_spike = parts[2].split(':', 1)[1].strip()
            next_check_str = parts[3].split(':', 1)[1].strip()

            # Parse next_check safely
            try:
                next_check_dt = datetime.strptime(next_check_str, "%Y-%m-%d %H:%M:%S")
                next_check_dt = timezone.make_aware(next_check_dt, timezone=timezone.get_current_timezone())
            except Exception:
                next_check_dt = None

            # --- Prediction dict ---
            predictions.append({
                "timestamp": timestamp_dt,
                "coin": coin.strip(),
                "signal": signal.strip(),
                "price": price,
                "ema200": float(ema200) if ema200 != "N/A" else None,
                "rsi": float(rsi) if rsi != "N/A" else None,
                "volume_spike": (volume_spike == "Yes"),
                "next_check": next_check_dt,
            })

        except Exception:
            continue  # Skip malformed lines

    context = {
        "bot_status": bot_status,
        "latest_results": getattr(strategy, "latest_results", []),
        "predictions": predictions,
    }
    return render(request, "bot.html", context)

@login_required
def start_bot(request):
    global bot_thread
    bot_status, _ = BotStatus.objects.get_or_create(id=1)

    if bot_status.is_running:
        messages.info(request, "Bot is already running!")
    else:
        bot_status.is_running = True
        bot_status.save()

        # Start bot in a background thread
        strategy.bot_running = True
        bot_thread = threading.Thread(target=strategy.main_loop, daemon=True)
        bot_thread.start()

        messages.success(request, "Bot started successfully!")

    return redirect("bot")

@login_required
def stop_bot(request):
    bot_status, _ = BotStatus.objects.get_or_create(id=1)

    if not bot_status.is_running:
        messages.info(request, "Bot is already stopped!")
    else:
        bot_status.is_running = False
        bot_status.save()

        # Stop bot
        strategy.bot_running = False
        messages.success(request, "Bot stopped successfully!")

    return redirect("bot")




@login_required
def start_trade(request):
    global trade_thread
    trade_status, _ = TradeStatus.objects.get_or_create(id=1)

    if trade_status.is_running:
        messages.info(request, "Trade is already running!")
    else:
        trade_status.is_running = True
        trade_status.save()

        # Start bot in a background thread
        code.bot_running = True
        bot_thread = threading.Thread(target=code.main_loop, daemon=True)
        bot_thread.start()

        messages.success(request, "Trade started successfully!")

    return redirect("trades")

@login_required
def stop_trade(request):
    trade_status, _ = TradeStatus.objects.get_or_create(id=1)

    if not trade_status.is_running:
        messages.info(request, "Trade is already stopped!")
    else:
        trade_status.is_running = False
        trade_status.save()

        # Stop bot
        code.bot_running = False
        messages.success(request, "Trade stopped successfully!")

    return redirect("trades")





