from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from . import main as strategy
from django.http import JsonResponse
from .models import CoinPrediction, BotStatus
import threading
from django.utils import timezone
from datetime import datetime

# Create your views here.
    
# Global variable to hold the bot thread
bot_thread = None

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
    # This will be the trading controller dashboard
    context = {
        "username": request.user.username,
        "trading_status": "Running",  # later connect to algo
        "last_signal": "BUY BTC",     # placeholder example
    }
    return render(request, "dashboard.html", context)


@login_required
def trades(request):
    return render(request, "trades.html")

@login_required
def strategies(request):
    return render(request, "strategies.html")

@login_required
def reports(request):
    return render(request, "reports.html")

@login_required
def bot(request):
    bot_status, _ = BotStatus.objects.get_or_create(id=1)

    # Get prediction log object (assumes only one row is used)
    log_obj = CoinPrediction.objects.first()
    log_lines = log_obj.log.strip().split('\n') if log_obj and log_obj.log else []

    predictions = []
    for line in log_lines:
        try:
            # Example log entry format:
            # [2025-09-14 15:30:00] BTC - BUY @ 26100.5 | EMA200: 25800, RSI: 35.4, Volume Spike: Yes, Next Check: 2025-09-14 16:30:00

            # --- Extract timestamp ---
            timestamp_part, rest = line.split('] ', 1)
            timestamp_str = timestamp_part.strip('[')

            # Convert string -> datetime (make timezone aware in Asia/Karachi)
            timestamp_dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
            timestamp_dt = timezone.make_aware(timestamp_dt, timezone=timezone.get_current_timezone())

            # --- Extract coin, signal, price ---
            coin_signal, indicators = rest.split('|', 1)
            coin, signal_price = coin_signal.strip().split(' - ')
            signal, price = signal_price.strip().split(' @ ')
            price = float(price)

            # --- Extract indicators ---
            parts = indicators.split(',')
            ema200 = parts[0].split(':')[1].strip()
            rsi = parts[1].split(':')[1].strip()
            volume_spike = parts[2].split(':')[1].strip()
            next_check_str = parts[3].split(':', 1)[1].strip()

            # Convert Next Check -> datetime (if possible)
            try:
                next_check_dt = datetime.strptime(next_check_str, "%Y-%m-%d %H:%M:%S")
                next_check_dt = timezone.make_aware(next_check_dt, timezone=timezone.get_current_timezone())
            except Exception:
                next_check_dt = None  # fallback if parsing fails

            # --- Build prediction dict ---
            predictions.append({
                "timestamp": timestamp_dt,
                "coin": coin,
                "signal": signal,
                "price": price,
                "ema200": float(ema200) if ema200 != "N/A" else None,
                "rsi": float(rsi) if rsi != "N/A" else None,
                "volume_spike": volume_spike == "Yes",
                "next_check": next_check_dt,
            })
        except Exception:
            continue  # skip malformed lines

    context = {
        'bot_status': bot_status,
        'latest_results': strategy.latest_results,
        'predictions': predictions,
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