from django.contrib import admin
from .models import CoinPrediction, BotStatus, StrategySettings, TradeStatus, TradeLog

# Register your models here.
admin.site.register(CoinPrediction)
admin.site.register(BotStatus)
admin.site.register(StrategySettings)
admin.site.register(TradeStatus)
admin.site.register(TradeLog)