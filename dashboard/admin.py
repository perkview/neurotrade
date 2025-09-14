from django.contrib import admin
from .models import CoinPrediction, BotStatus

# Register your models here.
admin.site.register(CoinPrediction)
admin.site.register(BotStatus)