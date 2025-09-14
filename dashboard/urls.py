from django.urls import path, include
from . import views

urlpatterns = [
    path('', views.login_view, name="login"),
    path('logout/', views.logout_view, name="logout"),
    path('dashboard/', views.dashboard, name="dashboard"),
    path('trades/', views.trades, name="trades"),
    path('strategies/', views.strategies, name="strategies"),
    path('reports/', views.reports, name="reports"),
    path('bot/', views.bot, name="bot"),
    path('start_bot/', views.start_bot, name="start_bot"),
    path('stop_bot/', views.stop_bot, name="stop_bot"),
]