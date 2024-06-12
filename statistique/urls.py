from django.urls import path
from .views import show_statistics

app_name = 'statistics'

urlpatterns = [
    path('show/', show_statistics, name='show_statistics'),
]