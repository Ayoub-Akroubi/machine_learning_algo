from django.urls import path
from .views import predict

app_name = 'prediction'

urlpatterns = [
    path('predict/', predict, name='predict'),
]