from django.urls import path
from .views import apply_algorithm

app_name = 'ml_algorithms'

urlpatterns = [
    path('apply/', apply_algorithm, name='apply_algorithm'),
]
