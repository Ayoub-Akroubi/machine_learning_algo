from django.urls import path
from .views import upload_dataset

app_name = 'data_upload'

urlpatterns = [
    path('upload/', upload_dataset, name='upload'),
]
