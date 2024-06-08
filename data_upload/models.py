# data_upload/models.py

from django.db import models

class UploadedDataset(models.Model):
    file_path = models.CharField(max_length=255)
    uploaded_at = models.DateTimeField(auto_now_add=True)
