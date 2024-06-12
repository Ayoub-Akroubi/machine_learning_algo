# data_upload/views.py

import os
from django.conf import settings
from django.shortcuts import render
import pandas as pd
from .models import UploadedDataset

def upload_dataset(request):
    if request.method == 'POST':
        if 'dataset' not in request.FILES:
            message = "No file uploaded."
            return render(request, 'data_upload/upload.html', {'message': message})

        dataset = request.FILES['dataset']

        try:
            df = pd.read_csv(dataset)
        except Exception as e:
            message = f"Error reading the CSV file: {str(e)}"
            return render(request, 'data_upload/upload.html', {'message': message})

        if df.empty:
            message = "The uploaded CSV file is empty."
            return render(request, 'data_upload/upload.html', {'message': message})
        else:
            file_path = os.path.join(settings.MEDIA_ROOT, dataset.name)
            with open(file_path, 'wb+') as destination:
                for chunk in dataset.chunks():
                    destination.write(chunk)
            UploadedDataset.objects.create(file_path=file_path)

            df_head = df.head()
            return render(request, 'data_upload/upload.html', {'df_head': df_head.to_html()})
    return render(request, 'data_upload/upload.html')
