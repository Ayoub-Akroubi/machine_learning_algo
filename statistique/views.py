from django.shortcuts import render
import pandas as pd
from data_upload.models import UploadedDataset

def show_statistics(request):
    try:
        dataset = UploadedDataset.objects.latest('uploaded_at')
        file_path = dataset.file_path
        df = pd.read_csv(file_path)
        statistics = df.describe().to_html()
        return render(request, 'statistique/statistique.html', {'statistics': statistics})
    except UploadedDataset.DoesNotExist:
        return render(request, 'statistique/statistique.html', {'message': 'No dataset uploaded yet.'})
    except Exception as e:
        return render(request, 'statistique/statistique.html', {'message': f'An error occurred: {str(e)}'})
