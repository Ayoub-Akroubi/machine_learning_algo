from django.shortcuts import render
import pandas as pd
import logging
from io import StringIO
from data_upload.models import UploadedDataset

# Set up logging
logger = logging.getLogger(__name__)

def show_statistics(request):
    try:
        dataset = UploadedDataset.objects.latest('uploaded_at')
        file_path = dataset.file_path

        # Load the dataset
        df = pd.read_csv(file_path)

        # Get basic statistics
        statistics = df.describe().to_html()

        # Get data info
        buffer = StringIO()
        df.info(buf=buffer)
        info = buffer.getvalue()

        # Get first few rows of the data
        head = df.head().to_html()

        # If reshaping is needed, perform it here
        
        shape = df.shape

        return render(request, 'statistique/statistique.html', {
            'statistics': statistics,
            'info': info,
            'head': head,
            'shape': shape
        })
    except UploadedDataset.DoesNotExist:
        logger.warning('No dataset uploaded yet.')
        return render(request, 'statistique/statistique.html', {'message': 'No dataset uploaded yet.'})
    except FileNotFoundError:
        logger.error(f'File not found: {file_path}')
        return render(request, 'statistique/statistique.html', {'message': 'Dataset file not found.'})
    except pd.errors.EmptyDataError:
        logger.error('The dataset is empty.')
        return render(request, 'statistique/statistique.html', {'message': 'The dataset is empty.'})
    except Exception as e:
        logger.exception('An unexpected error occurred.')
        return render(request, 'statistique/statistique.html', {'message': f'An error occurred: {str(e)}'})
