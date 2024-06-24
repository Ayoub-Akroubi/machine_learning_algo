from django.shortcuts import render
import pandas as pd
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO, StringIO
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
        statistics = df.describe(include='all').to_html()

        # Get data info
        buffer = StringIO()
        df.info(buf=buffer)
        info = buffer.getvalue()

        # Get first few rows of the data
        head = df.head().to_html()

        # Get data shape
        shape = df.shape

        # Convert boolean columns to numeric (0 and 1)
        df = df.applymap(lambda x: 1 if x is True else 0 if x is False else x)

        # Convert categorical columns to numeric using one-hot encoding
        df = pd.get_dummies(df, drop_first=True)

        # Calculate correlation matrix
        correlation_matrix = df.corr()

        # Generate heatmap
        plt.figure(figsize=(10, 8))
        heatmap = sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')

        # Save it to a bytes buffer
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        plt.close()
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()

        # Encode image to base64 string
        heatmap_base64 = base64.b64encode(image_png).decode('utf-8')

        return render(request, 'statistique/statistique.html', {
            'statistics': statistics,
            'info': info,
            'head': head,
            'shape': shape,
            'heatmap_base64': heatmap_base64,
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
