from django.shortcuts import render
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import io
import urllib, base64
from data_upload.models import UploadedDataset

def apply_algorithm(request):
    try:
        dataset = UploadedDataset.objects.latest('uploaded_at')
        file_path = dataset.file_path
        df = pd.read_csv(file_path)

        statistics = None
        plot_uri = None
        r2 = None
        algorithm_name = None

        if request.method == 'POST':
            # Preprocess the data
            if 'preprocess' in request.POST:
                # Convert boolean columns to integers
                bool_cols = df.select_dtypes(include=['bool']).columns
                df[bool_cols] = df[bool_cols].astype(int)

                # One-hot encode categorical columns
                categorical_cols = df.select_dtypes(include=['object']).columns
                df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

                # Assume target column is the last one
                X = df.iloc[:, :-1]
                y = df.iloc[:, -1]
                
                # Save the processed data for further use
                request.session['X'] = X.to_json()
                request.session['y'] = y.to_json()
                
                message = 'Data preprocessed successfully.'
                statistics = df.describe().to_html()  # Calculate statistics
                return render(request, 'ml_algorithms/algorithms.html', {
                    'message': message,
                    'data_ready': True,
                    'statistics': statistics,
                })

            # Apply Linear Regression
            elif 'linear_regression' in request.POST:
                X = pd.read_json(request.session['X'])
                y = pd.read_json(request.session['y'], typ='series')
                model = LinearRegression()
                model.fit(X, y)
                predictions = model.predict(X)
                r2 = r2_score(y, predictions)

                # Plotting
                plt.figure()
                plt.scatter(X.iloc[:, 0], y, color='blue')  # Assuming single feature for simplicity
                plt.plot(X.iloc[:, 0], predictions, color='red')
                plt.title('Linear Regression')
                plt.xlabel('Feature')
                plt.ylabel('Target')

                # Save plot to a string in base64 format
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                string = base64.b64encode(buf.read())
                plot_uri = urllib.parse.quote(string)
                algorithm_name = "Linear Regression"

            # Apply Support Vector Regression
            elif 'svr' in request.POST:
                X = pd.read_json(request.session['X'])
                y = pd.read_json(request.session['y'], typ='series')
                model = SVR()
                model.fit(X, y)
                predictions = model.predict(X)
                r2 = r2_score(y, predictions)

                # Plotting
                plt.figure()
                plt.scatter(X.iloc[:, 0], y, color='blue')  # Assuming single feature for simplicity
                plt.plot(X.iloc[:, 0], predictions, color='red')
                plt.title('Support Vector Regression')
                plt.xlabel('Feature')
                plt.ylabel('Target')

                # Save plot to a string in base64 format
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                string = base64.b64encode(buf.read())
                plot_uri = urllib.parse.quote(string)
                algorithm_name = "Support Vector Regression"

        return render(request, 'ml_algorithms/algorithms.html', {
            'message': 'Upload a dataset and preprocess it to apply algorithms.',
            'data_ready': False,
            'statistics': statistics,
            'plot': plot_uri,
            'r2': r2,
            'algorithm': algorithm_name
        })

    except UploadedDataset.DoesNotExist:
        return render(request, 'ml_algorithms/algorithms.html', {
            'message': 'No dataset uploaded yet.',
            'data_ready': False,
        })
    except Exception as e:
        return render(request, 'ml_algorithms/algorithms.html', {
            'message': f'An error occurred: {str(e)}',
            'data_ready': False,
        })
