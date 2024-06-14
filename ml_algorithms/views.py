# ml_algorithms/views.py

from django.shortcuts import render
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, silhouette_score
from sklearn.model_selection import GridSearchCV, cross_val_score
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
        linear_regression_plot = None
        svr_plot = None
        random_forest_plot = None
        kmeans_plot = None
        linear_regression_r2 = None
        svr_r2 = None
        random_forest_r2 = None
        kmeans_silhouette = None
        best_algorithm = None

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

                # Linear Regression with Cross-Validation
                lr_model = LinearRegression()
                lr_cv_scores = cross_val_score(lr_model, X, y, cv=5, scoring='r2')
                linear_regression_r2 = lr_cv_scores.mean()
                lr_model.fit(X, y)
                lr_predictions = lr_model.predict(X)

                # Plotting Linear Regression
                plt.figure()
                plt.scatter(X.iloc[:, 0], y, color='blue')  # Assuming single feature for simplicity
                plt.plot(X.iloc[:, 0], lr_predictions, color='red')
                plt.title('Linear Regression')
                plt.xlabel('Feature')
                plt.ylabel('Target')

                # Save plot to a string in base64 format
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                lr_string = base64.b64encode(buf.read())
                linear_regression_plot = urllib.parse.quote(lr_string)

                # Support Vector Regression with Grid Search
                svr_model = SVR()
                svr_params = {'C': [0.1, 1, 10], 'epsilon': [0.1, 0.2, 0.5]}
                svr_grid = GridSearchCV(svr_model, svr_params, cv=5, scoring='r2')
                svr_grid.fit(X, y)
                svr_r2 = svr_grid.best_score_
                svr_model = svr_grid.best_estimator_
                svr_predictions = svr_model.predict(X)

                # Plotting Support Vector Regression
                plt.figure()
                plt.scatter(X.iloc[:, 0], y, color='blue')  # Assuming single feature for simplicity
                plt.plot(X.iloc[:, 0], svr_predictions, color='red')
                plt.title('Support Vector Regression')
                plt.xlabel('Feature')
                plt.ylabel('Target')

                # Save plot to a string in base64 format
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                svr_string = base64.b64encode(buf.read())
                svr_plot = urllib.parse.quote(svr_string)

                # Random Forest with Grid Search
                rf_model = RandomForestRegressor()
                rf_params = {'n_estimators': [10, 50, 100], 'max_depth': [None, 10, 20]}
                rf_grid = GridSearchCV(rf_model, rf_params, cv=5, scoring='r2')
                rf_grid.fit(X, y)
                random_forest_r2 = rf_grid.best_score_
                rf_model = rf_grid.best_estimator_
                rf_predictions = rf_model.predict(X)

                # Plotting Random Forest
                plt.figure()
                plt.scatter(X.iloc[:, 0], y, color='blue')  # Assuming single feature for simplicity
                plt.plot(X.iloc[:, 0], rf_predictions, color='red')
                plt.title('Random Forest Regression')
                plt.xlabel('Feature')
                plt.ylabel('Target')

                # Save plot to a string in base64 format
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                rf_string = base64.b64encode(buf.read())
                random_forest_plot = urllib.parse.quote(rf_string)

                # K-Means Clustering with Silhouette Score
                kmeans_model = KMeans(n_clusters=3)  # Assuming 3 clusters for simplicity
                kmeans_model.fit(X)
                kmeans_labels = kmeans_model.labels_
                kmeans_silhouette = silhouette_score(X, kmeans_labels)

                # Plotting K-Means Clustering
                plt.figure()
                plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=kmeans_labels, cmap='viridis')  # Assuming first two features for simplicity
                plt.title('K-Means Clustering')
                plt.xlabel('Feature 1')
                plt.ylabel('Feature 2')

                # Save plot to a string in base64 format
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                kmeans_string = base64.b64encode(buf.read())
                kmeans_plot = urllib.parse.quote(kmeans_string)

                # Determine the best algorithm based on R2 score and silhouette score
                best_algorithm = max(
                    ('Linear Regression', linear_regression_r2),
                    ('Support Vector Regression', svr_r2),
                    ('Random Forest Regression', random_forest_r2),
                    key=lambda x: x[1]
                )[0]
                
                statistics = df.describe().to_html()  # Calculate statistics

                return render(request, 'ml_algorithms/algorithms.html', {
                    'statistics': statistics,
                    'linear_regression_plot': linear_regression_plot,
                    'svr_plot': svr_plot,
                    'random_forest_plot': random_forest_plot,
                    'kmeans_plot': kmeans_plot,
                    'linear_regression_r2': linear_regression_r2,
                    'svr_r2': svr_r2,
                    'random_forest_r2': random_forest_r2,
                    'kmeans_silhouette': kmeans_silhouette,
                    'best_algorithm': best_algorithm,
                })

        return render(request, 'ml_algorithms/algorithms.html', {
            'message': 'Upload a dataset and preprocess it to apply algorithms.',
        })

    except UploadedDataset.DoesNotExist:
        return render(request, 'ml_algorithms/algorithms.html', {
            'message': 'No dataset uploaded yet.',
        })
    except Exception as e:
        return render(request, 'ml_algorithms/algorithms.html', {
            'message': f'An error occurred: {str(e)}',
        })
