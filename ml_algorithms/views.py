import io
import base64
import urllib.parse
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, silhouette_score
from sklearn.model_selection import GridSearchCV, cross_val_score
from django.shortcuts import render
from data_upload.models import UploadedDataset


def preprocess_data(df):
    bool_cols = df.select_dtypes(include=['bool']).columns
    df[bool_cols] = df[bool_cols].astype(int)
    categorical_cols = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df


def plot_to_base64(plt):
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_string = base64.b64encode(buf.read()).decode('utf-8')
    return urllib.parse.quote(plot_string)


def apply_linear_regression(X, y):
    lr_model = LinearRegression()
    cv_scores = cross_val_score(lr_model, X, y, cv=5, scoring='r2')
    lr_model.fit(X, y)
    lr_predictions = lr_model.predict(X)
    linear_regression_r2 = cv_scores.mean()
    plt.figure()
    plt.scatter(X.iloc[:, 0], y, color='blue')
    plt.plot(X.iloc[:, 0], lr_predictions, color='red')
    plt.title('Linear Regression')
    plt.xlabel('Feature')
    plt.ylabel('Target')
    linear_regression_plot = plot_to_base64(plt)
    return linear_regression_r2, linear_regression_plot


def apply_svr(X, y):
    svr_model = SVR()
    svr_params = {'C': [0.1, 1, 10], 'epsilon': [0.1, 0.2, 0.5]}
    svr_grid = GridSearchCV(svr_model, svr_params, cv=5, scoring='r2')
    svr_grid.fit(X, y)
    svr_best_model = svr_grid.best_estimator_
    cv_scores = cross_val_score(svr_best_model, X, y, cv=5, scoring='r2')
    svr_predictions = svr_best_model.predict(X)
    svr_r2 = cv_scores.mean()
    plt.figure()
    plt.scatter(X.iloc[:, 0], y, color='blue')
    plt.plot(X.iloc[:, 0], svr_predictions, color='red')
    plt.title('Support Vector Regression')
    plt.xlabel('Feature')
    plt.ylabel('Target')
    svr_plot = plot_to_base64(plt)
    return svr_r2, svr_plot


def apply_random_forest(X, y):
    rf_model = RandomForestRegressor()
    rf_params = {'n_estimators': [10, 50, 100], 'max_depth': [None, 10, 20]}
    rf_grid = GridSearchCV(rf_model, rf_params, cv=5, scoring='r2')
    rf_grid.fit(X, y)
    rf_best_model = rf_grid.best_estimator_
    cv_scores = cross_val_score(rf_best_model, X, y, cv=5, scoring='r2')
    rf_predictions = rf_best_model.predict(X)
    random_forest_r2 = cv_scores.mean()
    plt.figure()
    plt.scatter(X.iloc[:, 0], y, color='blue')
    plt.plot(X.iloc[:, 0], rf_predictions, color='red')
    plt.title('Random Forest Regression')
    plt.xlabel('Feature')
    plt.ylabel('Target')
    random_forest_plot = plot_to_base64(plt)
    return random_forest_r2, random_forest_plot


def apply_kmeans(X):
    kmeans_model = KMeans(n_clusters=3)
    kmeans_model.fit(X)
    kmeans_labels = kmeans_model.labels_
    kmeans_silhouette = silhouette_score(X, kmeans_labels)
    plt.figure()
    if X.shape[1] > 1:
        plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=kmeans_labels, cmap='viridis')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
    else:
        plt.scatter(X.iloc[:, 0], [0] * len(X), c=kmeans_labels, cmap='viridis')
        plt.xlabel('Feature')
        plt.ylabel('Cluster')
    plt.title('K-Means Clustering')
    kmeans_plot = plot_to_base64(plt)
    return kmeans_silhouette, kmeans_plot


def apply_algorithm(request):
    try:
        dataset = UploadedDataset.objects.latest('uploaded_at')
        file_path = dataset.file_path
        df = pd.read_csv(file_path)
        columns = list(df.columns)

        if request.method == 'POST':
            if 'select_target' in request.POST:
                target_column = request.POST.get('target_column')
                if not target_column:
                    return render(request, 'ml_algorithms/algorithms.html', {
                        'message': 'Please select a target column.',
                        'columns': columns,
                    })

                request.session['target_column'] = target_column
                return render(request, 'ml_algorithms/algorithms.html', {
                    'columns': columns,
                    'selected_target': target_column,
                    'message': 'Target column selected. Now click "Preprocess Data" to proceed.'
                })

            if 'preprocess' in request.POST:
                target_column = request.session.get('target_column')
                if not target_column:
                    return render(request, 'ml_algorithms/algorithms.html', {
                        'message': 'Target column not selected.',
                        'columns': columns,
                    })

                df = preprocess_data(df)
                X = df.drop(columns=[target_column])
                y = df[target_column]

                linear_regression_r2, linear_regression_plot = apply_linear_regression(X, y)
                svr_r2, svr_plot = apply_svr(X, y)
                random_forest_r2, random_forest_plot = apply_random_forest(X, y)
                kmeans_silhouette, kmeans_plot = apply_kmeans(X)

                best_algorithm = max(
                    {
                        "Linear Regression": linear_regression_r2,
                        "Support Vector Regression": svr_r2,
                        "Random Forest Regression": random_forest_r2
                    },
                    key=lambda k: {
                        "Linear Regression": linear_regression_r2,
                        "Support Vector Regression": svr_r2,
                        "Random Forest Regression": random_forest_r2
                    }[k]
                )

                best_clustering_algorithm = "K-Means Clustering" if kmeans_silhouette > max(linear_regression_r2, svr_r2, random_forest_r2) else best_algorithm

                statistics = df.describe().to_html()

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
                    'best_clustering_algorithm': best_clustering_algorithm,
                })

        return render(request, 'ml_algorithms/algorithms.html', {
            'columns': columns,
        })

    except UploadedDataset.DoesNotExist:
        return render(request, 'ml_algorithms/algorithms.html', {
            'message': 'No dataset uploaded yet.',
        })
    except Exception as e:
        return render(request, 'ml_algorithms/algorithms.html', {
            'message': f'An error occurred: {str(e)}',
        })
