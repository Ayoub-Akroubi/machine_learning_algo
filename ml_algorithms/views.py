# ml_algorithms/views.py


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
    lr_model.fit(X, y)
    lr_predictions = lr_model.predict(X)
    linear_regression_r2 = r2_score(y, lr_predictions)
    plt.figure()
    plt.scatter(X.iloc[:, 0], y, color='blue')
    plt.plot(X.iloc[:, 0], lr_predictions, color='red')
    plt.title('Régression Linéaire')
    plt.xlabel('Caractéristique')
    plt.ylabel('Cible')
    linear_regression_plot = plot_to_base64(plt)
    return linear_regression_r2, linear_regression_plot


def apply_svr(X, y):
    svr_model = SVR()
    svr_model.fit(X, y)
    svr_predictions = svr_model.predict(X)
    svr_r2 = r2_score(y, svr_predictions)
    plt.figure()
    plt.scatter(X.iloc[:, 0], y, color='blue')
    plt.plot(X.iloc[:, 0], svr_predictions, color='red')
    plt.title('Régression par Vecteur de Support')
    plt.xlabel('Caractéristique')
    plt.ylabel('Cible')
    svr_plot = plot_to_base64(plt)
    return svr_r2, svr_plot


def apply_random_forest(X, y):
    rf_model = RandomForestRegressor()
    rf_model.fit(X, y)
    rf_predictions = rf_model.predict(X)
    random_forest_r2 = r2_score(y, rf_predictions)
    plt.figure()
    plt.scatter(X.iloc[:, 0], y, color='blue')
    plt.plot(X.iloc[:, 0], rf_predictions, color='red')
    plt.title('Régression par Forêt Aléatoire')
    plt.xlabel('Caractéristique')
    plt.ylabel('Cible')
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
        plt.xlabel('Caractéristique 1')
        plt.ylabel('Caractéristique 2')
    else:
        plt.scatter(X.iloc[:, 0], [0] * len(X), c=kmeans_labels, cmap='viridis')
        plt.xlabel('Caractéristique')
        plt.ylabel('Cluster')
    plt.title('Clustering K-Means')
    kmeans_plot = plot_to_base64(plt)
    return kmeans_silhouette, kmeans_plot


def apply_algorithm(request):
    try:
        dataset = UploadedDataset.objects.latest('uploaded_at')
        file_path = dataset.file_path
        df = pd.read_csv(file_path)

        if request.method == 'POST' and 'preprocess' in request.POST:
            df = preprocess_data(df)
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]

            linear_regression_r2, linear_regression_plot = apply_linear_regression(X, y)
            svr_r2, svr_plot = apply_svr(X, y)
            random_forest_r2, random_forest_plot = apply_random_forest(X, y)
            kmeans_silhouette, kmeans_plot = apply_kmeans(X)

            best_algorithm = max(
                {
                    "Régression Linéaire": linear_regression_r2,
                    "Régression par Vecteur de Support": svr_r2,
                    "Régression par Forêt Aléatoire": random_forest_r2
                },
                key=lambda k: {
                    "Régression Linéaire": linear_regression_r2,
                    "Régression par Vecteur de Support": svr_r2,
                    "Régression par Forêt Aléatoire": random_forest_r2
                }[k]
            )

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
            })

        return render(request, 'ml_algorithms/algorithms.html', {
            'message': 'Téléchargez un jeu de données et pré-traitez-le pour appliquer les algorithmes.',
        })

    except UploadedDataset.DoesNotExist:
        return render(request, 'ml_algorithms/algorithms.html', {
            'message': 'Aucun jeu de données n\'a encore été téléchargé.',
        })
    except Exception as e:
        return render(request, 'ml_algorithms/algorithms.html', {
            'message': f'Une erreur s\'est produite : {str(e)}',
        })
