import io
import base64
import urllib.parse
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import r2_score, silhouette_score
from sklearn.model_selection import GridSearchCV, cross_val_score
from django.shortcuts import render
from data_upload.models import UploadedDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import  train_test_split
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.linear_model import SGDClassifier

def preprocess_data(df):
    # Convertir les colonnes booléennes en entiers
    bool_cols = df.select_dtypes(include=['bool']).columns
    df[bool_cols] = df[bool_cols].astype(int)
    
    # Convertir les colonnes catégorielles en variables indicatrices (dummies)
    categorical_cols = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Assurez-vous que toutes les colonnes numériques sont bien de type float64 ou int64
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = df[numeric_cols].astype(float)  # Assurez-vous que toutes les colonnes numériques sont en float64
    
    return df



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
                    return render(request, 'ml_algorithms/apply_algorithms.html', {
                        'message': 'Please select a target column.',
                        'columns': columns,
                    })

                request.session['target_column'] = target_column
                return render(request, 'ml_algorithms/apply_algorithms.html', {
                    'message': 'Target column selected. Now click "Preprocess Data" to proceed.',
                    'selected_target': target_column,
                })

            if 'preprocess' in request.POST:
                target_column = request.session.get('target_column')
                if not target_column:
                    return render(request, 'ml_algorithms/apply_algorithms.html', {
                        'message': 'Target column not selected.',
                        'columns': columns,
                    })

                df = preprocess_data(df)
                X = df.drop(columns=[target_column])
                y = df[target_column]
                request.session['preprocessed'] = True
                request.session['columns'] = columns
                return render(request, 'ml_algorithms/apply_algorithms.html', {
                    'message': 'Data has been preprocessed.',
                    'preprocessed': True,
                })

            if 'algorithm' in request.POST:
                target_column = request.session.get('target_column')
                if not target_column:
                    return render(request, 'ml_algorithms/apply_algorithms.html', {
                        'message': 'Target column not selected.',
                        'columns': columns,
                    })

                if not request.session.get('preprocessed'):
                    return render(request, 'ml_algorithms/apply_algorithms.html', {
                        'message': 'Data not preprocessed.',
                        'columns': columns,
                    })

                df = preprocess_data(df)
                X = df.drop(columns=[target_column])
                y = df[target_column]

                algorithm = request.POST.get('algorithm')
                
                # Initialiser les variables
                r2 = 0
                silhouette = 0
                accuracy = 0
                report = ""
                plot = None
                
                if algorithm == 'linear_regression':
                    r2, plot = apply_linear_regression(X, y)
                elif algorithm == 'random_forest_regression':
                    r2, plot = apply_random_forest(X, y)
                elif algorithm == 'support_vector_regression':
                    r2, plot = apply_svr(X, y)
                elif algorithm == 'k_means_clustering':
                    silhouette, plot = apply_kmeans(X)
                elif algorithm == 'hierarchical_clustering':
                    silhouette, dendrogram_plot, cluster_plot = apply_hierarchical_clustering(X)
                    plot = cluster_plot
                elif algorithm == 'logistic_regression':
                    accuracy, report, plot = apply_logistic_regression(X, y)
                elif algorithm == 'sgd_classifier':
                    accuracy, plot = apply_sgd_classifier(X,y)
                else:
                    return render(request, 'ml_algorithms/apply_algorithms.html', {
                        'message': 'Invalid algorithm selected.',
                        'columns': columns,
                    })

                return render(request, 'ml_algorithms/apply_algorithms.html', {
                    'message': f'{algorithm.replace("_", " ").title()} applied.',
                    'preprocessed': True,
                    'plot': plot,
                    'r2': r2,
                    'silhouette': silhouette,
                    'accuracy': accuracy,
                    'report': report,
                })
            
            if 'all_algorithms' in request.POST:
                target_column = request.session.get('target_column')
                if not target_column:
                    return render(request, 'ml_algorithms/apply_algorithms.html', {
                        'message': 'Target column not selected.',
                        'columns': columns,
                    })

                if not request.session.get('preprocessed'):
                    return render(request, 'ml_algorithms/apply_algorithms.html', {
                        'message': 'Data not preprocessed.',
                        'columns': columns,
                    })

                df = preprocess_data(df)
                X = df.drop(columns=[target_column])
                y = df[target_column]

                linear_regression_r2, linear_regression_plot = apply_linear_regression(X, y)
                svr_r2, svr_plot = apply_svr(X, y)
                random_forest_r2, random_forest_plot = apply_random_forest(X, y)
                kmeans_silhouette, kmeans_plot = apply_kmeans(X)
                logistic_accuracy, logistic_report, logistic_plot = apply_logistic_regression(X, y)

                best_algorithm = max(
                    {
                        "Linear Regression": linear_regression_r2,
                        "Support Vector Regression": svr_r2,
                        "Random Forest Regression": random_forest_r2,
                        "Logistic Regression": logistic_accuracy
                    },
                    key=lambda k: {
                        "Linear Regression": linear_regression_r2,
                        "Support Vector Regression": svr_r2,
                        "Random Forest Regression": random_forest_r2,
                        "Logistic Regression": logistic_accuracy
                    }[k]
                )

                best_clustering_algorithm = "K-Means Clustering" if kmeans_silhouette > max(linear_regression_r2, svr_r2, random_forest_r2, logistic_accuracy) else best_algorithm

                return render(request, 'ml_algorithms/apply_algorithms.html', {
                    'message': 'All algorithms are applied.',
                    'preprocessed': True,
                    'linear_regression_plot': linear_regression_plot,
                    'svr_plot': svr_plot,
                    'random_forest_plot': random_forest_plot,
                    'kmeans_plot': kmeans_plot,
                    'logistic_plot': logistic_plot,
                    'linear_regression_r2': linear_regression_r2,
                    'svr_r2': svr_r2,
                    'random_forest_r2': random_forest_r2,
                    'kmeans_silhouette': kmeans_silhouette,
                    'logistic_accuracy': logistic_accuracy,
                    'best_algorithm': best_algorithm,
                    'best_clustering_algorithm': best_clustering_algorithm,
                })
            

        return render(request, 'ml_algorithms/apply_algorithms.html', {
            'columns': columns,
        })

    except UploadedDataset.DoesNotExist:
        return render(request, 'ml_algorithms/apply_algorithms.html', {
            'message': 'No dataset uploaded yet. Please upload data.',
        })
    except FileNotFoundError:
        return render(request, 'ml_algorithms/apply_algorithms.html', {'message': 'Dataset file not found.'})

    except Exception as e:
        return render(request, 'ml_algorithms/apply_algorithms.html', {
            'message': f'An error occurred: {str(e)}. Please upload data.',
        })

    


def plot_to_base64(plt):
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_string = base64.b64encode(buf.read()).decode('utf-8')
    return urllib.parse.quote(plot_string)


def apply_linear_regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    lr_model = LinearRegression()
    cv_scores = cross_val_score(lr_model, X_train, y_train, cv=5, scoring='r2')
    lr_model.fit(X_train, y_train)
    lr_predictions = lr_model.predict(X_test)
    linear_regression_r2 = cv_scores.mean()
    
    # Plot for multidimensional data
    if X.shape[1] > 1:
        plt.figure()
        plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual')
        plt.scatter(range(len(y_test)), lr_predictions, color='red', label='Predicted')
        plt.title('Linear Regression')
        plt.xlabel('Sample Index')
        plt.ylabel('Target')
        plt.legend()
        linear_regression_plot = plot_to_base64(plt)
    else:
        plt.figure()
        plt.scatter(X_test.iloc[:, 0], y_test, color='blue', label='Actual')
        plt.plot(X_test.iloc[:, 0], lr_predictions, color='red', label='Predicted')
        plt.title('Linear Regression')
        plt.xlabel('Feature')
        plt.ylabel('Target')
        plt.legend()
        linear_regression_plot = plot_to_base64(plt)
        
    plt.close()
    return linear_regression_r2, linear_regression_plot


def apply_svr(X, y):
    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Normalize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    svr_model = SVR()
    svr_params = {'C': [0.1, 1, 10, 100], 'epsilon': [0.1, 0.2, 0.5, 1], 'gamma': ['scale', 'auto']}
    svr_grid = GridSearchCV(svr_model, svr_params, cv=5, scoring='r2')
    svr_grid.fit(X_train_scaled, y_train)
    svr_best_model = svr_grid.best_estimator_
    
    cv_scores = cross_val_score(svr_best_model, X_train_scaled, y_train, cv=5, scoring='r2')
    svr_predictions = svr_best_model.predict(X_test_scaled)
    svr_r2 = cv_scores.mean()
    
    # Plot for multidimensional data
    if X.shape[1] > 1:
        plt.figure()
        plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual')
        plt.scatter(range(len(y_test)), svr_predictions, color='red', label='Predicted')
        plt.title('Support Vector Regression')
        plt.xlabel('Sample Index')
        plt.ylabel('Target')
        plt.legend()
        svr_plot = plot_to_base64(plt)
    else:
        plt.figure()
        plt.scatter(X_test.iloc[:, 0], y_test, color='blue', label='Actual')
        plt.plot(X_test.iloc[:, 0], svr_predictions, color='red', label='Predicted')
        plt.title('Support Vector Regression')
        plt.xlabel('Feature')
        plt.ylabel('Target')
        plt.legend()
        svr_plot = plot_to_base64(plt)
        
    plt.close()
    return svr_r2, svr_plot




def apply_random_forest(X, y):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create the model
    rf_model = RandomForestRegressor(random_state=42)
    
    # Cross-validation scores
    cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='r2')
    
    # Fit the model
    rf_model.fit(X_train, y_train)
    
    # Predictions
    rf_predictions = rf_model.predict(X_test)
    
    # Mean R^2 score from cross-validation
    random_forest_r2 = cv_scores.mean()
    
    # Plotting
    if X.shape[1] > 1:
        plt.figure()
        plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual')
        plt.scatter(range(len(y_test)), rf_predictions, color='red', label='Predicted')
        plt.title('Random Forest Regression')
        plt.xlabel('Sample Index')
        plt.ylabel('Target')
        plt.legend()
        random_forest_plot = plot_to_base64(plt)
    else:
        plt.figure()
        plt.scatter(X_test.iloc[:, 0], y_test, color='blue', label='Actual')
        plt.plot(X_test.iloc[:, 0], rf_predictions, color='red', label='Predicted')
        plt.title('Random Forest Regression')
        plt.xlabel('Feature')
        plt.ylabel('Target')
        plt.legend()
        random_forest_plot = plot_to_base64(plt)
        
    plt.close()
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





import numpy as np

def gradient_descent(X, y, learning_rate=0.01, epochs=1000):
    m, n = X.shape
    theta = np.zeros(n)  
    
    cost_history = []
    
    for epoch in range(epochs):
        h = np.dot(X, theta)
        
        error = h - y
        
        cost = np.sum(error ** 2) / (2 * m)
        cost_history.append(cost)
        
        gradient = np.dot(X.T, error) / m
        
        theta -= learning_rate * gradient
        
    return theta, cost_history

def linear_regression_gradient_descent(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train = X_train.astype(float)
    X_test = X_test.astype(float)
    y_train = y_train.astype(float)
    y_test = y_test.astype(float)
    
    X_train = np.c_[np.ones((len(X_train), 1)), X_train]
    
    learning_rate = 0.01
    epochs = 1000
    
    theta, cost_history = gradient_descent(X_train, y_train, learning_rate, epochs)
    
    X_test = np.c_[np.ones((len(X_test), 1)), X_test]
    predictions = np.dot(X_test, theta)
  
    plt.figure()
   
    plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual')
    plt.scatter(range(len(y_test)), predictions, color='red', label='Predicted')
    plt.plot(range(len(y_test)), predictions, color='red', linestyle='--', label='Linear Regression Line')
    
    plt.title('Linear Regression with Gradient Descent')
    plt.xlabel('Sample Index')
    plt.ylabel('Target')
    plt.legend()
    gradient_descent_plot = plot_to_base64(plt)
    
    plt.close()
    
    return theta, gradient_descent_plot


def apply_hierarchical_clustering(X):
    hier_model = AgglomerativeClustering(n_clusters=3)
    hier_labels = hier_model.fit_predict(X)

    hier_silhouette = silhouette_score(X, hier_labels)

    plt.figure(figsize=(10, 7))
    plt.title("Hierarchical Clustering Dendrogram")
    Z = linkage(X, 'ward')
    dendrogram(Z)
    hier_dendrogram_plot = plot_to_base64(plt)
    
    plt.close()

    plt.figure()
    if X.shape[1] > 1:
        plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=hier_labels, cmap='viridis')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
    else:
        plt.scatter(X.iloc[:, 0], [0] * len(X), c=hier_labels, cmap='viridis')
        plt.xlabel('Feature')
        plt.ylabel('Cluster')
    plt.title('Hierarchical Clustering')
    hier_cluster_plot = plot_to_base64(plt)

    plt.close()
    
    return hier_silhouette, hier_dendrogram_plot, hier_cluster_plot



from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

def apply_logistic_regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_train)
    y_pred = lr_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(10, 7))
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    confusion_matrix_plot = plot_to_base64(plt)
    
    plt.close()
    return accuracy, report, confusion_matrix_plot


def apply_sgd_classifier(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normaliser les caractéristiques
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Entraîner le classificateur SGD
    clf = SGDClassifier(alpha=0.01, max_iter=1000)
    clf.fit(X_train, y_train)
    
    # Prédictions sur l'ensemble de testES
    y_pred = clf.predict(X_test)
    
    # Calculer la précision
    accuracy = accuracy_score(y_test, y_pred)
    
    # Extraire les coefficients
    theta0, theta1 = clf.coef_[0]
    theta2 = clf.intercept_[0]
    
    # Visualisation
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', marker='x')
    plt.xlabel("Caractéristique 1 (Normalisée)")
    plt.ylabel("Caractéristique 2 (Normalisée)")

    # Tracer la frontière de décision
    x_values = np.linspace(X_train[:, 0].min(), X_train[:, 0].max(), 100)
    decision_boundary = -(theta0 * x_values + theta2) / theta1
    plt.plot(x_values, decision_boundary, label=r'Frontière de décision : $G(X) = a_1 \times x_1 + a_2 \times x_2 + b = 0$', color='magenta')
    
    plt.legend()
    plt.title("Classification des emails avec SGDclassifier")
    
    # Convertir le graphique en base64
    sgd_plot = plot_to_base64(plt)

    plt.close()
    
    return accuracy, sgd_plot

    
   
    
    

    
