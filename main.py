import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from knn import knn


# Generate a random dataset
X, y = make_classification(n_samples=200, n_features=2, n_redundant=0,
                           n_clusters_per_class=1, n_classes=4, flip_y=0.1,
                           class_sep=1, random_state=42)

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Example usage:
# Assuming X_train and y_train are defined elsewhere
plotter = knn(n_neighbors=10)
plotter.fit(X_train, y_train)
plotter.predict()
plotter.plot_decision_boundaries()
input_data,predicted_label,confusion_score = plotter.store_predictions_as_vec()
print(input_data)

