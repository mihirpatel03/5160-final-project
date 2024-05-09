from knn import knn
import numpy as np
import matplotlib.pyplot as plt
import info_gain as ig
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from knn import knn
from bagging import BaggedKNN

def main():
    # Generate a random dataset
    X, y = make_classification(n_features=100, n_redundant=0, n_informative=10,
                               n_clusters_per_class=1, n_samples=200, random_state=42, n_classes=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Non-Bagging KNN
    print("Evaluating single KNN model:")
    knn_model = knn(n_neighbors=5, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    knn_model.fit()
    knn_model_accuracy = knn_model.evaluate()

    # Bagging KNN
    print("\nEvaluating Bagged KNN model:")
    bagged_knn = BaggedKNN(knn, n_estimators=10, n_neighbors=5, X_train=X_train, y_train=y_train)
    bagged_knn.fit()
    bagged_knn_accuracy = bagged_knn.evaluate(X_test, y_test)


    #PCA

    #Info Gain
    ig_dict = ig.make_ig_dict(X_train,y_train)
    ig_fl = ig.best_feature_list(ig_dict)

    subset_X_train = X_train[:,ig_fl]
    subset_X_test = X_test[:,ig_fl]
    print("Evaluating feature-selected kNN model:")
    fs_knn = knn(n_neighbors=5, X_train=subset_X_train, X_test=subset_X_test, y_train=y_train, y_test=y_test)
    fs_knn.fit()
    fs_knn_accuracy = fs_knn.evaluate()


    # Comparing results
    print(f"\nSingle KNN Accuracy: {knn_model_accuracy:.2f}")
    print(f"Bagged KNN Accuracy: {bagged_knn_accuracy:.2f}")
    print(f"Feature-Selected KNN Accuracy: {fs_knn_accuracy:.2f}")





if __name__ == "__main__":
    main()

