from knn import knn
import numpy as np
import matplotlib.pyplot as plt
import info_gain as ig
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from knn import knn
from feature_subsampling import feature_subsampling

def main():
    # Generate a random dataset
    n_features = 100
    X, y = make_classification(n_features = n_features, n_redundant=0, n_informative=10,
                               n_clusters_per_class=1, n_samples=200, random_state=42, n_classes=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Non-Bagging
    knn_model = knn(n_neighbors=5, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    knn_model.fit()
    knn_model_accuracy = knn_model.evaluate()

    # Bagging KNN
    bagged_knn = feature_subsampling(n_features,20,X_train, X_test, y_train, y_test)
    bagged_knn_accuracy = bagged_knn.bagged_predictions()

    #Info Gain
    ig_dict = ig.make_ig_dict(X_train,y_train)
    ig_fl = ig.best_feature_list(ig_dict) #find the best features
    #split train and test with these features
    subset_X_train = X_train[:,ig_fl]
    subset_X_test = X_test[:,ig_fl]
    #create a knn model with these features
    fs_knn = knn(n_neighbors=5, X_train=subset_X_train, X_test=subset_X_test, y_train=y_train, y_test=y_test)
    fs_knn.fit()
    #get accuracy
    fs_knn_accuracy = fs_knn.evaluate()

    # Comparing results
    print(f"\nSingle KNN Accuracy: {knn_model_accuracy:.2f}")
    print(f"Bagged KNN Accuracy: {bagged_knn_accuracy:.2f}")
    print(f"Feature-Selected KNN Accuracy: {fs_knn_accuracy:.2f}")



if __name__ == "__main__":
    main()

