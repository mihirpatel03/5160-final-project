from knn import knn
import numpy as np
import matplotlib.pyplot as plt
import info_gain as ig
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from knn import knn
from feature_subsampling import feature_subsampling
import knn_pca as kp

def main():
    # Generate a random dataset
    n_features = 1000
    n_neighbors = 5
    X, y = make_classification(n_features = n_features, n_redundant=0, n_informative=600,
                               n_clusters_per_class=1, n_samples=200, random_state=42, n_classes=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Non-Bagging
    knn_model = knn(n_neighbors=n_neighbors, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    knn_model.fit()
    knn_model_accuracy = knn_model.evaluate()

    # Bagging KNN
    bagged_knn = feature_subsampling(n_features,500,X_train, X_test, y_train, y_test)
    bagged_knn_accuracy = bagged_knn.bagged_predictions()

    #Info Gain
    ig_dict = ig.make_ig_dict(X_train,y_train)
    ig_fl = ig.best_feature_list(ig_dict) #find the best features
    #split train and test with these features
    subset_X_train = X_train[:,ig_fl]
    subset_X_test = X_test[:,ig_fl]
    #create a knn model with these features
    fs_knn = knn(n_neighbors=n_neighbors, X_train=subset_X_train, X_test=subset_X_test, y_train=y_train, y_test=y_test)
    fs_knn.fit()
    #get accuracy
    fs_knn_accuracy = fs_knn.evaluate()

    # pca
    X_train_pca,X_test_pca = kp.knn_pca(int(np.sqrt(n_features)),X_train, X_test, y_train, y_test)
    knn_pca = knn(n_neighbors=n_neighbors, X_train=X_train_pca, X_test=X_test_pca, y_train=y_train, y_test=y_test)
    knn_pca.fit()
    knn_pca_accuracy = knn_model.evaluate()

    # Comparing results
    print(f"\nSingle KNN Accuracy: {knn_model_accuracy:.2f}")
    print(f"Bagged KNN Accuracy: {bagged_knn_accuracy:.2f}")
    print(f"Feature-Selected KNN Accuracy: {fs_knn_accuracy:.2f}")
    print(f"PCA KNN Accuracy: {knn_pca_accuracy:.2f}")

    return knn_model_accuracy, bagged_knn_accuracy, fs_knn_accuracy, knn_pca_accuracy


def plot(epochs):
    epochs_list = []
    knn_model_accuracy = []
    bagged_knn_accuracy = []
    fs_knn_accuracy = []
    knn_pca_accuracy = []
    for i in range(epochs):
        knn_model_accuracy_output, bagged_knn_accuracy_output, fs_knn_accuracy_output, knn_pca_accuracy_output = main()
        knn_model_accuracy.append(knn_model_accuracy_output)
        bagged_knn_accuracy.append(bagged_knn_accuracy_output)
        fs_knn_accuracy.append(fs_knn_accuracy_output)
        knn_pca_accuracy.append(knn_pca_accuracy_output)
        epochs_list.append(i)

    plt.figure(figsize=(10, 5))  # Set the figure size
    plt.plot(epochs_list, knn_model_accuracy, linestyle='--', color='black', label='KNN Model Accuracy')
    plt.plot(epochs_list, bagged_knn_accuracy, linestyle='-', color='blue', label='Bagged KNN Accuracy')
    plt.plot(epochs_list, fs_knn_accuracy, linestyle='-', color='blue', label='Information Gain KNN Accuracy')
    plt.plot(epochs_list, knn_pca_accuracy, linestyle='-', color='blue', label='PCA-Based Dimension Reduction KNN Accuracy')
    plt.title('Accuracy Comparison: KNN vs. Bagged KNN')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.xticks(epochs_list)

    plt.show()

if __name__ == "__main__":
    plot(10)
    main()
