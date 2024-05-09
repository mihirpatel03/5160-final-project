# input dataset
# input number of dimensions we want
# output pca dataset

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

def knn_pca(n_features, X_train,X_test,y_train,y_test):

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(X_train.shape[1])
    # train PCA
    pca = PCA(n_components=int(np.sqrt(n_features)))
    X_train_pca = pca.fit_transform(X_train_scaled)
    print(X_train_pca.shape[1])
    # apply to test
    X_test_pca = pca.transform(X_test_scaled)

    return X_train_pca,X_test_pca