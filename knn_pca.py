# input dataset
# input number of dimensions we want
# output pca dataset

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def knn_pca(n_features, X_train,X_test,y_train,y_test):

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # train PCA
    n_components = 2  # You can choose the number of components based on your requirements
    pca = PCA(n_components=n_features)
    X_train_pca = pca.fit_transform(X_train_scaled)

    # apply to test
    X_test_pca = pca.transform(X_test_scaled)

    return X_train_pca,X_test_pca