from knn import knn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np

class feature_subsampling:

    def __init__(self, n_features, n_bags, X_train, X_test, y_train, y_test):
        self.n_features = n_features #number of features
        self.n_features_subsetted = round(np.sqrt(self.n_features)) #num features in each abg
        self.n_bags = n_bags #number of bags we want to do 

        #data
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        #subsets that we will assign during each bag
        self.X_train_subset = 0
        self.X_test_subset = 0
        #number neighbors, vector of predictions
        self.num_of_neighbors = 5
        self.vector = 0

    def bagged_predictions(self):
        pred_array = np.zeros((self.n_bags,self.X_test.shape[0]))
        for i in range(self.n_bags):
            random_indices = np.random.choice(self.X_train.shape[1], size=self.n_features_subsetted, replace=False)
            self.X_train_subset = self.X_train[:, random_indices]
            self.X_test_subset = self.X_test[:, random_indices]
            model = knn(self.num_of_neighbors,  self.X_train_subset, self.X_test_subset, self.y_train, self.y_test)
            model.fit()
            vector = model.predict(self.X_test_subset)
            pred_array[i] = vector
        bagged_predictions = np.average(pred_array, axis = 0).round(decimals = 0)
        bagged_accuracy =  np.sum(self.y_test == bagged_predictions)/len(bagged_predictions)
        return bagged_accuracy