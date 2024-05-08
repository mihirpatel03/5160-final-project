from knn import knn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np

class feature_subsampling:

    def __init__(self, n_features, n_bags):
        self.n_features = n_features
        self.n_features_subsetted = round(np.sqrt(self.n_features))
        self.n_bags = n_bags
        self.X_train = 0
        self.X_test = 0
        self.y_train = 0
        self.y_test = 0
        self.X_train_subset = 0
        self.X_test_subset = 0
        self.num_of_neighbors = 5
        self.vector = 0

    def create_data(self):

        # Generate a random dataset
        X, y = make_classification(n_features=self.n_features, n_redundant=0, n_informative=2,
                                n_clusters_per_class=1, n_samples=1000, random_state=42, n_classes=2)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # def subset(self):
    #     random_indices = np.random.choice(self.X_train.shape[1], size=self.n_features_subsetted, replace=False)
    #     self.X_train_subset = self.X_train[:, random_indices]
    #     self.X_test_subset = self.X_test[:, random_indices]

    # def create_predictions(self):
    #     model = knn(self.num_of_neighbors,  self.X_train_subset, self.X_test_subset, self.y_train, self.y_test)
    #     model.fit()
    #     vector = model.predict()

    def bagged_predictions(self):
        pred_array = np.zeros((self.n_bags,self.X_test.shape[0]))
        for i in range(self.n_bags):
            random_indices = np.random.choice(self.X_train.shape[1], size=self.n_features_subsetted, replace=False)
            self.X_train_subset = self.X_train[:, random_indices]
            self.X_test_subset = self.X_test[:, random_indices]
            model = knn(self.num_of_neighbors,  self.X_train_subset, self.X_test_subset, self.y_train, self.y_test)
            model.fit()
            vector = model.predict()
            pred_array[i] = vector
        bagged_predictions = np.average(pred_array, axis = 0).round(decimals = 0)
        print(f"accuracy = {np.sum(self.y_test == bagged_predictions)/len(bagged_predictions)}")

model = feature_subsampling(50,20)
model.create_data()
model.bagged_predictions()