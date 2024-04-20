import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier,RadiusNeighborsClassifier

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



class knn:
    def __init__(self, n_neighbors):
        """
        Initialize the KNN classifier with n_neighbors.
        """
        self.n_neighbors = n_neighbors
        self.classifier = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        self.cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
        self.cmap_bold = ['darkorange', 'c', 'darkblue']

        # radius for confusion score
        self.radius_classifier = RadiusNeighborsClassifier(radius=0.75)  # Classifier to compute confusion score


    def fit(self, X_train, y_train):
        """
        Fit the KNN classifier on the training data.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.classifier.fit(X_train, y_train)
        self.train_predictions = self.classifier.predict(X_train)  # Predictions on the training data

        self.radius_classifier.fit(X_train, y_train)  # Fit the radius classifier

        # Dynamically create color maps based on the number of unique classes
        unique_classes = np.unique(y_train)
        self.num_classes = len(unique_classes)
        self.cmap_light = ListedColormap(plt.cm.viridis(np.linspace(0, 1, self.num_classes)))
        self.cmap_bold = plt.cm.viridis(np.linspace(0, 1, self.num_classes))


    def predict(self):
        """
        Predict using the trained classifier over a mesh grid.
        """

        # Mesh 2d space into grid to generate X_test and y_test

        h = 0.02  # step size in the mesh
        x_min, x_max = self.X_train[:, 0].min() - 1, self.X_train[:, 0].max() + 1
        y_min, y_max = self.X_train[:, 1].min() - 1, self.X_train[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        X_test = np.c_[xx.ravel(), yy.ravel()]

        # Predict the function value over the grid
        y_test = self.classifier.predict(X_test)
        self.xx, self.yy = xx, yy
        self.y_test = y_test.reshape(xx.shape)

    def confusion_score(self):
        """
        Calculate the confusion score for each point as the number of unique classes divided by
        the total number of neighbors within a 1-unit radius.
        """
        # Retrieve the indices of neighbors for each training point within the specified radius (1 unit here).
        neighbors = self.radius_classifier.radius_neighbors(self.X_train, return_distance=False)

        # Initialize an empty list to store the confusion scores for each training point.
        confusion_scores = []

        # Iterate through each set of neighbor indices (neigh_indices) for each training point.
        for idx, neigh_indices in enumerate(neighbors):
            # Check if there are any neighbors within the radius.
            if neigh_indices.size > 0:
                # Retrieve the class labels of the neighbors.
                classes = self.y_train[neigh_indices]
                # Count the number of unique class labels among the neighbors.
                unique_classes_count = np.unique(classes).size
                # Get the total number of neighbors.
                total_neighbors = classes.size
                # Calculate the confusion score: number of unique classes divided by total neighbors.
                score = unique_classes_count / total_neighbors
            else:
                # If no neighbors are found within the radius, assign a confusion score of 0.
                score = 0
            # Append the calculated score to the list of confusion scores.
            confusion_scores.append(score)

        # Convert the list of confusion scores to a numpy array and return it.
        return np.array(confusion_scores)


    def store_predictions_as_vec(self):
        """
        Return an array that includes each training point's coordinates rounded to two decimals,
        true label, predicted label as integers, and confusion score rounded to two decimals.
        """
        # Calculate the confusion score using the custom method
        confusion_scores = self.confusion_score()

        # Stack the training data points, true labels, predicted labels, and confusion scores
        data = np.column_stack((self.X_train, self.y_train, self.train_predictions, confusion_scores))

        # Round coordinates and confusion scores to two decimal places
        data[:, [0, 1, -1]] = np.round(data[:, [0, 1, -1]], 2)

        # Convert true and predicted labels to integers
        data[:, [2, 3]] = data[:, [2, 3]].astype(int)

        input_data = data[:, :3]
        predicted_label = data[:,3]
        confusion_score = data[:,4]


        return input_data,predicted_label,confusion_score


    def plot_decision_boundaries(self):
        """
        Plot the decision boundaries and training points.
        """
        plt.figure(figsize=(8, 6))
        plt.contourf(self.xx, self.yy, self.y_test, cmap=self.cmap_light)

        # Plot the training points
        sns.scatterplot(x=self.X_train[:, 0], y=self.X_train[:, 1],
                        hue=self.y_train, palette=list(self.cmap_bold),
                        alpha=1.0, edgecolor="black", legend=None)  # Removed legend for clarity
        plt.xlim(self.xx.min(), self.xx.max())
        plt.ylim(self.yy.min(), self.yy.max())
        plt.title(f"{self.num_classes}-Class classification (k = {self.n_neighbors})")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.show()