from knn import knn
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main():

    # Generate a random dataset
    X, y = make_classification(n_samples=200, n_features=2, n_redundant=0,
                            n_clusters_per_class=1, n_classes=4, flip_y=0.1,
                            class_sep=1, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    k_pred_arr = []
    for i in range(2):
        plotter = knn(i+1)
        plotter.fit(X_train, y_train)
        plotter.predict()
        plotter.plot_decision_boundaries()
        input_data,predicted_label,confusion_score = plotter.get_training_data_predictions()
        actual_label = input_data[:,-1]
        k_pred_arr.append(get_pred_accuracy(actual_label,predicted_label))
        print(i)

    print(k_pred_arr)


def get_pred_accuracy(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i]==predicted[i]:
            correct+=1
    return correct/len(predicted)

def calculate_epsilon():
    pass


def calculate_beta():
    pass



if __name__ == "__main__":
    main()

