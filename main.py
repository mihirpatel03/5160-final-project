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

    ##this one creates the plot of number of informative features out of 50 vs accuracy
    #plot_inf_features_to_accuracy(num_features = 50)

    ##this one creates the plot of bag size out of 50 vs accuracy
    #plot_bag_size_to_accuracy(num_features = 50)

    #this one plots the accuracy of each different method as number of features increases
    fix_num_informative = True #set to true if you want to fix the number of informative features at 20, else it will increase at 1/2 of num_features
    plot_numfeature_to_accuracy(fix_num_informative)




def plot_inf_features_to_accuracy(num_features):
    features_list = []
    knn_model_accuracy_list = []
    bagged_knn_accuracy_list = []

    for i in range(1,num_features):
        print(i)
        n_neighbors = 3
        n_clusters_per = 1

        X, y = make_classification(n_features = num_features, n_redundant=0, n_informative=i,
                                n_clusters_per_class=n_clusters_per, n_samples=200, random_state=42, n_classes=2)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Non-Bagging
        knn_model = knn(n_neighbors=n_neighbors, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
        knn_model.fit()
        knn_model_accuracy = knn_model.evaluate()

        # Bagging KNN
        number_bags = 50
        bagged_knn = feature_subsampling(num_features,number_bags,num_features/2,X_train, X_test, y_train, y_test)
        bagged_knn_accuracy = bagged_knn.bagged_predictions()
        
        features_list.append(i)
        knn_model_accuracy_list.append(knn_model_accuracy)
        bagged_knn_accuracy_list.append(bagged_knn_accuracy)

    
    plt.figure(figsize=(10, 5))  # Set the figure size
    plt.plot(features_list, knn_model_accuracy_list, linestyle='dashed', color='blue', label='KNN Model Accuracy')
    plt.plot(features_list, bagged_knn_accuracy_list, linestyle='dotted', color='orange', label='Bagged KNN Accuracy')
    plt.title('Accuracy Comparisons Across Number of Informative Features out of 50')
    plt.xlabel('Number of Informative Features out of 50')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.ylim(0, 1)

    plt.savefig("figures/increasing_informative.pdf", format='pdf')
 
def plot_bag_size_to_accuracy(num_features):
    features_list = []
    knn_model_accuracy_list = []
    bagged_knn_accuracy_list = []

    for i in range(1,num_features):
        print(i)
        n_neighbors = 3
        bag_size = i
        n_clusters_per = 1
        n_informative = 10

        X, y = make_classification(n_features = num_features, n_redundant=0, n_informative=n_informative,
                                n_clusters_per_class=n_clusters_per, n_samples=200, random_state=42, n_classes=2)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Non-Bagging
        knn_model = knn(n_neighbors=n_neighbors, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
        knn_model.fit()
        knn_model_accuracy = knn_model.evaluate()

        # Bagging KNN
        number_bags = 50
        bagged_knn = feature_subsampling(num_features,number_bags,bag_size,X_train, X_test, y_train, y_test)
        bagged_knn_accuracy = bagged_knn.bagged_predictions()
        
        features_list.append(i)
        knn_model_accuracy_list.append(knn_model_accuracy)
        bagged_knn_accuracy_list.append(bagged_knn_accuracy)

    
    plt.figure(figsize=(10, 5))  # Set the figure size
    plt.plot(features_list, knn_model_accuracy_list, linestyle='dashed', color='blue', label='KNN Model Accuracy')
    plt.plot(features_list, bagged_knn_accuracy_list, linestyle='dotted', color='orange', label='Bagged KNN Accuracy')
    plt.title('Accuracy Comparisons Across Number of Features Subsampled per Bag')
    plt.xlabel('Number of Features Subsampled per Bag')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.ylim(0, 1)

    #plt.show()
    plt.savefig("figures/increasing_bag_size.pdf", format='pdf')



def plot_numfeature_to_accuracy(fix_num_inf):
    features_list = []
    knn_model_accuracy_list = []
    bagged_knn_accuracy_list = []
    fs_knn_accuracy_list = []
    knn_pca_accuracy_list = []

    for i in range(20,500,20):
        print(i)
        n_features = i
        if fix_num_inf:
            n_informative = 10
        else:
            n_informative = n_features//2
        n_neighbors = 3
        n_clusters_per = 1

        knn_model_accuracy, bagged_knn_accuracy, fs_knn_accuracy, knn_pca_accuracy = compute_accuracies(n_features, n_neighbors, n_informative, n_clusters_per)
        
        features_list.append(i)
        knn_model_accuracy_list.append(knn_model_accuracy)
        bagged_knn_accuracy_list.append(bagged_knn_accuracy)
        fs_knn_accuracy_list.append(fs_knn_accuracy)
        knn_pca_accuracy_list.append(knn_pca_accuracy)


    plt.figure(figsize=(10, 5))  # Set the figure size
    plt.plot(features_list, knn_model_accuracy_list, linestyle='dashed', color='black', label='KNN Model Accuracy')
    plt.plot(features_list, fs_knn_accuracy_list, linestyle='dotted', color='red', label='Information Gain KNN Accuracy')
    plt.plot(features_list, knn_pca_accuracy_list, linestyle='dotted', color='green', label='PCA-Based Dimension Reduction KNN Accuracy')
    plt.plot(features_list, bagged_knn_accuracy_list, linestyle='dotted', color='orange', label='Bagged KNN Accuracy')

    if fix_num_inf: #if we are fixing the number of informative features at 10
        plt.title('Accuracy Comparisons Across Number of Features (with Fixed Number of Informative Features)')
        plt.xlabel('Number of Features')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.legend(loc='lower right')
        plt.ylim(0, 1)

        plt.show()
        #plt.savefig("figures/fixed_inform_percent.pdf", format='pdf')


    else: #if the number of informative features increases along with total number of features
        plt.title('Accuracy Comparisons Across Number of Features (with Increasing Number of Informative Features)')
        plt.xlabel('Number of Features')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.legend(loc='lower right')
        plt.ylim(0, 1)

        plt.show()
        #plt.savefig("figures/increasing_inform.pdf", format='pdf')


def compute_accuracies(n_features, n_neighbors, n_informative,n_clusters_per):
    # Generate a random dataset
    X, y = make_classification(n_features = n_features, n_redundant=0, n_informative=n_informative,
                               n_clusters_per_class=n_clusters_per, n_samples=200, random_state=42, n_classes=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Non-Bagging
    knn_model = knn(n_neighbors=n_neighbors, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    knn_model.fit()
    knn_model_accuracy = knn_model.evaluate()

    # Bagging KNN
    number_bags = 50
    bagged_knn = feature_subsampling(n_features,number_bags,n_features//2,X_train, X_test, y_train, y_test)
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
    X_train_pca,X_test_pca = kp.knn_pca(n_features,X_train, X_test, y_train, y_test)
    knn_pca = knn(n_neighbors=n_neighbors, X_train=X_train_pca, X_test=X_test_pca, y_train=y_train, y_test=y_test)
    knn_pca.fit()
    knn_pca_accuracy = knn_pca.evaluate()

    return knn_model_accuracy, bagged_knn_accuracy, fs_knn_accuracy, knn_pca_accuracy


if __name__ == "__main__":
    main()
