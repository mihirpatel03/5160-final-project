from sklearn.feature_selection import mutual_info_classif
import numpy as np

#make a dictionary of each feature as a key, associated value is the mutual info gain
def make_ig_dict(X_train, y_train): 
    info_gain_dict = {}
    num_features = X_train.shape[1]
    #calculate the mutual information gain of each feature...
    for i in range(num_features):
        #between feature and target value
        mi = mutual_info_classif(X_train[:, i:i+1], y_train)[0]
        info_gain_dict[i] = mi
    return info_gain_dict
    
#create a list of the best features using our dictionary
def best_feature_list(ig_dict):
    #create an inverted dictionary that has the info gain values as keys, the feature names as values
    inverted_dict = {}
    for key, value in ig_dict.items():
        if value not in inverted_dict:
            inverted_dict[value] = [key]
        else:
            #if there are two features with the same info gain, make a list as this value which will store 
            #multiple feature names for this key of info gain
            inverted_dict[value].append(key)

    #duplicate the info gain values and sort them in descending order
    ig_vals = sorted(list(ig_dict.values()), reverse=True)

    best_feature_list = []
    num_to_select = int(len(ig_vals)** 0.5) #want to select sqrt(d) features

    for i in range(num_to_select):
        #find the values in our inverted dict for this info gain number (i.e. a list of feature names)
        feature_list = inverted_dict[ig_vals[i]]
        if len(feature_list)>1:
            #if there are multiple, take one and remove it
            best_feature_list.append(feature_list[-1])
            feature_list.pop()
        else:
            best_feature_list.append(feature_list[0])

    return best_feature_list