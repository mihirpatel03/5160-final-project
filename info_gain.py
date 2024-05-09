from sklearn.feature_selection import mutual_info_classif
import numpy as np

def make_ig_dict(X_train, y_train):
    info_gain_dict = {}
    num_features = X_train.shape[1]
    for i in range(num_features):
        # calculate mutual information between feature and target
        mi = mutual_info_classif(X_train[:, i:i+1], y_train)[0]
        info_gain_dict[i] = mi
    
    return info_gain_dict
    

def best_feature_list(ig_dict):
    inverted_dict = {}
    for key, value in ig_dict.items():
        if value not in inverted_dict:
            inverted_dict[value] = [key]
        else:
            inverted_dict[value].append(key)

    ig_vals = sorted(list(ig_dict.values()), reverse=True)

    best_feature_list = []
    num_to_select = int(len(ig_vals)** 0.5)
    for i in range(num_to_select):
        feature_list = inverted_dict[ig_vals[i]]
        if len(feature_list)>1:
            best_feature_list.append(feature_list[-1])
            feature_list.pop()
        else:
            best_feature_list.append(feature_list[0])

    return best_feature_list

    


def create_new_trainset():
    pass