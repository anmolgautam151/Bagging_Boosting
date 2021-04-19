#!/usr/bin/env python
# coding: utf-8

# The plt.show() is commented in main because there are a total of 16 confusion matrices.
# To see those marices, plt.show() can be uncommented in the main
# All the accuracies of combinations of depth and bag size are printed out as a table

import math
import numpy as np
import pandas as pd
import category_encoders as ce
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt


def partition(x):
    """
    Partition the column vector x into subsets indexed by its unique values (v1, ... vk)
    Returns a dictionary of the form
    { v1: indices of x == v1,
      v2: indices of x == v2,
      ...
      vk: indices of x == vk }, where [v1, ... vk] are all the unique values in the vector z.
    """
    res = {i: [] for i in set(x)}
    for index, i in enumerate(x):
        res[i].append(index)
    return res


def bagging(x, y, max_depth, num_trees):
    alpha_hypothesis_pair = []
    alpha = 1 / num_trees
    for f in range(num_trees):
        idx = np.random.choice(np.arange(len(x)), len(x), replace=True)
        x_new = x[idx]
        y_new = y[idx]
        attr_value = []
        for i in range(x_new.shape[1]):
            attr_value.extend([(i, k) for k in set(x_new[:, i])])
        decision_tree = id3(x_new, y_new, attribute_value_pairs=attr_value, max_depth=max_depth)
        alpha_hypothesis_pair.append((alpha, decision_tree))

    return alpha_hypothesis_pair


def boosting(x, y, max_depth, num_stumps):
    """
    Return  multiple hypothesis  and its weights based on num_stumps
    :param x:
    :param y:
    :param max_depth:
    :param num_stumps:
    :return:
    """
    alpha_hypothesis_pair = []
    n = len(y)
    for i in range(0, num_stumps):
        weight = np.full((n, 1), 1 / n, dtype=float).flatten()
        attr_value = []
        for z in range(x.shape[1]):
            attr_value.extend([(z, k) for k in set(x[:, z])])
        dtree = id3(x, y, attribute_value_pairs=attr_value, max_depth=max_depth)
        y_pred = [predict_with_single_tree(t, dtree) for t in x]
        error = errorboost(x, y, y_pred, weight)
        mis_index, cor_index = getmisprediction(y, y_pred)
        alpha = 0.5 * (math.log((1 - error) / (error + 1e-5)))
        for d in mis_index:
            weight[d] = weight[d] * math.exp(alpha)
        for d in cor_index:
            weight[d] = weight[d] * math.exp(-alpha)
        sum_of_weights = sum(weight)
        for u in range(len(weight)):
            weight[u] = weight[u] / sum_of_weights
        alpha_hypothesis_pair.append([alpha, dtree])
        x, y = resample(x, y, weight)
    return alpha_hypothesis_pair


def resample(x, y, weight):
    merge = (np.column_stack((x, y)))[np.random.choice(x.shape[0], y.shape[0], replace=True, p=weight)]
    x = merge[:, :-1].copy()
    y = merge[:, -1].copy()
    return x, y


def errorboost(x, y, y_pred, weight):
    """
    Weightade error for boosting tree
    :param x:
    :param y:
    :param y_pred:
    :param weight:
    :return:
    """
    error = 0
    for i in range(len(x)):
        if y_pred[i] != y[i]:
            error = error + weight[i]
    return error


def getmisprediction(a, b):
    mispred = []
    pred = []
    for i in range(0, len(a)):
        if (a[i] != b[i]):
            mispred.append(i)
        else:
            pred.append(i)
    return mispred, pred


def entropy(y):
    """
    Compute the entropy of a vector y by considering the counts of the unique values (v1, ... vk), in z

    Returns the entropy of z: H(z) = p(z=v1) log2(p(z=v1)) + ... + p(z=vk) log2(p(z=vk))
    """
    rep_dict = partition(y)
    H_z = 0
    for k, v in rep_dict.items():
        p = len(v) / len(y)
        H_z += -p * np.log2(p)
    return H_z


def mutual_information(x, y):
    """
    Compute the mutual information between a data column (x) and the labels (y). The data column is a single attribute
    over all the examples (n x 1). Mutual information is the difference between the entropy BEFORE the split set, and
    the weighted-average entropy of EACH possible split.

    Returns the mutual information: I(x, y) = H(y) - H(y | x)
    """
    H_y = entropy(y)
    feature_partition = partition(x)
    H_y_x = 0
    for k, v in feature_partition.items():
        weightage = len(v) / len(y)
        H_y_x += weightage * entropy([y[k] for k in v])
    I = H_y - H_y_x
    return I


def get_majority(y):
    res = partition(y)
    max = 0
    majority = 0
    for k, v in res.items():
        if len(v) > max:
            max = len(v)
            majority = k
    return majority


def id3(x, y, attribute_value_pairs=None, depth=0, max_depth=5):
    """
    Implements the classical ID3 algorithm given training data (x), training labels (y) and an array of
    attribute-value pairs to consider. This is a recursive algorithm that depends on three termination conditions
        1. If the entire set of labels (y) is pure (all y = only 0 or only 1), then return that label
        2. If the set of attribute-value pairs is empty (there is nothing to split on), then return the most common
           value of y (majority label)
        3. If the max_depth is reached (pre-pruning bias), then return the most common value of y (majority label)
    Otherwise the algorithm selects the next best attribute-value pair using INFORMATION GAIN as the splitting criterion
    and partitions the data set based on the values of that attribute before the next recursive call to ID3.

    The tree we learn is a BINARY tree, which means that every node has only two branches. The splitting criterion has
    to be chosen from among all possible attribute-value pairs. That is, for a problem with two features/attributes x1
    (taking values a, b, c) and x2 (taking values d, e), the initial attribute value pair list is a list of all pairs of
    attributes with their corresponding values:
    [(x1, a),
     (x1, b),
     (x1, c),
     (x2, d),
     (x2, e)]
     If we select (x2, d) as the best attribute-value pair, then the new decision node becomes: [ (x2 == d)? ] and
     the attribute-value pair (x2, d) is removed from the list of attribute_value_pairs.

    The tree is stored as a nested dictionary, where each entry is of the form
                    (attribute_index, attribute_value, True/False): subtree
    * The (attribute_index, attribute_value) determines the splitting criterion of the current node. For example, (4, 2)
    indicates that we test if (x4 == 2) at the current node.
    * The subtree itself can be nested dictionary, or a single label (leaf node).
    * Leaf nodes are (majority) class labels

    Returns a decision tree represented as a nested dictionary, for example
    {(4, 1, False):
        {(0, 1, False):
            {(1, 1, False): 1,
             (1, 1, True): 0},
         (0, 1, True):
            {(1, 1, False): 0,
             (1, 1, True): 1}},
     (4, 1, True): 1}
    """
    tree = {}
    # entire set of label is pure
    e = partition(y)
    if len(e) == 1:
        return y[0]
    # attribute set is empty
    elif not attribute_value_pairs or depth == max_depth:
        return get_majority(y)
    else:
        # check best information gain among list of features
        ml_info_gain = -1.0
        best_feature, best_value = 0, 0
        pos_subset_indx, neg_subset_indx = [], []
        for i in attribute_value_pairs:
            feature, value = i[0], i[1]
            temp_x = [1 if k == value else 0 for k in x[:, feature]]
            m = mutual_information(temp_x, y)
            if m > ml_info_gain:
                ml_info_gain = m
                best_feature, best_value = feature, value
                pos_subset_indx = np.where(x[:, feature] == value)[0]
                neg_subset_indx = np.where(x[:, feature] != value)[0]

        modified_x_pos = np.take(x, pos_subset_indx, axis=0)
        modified_y_pos = np.take(y, pos_subset_indx)

        modified_x_neg = np.take(x, neg_subset_indx, axis=0)
        modified_y_neg = np.take(y, neg_subset_indx)

        temp_attr_value = (best_feature, best_value)

        av1, av2 = attribute_value_pairs.copy(), attribute_value_pairs.copy()
        av1.remove(temp_attr_value)
        av2.remove(temp_attr_value)
        tree[(best_feature, best_value, True)] = id3(modified_x_pos,
                                                     modified_y_pos,
                                                     av1,
                                                     depth + 1, max_depth)
        tree[(best_feature, best_value, False)] = id3(modified_x_neg,
                                                      modified_y_neg,
                                                      av2,
                                                      depth + 1, max_depth)
        return tree


def predict_with_single_tree(x, tree):
    """
    Predicts the classification label for a single example x using tree by recursively descending the tree until
    a label/leaf node is reached.

    Returns the predicted label of x according to tree
    """
    if tree == 0 or tree == 1:
        return tree
    attr_pair = list(tree.keys())[0]
    key, val = attr_pair[0], attr_pair[1]
    res = bool(x[key] == val)
    return predict_with_single_tree(x, tree[key, val, res])


def predict_example(x, h_ens):
    predictions = []
    neg_alpha, pos_alpha = 0, 0
    for i in h_ens:
        res = predict_with_single_tree(x, i[1])
        if res == 0:
            neg_alpha += i[0]
        else:
            pos_alpha += i[0]
    if neg_alpha >= pos_alpha:
        return 0
    else:
        return 1


def predict_test_set(X_test, y_test, alpha_hypo_pair):
    y_hat = []
    for i in X_test:
        y_hat.append(predict_example(i, alpha_hypo_pair))
    cf_matrix = confusion_matrix(y_test, y_hat)
    return cf_matrix, y_hat


def compute_error(y_true, y_pred):
    """
    Computes the average error between the true labels (y_true) and the predicted labels (y_pred)

    Returns the error = (1/n) * sum(y_true != y_pred)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    error = np.sum(y_true != y_pred)
    error = (1 / len(y_true)) * error
    return error


if __name__ == '__main__':

    train_file, test_file = './mushroom.train', './mushroom.test'
    M_train = np.genfromtxt(train_file, missing_values=0, skip_header=0, delimiter=',', dtype='U')
    M_test = np.genfromtxt(test_file, missing_values=0, skip_header=0, delimiter=',', dtype='U')

    column_names = ['bruises', 'poisnous', 'cap-shape', 'cap-surface', 'cap-color', 'odor', 'gill-attachment',
                    'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-surface-above-ring',
                    'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring',
                    'veil-type', 'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat']

    train_df = pd.DataFrame(data=M_train, columns=column_names)
    test_df = pd.DataFrame(data=M_test, columns=column_names)
    train_df.head()

    train_data = train_df.drop(['bruises'], axis=1)
    test_data = test_df.drop(['bruises'], axis=1)
    y_train_label = train_df['bruises']
    y_test_label = test_df['bruises']

    # ### Encoding categorical variables
    categorical_transformer = Pipeline(steps=[('woe', ce.OrdinalEncoder())])
    categorical_features = train_data.select_dtypes(include=['object']).columns
    preprocessor = ColumnTransformer(transformers=[('cat', categorical_transformer, categorical_features)])

    # Feature transformation
    preprocessor.fit(train_data)
    X_train = preprocessor.transform(train_data)
    X_test = preprocessor.transform(test_data)

    le = LabelEncoder()
    label_encoder = le.fit(y_train_label)
    y_train = label_encoder.transform(y_train_label)
    y_test = label_encoder.transform(y_test_label)

    print("Training Size:", X_train.shape, "Testing Size:", X_test.shape)

    comparision_df = pd.DataFrame(columns=['Algorithm', 'Implementation', 'Depth', 'Bag Size', 'Accuracy'])
    MAX_TREE_DEPTH = [3, 5]
    BAG_SIZE = [10, 20]
    COMBINATION = [(i, j) for i in MAX_TREE_DEPTH for j in BAG_SIZE]

    ## Part a. Bagging: Custom Implementation
    for index, i in enumerate(COMBINATION):
        d, k = i
        temp = {'Algorithm': "Bagging", 'Depth': d, 'Implementation': "Custom", 'Bag Size': k, 'Accuracy': 0}
        alpha_hypo_pair = bagging(X_train, y_train, d, k)
        cf_matrix, y_pred = predict_test_set(X_test, y_test, alpha_hypo_pair)
        temp['Accuracy'] = accuracy_score(y_test, y_pred)
        comparision_df = comparision_df.append(temp, ignore_index=True)
        fig, ax = plot_confusion_matrix(conf_mat=cf_matrix, figsize=(5, 8))
        ax.set_title('Confusion Matrix - Max Depth {} Bag Size {}'.format(d, k))
    # plt.show()
    print("Bagging Custom Implementation Completed")

    # # Part c.1 Bagging: Sci-Kit Implementation
    for index, i in enumerate(COMBINATION):
        d, k = i
        temp = {'Algorithm': "Bagging", 'Depth': d, 'Implementation': 'Sci-Kit', 'Bag Size': k, 'Accuracy': 0}
        clf = BaggingClassifier(DecisionTreeClassifier(max_depth=d), n_estimators=k)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        temp['Accuracy'] = accuracy_score(y_test, y_pred)
        cf_matrix = confusion_matrix(y_test, y_pred)
        comparision_df = comparision_df.append(temp, ignore_index=True)
        fig, ax = plot_confusion_matrix(conf_mat=cf_matrix, figsize=(5, 8))
        ax.set_title('Confusion Matrix - Max Depth {} Bag Size {}'.format(d, k))
    # plt.show()
    print("Bagging Sci-kit Implementation Completed")

    MAX_TREE_DEPTH = [1, 2]
    BAG_SIZE = [20, 40]
    COMBINATION = [(i, j) for i in MAX_TREE_DEPTH for j in BAG_SIZE]

    # # Part b. Boosting: Custom Implementation
    for index, i in enumerate(COMBINATION):
        d, k = i
        temp = {'Algorithm': "Boosting", 'Depth': d, 'Implementation': 'Custom', 'Bag Size': k, 'Accuracy': 0}
        alpha_hypo_pair = boosting(X_train, y_train, d, k)
        cf_matrix, y_pred = predict_test_set(X_test, y_test, alpha_hypo_pair)
        temp['Accuracy'] = accuracy_score(y_test, y_pred)
        comparision_df = comparision_df.append(temp, ignore_index=True)
        cf_matrix = confusion_matrix(y_test, y_pred)
        fig, ax = plot_confusion_matrix(conf_mat=cf_matrix, figsize=(5, 8))
        ax.set_title('Confusion Matrix - Max Depth {} Bag Size {}'.format(d, k))
    # plt.show()
    print("Boosting Custom Implementation Completed")

    ## Part c.2 Boosting: Sci-Kit Implementation
    for index, i in enumerate(COMBINATION):
        d, k = i
        temp = {'Algorithm': "Boosting", 'Depth': d, 'Implementation': 'Sci-Kit', 'Bag Size': k, 'Accuracy': 0}
        clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=d), n_estimators=k, random_state=0)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        temp['Accuracy'] = accuracy_score(y_test, y_pred)
        comparision_df = comparision_df.append(temp, ignore_index=True)
        cf_matrix = confusion_matrix(y_test, y_pred)
        fig, ax = plot_confusion_matrix(conf_mat=cf_matrix, figsize=(5, 8))
        ax.set_title('Confusion Matrix - Max Depth {} Bag Size {}'.format(d, k))
    # plt.show()
    print("Boosting Sci-Kit Implementation Completed")

    print("Comparision Table")
    print(comparision_df)
