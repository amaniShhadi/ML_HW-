# ID3 (Iterative Dichotomiser 3) Algorithm implementation from scratch

import math
from collections import deque
import pandas as pd
import numpy as np
import math
from collections import deque


class Node:
    """Contains the information of the node and another nodes of the Decision Tree."""

    def __init__(self):
        self.value = None
        self.next = None
        self.childs = None


class DecisionTreeClassifier:
    """Decision Tree Classifier using ID3 algorithm."""

    def __init__(self, X, feature_names, labels):
        self.X = X
        self.feature_names = feature_names
        self.labels = labels
        self.labelCategories = list(set(labels))

        self.labelCategoriesCount = [list(labels).count(x) for x in self.labelCategories]
        self.node = None
        self.entropy = self._get_entropy([x for x in range(len(self.labels))])  # calculates the initial entropy

    def _get_entropy(self, x_ids):
        """ Calculates the entropy.
        Parameters
        __________
        :param x_ids: list, List containing the instances ID's
        __________
        :return: entropy: float, Entropy.
        """
        # sorted labels by instance id
        labels = [self.labels[i] for i in x_ids]
        # count number of instances of each category
        label_count = [labels.count(x) for x in self.labelCategories]
        # calculate the entropy for each category and sum them
        entropy = sum([-count / len(x_ids) * math.log(count / len(x_ids), 2) if count else 0 for count in label_count])
        return entropy

    def _get_information_gain(self, x_ids, feature_id):
        """Calculates the information gain for a given feature based on its entropy and the total entropy of the system.
        Parameters
        __________
        :param x_ids: list, List containing the instances ID's
        :param feature_id: int, feature ID
        __________
        :return: info_gain: float, the information gain for a given feature.
        """
        # calculate total entropy
        info_gain = self._get_entropy(x_ids)
        # store in a list all the values of the chosen feature
        x_features = [self.X[x][feature_id] for x in x_ids]
        # get unique values
        feature_vals = list(set(x_features))
        # get frequency of each value
        feature_vals_count = [x_features.count(x) for x in feature_vals]
        # get the feature values ids
        feature_vals_id = [
            [x_ids[i]
            for i, x in enumerate(x_features)
            if x == y]
            for y in feature_vals
        ]

        # compute the information gain with the chosen feature
        info_gain = info_gain - sum([val_counts / len(x_ids) * self._get_entropy(val_ids)
                                     for val_counts, val_ids in zip(feature_vals_count, feature_vals_id)])

        return info_gain

    def _get_feature_max_information_gain(self, x_ids, feature_ids):
        """Finds the attribute/feature that maximizes the information gain.
        Parameters
        __________
        :param x_ids: list, List containing the samples ID's
        :param feature_ids: list, List containing the feature ID's
        __________
        :returns: string and int, feature and feature id of the feature that maximizes the information gain
        """
        # get the entropy for each feature
        features_entropy = [self._get_information_gain(x_ids, feature_id) for feature_id in feature_ids]
        # find the feature that maximises the information gain
        max_id = feature_ids[features_entropy.index(max(features_entropy))]

        return self.feature_names[max_id], max_id

    def id3(self):
        """Initializes ID3 algorithm to build a Decision Tree Classifier.
        :return: None
        """
        x_ids = [x for x in range(len(self.X))]
        feature_ids = [x for x in range(len(self.feature_names))]
        self.node = self._id3_recv(x_ids, feature_ids, self.node)
        print('')

    def _id3_recv(self, x_ids, feature_ids, node):
        """ID3 algorithm. It is called recursively until some criteria is met.
        Parameters
        __________
        :param x_ids: list, list containing the samples ID's
        :param feature_ids: list, List containing the feature ID's
        :param node: object, An instance of the class Nodes
        __________
        :returns: An instance of the class Node containing all the information of the nodes in the Decision Tree
        """
        if not node:
            node = Node()  # initialize nodes
        # sorted labels by instance id
        labels_in_features = [self.labels[x] for x in x_ids]
        # if all the example have the same class (pure node), return node
        if len(set(labels_in_features)) == 1:
            node.value = self.labels[x_ids[0]]
            return node
        # if there are not more feature to compute, return node with the most probable class
        if len(feature_ids) == 0:
            node.value = max(set(labels_in_features), key=labels_in_features.count)  # compute mode
            return node
        # else...
        # choose the feature that maximizes the information gain
        best_feature_name, best_feature_id = self._get_feature_max_information_gain(x_ids, feature_ids)
        node.value = best_feature_name
        node.childs = []
        # value of the chosen feature for each instance
        feature_values = list(set([self.X[x][best_feature_id] for x in x_ids]))
        # loop through all the values
        for value in feature_values:
            child = Node()
            child.value = value  # add a branch from the node to each feature value in our feature
            node.childs.append(child)  # append new child node to current node
            child_x_ids = [x for x in x_ids if self.X[x][best_feature_id] == value]
            if not child_x_ids:
                print("child next")
                child.next = max(set(labels_in_features), key=labels_in_features.count)
                print(child.next)
                exit(0)
                print('')
            else:
                if feature_ids and best_feature_id in feature_ids:
                    to_remove = feature_ids.index(best_feature_id)
                    feature_ids.pop(to_remove)
                # recursively call the algorithm
                print("before child next")
                print(child.next)
                child.next = self._id3_recv(child_x_ids, feature_ids, child.next)
                print("after child next")
                print(child.next.value)
        return node

    def printTree(self):
        if not self.node:
            return
        nodes = deque()
        nodes.append(self.node)
        while len(nodes) > 0:
            node = nodes.popleft()
            print(node.value)
            if node.childs:
                for child in node.childs:
                    print('({})'.format(child.value))
                    nodes.append(child.next)
            elif node.next:
                print(node.next)


if __name__ == "__main__":

    # generate some data
    # define features and target values
    data = {
        'wind_direction': ['N', 'S', 'E', 'W'],
        'tide': ['Low', 'High'],
        'swell_forecasting': ['small', 'medium', 'large'],
        'good_waves': ['Yes', 'No']
    }

    # create an empty dataframe
    data_df = pd.DataFrame(columns=data.keys())

    np.random.seed(42)
    # randomnly create 1000 instances
    for i in range(5):
        data_df.loc[i, 'wind_direction'] = str(np.random.choice(data['wind_direction'], 1)[0])
        data_df.loc[i, 'tide'] = str(np.random.choice(data['tide'], 1)[0])
        data_df.loc[i, 'swell_forecasting'] = str(np.random.choice(data['swell_forecasting'], 1)[0])
        data_df.loc[i, 'good_waves'] = str(np.random.choice(data['good_waves'], 1)[0])

    data_df.head()
    print("--------------- data_df ----------------")
    print(data_df)

    # separate target from predictors
    X = np.array(data_df.drop('good_waves', axis=1).copy())
    print("--------------- X ----------------")
    print(X)
    y = np.array(data_df['good_waves'].copy())
    print("--------------- y ----------------")
    print(y)

    feature_names = list(data_df.keys())[:3]
    print("--------------- feature_names ----------------")
    print(feature_names)
    # import and instantiate our DecisionTreeClassifier class
    # from ID3 import DecisionTreeClassifier

    # instantiate DecisionTreeClassifier
    tree_clf = DecisionTreeClassifier(X=X, feature_names=feature_names, labels=y)
    print("System entropy {:.4f}".format(tree_clf.entropy))
    # run algorithm id3 to build a tree
    tree_clf.id3()
    tree_clf.printTree()
