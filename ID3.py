import AbstractClassifier
import math
import pandas as pd
import numpy as np

from collections import deque


def entropy(col):
    diff_values = sorted(col.unique())
    num_diff_values = len(diff_values)

    normalized_counts = col.value_counts(bins=num_diff_values-1, sort=False, normalize=True)
    res = sum([-count * math.log(count, 2)
               if count else 0
               for count in normalized_counts])
    return res


class ClassifyNode:
    def __init__(self, data_ids, feature_id, vals_range):
        self.data_ids = data_ids
        self.feature_id = feature_id
        self.children = []
        self.majority = None #calc_majority()
        self.parent = None
        self.vals_range = vals_range


class ID3_Classifier:
    def __init__(self, data, features, output):
        self.data = data
        self.data_output = output
        self.features = features
        self.root = None

    def fit(self, x, y):
        """abstract function
           check what is the x, y
        """
        raise NotImplementedError

    def predict(self, data):
        """abstract function
           check what is the x
        """
        assert self.root is not None
        return self.predict_aux(self.root, data)

    def predict_aux(self, node, data):
        if not node.children or len(node.children) == 0:
            return node.majority

        for child in node.children:
            if data[child.feature_id] in child.vals_range:
                return self.predict_aux(child, data)




    def calc_entropy(self, data_ids):
        # sorted labels by instance id
        output = [self.data_output[i] for i in data_ids]
        # count number of instances of each category
        output_count = [output.count(x) for x in [0, 1]]
        # calculate the entropy for each category and sum them
        entropy = sum([-count / len(data_ids) * math.log(count / len(data_ids), 2) if count else 0 for count in output_count])
        return entropy

    def cal_feature_vals_ranges(self, data_ids, feature_id):
        x_features = [self.data.iloc[x][feature_id] for x in data_ids]
        # get unique values
        feature_vals = list(set(x_features))
        # if not feature_vals:
        #     print("f vals ---->: ", feature_vals)
        #     print("feature_id:", feature_id)
        #     print("data_ids:",data_ids)
        #
        #     exit(0)

        bins = 1 if len(feature_vals) < 2 else len(feature_vals) - 1
        feature_vals_ranges = pd.unique(pd.cut(np.array(feature_vals), bins, right=False, ordered=True, duplicates='drop'))
        # print("f vals ---->: ",feature_vals)
        # print("------->feature_vals_ranges:")
        # print(feature_vals_ranges)

        return feature_vals_ranges

    def calc_information_gain(self, data_ids, feature_id):
        # calculate total entropy
        curr_IG = self.calc_entropy(data_ids)
        # store in a list all the values of the chosen feature
        # print("--------> feature_id  ", feature_id)
        # print("--------> data_ids  ", data_ids)
        # print("----amani ", self.data.iloc[0][0])
        # exit(0)

        feature_vals_ranges = self.cal_feature_vals_ranges(data_ids, feature_id)
        feature_vals_ranges_ids = []

        # get frequency of each value
        for val_range in feature_vals_ranges:
            val_ids = []
            for id in data_ids:
                if self.data.iloc[id][feature_id] in val_range:
                    val_ids.append(id)
            feature_vals_ranges_ids.append(val_ids)

        # compute the information gain with the chosen feature
        entropy_sum = 0
        for i in range(len(feature_vals_ranges_ids)):
            entropy_sum += len(feature_vals_ranges_ids[i])/len(data_ids) * self.calc_entropy(feature_vals_ranges_ids[i])

        return curr_IG - entropy_sum

    def get_best_feature_id(self, data_ids, feature_ids):
        # for each feature calc entropy
        features_info_gain = [self.calc_information_gain(data_ids, feature_id) for feature_id in feature_ids]
        # find the feature that maximises the information gain
        # print("amanaaaaaaaaani 27sn w7de! ")
        # print(features_info_gain)
        best_id = feature_ids[features_info_gain.index(max(features_info_gain))]

        return best_id


    def build_id3_tree(self):
        data_ids = [x for x in range(len(self.data))]
        if not data_ids:
            print(self.data)
            exit(0)

        assert data_ids is not None

        feature_ids = [x for x in range(len(self.features))]
        assert feature_ids is not None

        best_feature_id = self.get_best_feature_id(data_ids, feature_ids)
        self.root = ClassifyNode(data_ids, best_feature_id, (float('-inf'), float('inf')))
        self.build_id3_tree_aux(self.root, data_ids, feature_ids)

    def calc_majority(self, data_ids):
        outputs = [self.data_output[id] for id in data_ids]
        positive_cnt = len([x for x in outputs if (x == 1)])
        negative_cnt = len([x for x in outputs if (x == 0)])
        if positive_cnt < negative_cnt:
            return 0
        return 1

    def build_id3_tree_aux(self, root, data_ids, feature_ids):
        # sorted labels by instance id
        outputs = [self.data_output[id] for id in data_ids]

        # if all the example have the same class (pure node), return node
        if len(set(outputs)) == 1:
            node = ClassifyNode(data_ids, None, None)
            node.majority = self.calc_majority(data_ids)
            return node

        # if there are not more feature to compute, return node with the most probable class
        # if len(feature_ids) == 0:
        #     root.value = max(set(labels_in_features), key=labels_in_features.count)  # compute mode
        #     return root
        # else...
        # choose the feature that maximizes the information gain
        best_feature_id = self.get_best_feature_id(data_ids, feature_ids)
        self.root.feature_id = best_feature_id

        feature_vals_ranges = self.cal_feature_vals_ranges(data_ids, best_feature_id)

        # value of the chosen feature for each instance
        # feature_values = list(set([self.data.iloc[x][best_feature_id] for x in data_ids]))
        # loop through all the values


        for vals_range in feature_vals_ranges:
            child_data_ids = [x for x in data_ids if self.data.iloc[x][best_feature_id] in vals_range]
            if child_data_ids is None or len(child_data_ids) == 0:
                continue

            child = ClassifyNode(child_data_ids, best_feature_id, vals_range)
            child.majority = self.calc_majority(child_data_ids)
            root.children.append(child)

            # todo check with Sammar
            # self._id3_recv(child_data_ids, feature_ids, child)
            self.build_id3_tree_aux(child, child_data_ids, feature_ids )


    def printTree(self):
        if not self.root:
            return
        nodes = deque()
        nodes.append(self.root)
        while len(nodes) > 0:
            node = nodes.popleft()
            print("*****************************************")
            print("node address = ", node)
            print("child.feature_id", '({})'.format(node.feature_id))
            print("child.data_ids", '({})'.format(node.data_ids))
            print("child.children", '({})'.format(node.children))
            print("child.vals_range", '({})'.format(node.vals_range))
            print("child.majority", '({})'.format(node.majority))


            if node.children:
                # print("children: ")
                for child in node.children:

                    nodes.append(child)

