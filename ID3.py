import math
import pandas as pd
import numpy as np
from collections import deque


class ClassifyNode:
    def __init__(self, data_ids, feature_id, feature_bin, majority):
        self.data_ids = data_ids
        self.feature_id = feature_id
        self.children = []  # binary array
        self.majority = majority
        self.feature_bin = feature_bin


class ID3_Classifier:
    def __init__(self, data, features, output):
        self.data = data
        self.data_output = output
        self.features = features
        self.root = None

    # todo: call build tree function
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

    def predict_aux(self, node, data_line):
        if not node.children or len(node.children) == 0:
            return node.majority

        if data_line[node.feature_id] < node.feature_bin:
            return self.predict_aux(node.children[0], data_line)
        else:
            return self.predict_aux(node.children[1], data_line)

    def calc_majority(self, data_ids):
        outputs = [self.data_output[id] for id in data_ids]
        positive_cnt = len([x for x in outputs if (x == 1)])
        negative_cnt = len([x for x in outputs if (x == 0)])
        if positive_cnt < negative_cnt:
            return 0
        return 1

    def calc_entropy(self, data_ids):
        # calc the outputs the of the data_ids
        outputs = [self.data_output[i] for i in data_ids]
        # count the number of the different outputs
        output_count = [outputs.count(x) for x in [0, 1]]

        entropy = 0
        for count in output_count:
            if count:
                p_i = count / len(data_ids)
                entropy += -p_i * math.log(p_i, 2)
        return entropy

    def cal_feature_vals_bins(self, data_ids, feature_id):
        feature_vals = sorted(list(set([self.data.iloc[x][feature_id] for x in data_ids])))
        if len(feature_vals) == 1:
            return [feature_vals[0] + 1]

        bins = []
        for i in range(len(feature_vals)-1):
            bins.append( feature_vals[i] + ((feature_vals[i+1] - feature_vals[i]) / 2))

        return bins

    def calc_information_gain(self, data_ids, feature_id):
        # calculate total entropy
        curr_IG = self.calc_entropy(data_ids)
        feature_vals_bins = self.cal_feature_vals_bins(data_ids, feature_id)

        max_info_gain = float('-inf')
        best_bin = float('-inf')
        # get frequency of each value
        for bin in feature_vals_bins:
            smaller_ids = []
            bigger_ids = []

            for id in data_ids:
                if self.data.iloc[id][feature_id] < bin:
                    smaller_ids.append(id)
                else:
                    bigger_ids.append(id)
            # compute the information gain with the chosen feature
            smaller_entropy = len(smaller_ids)/len(data_ids) * self.calc_entropy(smaller_ids)
            bigger_entropy = len(bigger_ids)/len(data_ids) * self.calc_entropy(bigger_ids)
            info_gain = curr_IG - smaller_entropy - bigger_entropy
            if info_gain >= max_info_gain:
                max_info_gain = info_gain
                best_bin = bin
        return max_info_gain, best_bin


    def get_best_feature_id(self, data_ids, feature_ids):
        # for each feature calc entropy
        features_info_gain = []
        for feature_id in feature_ids:
            features_info_gain.append(self.calc_information_gain(data_ids, feature_id))

        best_indx = 0
        best_info_gain = features_info_gain[0][0]

        for i in range(len(features_info_gain)):
            if features_info_gain[i][0] >= best_info_gain:
                best_indx = i
                best_info_gain = features_info_gain[i][0]

        return feature_ids[best_indx], features_info_gain[best_indx][1]


    def build_id3_tree(self):
        data_ids = [x for x in range(len(self.data))]
        if not data_ids:
            print(self.data)
            exit(0)

        assert data_ids is not None

        feature_ids = [x for x in range(len(self.features))]
        assert feature_ids is not None

        best_feature_id, best_feature_bin = self.get_best_feature_id(data_ids, feature_ids)

        self.root = ClassifyNode(data_ids, best_feature_id, best_feature_bin, self.calc_majority(data_ids))
        self.build_id3_tree_aux(self.root, data_ids, feature_ids)

    def build_id3_tree_aux(self, root, data_ids, feature_ids):
        if len(data_ids) == 0:
            return ClassifyNode(data_ids, None, None, root.majority)

        # sorted labels by instance id
        outputs = [self.data_output[id] for id in data_ids]

        # if all the example have the same class (pure node), return node
        if len(set(outputs)) == 1:
            node = ClassifyNode(data_ids, None, None, self.calc_majority(data_ids))
            return node

        best_feature_id, best_feature_bin = self.get_best_feature_id(data_ids, feature_ids)
        root.feature_id = best_feature_id
        root.feature_bin = best_feature_bin

        left_child_ids = []
        right_child_ids = []

        for x in data_ids:
            if self.data.iloc[x][best_feature_id] < best_feature_bin:
                left_child_ids.append(x)
            else:
                right_child_ids.append(x)

        left_child = ClassifyNode(left_child_ids, None, None, self.calc_majority(left_child_ids))
        root.children.append(left_child)

        right_child = ClassifyNode(right_child_ids, None, None, self.calc_majority(right_child_ids))
        root.children.append(right_child)

        # todo check with Sammar
        # self._id3_recv(child_data_ids, feature_ids, child)
        self.build_id3_tree_aux(left_child, left_child_ids, feature_ids)
        self.build_id3_tree_aux(right_child, right_child_ids, feature_ids)


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
            print("child.bin", '({})'.format(node.feature_bin))
            print("child.children", '({})'.format(node.children))
            print("child.majority", '({})'.format(node.majority))

            if node.children:
                # print("children: ")
                for child in node.children:

                    nodes.append(child)

