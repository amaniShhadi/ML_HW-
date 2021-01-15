import numpy as np
import pandas as pd
import math
from ID3 import ID3_Classifier

from collections import deque
import sklearn, random, matplotlib

""" :parameter df: data frame
    :return the entropy of the given data frame"""
# def calc_entropy(df):
#     output_col = df[df.columns[-1]]
#     diff_values = sorted(output_col.unique())
#     num_diff_values = len(diff_values)
#     # print("----    num_diff_values  ", num_diff_values)
#     # total_cnt = col.count()
#     # print("---- total_cnt  ", total_cnt)
#     frequency_table = output_col.value_counts(bins=num_diff_values, sort=False, normalize=True)
#     # print("+++++++++++> ranges")
#     # print(frequency_table)
#     res = sum([-count * math.log(count, 2)
#                if count else 0
#                for count in frequency_table])
#     return res


def calc_IG(df, feature):
    curr_entropy = calc_entropy(df)

    feature_col = df[feature]
    diff_values = sorted(feature_col.unique())
    print(diff_values)
    print("newwww")
    print( df[(df[feature] > 1)])
    # normalized_counts = feature_col.value_counts(bins=num_diff_values-1, sort=False, normalize=True)
    aa = pd.qcut(feature_col, q=len(diff_values) -1)
    # print(aa)

    # num_diff_values = len(diff_values)
    # # print("----", diff_values)
    # # total_cnt = col.count()
    #
    # frequency_table = col.value_counts(bins=num_diff_values - 1, sort=False, normalize=True)
    # # print("+++++++++++> ranges")
    # # print(normalized_counts)
    # res = sum([-count * math.log(count, 2)
    #            if count else 0
    #            for count in frequency_table])
    # return res


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
    feature_v_count = [x_features.count(x) for x in feature_vals]
    # get the feature values ids
    feature_v_id = [
        [x_ids[i]
        for i, x in enumerate(x_features)
        if x == y]
        for y in feature_vals
    ]

    # compute the information gain with the chosen feature
    info_gain_feature = sum([v_counts / len(x_ids) * self._get_entropy(v_ids)
                             for v_counts, v_ids in zip(feature_v_count, feature_v_id)])

    info_gain = info_gain - info_gain_feature
    return info_gain


if __name__ == "__main__":

    data = pd.read_csv("train.csv")
    feature_names = list(data.keys())[:-1]
    data_features = np.array(data.drop('Outcome', axis=1).copy())
    data_outcome = np.array(data['Outcome'].copy())

    # print("=======  ", data.iloc[0][0])
    # exit(0)
    # aa = [1, 1, 1, 1, 1, 1, 1, 1, 2]
    # feature_vals = list(set(aa))
    # print("====1   ", feature_vals)
    # feature_vals_ranges = pd.cut(np.array(feature_vals),  1, right=False, ordered=True)
    # print("====2   ", feature_vals_ranges)
    # exit(0)

    id3_tree = ID3_Classifier(data, feature_names, data_outcome)
    id3_tree.build_id3_tree()
    id3_tree.printTree()

    test = pd.read_csv("test.csv")
    test_features = np.array(data.drop('Outcome', axis=1).copy())
    test_outcome = np.array(data['Outcome'].copy())
    match_cnt = 0
    for i in range(len(test_features)):
        res = id3_tree.predict(test_features[i])
        print ("res : ", res, "  predict : ", test_outcome[i])
        if res == test_outcome[i]:
            match_cnt += 1

    print("percision = ", match_cnt/len(test_features))





    # df = np.genfromtxt('amani.csv', delimiter=',', dtype=None)
    # print(X[0][1])
    # # # load csv file
    # features = df.columns
    # #
    # # # for row in labels.iterrows():
    # # #     print(row[0])
    # # #     print("***************")
    # # #
    # # # for i in range(len(labels.index)):
    # # #     print(labels.loc[i, [features[0]]])
    # # #     print("&&&&&&&&&&&&")
    # # #
    # # # for f in features:
    # # #     print("=======   " + f)
    # # #
    # # # print("-------------> features")
    # # # print(features)
    # # # print("-------------> features")
    # #
    # # """array of arrays where each array has the unique value of that feature
    # # """
    # # all_ranges = [sorted(df[x].unique()) for x in features]
    # # # labelCategoriesCount = sorted(df.Pregnancies.unique())
    # # print(all_ranges)
    # # print(features)
    # # # for feature in features:
    # # # for limit in labelCategoriesCount[0]:
    # # #     print(df[df[features[0]] <= limit ].count()[0])
    #
    # res = calc_IG(df, features[0])
    # # first_col = df[df.columns[-1]]
    # # # print(first_col)
    # # #
    # # res = calc_entropy(df)
    # print("[[[[[[[[[[[[[[[[[[[[[[[[[[[", res, "]]]]]]]]]]]]]]]]]]]]]]")
    #
    # # total_attributes = first_col.count()
    # # # ranges = first_col.groupby(pd.cut(df[features[0]], all_ranges[0])).count()
    # # # print("777777   -----> ", all_ranges[0])
    # # ranges_2 = first_col.value_counts(bins=all_ranges[0], sort=False)
    # # print(ranges_2)
    # # # print("yyyyyyyyyyyyyyyyyyyyyy")
    # # print("total_attributes =  ", total_attributes)
    # # entropy = sum([-count / total_attributes * math.log(count / total_attributes, 2)
    # #                if count else 0
    # #                for count in ranges_2])
    # #
    # # print("entropy  =   ", entropy)
    #
    # # ent = 0
    # # for count in ranges_2:
    # #     ent += -count / total_attributes * math.log(count / total_attributes, 2)
    # #     print("---> ", count)
    # # print("ent  =   ", ent)
    #
    #
    # # print(first_col.groupby(pd.cut(df[features[0]], ranges[0])).count())
    # # print("====", df[df.Outcome == 1].count()[1])
    # # true = df[df.Outcome == 1].count()[1]
    # # print ("===1==  ", true)
    # # false = df[df.Outcome == 0].count()[1]
    # # print ("===0==  ", false)
    # #
    # # res = true > false
    # # print ("....res", res)
    # # print(sorted(df.Pregnancies.unique()))