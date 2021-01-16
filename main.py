import numpy as np
import pandas as pd
import math
from ID3 import ID3_Classifier


if __name__ == "__main__":

    data = pd.read_csv("amani.csv")
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

    test = pd.read_csv("amani_test.csv")
    test_features = np.array(test.drop('Outcome', axis=1).copy())
    test_outcome = np.array(test['Outcome'].copy())
    match_cnt = 0
    none_cnt = 0
    for i in range(len(test_features)):
        res = id3_tree.predict(test_features[i])
        print ("res : ", res, "  predict : ", test_outcome[i])
        if res == test_outcome[i]:
            match_cnt += 1
        if res is None:
            none_cnt += 1

    print("percision = ", match_cnt/len(test_features))
    print("none_cnt  = ", none_cnt)

