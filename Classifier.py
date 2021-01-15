# fit(x,y)
# predict(x)
import numpy as np
import pandas as pd
import sklearn, random, matplotlib

# csv_reader = csv.reader(csv_file)
# class dr_dump_action_qp(dr_obj):
#     def __init__(self, data):
#         keys = ["dr_dump_rec_type", "id", "rule_id", "qp_num"]
#         self.data = dict(zip(keys, data))

from sklearn import *

df = pd.read_csv("train.csv")
print(df)

def MajorityClass(examples):

    return []


def TDIDT(examples, features, default_value, select_feature):
    if len(examples) == 0:
        return (None, [], default_value)

    majority = MajorityClass(examples)

    if len(features) == 0:
        return None, [], majority

    is_consistent_node = True
    for example in examples:
        if example.output != majority:
            is_consistent_node = False

    if is_consistent_node:
        return None, [], majority

    function = select_feature(examples, features)
    # todo: define subtrees while the values are רציפים
    subtrees = None

    return function, subtrees, majority


def DT_Classify(obj, tree):
    if len(tree.children) == 0:
        return tree.classfication

    for child in tree.children:
        if tree.feature(obj) == child.value:
            return DT_Classify(obj, child)


""" ---------------------------------------------   today
 """





