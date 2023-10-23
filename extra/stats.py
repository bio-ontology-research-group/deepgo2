#!/usr/bin/env python

import numpy as np
import pandas as pd
import click as ck
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity
import sys
from collections import deque
import time
import logging
from sklearn.metrics import roc_curve, auc, matthews_corrcoef
from scipy.spatial import distance
from scipy import sparse
import math
from utils import FUNC_DICT, Ontology, NAMESPACES
from matplotlib import pyplot as plt
from collections import Counter

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


@ck.command()
@ck.option(
    '--data-root', '-dr', default='data',
    help='Prediction model')
@ck.option(
    '--ont', '-ont', default='mf',
    help='Prediction model')
@ck.option(
    '--model', '-m', default='deepgo2',
    help='Prediction model')
def main(data_root, ont, model):
    train_data_file = f'{data_root}/{ont}/train_data.pkl'
    valid_data_file = f'{data_root}/{ont}/valid_data.pkl'
    test_data_file = f'{data_root}/{ont}/test_data.pkl'
    terms_file = f'{data_root}/{ont}/terms.pkl'
    go_rels = Ontology(f'{data_root}/go.obo', with_rels=True)
    terms_df = pd.read_pickle(terms_file)
    terms = terms_df['gos'].values.flatten()
    terms_dict = {v: i for i, v in enumerate(terms)}

    train_df = pd.read_pickle(train_data_file)
    valid_df = pd.read_pickle(valid_data_file)
    # train_df = pd.concat([train_df, valid_df])
    test_df = pd.read_pickle(test_data_file)

    train = Counter()
    valid = Counter()
    test = Counter()
    for i, row in enumerate(train_df.itertuples()):
        for go_id in row.prop_annotations:
            train[go_id] += 1
    for i, row in enumerate(valid_df.itertuples()):
        for go_id in row.prop_annotations:
            valid[go_id] += 1
    for i, row in enumerate(test_df.itertuples()):
        for go_id in row.prop_annotations:
            test[go_id] += 1
    for go_id in terms:
#        if train[go_id] == 0 or valid[go_id] == 0 or test[go_id] == 0:
        print(go_id, train[go_id], valid[go_id], test[go_id])
        
    


if __name__ == '__main__':
    main()
