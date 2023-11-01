#!/usr/bin/env python

import numpy as np
import pandas as pd
import click as ck
import sys
from collections import deque, Counter
import time
import logging
from deepgo.utils import FUNC_DICT, Ontology, NAMESPACES

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


@ck.command()
@ck.option(
    '--data-root', '-dr', default='data',
    help='Prediction model')
@ck.option(
    '--ont', '-ont', default='mf',
    help='GO subontology (bp, mf, cc)')
@ck.option(
    '--test-data-name', '-td', default='test', type=ck.Choice(['test', 'nextprot']),
    help='Test data set name')
def main(data_root, ont, test_data_name):
    train_data_file = f'{data_root}/{ont}/train_data.pkl'
    valid_data_file = f'{data_root}/{ont}/valid_data.pkl'
    test_data_file = f'{data_root}/{ont}/{test_data_name}_data.pkl'
    out_file = f'{data_root}/{ont}/{test_data_name}_predictions_naive.pkl'
    
    go_rels = Ontology(f'{data_root}/go.obo', with_rels=True)
    terms_df = pd.read_pickle(f'{data_root}/{ont}/terms.pkl')

    terms = terms_df['gos'].values.flatten()
    terms_dict = {v: i for i, v in enumerate(terms)}

    train_df = pd.read_pickle(train_data_file)
    valid_df = pd.read_pickle(valid_data_file)
    train_df = pd.concat([train_df, valid_df])
    annotations = train_df['prop_annotations'].values
    annotations = list(map(lambda x: set(x), annotations))

    test_df = pd.read_pickle(test_data_file)

    terms_set = set(terms_dict)
    
    cnt = Counter()
    max_n = 0
    for x in annotations:
        cnt.update(x & terms_set)
        
    max_n = cnt.most_common(1)[0][1]
    print(max_n)

    scores = {}
    for go_id, n in cnt.items():
        score = n / max_n
        scores[go_id] = score

    pred_scores = np.zeros(len(terms), dtype=np.float32)
    for i, go_id in enumerate(terms):
        if go_id in scores:
            pred_scores[i] = scores[go_id]
    preds = [pred_scores] * len(test_df)
    test_df['preds'] = preds
    test_df.to_pickle(out_file)


if __name__ == '__main__':
    main()
