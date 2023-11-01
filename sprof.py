#!/usr/bin/env python

import numpy as np
import pandas as pd
import click as ck
import sys
from collections import deque
import time
import logging
from deepgo.utils import FUNC_DICT, Ontology, NAMESPACES
import os

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


@ck.command()
@ck.option(
    '--data-root', '-dr', default='data',
    help='Prediction model')
@ck.option(
    '--ont', '-ont', default='mf',
    help='Prediction model')
def main(data_root, ont):
    train_data_file = f'{data_root}/{ont}/train_data.pkl'
    valid_data_file = f'{data_root}/{ont}/valid_data.pkl'
    test_data_file = f'{data_root}/{ont}/nextprot_data.pkl'
    terms_file = f'{data_root}/{ont}/terms.pkl'
    
    go_rels = Ontology(f'data/go.obo', with_rels=True)
    terms_df = pd.read_pickle(terms_file)
    terms = terms_df['gos'].values.flatten()
    terms_dict = {v: i for i, v in enumerate(terms)}
    print(ont, len(terms))
    
    train_df = pd.read_pickle(train_data_file)
    valid_df = pd.read_pickle(valid_data_file)
    test_df = pd.read_pickle(test_data_file)
    
    annotations = train_df['prop_annotations'].values
    annotations = list(map(lambda x: set(x), annotations))
    
    prot_index = {}
    for i, row in enumerate(test_df.itertuples()):
        prot_index[row.proteins] = i

    with open(f'data/sprof/nextprot_{ont}/nextprot_data_all_preds.txt') as f:
        data = f.read().split('\n\n')    
        headers = data[0].splitlines()
        ont_ind = 2
        if ont == 'bp':
            ont_ind = 4
        elif ont == 'cc':
            ont_ind = 6
        sprof_terms = headers[ont_ind + 1].split('; ')
        print(sprof_terms[:10])
        print(len(set(sprof_terms).intersection(set(terms_dict))))
        sprof_preds = {}
        for item in data[1:]:
            it = item.splitlines()
            if len(it) == 0:
                continue
            prot_id = it[0]
            scores = it[ont_ind].split('; ')
            preds = {}
            for go_id, score in zip(sprof_terms, scores):
                score = float(score)
                if score >= 0.01:
                    preds[go_id] = score
            sprof_preds[prot_id] = preds
    preds = []
    for i, row in enumerate(test_df.itertuples()):
        prop_annots = sprof_preds[row.proteins]
        pred_scores = np.zeros(len(terms), dtype=np.float32)
        for i, go_id in enumerate(terms):
            if go_id in prop_annots:
                pred_scores[i] = prop_annots[go_id]
        preds.append(pred_scores)
    test_df['preds'] = preds
    test_df.to_pickle(f'{data_root}/{ont}/nextprot_predictions_sprof.pkl')
    
if __name__ == '__main__':
    main()
