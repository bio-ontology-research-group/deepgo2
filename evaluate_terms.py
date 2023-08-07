#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd
import gzip
import os

from collections import Counter
import logging

from sklearn.metrics import roc_curve, auc, matthews_corrcoef
from utils import Ontology, MOLECULAR_FUNCTION, CELLULAR_COMPONENT, BIOLOGICAL_PROCESS

logging.basicConfig(level=logging.INFO)


@ck.command()
@ck.option(
    '--data-root', '-dr', default='data',
    help='Prediction model')
@ck.option(
    '--ont', '-ont', default='mf',
    help='Prediction model')
@ck.option(
    '--model', '-m', default='mlp',
    help='Prediction model')
@ck.option(
    '--combine', '-c', is_flag=True,
    help='Prediction model')
def main(data_root, ont, model, combine):
    train_data_file = f'{data_root}/{ont}/train_data.pkl'
    valid_data_file = f'{data_root}/{ont}/valid_data.pkl'
    test_data_file = f'{data_root}/{ont}/predictions_{model}.pkl'
    terms_file = f'{data_root}/{ont}/terms.pkl'
    go_rels = Ontology(f'{data_root}/go.obo', with_rels=True)
    terms_df = pd.read_pickle(terms_file)
    terms = terms_df['gos'].values.flatten()
    terms_dict = {v: i for i, v in enumerate(terms)}

    train_df = pd.read_pickle(train_data_file)
    valid_df = pd.read_pickle(valid_data_file)
    train_df = pd.concat([train_df, valid_df])
    test_df = pd.read_pickle(test_data_file)
    
    annotations = train_df['prop_annotations'].values
    annotations = list(map(lambda x: set(x), annotations))
    test_annotations = test_df['prop_annotations'].values
    test_annotations = list(map(lambda x: set(x), test_annotations))
    go_rels.calculate_ic(annotations + test_annotations)

    # Print IC values of terms
    ics = {}
    for term in terms:
        ics[term] = go_rels.get_ic(term)


    preds = np.empty((len(test_df), len(terms)), dtype=np.float32)
    labels = np.zeros((len(test_df), len(terms)), dtype=np.float32)
        
    for i, row in enumerate(test_df.itertuples()):
        preds[i, :] = row.preds
        for go_id in row.prop_annotations:
            if go_id in terms_dict:
                labels[i, terms_dict[go_id]] = 1

    total_n = 0
    total_sum = 0
    aucs = []
    anns = []
    for go_id, i in terms_dict.items():
        if ics[go_id] < 0.5:
            continue
        pos_n = np.sum(labels[:, i])
        if pos_n > 0 and pos_n < len(test_df):
            total_n += 1
            roc_auc, fpr, tpr = compute_roc(labels[:, i], preds[:, i])
            print(go_id, roc_auc)
            total_sum += roc_auc
    print(f'Average AUC for {ont} {total_sum / total_n:.3f}')
        
def compute_roc(labels, preds):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)

    return roc_auc, fpr, tpr

if __name__ == '__main__':
    main()
