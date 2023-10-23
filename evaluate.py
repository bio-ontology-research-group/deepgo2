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
from deepgo.utils import FUNC_DICT, Ontology, NAMESPACES, EXP_CODES
from deepgo.metrics import compute_roc, evaluate_annotations

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


@ck.command()
@ck.option(
    '--data-root', '-dr', default='data',
    help='Prediction model')
@ck.option(
    '--ont', '-ont', default='mf',
    help='Prediction model')
@ck.option(
    '--model', '-m', default='deepgozero',
    help='Prediction model')
@ck.option(
    '--combine', '-c', is_flag=True,
    help='Prediction model')
@ck.option(
    '--alpha', '-a', default=0.50,
    help='Combining weight')
@ck.option(
    '--num-preds', '-np', default=50,
    help='Combining weight')
def main(data_root, ont, model, combine, alpha, num_preds):
    train_data_file = f'{data_root}/{ont}/train_data.pkl'
    valid_data_file = f'{data_root}/{ont}/valid_data.pkl'
    test_data_file = f'{data_root}/{ont}/predictions_{model}.pkl'
    terms_file = f'{data_root}/{ont}/terms.pkl'
    go = Ontology(f'data/go.obo', with_rels=True)
    terms_df = pd.read_pickle(terms_file)
    terms = terms_df['gos'].values.flatten()
    terms_dict = {v: i for i, v in enumerate(terms)}

    train_df = pd.read_pickle(train_data_file)
    valid_df = pd.read_pickle(valid_data_file)
    train_df = pd.concat([train_df, valid_df])
    test_df = pd.read_pickle(test_data_file)
    # diam_df = pd.read_pickle(diam_data_file)
    
    annotations = train_df['prop_annotations'].values
    annotations = list(map(lambda x: set(x), annotations))
    test_annotations = test_df['prop_annotations'].values
    test_annotations = list(map(lambda x: set(x), test_annotations))
    go.calculate_ic(annotations + test_annotations)

    # Print IC values of terms
    ics = {}
    for i, term in enumerate(terms):
        ics[term] = go.get_ic(term)
    
    # Combine scores for diamond and deepgo
    eval_preds = []
    
    for i, row in enumerate(test_df.itertuples()):
        preds = row.preds
        eval_preds.append(preds)

    eval_preds = np.concatenate(eval_preds).reshape(-1, len(terms))

    fmax, smin, tmax, wfmax, wtmax, avg_auc, avgic, fmax_spec_match = compute_metrics(
        test_df, go, terms_dict, eval_preds)
    print(model, ont)
    print(f'Fmax: {fmax:0.3f}, Smin: {smin:0.3f}, threshold: {tmax}, spec: {fmax_spec_match}')
    print(f'WFmax: {wfmax:0.3f}, threshold: {wtmax}')
    print(f'AUC: {avg_auc:0.3f}')
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    sorted_index = np.argsort(recalls)
    recalls = recalls[sorted_index]
    precisions = precisions[sorted_index]
    aupr = np.trapz(precisions, recalls)
    print(f'AUPR: {aupr:0.3f}')
    print(f'AVGIC: {avgic:0.3f}')


if __name__ == '__main__':
    main()
