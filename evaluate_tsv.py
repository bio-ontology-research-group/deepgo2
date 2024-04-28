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
from deepgo.metrics import compute_metrics

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


@ck.command()
@ck.option(
    '--data-root', '-dr', default='data',
    help='Data folder')
@ck.option(
    '--ont', '-ont', default='mf', type=ck.Choice(['mf', 'bp', 'cc']),
    help='GO subontology')
@ck.option(
    '--annotations-file', '-a', required=True, help='Annotations')
@ck.option(
    '--test-predictions', '-tp', default='test_preds_mf.tsv',
    help='Test data set name')
def main(data_root, ont, annotations_file, test_predictions):
    train_data_file = f'{data_root}/{ont}/train_data.pkl'
    valid_data_file = f'{data_root}/{ont}/valid_data.pkl'

    predictions = {}
    with open(test_predictions) as f:
        for line in f:
            it = line.strip().split('\t')
            prot_id = it[0]
            go_id = it[1]
            score = float(it[2])
            if prot_id not in predictions:
                predictions[prot_id] = {}
            predictions[prot_id][go_id] = score
    annotations = {}
    with open(annotations_file) as f:
        for line in f:
            line = line.strip()
            if line == 'bp' or line == 'mf' or line == 'cc':
                continue
            prot_id = line
            line = next(f).strip()
            annots = set(line.split('\t'))
            if prot_id not in annotations:
                annotations[prot_id] = set()
            annotations[prot_id] |= annots
    terms_file = f'{data_root}/{ont}/terms.pkl'
    go = Ontology(f'{data_root}/go.obo', with_rels=True)
    terms_df = pd.read_pickle(terms_file)
    terms = terms_df['gos'].values.flatten()
    terms_dict = {v: i for i, v in enumerate(terms)}

    train_df = pd.read_pickle(train_data_file)
    valid_df = pd.read_pickle(valid_data_file)
    train_df = pd.concat([train_df, valid_df])

    # Filter proteins for GO subset
    proteins = []
    prop_annotations = []
    exp_annotations = []
    preds = []
    go_set = set(terms_dict)
    for prot_id, scores in predictions.items():
        if prot_id not in annotations:
            continue
        annots = go_set.intersection(annotations[prot_id])
        if len(annots) == 0:
            continue
        prop_annots = set()
        for go_id in annots:
            prop_annots |= go.get_ancestors(go_id)
        proteins.append(prot_id)
        prop_annotations.append(prop_annots)
        exp_annotations.append(annots)
        pred_scores = np.zeros((len(terms),), dtype=np.float32)
        for go_id, score in scores.items():
            pred_scores[terms_dict[go_id]] = score
        preds.append(pred_scores)

    test_df = pd.DataFrame({
        'proteins': proteins, 'preds': preds,
        'prop_annotations': prop_annotations,
        'exp_annotations': exp_annotations})
    print(test_df)
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
    # np.save(f'{data_root}/{ont}/{model_name}_preds.npy', eval_preds)
    # return
    fmax, smin, tmax, wfmax, wtmax, avg_auc, aupr, avgic, fmax_spec_match = compute_metrics(
        test_df, go, terms_dict, terms, ont, eval_preds)
    print(ont)
    print(f'Fmax: {fmax:0.3f}, Smin: {smin:0.3f}, threshold: {tmax}, spec: {fmax_spec_match}')
    print(f'WFmax: {wfmax:0.3f}, threshold: {wtmax}')
    print(f'AUC: {avg_auc:0.3f}')
    print(f'AUPR: {aupr:0.3f}')
    print(f'AVGIC: {avgic:0.3f}')


if __name__ == '__main__':
    main()
