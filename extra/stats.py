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
from scipy import stats
from scipy.stats import rankdata

@ck.command()
@ck.option(
    '--data-root', '-dr', default='data',
    help='Prediction model')
@ck.option(
    '--ont', '-ont', default='mf',
    help='Prediction model')
def main(data_root, ont):
    terms_file = f'{data_root}/{ont}/terms.pkl'
    terms_df = pd.read_pickle(terms_file)
    terms = terms_df['gos'].values.flatten()
    terms_dict = {v: i for i, v in enumerate(terms)}

    # Get labels
    df = pd.read_pickle(f'{data_root}/{ont}/nextprot_data.pkl')
    labels = np.zeros((len(df), len(terms)), dtype=np.float32)
    for i, row in enumerate(df.itertuples()):
        for go_id in row.prop_annotations:
            if go_id in terms_dict:
                labels[i, terms_dict[go_id]] = 1
    n_prots = len(df)
    ranks = {}
    models = (
        'deepgozero_esm_plus', 'deepgozero_gat_plus', 'deepgocnn',
        'deepgozero', 'tale', 'sprof', 'mlp', 'naive')
    model_names = (
        'DeepGO-SE', 'DeepGOGAT-SE', 'DeepGOCNN',
        'DeepGOZero', 'Tale', 'SPROF-GO', 'MLP', 'Naive')
    for model in models:
        preds = np.load(f'{data_root}/{ont}/{model}_preds.npy')
        # Compute ranks
        ranks[model] = []
        for i in range(len(terms)):
            pos_prots = np.argwhere(labels[:, i] == 1).flatten()
            if len(pos_prots) == 0:
                continue
            scores = preds[:, i]
            for p_id in pos_prots:
                scores[p_id] = preds[p_id, i]
                index = rankdata(-scores, method='average')
                rank = index[p_id]
                ranks[model].append(rank)
    for i in range(2):
        for j in range(2, len(models)):
            r1 = ranks[models[i]]
            r2 = ranks[models[j]]
            _, pval = stats.wilcoxon(r1, r2)
            pval *= 7
            print(f'{model_names[i]} & {model_names[j]} & {pval:1.0e} \\\\')
            
    
if __name__ == '__main__':
    main()
