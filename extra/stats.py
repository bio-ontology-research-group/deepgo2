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
def main(data_root):
    # Get labels
    ranks = {'mf': {}, 'bp': {}}
    models = (
        'deepgozero_esm_plus', 'deepgozero_gat_plus', 'deepgocnn',
        'deepgozero', 'tale', 'sprof', 'mlp_esm', 'mlp', 'naive')
    model_names = (
        'DeepGO-SE', 'DeepGOGAT-SE', 'DeepGOCNN',
        'DeepGOZero', 'Tale', 'SPROF-GO', 'MLP (ESM2)', 'MLP', 'Naive')
    for ont in ['mf', 'bp']:
        df = pd.read_pickle(f'{data_root}/{ont}/nextprot_data.pkl')
        n_prots = len(df)
        terms_file = f'{data_root}/{ont}/terms.pkl'
        terms_df = pd.read_pickle(terms_file)
        terms = terms_df['gos'].values.flatten()
        terms_dict = {v: i for i, v in enumerate(terms)}
        labels = np.zeros((len(df), len(terms)), dtype=np.float32)
        for i, row in enumerate(df.itertuples()):
            for go_id in row.prop_annotations:
                if go_id in terms_dict:
                    labels[i, terms_dict[go_id]] = 1
        
        for model in models:
            preds = np.load(f'{data_root}/{ont}/{model}_preds.npy')
            # Compute ranks
            ranks[ont][model] = []
            for i in range(len(terms)):
                pos_prots = np.argwhere(labels[:, i] == 1).flatten()
                if len(pos_prots) == 0:
                    continue
                scores = preds[:, i]
                for p_id in pos_prots:
                    scores[p_id] = preds[p_id, i]
                    index = rankdata(-scores, method='average')
                    rank = index[p_id]
                    ranks[ont][model].append(rank)
    for i in range(2):
        for j in range(2, len(models)):
            pvals = {}
            for ont in ['mf', 'bp']:
                r1 = ranks[ont][models[i]]
                r2 = ranks[ont][models[j]]
                _, pval = stats.wilcoxon(r1, r2)
                pval *= 8
                if pval > 1:
                    pval = 1.0
                if pval >= 0.01 or pval == 0.0:
                    pval = f'{pval:0.2f}'
                else:
                    pval = f'{pval:1.0e}'.split('e')
                    a, b = pval[0], '{' + pval[1] + '}'
                    pval = f'${a} \cdot 10^{b} $'
                pvals[ont] = pval
                
            print(f'{model_names[i]} & {model_names[j]} & {pvals["mf"]} & {pvals["bp"]} \\\\')
            
    
if __name__ == '__main__':
    main()
