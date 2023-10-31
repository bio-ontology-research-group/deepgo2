#!/usr/bin/env python

import numpy as np
import pandas as pd
import click as ck
import sys
from collections import deque
import time
import logging
import math
from deepgo.utils import FUNC_DICT, Ontology, NAMESPACES
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
    '--model-name', '-m', required=True, help='Prediction model name')
@ck.option(
    '--test-data-name', '-td', default='test', type=ck.Choice(['test', 'nextprot']),
    help='Test data set name')
@ck.option(
    '--combine', '-c', default='avg', type=ck.Choice(['avg', 'min', 'max']),
    help='Combination strategy')
@ck.option(
    '--n-models', '-nm', default=6,
    help='Top N models for semantic entailment')
def main(data_root, ont, model_name, test_data_name, combine, n_models):
    train_data_file = f'{data_root}/{ont}/train_data.pkl'
    valid_data_file = f'{data_root}/{ont}/valid_data.pkl'
    test_data_file = f'{data_root}/{ont}/{test_data_name}_predictions_{model_name}_0.pkl'
    terms_file = f'{data_root}/{ont}/terms.pkl'
    go = Ontology(f'{data_root}/go.obo', with_rels=True)
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
    go.calculate_ic(annotations + test_annotations)
    
    
    # Print IC values of terms
    ics = {}
    for term in terms:
        ics[term] = go.get_ic(term)

    eval_preds = []
    top_models = get_top_models(ont, model_name, n_models)
    print(top_models)
    
    for i in top_models: #range(6):#[0, 5, 6, 8]:
        #if i not in top_models:
        #    continue
        test_df = pd.read_pickle(f'{data_root}/{ont}/nextprot_predictions_{model_name}_{i}.pkl')
        for j, row in enumerate(test_df.itertuples()):
            if j == len(eval_preds):
                eval_preds.append(row.preds)
            else:
                if combine == 'max':
                    eval_preds[j] = np.maximum(eval_preds[j], row.preds)
                elif combine == 'min':
                    eval_preds[j] = np.minimum(eval_preds[j], row.preds)
                elif combine == 'avg':
                    eval_preds[j] = eval_preds[j] + row.preds
                else:
                    raise NotImplementedError()
                
                
    eval_preds = np.stack(eval_preds).reshape(-1, len(terms))
    if combine == 'avg':
        eval_preds /= len(top_models) # taking mean

    fmax, smin, tmax, wfmax, wtmax, avg_auc, aupr, avgic, fmax_spec_match = compute_metrics(
        test_df, go, terms_dict, terms, ont, eval_preds)

    print(ont)
    print(f'Fmax: {fmax:0.3f}, Smin: {smin:0.3f}, threshold: {tmax}, spec: {fmax_spec_match}')
    print(f'WFmax: {wfmax:0.3f}, threshold: {wtmax}')
    print(f'AUPR: {aupr:0.3f}')
    print(f'AVGIC: {avgic:0.3f}')


def get_top_models(ont, model, n_models):
    valid_losses = []
    for ind in range(10):
        with open(f'data/{ont}/valid_{model}_{ind}.pf') as f:
            lines = f.readlines()
            it = lines[-1].strip().split(', ')[0].split(' - ')
            loss = float(it[-1])
            valid_losses.append((ind, loss))
    valid_losses = sorted(valid_losses, key=lambda x: x[1])
    valid_losses = valid_losses[:n_models]
    result = [m_id for m_id, loss in valid_losses]
    print(valid_losses)
    return set(result)

if __name__ == '__main__':
    main()
