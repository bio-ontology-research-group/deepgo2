#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd
import gzip
import os
import torch as th

from collections import Counter
from aminoacids import MAXLEN, to_ngrams
import logging
import json

from sklearn.metrics import roc_curve, auc, matthews_corrcoef
from utils import get_goplus_defs, Ontology, NAMESPACES

logging.basicConfig(level=logging.INFO)

from deepgoel import DGELModel, load_normal_forms
from torch_utils import FastTensorDataLoader

ont = 'mf'

@ck.command()
@ck.option(
    '--data-root', '-dr', default=f'data/',
    help='Data root')
@ck.option(
    '--ont', '-ont', default='mf',
    help='Subontology')
@ck.option(
    '--data-file', '-df', default=f'swissprot.pkl',
    help='Pandas pkl file with proteins and their interpo annotations')
@ck.option(
    '--device', '-d', default='cuda:1',
    help='Device')
def main(data_root, ont, data_file, device):
    terms_file = f'{data_root}/{ont}/terms.pkl'
    model_file = f'{data_root}/{ont}/deepgozero.th'
    go = Ontology(f'{data_root}/go.obo', with_rels=True)

    # Load interpro data
    df = pd.read_pickle(data_file)
    terms_df = pd.read_pickle(terms_file)
    terms = terms_df['gos'].values.flatten()
    terms_dict = {v: i for i, v in enumerate(terms)}
    
    ipr_df = pd.read_pickle(f'{data_root}/{ont}/interpros.pkl')
    iprs = ipr_df['interpros'].values
    iprs_dict = {v:k for k, v in enumerate(iprs)}

    nf1, nf2, nf3, nf4, rels_dict, zero_classes = load_normal_forms(
        f'{data_root}/go.norm', terms_dict)

    defins = get_goplus_defs(f'{data_root}/definitions_go.txt')
    zero_terms = [term for term in zero_classes if term in defins and go.get_namespace(term) == NAMESPACES[ont]]
    print(len(zero_terms))
    
    net = DGELModel(len(iprs_dict), len(terms), len(zero_classes), len(rels_dict), device).to(device)
    # Loading best model
    print('Loading the best model')
    net.load_state_dict(th.load(model_file))
    net.eval()

    zero_terms_dict = {v: k for k, v in enumerate(zero_terms)}
    data = get_data(df, iprs_dict, zero_terms_dict)
    # data = data.to(device)
    batch_size = 1000
    data_loader = FastTensorDataLoader(*data, batch_size=batch_size, shuffle=False)

    go_data = th.zeros(len(zero_terms), dtype=th.long).to(device)
    for i, term in enumerate(zero_terms):
        go_data[i] = zero_classes[term]
    scores = np.zeros((data[0].shape[0], len(zero_terms)), dtype=np.float32)
    for i, batch_data in enumerate(data_loader):
        batch_data, _ = batch_data
        zero_score = net.predict_zero(
            batch_data.to(device), go_data).cpu().detach().numpy()
        scores[i * batch_size: (i + 1) * batch_size] = zero_score
    
    for i, row in enumerate(df.itertuples()):
        for j, go_id in enumerate(zero_terms):
            if scores[i, j] >= 0.01:
                print(row.proteins, go_id, scores[i, j])
                
def compute_roc(labels, preds):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)
    return roc_auc, fpr, tpr


def compute_fmax(labels, preds):
    fmax = 0.0
    pmax = 0
    rmax = 0
    patience = 0
    precs = []
    recs = []
    for t in range(0, 101):
        threshold = t / 100.0
        predictions = (preds >= threshold).astype(np.float32)
        tp = np.sum(labels * predictions, axis=1)
        fp = np.sum(predictions, axis=1) - tp
        fn = np.sum(labels, axis=1) - tp
        tp_ind = tp > 0
        tp = tp[tp_ind]
        fp = fp[tp_ind]
        fn = fn[tp_ind]
        if len(tp) == 0:
            continue
        p = np.mean(tp / (tp + fp))
        r = np.sum(tp / (tp + fn)) / len(tp_ind)
        precs.append(p)
        recs.append(r)
        f = 2 * p * r / (p + r)
        if fmax <= f:
            fmax = f
    return fmax, precs, recs


def get_data(df, iprs_dict, terms_dict):
    data = th.zeros((len(df), len(iprs_dict)), dtype=th.float32)
    labels = th.zeros((len(df), len(terms_dict)), dtype=th.float32)
    for i, row in enumerate(df.itertuples()):
        for ipr in row.interpros:
            if ipr in iprs_dict:
                data[i, iprs_dict[ipr]] = 1
        for go_id in row.prop_annotations:
            if go_id in terms_dict:
                g_id = terms_dict[go_id]
                labels[i, g_id] = 1
    return data, labels

if __name__ == '__main__':
    main()
