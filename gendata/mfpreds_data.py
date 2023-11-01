#!/usr/bin/env python
import os
import sys
sys.path.append('.')

import click as ck
import numpy as np
import pandas as pd
import gzip
import logging
from deepgo.utils import NAMESPACES, Ontology
from collections import Counter
import json
import os
import math
import torch as th

from deepgo.models import DeepGOModel
from deepgo.data import load_normal_forms

from evaluate_entailment import get_top_models

logging.basicConfig(level=logging.INFO)


@ck.command()
@ck.option(
    '--data-file', '-df', default='data/nextprot_interactions.pkl',
    help='DataFrame with proteins, sequences and annotations')
@ck.option(
    '--out-file', '-of', default='data/nextprot_mf.pkl',
    help='with ESM features')
@ck.option(
    '--device', '-d', default='cuda:1',
    help='Device')
def main(data_file, out_file, device):
    df = pd.read_pickle(data_file)
    
    mf_preds = None
    go_file = f'data/go-plus.norm'
    terms_file = f'data/mf/terms.pkl'
    terms_df = pd.read_pickle(terms_file)
    terms = terms_df['gos'].values.flatten()
    terms_dict = {v: i for i, v in enumerate(terms)}
    
    go = Ontology(f'data/go.obo', with_rels=True)
    n_terms = len(terms_dict)
    
    nf1, nf2, nf3, nf4, relations, zero_classes = load_normal_forms(
        go_file, terms_dict)
    n_rels = len(relations)
    n_zeros = len(zero_classes)

    net = DGZeroModel(2560, n_terms, n_zeros, n_rels, device).to(device)

    data = th.zeros((len(df), 2560), dtype=th.float32)
    for i, row in enumerate(df.itertuples()):
        data[i, :] = th.FloatTensor(row.esm2)
    batch_size = 64
    steps = int(math.ceil(len(df) / batch_size))
    top_models = get_top_models('mf', 'deepgozero_esm_plus', 4)
    for m in top_models:
        model_file = f'data/mf/deepgozero_esm_plus_{m}.th'
        net.load_state_dict(th.load(model_file))
        net.eval()
        preds = []
        with ck.progressbar(length=steps, show_pos=True) as bar:
            for i in range(steps):
                start = i * batch_size
                end = start + batch_size
                batch_data = data[start: end].to(device)
                logits = net(batch_data)
                preds = np.append(preds, logits.detach().cpu().numpy())
                
                bar.update(1)
        if m == 0:
            mf_preds = preds
        else:
            mf_preds += preds

    mf_preds /= len(top_models)
    mf_preds = mf_preds.reshape(-1, n_terms)
    df['mf_preds'] = list(mf_preds)
    df.to_pickle(out_file)
    
if __name__ == '__main__':
    main()
