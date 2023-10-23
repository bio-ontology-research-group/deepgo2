#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd
import time
import math
from utils import Ontology, NAMESPACES
import gzip
from deepgo.models import DeepGOModel
from deepgo.data import load_normal_forms
from deepgo.extract_esm import extract_esm
from pathlib import Path
import torch as th
from Bio import SeqIO
import os

@ck.command()
@ck.option('--in-file', '-if', help='Input FASTA file', required=True)
@ck.option(
    '--data-root', '-dr', default='data',
    help='Prediction model')
@ck.option('--threshold', '-t', default=0.1, help='Prediction threshold')
@ck.option('--batch-size', '-bs', default=32, help='Batch size for prediction model')
@ck.option(
    '--device', '-d', default='cuda:0',
    help='Device')
def main(in_file, data_root, threshold, batch_size, device):

    # Extract ESM features
    fn = os.path.splitext(in_file)[0]
    out_file_esm = f'{fn}_esm.pkl'
    proteins, data = extract_esm(in_file, out_file=out_file_esm)
    # proteins, data = load_esm(in_file)
    
    # Load GO and read list of all terms
    go_file = f'{data_root}/go-plus.obo'
    go_norm = f'{data_root}/go-plus.norm'
    go = Ontology(go_file, with_rels=True)
    ent_models = {
        'mf': [3, 6, 9, 16, 0, 4, 13, 14, 7, 12],
        'bp': [18, 6, 13, 1, 10, 0, 12, 5, 14, 8],
        'cc': [6, 12, 14, 0, 13, 18, 19, 2, 7, 8]
    }
    for ont in ['mf', 'cc', 'bp']:
        terms_file = f'{data_root}/{ont}/terms.pkl'
        out_file = f'{fn}_preds_{ont}.tsv.gz'
        terms_df = pd.read_pickle(terms_file)
        terms = terms_df['gos'].values.flatten()
        terms_dict = {v: i for i, v in enumerate(terms)}

        n_terms = len(terms_dict)

        nf1, nf2, nf3, nf4, relations, zero_classes = load_normal_forms(
            go_norm, terms_dict)
        n_rels = len(relations)
        n_zeros = len(zero_classes)
        sum_preds = np.zeros((len(proteins), n_terms), dtype=np.float32)
        model = DeepGOModel(2560, n_terms, n_zeros, n_rels, device).to(device)
        for mn in ent_models[ont]:
            model_file = f'{data_root}/{ont}/deepgozero_esm_plus_{mn}.th'
            model.load_state_dict(th.load(model_file, map_location=device))
            model.eval()
            
            with th.no_grad():
                steps = int(math.ceil(len(proteins) / batch_size))
                preds = []
                with ck.progressbar(length=steps, show_pos=True) as bar:
                    for i in range(steps):
                        bar.update(1)
                        start, end = i * batch_size, (i + 1) * batch_size
                        batch_features = data[start:end].to(device)
                        logits = model(batch_features)
                        preds.append(logits.detach().cpu().numpy())
                preds = np.concatenate(preds)
            sum_preds += preds
        preds = sum_preds / len(ent_models[ont])
        go_ind = [None] * len(proteins)
        scores = [None] * len(proteins)
        with gzip.open(out_file, 'wt') as f:
            for i in range(len(proteins)):
                above_threshold = np.argwhere(preds[i] >= threshold).flatten()
                for j in above_threshold:
                    name = go.get_term(terms[j])['name']
                    f.write(f'{proteins[i]}\t{terms[j]}\t{preds[i,j]:0.3f}\n')
       
def load_esm(in_file):
    npz_data = np.load(in_file)
    n = len(npz_data.keys())
    data = th.zeros((n, 2560), dtype=th.float32)
    keys = list(npz_data.keys())
    proteins = [None] * n
    for i, key in enumerate(keys):
        emb = npz_data.get(key)
        proteins[i] = key.split('/')[-1]
        data[i, :] = th.FloatTensor(emb)
    return proteins, data
    
if __name__ == '__main__':
    main()
