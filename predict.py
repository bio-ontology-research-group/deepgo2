#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd
import time
import math
from utils import Ontology, NAMESPACES
import gzip
from deepgozero_esm import DGZeroModel, load_normal_forms
from extract_esm import extract_esm
from pathlib import Path
import torch as th
from Bio import SeqIO

@ck.command()
@ck.option('--in-file', '-if', help='Input FASTA file', required=True)
@ck.option('--out-file', '-of', default='results.tsv', help='Output result file')
@ck.option(
    '--data-root', '-dr', default='data',
    help='Prediction model')
@ck.option('--ont', '-ont', default='mf', help='GO subontology')
@ck.option('--threshold', '-t', default=0.1, help='Prediction threshold')
@ck.option('--batch-size', '-bs', default=32, help='Batch size for prediction model')
@ck.option(
    '--device', '-d', default='cuda:0',
    help='Device')
def main(in_file, out_file, data_root, ont, threshold, batch_size, device):

    # Extract ESM features
    esm_dir = Path(f'{data_root}/esm')
    proteins, data = extract_esm(in_file, output_dir=esm_dir)
    
    # Load GO and read list of all terms
    go_file = f'{data_root}/go.obo'
    go_norm = f'{data_root}/go.norm'
    model_file = f'{data_root}/{ont}/deepgozero_esm.th'
    terms_file = f'{data_root}/{ont}/terms.pkl'
    go = Ontology(go_file, with_rels=True)

    terms_df = pd.read_pickle(terms_file)
    terms = terms_df['gos'].values.flatten()
    terms_dict = {v: i for i, v in enumerate(terms)}

    n_terms = len(terms_dict)
    
    nf1, nf2, nf3, nf4, relations, zero_classes = load_normal_forms(
        go_norm, terms_dict)
    n_rels = len(relations)
    n_zeros = len(zero_classes)
    

    model = DGZeroModel(2560, n_terms, n_zeros, n_rels, device).to(device)
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
                preds = np.append(preds, logits.detach().cpu().numpy())
        preds = preds.reshape(-1, n_terms)
        
    with gzip.open(out_file, 'w') as f:
        for i in range(len(proteins)):
            for j in range(n_terms):
                if preds[i, j] > threshold:
                    f.write(f'{proteins[i].id}\t{terms[j]}\t{preds[i,j]:0.3f}\n') 

if __name__ == '__main__':
    main()
