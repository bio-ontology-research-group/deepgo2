#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd
import gzip
import logging
from utils import NAMESPACES, Ontology, get_goplus_defs
from collections import Counter
import json

import torch as th

logging.basicConfig(level=logging.INFO)


@ck.command()
@ck.option(
    '--data-file', '-df', default='data/swissprot_exp.pkl',
    help='DataFrame with proteins, sequences and annotations')
@ck.option(
    '--out-file', '-of', default='data/swissprot_exp_esm.pkl',
    help='with ESM features')
def main(data_file, out_file):
    df = pd.read_pickle(data_file)

    esm = []
    data_root = 'data/swissprot_esm1b/'
    with ck.progressbar(length=len(df), show_pos=True) as bar:
        for i, row in df.iterrows():
            bar.update(1)
            filename = row.proteins + '.pt'
            data = th.load(data_root + filename)
            emb = data['mean_representations'][33].numpy()
            esm.append(emb)
    df['esm'] = esm
    df.to_pickle(out_file)
    
if __name__ == '__main__':
    main()
