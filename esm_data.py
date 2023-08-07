#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd
import gzip
import logging
from utils import NAMESPACES, Ontology, get_goplus_defs
from collections import Counter
import json
import os

import torch as th

logging.basicConfig(level=logging.INFO)


@ck.command()
@ck.option(
    '--data-file', '-df', default='data/swissprot_exp_2022_04.pkl',
    help='DataFrame with proteins, sequences and annotations')
@ck.option(
    '--out-file', '-of', default='data/swissprot_exp_2022_04_esm2.pkl',
    help='with ESM features')
def main(data_file, out_file):
    df = pd.read_pickle(data_file)

    esm = []
    data_root = 'data/swissprot_exp_esm2/'
    # mis = open('data/missing.fasta', 'w')
    with ck.progressbar(length=len(df), show_pos=True) as bar:
        for i, row in df.iterrows():
            bar.update(1)
            filename = row.proteins + '.pt'
            # if not os.path.exists(data_root + filename):
            #     mis.write('>' + row.proteins + '\n')
            #     mis.write(row.sequences + '\n')
            #     print('Missing')
            data = th.load(data_root + filename)
            emb = data['mean_representations'][36].numpy()
            esm.append(emb)
    df['esm2'] = esm
    df.to_pickle(out_file)
    
if __name__ == '__main__':
    main()
