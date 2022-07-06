#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd
import gzip
import logging
from utils import NAMESPACES, Ontology, get_goplus_defs
from collections import Counter
import json
from gensim.models import KeyedVectors

import torch as th

logging.basicConfig(level=logging.INFO)


@ck.command()
@ck.option(
    '--data-file', '-df', default='data/swissprot_exp_esm_dl2vec.pkl',
    help='DataFrame with proteins, sequences and annotations')
@ck.option(
    '--out-file', '-of', default='data/swissprot_exp_esm_dl2vec_text.pkl',
    help='with DL2Vec features')
def main(data_file, out_file):
    df = pd.read_pickle(data_file)

    sim = {}
    with open('data/swissprot_exp.sim') as f:
        for line in f:
            it = line.strip().split('\t')
            p1, p2, score = it[0], it[1], float(it[2]) / 100.0
            if p1 == p2:
                continue
            if p1 not in sim:
                sim[p1] = []
            sim[p1].append(p2)

    text = []
    embeds = pd.read_pickle(f'data/text_vectors.pkl')

    print(len(embeds))
    missing = 0
    for i, row in enumerate(df.itertuples()):
        p_id = row.proteins
        if p_id in embeds:
            embed = embeds[p_id]
        elif p_id in sim:
            ok = False
            for pt_id in sim[p_id]:
                if pt_id in embeds:
                    embed = embeds[pt_id]
                    ok = True
                    break

            if not ok:
                missing += 1
                embed = np.zeros(100, dtype=np.float32)
        else:
            missing += 1
            embed = np.zeros(100, dtype=np.float32)
        text.append(embed)

    print('Missing', missing)
    df[f'text'] = text
    df.to_pickle(out_file)
    
if __name__ == '__main__':
    main()
