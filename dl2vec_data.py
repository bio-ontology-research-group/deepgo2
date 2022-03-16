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
    '--data-file', '-df', default='data/swissprot_exp_esm.pkl',
    help='DataFrame with proteins, sequences and annotations')
@ck.option(
    '--out-file', '-of', default='data/swissprot_exp_esm_dl2vec.pkl',
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

    id_map = {}
    for i, row in enumerate(df.itertuples()):
        for st_id in row.string_ids:
            id_map[st_id] = row.proteins
        
    for ont in ("cc",): #('mf', 'bp', 'cc', 'all'):
        dl2vec = []
        embeddings_file = f'data/{ont}/wv_embeddings_node2vec'
        
        embeds = {}
        wv = KeyedVectors.load(embeddings_file)
        wv = wv.wv
        for i, w in enumerate(wv.index_to_key):

            if not w.startswith('GO'):
                w = w.replace(':', '_')
                embeds[w] = wv[i]

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
                    embed = np.zeros(wv[0].shape, dtype=np.float32)
            else:
                missing += 1
                embed = np.zeros(wv[0].shape, dtype=np.float32)
            dl2vec.append(embed)

        print(ont, missing)
        df[f'{ont}_dl2vec'] = dl2vec
    df.to_pickle(out_file)
    
if __name__ == '__main__':
    main()
