#!/usr/bin/env python
import os
import sys
sys.path.append('.')

import click as ck
import numpy as np
import pandas as pd
import gzip
import logging
import torch as th


#DGL imports
import dgl


@ck.command()
@ck.option(
    '--string-db-actions-file', '-sdb', default='data/protein.actions.v11.0.txt.gz'),
    help='String Database Actions file')
@ck.option(
    '--data-file', '-df', default='data/swissprot_exp.pkl'),
    help='Swissprot pandas DataFrame')
def main(string_db_actions_file, data_file):
    df = pd.read_pickle(data_file)
    proteins = df['proteins']
    prot_idx = {v: k for k, v in enumerate(proteins)}

    mapping = {}
    for i, row in enumerate(df.itertuples()):
        for st_id in row.string_ids:
            mapping[st_id] = row.proteins
    relations = {}
    inters = {}
    with gzip.open(string_db_actions_file, 'rt') as f:
        next(f)
        for line in f:
            it = line.strip().split('\t')
            p1, p2 = it[0], it[1]
            if p1 not in mapping or p2 not in mapping:
                continue
            score = int(it[6])
            if score < 700:
                continue
            p1, p2 = mapping[p1], mapping[p2]
            rel = it[2]
            if rel not in relations:
                relations[rel] = len(relations)
            is_dir = it[4] == 't'
            a_is_act = it[5] == 't'
            if p1 not in inters:
                inters[p1] = set()
            inters[p1].add((rel, p2))
    interactions = []
    for i, row in enumerate(df.itertuples()):
        p_id = row.proteins
        if p_id in inters:
            interactions.append(inters[p_id])
        else:
            interactions.append([])
    df['interactions'] = interactions
    df.to_pickle(data_file)
            
if __name__ == '__main__':
    main()
