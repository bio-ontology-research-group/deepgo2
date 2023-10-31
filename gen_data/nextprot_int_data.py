#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd
import time
import math
from utils import Ontology, NAMESPACES
import gzip
from extract_esm import extract_esm
from pathlib import Path
import torch as th
from Bio import SeqIO

@ck.command()
@ck.option(
    '--data-root', '-dr', default='data',
    help='Data root folder')
def main(data_root):
    sw_df = pd.read_pickle('data/swissprot_exp_esm2.pkl')
    mapping = {}
    for i, row in enumerate(sw_df.itertuples()):
        for st_id in row.string_ids:
            mapping[st_id] = row.proteins
    
    df = pd.read_pickle('data/nextprot.pkl')
    for i, row in enumerate(df.itertuples()):
        for st_id in row.string_ids:
            mapping[st_id] = row.proteins
    
    string_ids = set()
    for ids in df['string_ids']:
        string_ids |= set(ids)
    inters = {}
    relations = {'binding': 0, 'catalysis': 1, 'reaction': 2}
    with gzip.open('data/9606.protein.actions.v11.0.txt.gz', 'rt') as f:
        next(f)
        for line in f:
            it = line.strip().split('\t')
            p1, p2 = it[0], it[1]
            score = int(it[6])
            if score < 300:
                continue
            if p1 not in string_ids:
                continue
            if p2 not in mapping:
                continue
            p1, p2 = mapping[p1], mapping[p2]
            rel = it[2]
            if p1 not in inters:
                inters[p1] = set()
            inters[p1].add((rel, p2))
    interactions = []
    for row in df.itertuples():
        prot_id = row.proteins
        if prot_id in inters:
            interactions.append(inters[prot_id])
        else:
            interactions.append([])
    df['interactions'] = interactions
    df.to_pickle('data/nextprot_interactions.pkl')

    print(inters)

if __name__ == '__main__':
    main()
