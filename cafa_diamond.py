#!/usr/bin/env python

import numpy as np
import pandas as pd
import click as ck
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity
import sys
from collections import deque
import time
import logging
from sklearn.metrics import roc_curve, auc, matthews_corrcoef
from scipy.spatial import distance
from scipy import sparse
import math
from utils import FUNC_DICT, Ontology, NAMESPACES
from matplotlib import pyplot as plt
import os

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


@ck.command()
@ck.option(
    '--input-file', '-if', default='data/cafa5/diamond.res',
    help='Diamond results')
@ck.option(
    '--output-file', '-of', default='data/cafa5/diamond_preds.tsv',
    help='Diamond results')
def main(input_file, output_file):
    data_file = f'data/swissprot_exp_2022_04.pkl'
    go_rels = Ontology(f'data/go.obo', with_rels=True)
    
    df = pd.read_pickle(data_file)
    
    annotations = df['prop_annotations'].values
    annotations = list(map(lambda x: set(x), annotations))
    
    prot_index = {}
    for i, row in enumerate(df.itertuples()):
        prot_index[row.proteins] = i

    
    diamond_scores = {}
    with open(input_file) as f:
        for line in f:
            it = line.strip().split()
            if it[0] not in diamond_scores:
                diamond_scores[it[0]] = {}
            diamond_scores[it[0]][it[1]] = float(it[2])
            
    with open(output_file, 'w') as f:
        diam_preds = []
        for prot_id, scores in diamond_scores.items():
            annots = {}
            prop_annots = {}
            sim_prots = scores
            allgos = set()
            total_score = 0.0
            for p_id, score in sim_prots.items():
                allgos |= annotations[prot_index[p_id]]
                total_score += score
            allgos = list(sorted(allgos))
            sim = np.zeros(len(allgos), dtype=np.float32)
            for j, go_id in enumerate(allgos):
                s = 0.0
                for p_id, score in sim_prots.items():
                    if go_id in annotations[prot_index[p_id]]:
                        s += score
                sim[j] = s / total_score
            ind = np.argsort(-sim)
            for go_id, score in zip(allgos, sim):
                annots[go_id] = score

            prop_annots = annots.copy()
            for go_id, score in annots.items():
                for sup_go in go_rels.get_anchestors(go_id):
                    if sup_go in prop_annots:
                        prop_annots[sup_go] = max(prop_annots[sup_go], score)
                    else:
                        prop_annots[sup_go] = score
            for go_id, score in prop_annots.items():
                f.write(f'{prot_id}\t{go_id}\t{score}\n')

        
if __name__ == '__main__':
    main()
