#!/usr/bin/env python

import numpy as np
import pandas as pd
import click as ck
import sys
from collections import deque
import time
import logging
import math
from deepgo.utils import FUNC_DICT, Ontology, NAMESPACES
import os

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


ont = 'mf'

@ck.command()
@ck.option(
    '--test-data-file', '-tsdf', default=f'data/{ont}/nextprot_data.pkl',
    help='Test data file')
@ck.option(
    '--terms-file', '-tf', default=f'data/{ont}/terms.pkl',
    help='Data file with sequences and complete set of annotations')
@ck.option(
    '--tale-scores-file', '-tsf', default=f'data/tale/deepgo2_nextprot_{ont}.txt',
    help='TALE predictions')
@ck.option(
    '--out_file', '-of', default=f'data/{ont}/nextprot_predictions_tale.pkl', help='Output file')
def main(test_data_file, terms_file,
         tale_scores_file, out_file):

    go_rels = Ontology('data/go.obo', with_rels=True)
    terms_df = pd.read_pickle(terms_file)
    terms = terms_df['gos'].values.flatten()
    terms_dict = {v: i for i, v in enumerate(terms)}

    test_df = pd.read_pickle(test_data_file)
    
    
    tale_scores = {}
    with open(tale_scores_file) as f:
        for line in f:
            it = line.strip().split()
            p_id, go_id, score = it[0], it[1][2:-2], float(it[-1])
            if p_id not in tale_scores:
                tale_scores[p_id] = {}
            tale_scores[p_id][go_id] = score
    preds = []
    print('Tale preds')
    
    for i, row in enumerate(test_df.itertuples()):
        annots = {}
        prop_annots = {}
        prot_id = row.proteins
        # BlastKNN
        if prot_id in tale_scores:
            annots = tale_scores[prot_id]
            prop_annots = annots.copy()
            for go_id, score in annots.items():
                for sup_go in go_rels.get_ancestors(go_id):
                    if sup_go in prop_annots:
                        prop_annots[sup_go] = max(prop_annots[sup_go], score)
                    else:
                        prop_annots[sup_go] = score
        pred_scores = np.zeros(len(terms), dtype=np.float32)
        for i, go_id in enumerate(terms):
            if go_id in prop_annots:
                pred_scores[i] = prop_annots[go_id]

        preds.append(pred_scores)

    test_df['preds'] = preds
    test_df.to_pickle(out_file)

if __name__ == '__main__':
    main()
