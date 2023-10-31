#!/usr/bin/env python

import numpy as np
import pandas as pd
import click as ck
import sys
from collections import deque, Counter
import time
import logging
from utils import FUNC_DICT, Ontology, NAMESPACES
from matplotlib import pyplot as plt

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


@ck.command()
@ck.option(
    '--data-root', '-dr', default='data',
    help='Prediction model')
@ck.option(
    '--ont', '-ont', default='mf',
    help='GO subontology (bp, mf, cc)')
def main(data_root, ont):
    train_data_file = f'{data_root}/{ont}/train_data.pkl'
    valid_data_file = f'{data_root}/{ont}/valid_data.pkl'
    out_file = f'{data_root}/{ont}/predictions_nextprot.pkl'
    
    go_rels = Ontology(f'{data_root}/go.obo', with_rels=True)
    terms_df = pd.read_pickle(f'{data_root}/{ont}/terms.pkl')

    terms = terms_df['gos'].values.flatten()
    terms_dict = {v: i for i, v in enumerate(terms)}

    train_df = pd.read_pickle(train_data_file)
    valid_df = pd.read_pickle(valid_data_file)
    train_df = pd.concat([train_df, valid_df])
    annotations = train_df['prop_annotations'].values
    annotations = list(map(lambda x: set(x), annotations))

    with open('data/nextprot/nextprot_annots.tsv') as f:
        next(f)
        annots = {}
        prop_annots = {}
        c = 0
        n = 0
        for line in f:
            it = line.strip().split('\t')
            prot_id = it[0]
            go_id = it[2]
            if go_rels.has_term(go_id):
                c += 1
                if go_rels.get_term(go_id)['namespace'] == NAMESPACES[ont]:
                    n += 1
            if prot_id not in annots:
                annots[prot_id] = set()
                prop_annots[prot_id] = set()
            annots[prot_id].add(go_id)
            prop_annots[prot_id] |= go_rels.get_anchestors(go_id)
    print(c, n)

    proteins = []
    annotations = []
    prop_annotations = []
    for prot_id, ann in annots.items():
        ok = False
        for go_id in prop_annots[prot_id]:
            if go_id in terms_dict:
                ok = True
                break
        if not ok:
            continue
        proteins.append(prot_id)
        annotations.append(list(ann))
        prop_annotations.append(list(prop_annots[prot_id]))

    preds = {}
    count = 0
    with open(f'data/nextprot/results_{ont}.tsv') as f:
        for line in f:
            it = line.strip().split('\t')
            prot_id = it[0].split('|')[1]
            go_id = it[1]
            score = float(it[2])
            if score >= 0.1 and go_id in annots[prot_id]:
                count += 1
            if prot_id not in preds:
                preds[prot_id] = {}
            preds[prot_id][go_id] = score
    print(count)
    predictions = []
    for prot_id in proteins:
        pred_scores = np.zeros((len(terms),), dtype=np.float32)
        for go_id, score in preds[prot_id].items():
            pred_scores[terms_dict[go_id]] = score
        predictions.append(pred_scores)
        
    df = pd.DataFrame({'proteins': proteins, 'annotations': annotations,
                       'prop_annotations': prop_annotations, 'preds': predictions})
    # df.to_pickle(out_file)
    print(df)

if __name__ == '__main__':
    main()
