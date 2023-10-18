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
from utils import FUNC_DICT, Ontology, NAMESPACES, EXP_CODES

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


@ck.command()
@ck.option(
    '--data-root', '-dr', default='data',
    help='Prediction model')
@ck.option(
    '--ont', '-ont', default='mf',
    help='Prediction model')
@ck.option(
    '--model', '-m', default='deepgo2',

    help='Prediction model')
@ck.option(
    '--combine', '-c', is_flag=True,
    help='Prediction model')
@ck.option(
    '--alpha', '-a', default=0.50,
    help='Combining weight')
@ck.option(
    '--num-preds', '-np', default=50,
    help='Combining weight')
def main(data_root, ont, model, combine, alpha, num_preds):
    train_data_file = f'{data_root}/{ont}/train_data.pkl'
    valid_data_file = f'{data_root}/{ont}/valid_data.pkl'
    test_data_file = f'{data_root}/{ont}/predictions_{model}.pkl'
    # diam_data_file = f'{data_root}/{ont}/test_data_diam.pkl'
    terms_file = f'{data_root}/{ont}/terms.pkl'
    go_rels = Ontology(f'data/go.obo', with_rels=True)
    terms_df = pd.read_pickle(terms_file)
    terms = terms_df['gos'].values.flatten()
    terms_dict = {v: i for i, v in enumerate(terms)}

    train_df = pd.read_pickle(train_data_file)
    valid_df = pd.read_pickle(valid_data_file)
    train_df = pd.concat([train_df, valid_df])
    test_df = pd.read_pickle(test_data_file)
    # diam_df = pd.read_pickle(diam_data_file)
    
    annotations = train_df['prop_annotations'].values
    annotations = list(map(lambda x: set(x), annotations))
    test_annotations = test_df['prop_annotations'].values
    test_annotations = list(map(lambda x: set(x), test_annotations))
    go_rels.calculate_ic(annotations + test_annotations)

    # Print IC values of terms
    ics = {}
    for i, term in enumerate(terms):
        ics[term] = go_rels.get_ic(term)
    
    # Combine scores for diamond and deepgo
    eval_preds = []
    
    for i, row in enumerate(test_df.itertuples()):
        # diam_preds = np.zeros((len(terms),), dtype=np.float32)
        # for go_id, score in diam_df.iloc[i]['diam_preds'].items():
        #     if go_id in terms_dict:
        #         diam_preds[terms_dict[go_id]] = score
        preds = row.preds # diam_preds * alpha + row.preds * (1 - alpha)
        eval_preds.append(preds)

    labels = np.zeros((len(test_df), len(terms)), dtype=np.float32)
    eval_preds = np.concatenate(eval_preds).reshape(-1, len(terms))

    for i, row in enumerate(test_df.itertuples()):
        for go_id in row.prop_annotations:
            if go_id in terms_dict:
                labels[i, terms_dict[go_id]] = 1

    total_n = 0
    total_sum = 0
    for go_id, i in terms_dict.items():
        pos_n = np.sum(labels[:, i])
        if pos_n > 0 and pos_n < len(test_df):
            total_n += 1
            roc_auc  = compute_roc(labels[:, i], eval_preds[:, i])
            total_sum += roc_auc

    avg_auc = total_sum / total_n
    
    print('Computing Fmax')
    fmax = 0.0
    tmax = 0.0
    wfmax = 0.0
    wtmax = 0.0
    avgic = 0.0
    precisions = []
    recalls = []
    smin = 1000000.0
    rus = []
    mis = []
    go_set = go_rels.get_namespace_terms(NAMESPACES[ont])
    go_set.remove(FUNC_DICT[ont])
    labels = test_df['prop_annotations'].values
    labels = list(map(lambda x: set(filter(lambda y: y in go_set, x)), labels))
    spec_labels = test_df['exp_annotations'].values
    spec_labels = list(map(lambda x: set(filter(lambda y: y in go_set, x)), spec_labels))
    fmax_spec_match = 0
    for t in range(0, 101):
        threshold = t / 100.0
        preds = [set() for _ in range(len(test_df))]
        for i in range(len(test_df)):
            annots = set()
            above_threshold = np.argwhere(eval_preds[i] >= threshold).flatten()
            for j in above_threshold:
                annots.add(terms[j])
        
            if t == 0:
                preds[i] = annots
                continue
            # new_annots = set()
            # for go_id in annots:
            #     new_annots |= go_rels.get_anchestors(go_id)
            preds[i] = annots
            
        # Filter classes
        preds = list(map(lambda x: set(filter(lambda y: y in go_set, x)), preds))
        fscore, prec, rec, s, ru, mi, fps, fns, avg_ic, wf = evaluate_annotations(go_rels, labels, preds)
        spec_match = 0
        for i, row in enumerate(test_df.itertuples()):
            spec_match += len(spec_labels[i].intersection(preds[i]))
        # print(f'AVG IC {avg_ic:.3f}')
        precisions.append(prec)
        recalls.append(rec)
        # print(f'Fscore: {fscore}, Precision: {prec}, Recall: {rec} S: {s}, RU: {ru}, MI: {mi} threshold: {threshold}, WFmax: {wf}')
        if fmax < fscore:
            fmax = fscore
            tmax = threshold
            avgic = avg_ic
            fmax_spec_match = spec_match
        if wfmax < wf:
            wfmax = wf
            wtmax = threshold
        if smin > s:
            smin = s
    if combine:
        model += '_diam'
    print(model, ont)
    print(f'Fmax: {fmax:0.3f}, Smin: {smin:0.3f}, threshold: {tmax}, spec: {fmax_spec_match}')
    print(f'WFmax: {wfmax:0.3f}, threshold: {wtmax}')
    print(f'AUC: {avg_auc:0.3f}')
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    sorted_index = np.argsort(recalls)
    recalls = recalls[sorted_index]
    precisions = precisions[sorted_index]
    aupr = np.trapz(precisions, recalls)
    print(f'AUPR: {aupr:0.3f}')
    print(f'AVGIC: {avgic:0.3f}')

    df = pd.DataFrame({'precisions': precisions, 'recalls': recalls})
    df.to_pickle(f'{data_root}/{ont}/pr_{model}.pkl')

def compute_roc(labels, preds):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)
    return roc_auc

def compute_mcc(labels, preds):
    # Compute ROC curve and ROC area for each class
    mcc = matthews_corrcoef(labels.flatten(), preds.flatten())
    return mcc

def evaluate_annotations(go, real_annots, pred_annots):
    total = 0
    p = 0.0
    r = 0.0
    wp = 0.0
    wr = 0.0
    p_total= 0
    ru = 0.0
    mi = 0.0
    avg_ic = 0.0
    fps = []
    fns = []
    for i in range(len(real_annots)):
        if len(real_annots[i]) == 0:
            continue
        tp = set(real_annots[i]).intersection(set(pred_annots[i]))
        fp = pred_annots[i] - tp
        fn = real_annots[i] - tp
        tpic = 0.0
        for go_id in tp:
            tpic += go.get_norm_ic(go_id)
            avg_ic += go.get_ic(go_id)
        fpic = 0.0
        for go_id in fp:
            fpic += go.get_norm_ic(go_id)
            mi += go.get_ic(go_id)
        fnic = 0.0
        for go_id in fn:
            fnic += go.get_norm_ic(go_id)
            ru += go.get_ic(go_id)
        fps.append(fp)
        fns.append(fn)
        tpn = len(tp)
        fpn = len(fp)
        fnn = len(fn)
        total += 1
        recall = tpn / (1.0 * (tpn + fnn))
        r += recall
        wrecall = tpic / (tpic + fnic)
        wr += wrecall
        if len(pred_annots[i]) > 0:
            p_total += 1
            precision = tpn / (1.0 * (tpn + fpn))
            p += precision
            if tpic + fpic > 0:
                wp += tpic / (tpic + fpic)
    avg_ic = (avg_ic + mi) / total
    ru /= total
    mi /= total
    r /= total
    wr /= total
    if p_total > 0:
        p /= p_total
        wp /= p_total
    f = 0.0
    wf = 0.0
    if p + r > 0:
        f = 2 * p * r / (p + r)
        wf = 2 * wp * wr / (wp + wr)
    s = math.sqrt(ru * ru + mi * mi)
    return f, p, r, s, ru, mi, fps, fns, avg_ic, wf


if __name__ == '__main__':
    main()
