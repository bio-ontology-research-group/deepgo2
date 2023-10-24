import numpy as np
import pandas as pd
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
from .utils import NAMESPACES, FUNC_DICT

def compute_metrics(test_df, go, terms_dict, terms, ont, eval_preds):
    labels = np.zeros((len(test_df), len(terms_dict)), dtype=np.float32)
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
    go_set = go.get_namespace_terms(NAMESPACES[ont])
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
            preds[i] = annots
            
        # Filter classes
        preds = list(map(lambda x: set(filter(lambda y: y in go_set, x)), preds))
        fscore, prec, rec, s, ru, mi, fps, fns, avg_ic, wf = evaluate_annotations(go, labels, preds)
        spec_match = 0
        for i, row in enumerate(test_df.itertuples()):
            spec_match += len(spec_labels[i].intersection(preds[i]))
        precisions.append(prec)
        recalls.append(rec)
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
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    sorted_index = np.argsort(recalls)
    recalls = recalls[sorted_index]
    precisions = precisions[sorted_index]
    aupr = np.trapz(precisions, recalls)
    

    return fmax, smin, tmax, wfmax, wtmax, avg_auc, aupr, avgic, fmax_spec_match


def compute_roc(labels, preds):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)

    return roc_auc

def compute_mcc(labels, preds):
    # Computes MCC for each class
    mcc = matthews_corrcoef(labels.flatten(), preds.flatten())
    return mcc

def evaluate_annotations(go, real_annots, pred_annots):
    """
    Computes Fmax, Smin, WFmax and Average IC
    Args:
       go (utils.Ontology): Ontology class instance with go.obo
       real_annots (set): Set of real GO classes
       pred_annots (set): Set of predicted GO classes
    """
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
