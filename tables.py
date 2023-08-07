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
from collections import Counter
import os

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


@ck.command()
@ck.option(
    '--data-root', '-dr', default='data',
    help='Prediction model')
def main(data_root):
    models = ('valid_deepgozero_gat_mfpreds', 'valid_deepgozero_gat_mfpreds_plus',)
    strategies = ('min', 'max', 'avg')
    num_models = range(1, 11)

    for model in models:
        out = model
        for strat in strategies:
            strat_out = strat
            for n in num_models:
                fmaxs, smins, auprs, aucs = [], [], [], []
                for ont in ('bp', 'cc'):
                    fname = f'{data_root}/{ont}/{model}_{strat}_{n}.res'
                    with open(fname) as f:
                        lines = f.read().splitlines()
                    if len(lines) == 0:
                        print(fname)
                    auc, fmax, smin, aupr = 0, 0, 0, 0
                    for line in lines:
                        if line.startswith('Average AUC'):
                            auc = line.split()[-1]
                        elif line.startswith('Fmax'):
                            fmax, smin, _ = line.split(', ')
                            fmax = fmax.split()[-1]
                            smin = smin.split()[-1]
                        elif line.startswith('AUPR'):
                            aupr = line.split()[-1]
                    fmaxs.append(fmax)
                    smins.append(smin)
                    auprs.append(aupr)
                    aucs.append(auc)
                res = ' & '.join(fmaxs + smins + auprs + aucs)
                print('\\hline')
                print(f'{out} & {strat_out} & {n} & {res} \\\\')
                out = ''
                strat_out = ''

if __name__ == '__main__':
    main()
