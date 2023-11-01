#!/usr/bin/env python
import os
import sys
sys.path.append('.')

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
@ck.option('--fasta-file', '-if', default='data/nextprot.fa', help='Input FASTA file', required=True)
@ck.option('--tsv-file', '-tf', default='data/nextprot.tsv', help='TSV FASTA file', required=True)
@ck.option(
    '--data-root', '-dr', default='data',
    help='Prediction model')
def main(fasta_file, tsv_file, data_root):
    go_rels = Ontology(f'{data_root}/go.obo', with_rels=True)

    proteins = []
    accessions = []
    interpros = []
    string_ids = []
    genes = []
    sequences = []
    acc_map = {}
    with open(tsv_file) as f:
        next(f)
        for line in f:
            it = line.strip().split('\t', -1)
            accessions.append(it[1])
            proteins.append(it[3])
            acc_map[it[1]] = it[3]
            if len(it) >= 9:
                interpros.append(it[8].split(';'))
            else:
                interpros.append([])
            if len(it) == 10:
                string_ids.append(it[9].split(';'))
            else:
                string_ids.append([])
    df = pd.DataFrame({'proteins': proteins, 'accessions': accessions,
                       'interpros': interpros, 'string_ids': string_ids})
    seqs = {}
    with open(fasta_file) as f:
        records = SeqIO.parse(f, 'fasta')
        for rec in records:
            prot_id = rec.id.split('|')[-1]
            seqs[prot_id] = str(rec.seq)
    for prot_id in proteins:
        sequences.append(seqs[prot_id])

    df['sequences'] = sequences
    
    with open('data/nextprot_annots.tsv') as f:
        next(f)
        annots = {}
        prop_annots = {}
        for line in f:
            it = line.strip().split('\t')
            prot_id = acc_map[it[0]]
            go_id = it[3]
            if prot_id not in annots:
                annots[prot_id] = set()
                prop_annots[prot_id] = set()
            annots[prot_id].add(go_id)
            prop_annots[prot_id] |= go_rels.get_anchestors(go_id)

    annotations = []
    prop_annotations = []
    for prot_id in proteins:
        annotations.append(list(annots[prot_id]))
        prop_annotations.append(list(prop_annots[prot_id]))
    df['annotations'] = annotations
    df['exp_annotations'] = annotations
    df['prop_annotations'] = prop_annotations

    prots, data = extract_esm(fasta_file)
    data = list(data)
    esm2 = {}
    for prot, embed in zip(prots, data):
        prot_id = prot.split()[0].split('|')[-1]
        esm2[prot_id] = embed
    esms = []
    for prot_id in proteins:
        esms.append(esm2[prot_id])
    df['esm2'] = esms

    df.to_pickle('data/nextprot.pkl')
    

if __name__ == '__main__':
    main()
