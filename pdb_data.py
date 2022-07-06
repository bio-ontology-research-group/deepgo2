#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd
import gzip
import logging
from utils import NAMESPACES, Ontology, get_goplus_defs
from collections import Counter
import json
from Bio.PDB import PDBParser
import torch as th
import os
import dgl

logging.basicConfig(level=logging.INFO)


def calc_residue_dist(residue_one, residue_two) :
    """Returns the C-alpha distance between two residues"""
    diff_vector  = residue_one["CA"].coord - residue_two["CA"].coord
    return np.sqrt(np.sum(diff_vector * diff_vector))

def calc_dist_matrix(chain_one, chain_two) :
    """Returns a matrix of C-alpha distances between two chains"""
    answer = np.zeros((len(chain_one), len(chain_two)), np.float32)
    for row, residue_one in enumerate(chain_one) :
        for col, residue_two in enumerate(chain_two) :
            answer[row, col] = calc_residue_dist(residue_one, residue_two)
    return answer


@ck.command()
@ck.option(
    '--data-file', '-df', default='data/swissprot_exp.pkl',
    help='DataFrame with proteins, sequences and annotations')
def main(data_file):
    df = pd.read_pickle(data_file)

    data_root = 'data/pdb/'
    parser = PDBParser()
    missing = 0
    with ck.progressbar(length=len(df), show_pos=True) as bar:
        for i, row in df.iterrows():
            bar.update(1)
            accessions = row.accessions.split(';')
            p_id = accessions[0]
            seq_len = min(1000, len(row.sequences))
            filename = 'AF-' + p_id + '-F1-model_v2.pdb.gz'
            if os.path.exists(data_root + filename):
                pdb_file = gzip.open(data_root + filename, 'rt')
                structure = parser.get_structure(p_id, pdb_file)
                pdb_file.close()
                model = structure[0]
                chain = model['A']
                dist_matrix = calc_dist_matrix(chain, chain)
                dist_matrix = dist_matrix[:1000,:1000]
                if dist_matrix.shape[0] == seq_len:
                    src, dst = (dist_matrix < 8).nonzero()
                    graph = dgl.graph((src, dst))
                else:
                    graph = dgl.graph(([], []), num_nodes=seq_len)
            else:
                graph = dgl.graph(([], []), num_nodes=seq_len)
            graph = graph.add_self_loop()
            
            filename = row.proteins + '.pt'
            data = th.load('data/swissprot_esm1b/' + filename)
            features = data['representations'][33]
            graph.ndata['h'] = features
            dgl.save_graphs('data/graphs/' + row.proteins + '.bin', graph)
            
            
if __name__ == '__main__':
    main()
