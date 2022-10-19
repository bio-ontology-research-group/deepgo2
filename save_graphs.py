#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd
import gzip
import logging
import torch as th
from aminoacids import MAXLEN
import torch
import os

#DGL imports
import dgl


@ck.command()
def main():
    df = pd.read_pickle('data/swissprot_exp.pkl')
    mask = torch.zeros((len(df), MAXLEN, MAXLEN), dtype=torch.uint8)
    with ck.progressbar(length=len(df), show_pos=True) as bar:
        for i, row in enumerate(df.itertuples()):
            seq_len = min(MAXLEN, len(row.sequences))
            graph_path = 'data/graphs/' + row.proteins + '.bin'
            if os.path.exists(graph_path):
                gs, _ = dgl.load_graphs(graph_path)
                g = gs[0]
                nnodes = g.num_nodes()
                mask[i, :nnodes, :nnodes] = g.adj().to_dense()
            else:
                mask[i, :seq_len, :seq_len] = 1
            bar.update(1)
    mask = mask.to_sparse()
    torch.save(mask, 'data/mask.pt')
    
    
def save_ppi():

    dgl.save_graphs('data/go.bin', graph, {'etypes': etypes})
    df = pd.read_pickle('data/swissprot_interactions.pkl')
    proteins = df['proteins']
    prot_idx = {v: k for k, v in enumerate(proteins)}
    src = []
    dst = []
    for i, row in enumerate(df.itertuples()):
        p_id = prot_idx[row.proteins]
        for p2_id in row.interactions:
            if p2_id in prot_idx:
                p2_id = prot_idx[p2_id]
                src.append(p_id)
                dst.append(p2_id)

    graph = dgl.graph((src, dst), num_nodes=len(proteins))
    graph = dgl.add_self_loop(graph)
    dgl.save_graphs('data/ppi.bin', graph)

    


def to_go(uri):
    return uri[1:-1].replace('http://purl.obolibrary.org/obo/GO_', 'GO:')

def to_rel(uri):
    return uri[len('ObjectSomeValuesFrom(<http://purl.obolibrary.org/obo/'):-1]

if __name__ == '__main__':
    main()
