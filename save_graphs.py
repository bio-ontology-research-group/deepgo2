#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd
import gzip
import logging
import torch as th


#DGL imports
import dgl


@ck.command()
def main():

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
