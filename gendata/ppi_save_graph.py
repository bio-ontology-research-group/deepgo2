#!/usr/bin/env python
import os
import sys
sys.path.append('.')

import click as ck
import numpy as np
import pandas as pd
import gzip
import logging
import torch as th


#DGL imports
import dgl


@ck.command()
@ck.option(
    '--test-data-name', '-td', default='test', type=ck.Choice(['test', 'nextprot']),
    help='Test data set name')
def main(test_data_name):

    for ont in ['mf', 'bp', 'cc']:
        train_df = pd.read_pickle(f'data/{ont}/train_data.pkl')
        proteins = train_df['proteins']
        prot_idx = {v: k for k, v in enumerate(proteins)}
        src = []
        dst = []
        edge_types = []
        rels = {}
        for i, row in enumerate(train_df.itertuples()):
            p_id = prot_idx[row.proteins]
            for rel, p2_id in row.interactions:
                if rel not in rels:
                    rels[rel] = len(rels)
                if p2_id in prot_idx:
                    p2_id = prot_idx[p2_id]
                    src.append(p_id)
                    dst.append(p2_id)
                    edge_types.append(rels[rel])
        
        print(len(src), len(proteins))
        
        train_n = len(proteins)
        valid_df = pd.read_pickle(f'data/{ont}/valid_data.pkl')
        valid_proteins = valid_df['proteins']
        for i, p_id in enumerate(valid_proteins):
            prot_idx[p_id] = train_n + i

        valid_proteins = set(valid_proteins)
        valid_n = len(valid_proteins)

        for i, row in enumerate(train_df.itertuples()):
            p_id = prot_idx[row.proteins]
            for rel, p2_id in row.interactions:
                if p2_id in valid_proteins:
                    p2_id = prot_idx[p2_id]
                    src.append(p_id)
                    dst.append(p2_id)
                    edge_types.append(rels[rel])

        for i, row in enumerate(valid_df.itertuples()):
            p_id = prot_idx[row.proteins]
            for rel, p2_id in row.interactions:
                if rel not in rels:
                    rels[rel] = len(rels)
                if p2_id in prot_idx:
                    p2_id = prot_idx[p2_id]
                    src.append(p_id)
                    dst.append(p2_id)
                    edge_types.append(rels[rel])


        train_df = pd.concat([train_df, valid_df])

        test_df = pd.read_pickle(f'data/{ont}/{test_data_name}_data.pkl')
        test_proteins = test_df['proteins']
        for i, p_id in enumerate(test_proteins):
            prot_idx[p_id] = train_n + valid_n + i

        test_proteins = set(test_proteins)
        test_n = len(test_proteins)
        
        for i, row in enumerate(train_df.itertuples()):
            p_id = prot_idx[row.proteins]
            for rel, p2_id in row.interactions:
                if p2_id in test_proteins:
                    p2_id = prot_idx[p2_id]
                    src.append(p_id)
                    dst.append(p2_id)
                    edge_types.append(rels[rel])

        for i, row in enumerate(test_df.itertuples()):
            p_id = prot_idx[row.proteins]
            for rel, p2_id in row.interactions:
                if rel not in rels:
                    rels[rel] = len(rels)
                if p2_id in prot_idx:
                    p2_id = prot_idx[p2_id]
                    src.append(p_id)
                    dst.append(p2_id)
                    edge_types.append(rels[rel])

        print(len(prot_idx))
        graph = dgl.graph((src, dst), num_nodes=len(prot_idx))
        graph.edata['etypes'] = th.LongTensor(edge_types)
        graph = dgl.add_self_loop(graph)
        dgl.save_graphs(
            f'data/{ont}/ppi_{test_data_name}.bin', graph,
            {
                'train_nids': th.LongTensor(np.arange(train_n)),
                'valid_nids': th.LongTensor(np.arange(train_n, train_n + valid_n)),
                'test_nids': th.LongTensor(np.arange(train_n + valid_n, train_n + valid_n + test_n))
            })

if __name__ == '__main__':
    main()
