import pandas as pd
import torch as th
import numpy as np
import dgl


def get_data(df, features_dict, terms_dict, features_length, features_column):
    """
    Converts dataframe file with protein information and returns
    PyTorch tensors
    """
    data = th.zeros((len(df), features_length), dtype=th.float32)
    labels = th.zeros((len(df), len(terms_dict)), dtype=th.float32)
    for i, row in enumerate(df.itertuples()):
        # Data vector
        if features_column == 'esm2':
            data[i, :] = th.FloatTensor(row.esm2)
        elif features_column == 'interpros':
            for feat in row.interpros:
                if feat in features_dict:
                    data[i, features_dict[feat]]
        elif features_column == 'mf_preds':
            data[i, :] = th.FloatTensor(row.mf_preds)
        elif features_column == 'prop_annotations':
            for feat in row.prop_annotations:
                if feat in features_dict:
                    data[i, features_dict[feat]] = 1
        # Labels vector
        for go_id in row.prop_annotations:
            if go_id in terms_dict:
                g_id = terms_dict[go_id]
                labels[i, g_id] = 1
    return data, labels

def load_data(
        data_root, ont, terms_file, features_length=2560,
        features_column='esm2', test_data_file='test_data.pkl'):
    terms_df = pd.read_pickle(terms_file)
    terms = terms_df['gos'].values.flatten()
    terms_dict = {v: i for i, v in enumerate(terms)}
    print('Terms', len(terms))
    
    ipr_df = pd.read_pickle(f'{data_root}/{ont}/interpros.pkl')
    iprs = ipr_df['interpros'].values
    iprs_dict = {v:k for k, v in enumerate(iprs)}
    if features_column == 'interpros':
        features_length = len(iprs_dict)
    

    train_df = pd.read_pickle(f'{data_root}/{ont}/train_data.pkl')
    valid_df = pd.read_pickle(f'{data_root}/{ont}/valid_data.pkl')
    test_df = pd.read_pickle(f'{data_root}/{ont}/{test_data_file}')

    train_data = get_data(train_df, iprs_dict, terms_dict, features_length, features_column)
    valid_data = get_data(valid_df, iprs_dict, terms_dict, features_length, features_column)
    test_data = get_data(test_df, iprs_dict, terms_dict, features_length, features_column)

    return iprs_dict, terms_dict, train_data, valid_data, test_data, test_df


def load_ppi_data(data_root, ont, features_length=2560,
                  features_column='esm2', test_data_file='test_data.pkl',
                  ppi_graph_file='ppi_test.bin'):
    
    terms_df = pd.read_pickle(f'{data_root}/{ont}/terms.pkl')
    terms = terms_df['gos'].values.flatten()
    terms_dict = {v: i for i, v in enumerate(terms)}
    print('Terms', len(terms))
    
    mf_df = pd.read_pickle(f'{data_root}/mf/terms.pkl')
    mfs = mf_df['gos'].values
    mfs_dict = {v:k for k, v in enumerate(mfs)}

    ipr_df = pd.read_pickle(f'{data_root}/{ont}/interpros.pkl')
    iprs = ipr_df['interpros'].values
    iprs_dict = {v:k for k, v in enumerate(iprs)}

    feat_dict = None
    
    if features_column == 'interpros':
        features_length = len(iprs_dict)
        feat_dict = iprs_dict
    elif features_column != 'esm2':
        features_length = len(mfs_dict)
        feat_dict = mfs_dict
    
    
    train_df = pd.read_pickle(f'{data_root}/{ont}/train_data.pkl')
    valid_df = pd.read_pickle(f'{data_root}/{ont}/valid_data.pkl')
    test_df = pd.read_pickle(f'{data_root}/{ont}/{test_data_file}')

    df = pd.concat([train_df, valid_df, test_df])
    graphs, nids = dgl.load_graphs(f'{data_root}/{ont}/{ppi_graph_file}')

    data, labels = get_data(df, feat_dict, terms_dict, features_length, features_column)
    graph = graphs[0]
    graph.ndata['feat'] = data
    graph.ndata['labels'] = labels
    train_nids, valid_nids, test_nids = nids['train_nids'], nids['valid_nids'], nids['test_nids']
    return feat_dict, terms_dict, graph, train_nids, valid_nids, test_nids, data, labels, test_df



def load_normal_forms(go_file, terms_dict):
    """
    Parses and loads normalized (using Normalize.groovy script)
    ontology axioms file
    Args:
        go_file (string): Path to a file with normal forms
        terms_dict (dict): Dictionary with GO classes that are predicted
    Returns:
        
    """
    nf1 = []
    nf2 = []
    nf3 = []
    nf4 = []
    relations = {}
    zclasses = {}
    
    def get_index(go_id):
        if go_id in terms_dict:
            index = terms_dict[go_id]
        elif go_id in zclasses:
            index = zclasses[go_id]
        else:
            zclasses[go_id] = len(terms_dict) + len(zclasses)
            index = zclasses[go_id]
        return index

    def get_rel_index(rel_id):
        if rel_id not in relations:
            relations[rel_id] = len(relations)
        return relations[rel_id]
                
    with open(go_file) as f:
        for line in f:
            line = line.strip().replace('_', ':')
            if line.find('SubClassOf') == -1:
                continue
            left, right = line.split(' SubClassOf ')
            # C SubClassOf D
            if len(left) == 10 and len(right) == 10:
                go1, go2 = left, right
                nf1.append((get_index(go1), get_index(go2)))
            elif left.find('and') != -1: # C and D SubClassOf E
                go1, go2 = left.split(' and ')
                go3 = right
                nf2.append((get_index(go1), get_index(go2), get_index(go3)))
            elif left.find('some') != -1:  # R some C SubClassOf D
                rel, go1 = left.split(' some ')
                go2 = right
                nf3.append((get_rel_index(rel), get_index(go1), get_index(go2)))
            elif right.find('some') != -1: # C SubClassOf R some D
                go1 = left
                rel, go2 = right.split(' some ')
                nf4.append((get_index(go1), get_rel_index(rel), get_index(go2)))
    return nf1, nf2, nf3, nf4, relations, zclasses
