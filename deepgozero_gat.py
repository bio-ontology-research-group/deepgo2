import click as ck
import pandas as pd
from utils import Ontology
import torch as th
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch import optim
from sklearn.metrics import roc_curve, auc, matthews_corrcoef
import copy
from torch.utils.data import DataLoader, IterableDataset, TensorDataset
from itertools import cycle
import math
from aminoacids import to_onehot, MAXLEN
from dgl.nn import GraphConv, GATConv
import dgl
from torch_utils import FastTensorDataLoader
import csv
from torch.optim.lr_scheduler import MultiStepLR


@ck.command()
@ck.option(
    '--data-root', '-dr', default='data',
    help='Prediction model')
@ck.option(
    '--ont', '-ont', default='mf',
    help='Prediction model')
@ck.option(
    '--model-name', '-m', default='deepgozero_gat',
    help='Prediction model')
@ck.option(
    '--batch-size', '-bs', default=37,
    help='Batch size for training')
@ck.option(
    '--epochs', '-ep', default=256,
    help='Training epochs')
@ck.option(
    '--load', '-ld', is_flag=True, help='Load Model?')
@ck.option(
    '--device', '-d', default='cuda:1',
    help='Device')
def main(data_root, ont, model_name, batch_size, epochs, load, device):
    go_file = f'{data_root}/go.obo'
    model_file = f'{data_root}/{ont}/{model_name}.th'
    terms_file = f'{data_root}/{ont}/terms.pkl'
    out_file = f'{data_root}/{ont}/nextprot_predictions_{model_name}.pkl'

    go = Ontology(go_file, with_rels=True)

    loss_func = nn.BCELoss()
    terms_dict, graph, train_nids, valid_nids, test_nids, data, labels, test_df = load_data(data_root, ont)
    n_terms = len(terms_dict)
    
    
    valid_labels = labels[valid_nids].numpy()
    test_labels = labels[test_nids].numpy()

    labels = labels.to(device)

    print(valid_labels.shape)
    
    graph = graph.to(device)

    train_nids = train_nids.to(device)
    valid_nids = valid_nids.to(device)
    test_nids = test_nids.to(device)

    nf1, nf2, nf3, nf4, relations, zero_classes = load_normal_forms(
        f'{data_root}/go.norm', terms_dict)
    n_rels = len(relations)
    n_zeros = len(zero_classes)

    
    normal_forms = nf1, nf2, nf3, nf4
    nf1 = th.LongTensor(nf1).to(device)
    nf2 = th.LongTensor(nf2).to(device)
    nf3 = th.LongTensor(nf3).to(device)
    nf4 = th.LongTensor(nf4).to(device)
    normal_forms = nf1, nf2, nf3, nf4


    net = DeepGO2Model(2560, n_terms, n_rels, n_zeros, device).to(device)

    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
    train_dataloader = dgl.dataloading.DataLoader(
        graph, train_nids, sampler,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0)

    valid_dataloader = dgl.dataloading.DataLoader(
        graph, valid_nids, sampler,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0)

    test_dataloader = dgl.dataloading.DataLoader(
        graph, test_nids, sampler,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0)
    
    
    optimizer = th.optim.Adam(net.parameters(), lr=1e-3)
    scheduler = MultiStepLR(optimizer, milestones=[5, 10,], gamma=0.1)

    best_loss = 10000.0
    if not load:
        print('Training the model')
        log_file = open(f'{data_root}/train_logs.tsv', 'w')
        logger = csv.writer(log_file, delimiter='\t')
        for epoch in range(epochs):
            net.train()
            train_loss = 0
            train_steps = int(math.ceil(len(train_nids) / batch_size))
            with ck.progressbar(length=train_steps, show_pos=True) as bar:
                for input_nodes, output_nodes, blocks in train_dataloader:
                    bar.update(1)
                    logits = net(input_nodes, output_nodes, blocks)
                    batch_labels = labels[output_nodes]
                    loss = F.binary_cross_entropy(logits, batch_labels)
                    el_loss = net.el_loss(normal_forms)
                    total_loss = loss + el_loss
                    train_loss += loss.detach().item()
                    train_elloss = el_loss.detach().item()
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()
                    
            train_loss /= train_steps
            
            print('Validation')
            net.eval()
            with th.no_grad():
                valid_steps = int(math.ceil(len(valid_nids) / batch_size))
                valid_loss = 0
                preds = []
                with ck.progressbar(length=valid_steps, show_pos=True) as bar:
                    for input_nodes, output_nodes, blocks in valid_dataloader:
                        bar.update(1)
                        logits = net(input_nodes, output_nodes, blocks)
                        batch_labels = labels[output_nodes]
                        batch_loss = F.binary_cross_entropy(logits, batch_labels)
                        valid_loss += batch_loss.detach().item()
                        preds = np.append(preds, logits.detach().cpu().numpy())
                valid_loss /= valid_steps
                roc_auc = compute_roc(valid_labels, preds)
                print(f'Epoch {epoch}: Loss - {train_loss}, Valid loss - {valid_loss}, AUC - {roc_auc}')
                logger.writerow([epoch, train_loss, valid_loss, roc_auc])
            if valid_loss < best_loss:
                best_loss = valid_loss
                print('Saving model')
                th.save(net.state_dict(), model_file)

            scheduler.step()
            
        log_file.close()

    # Loading best model
    print('Loading the best model')
    net.load_state_dict(th.load(model_file))
    net.eval()

    with th.no_grad():
        valid_steps = int(math.ceil(len(valid_nids) / batch_size))
        valid_loss = 0
        preds = []
        with ck.progressbar(length=valid_steps, show_pos=True) as bar:
            for input_nodes, output_nodes, blocks in valid_dataloader:
                bar.update(1)
                logits = net(input_nodes, output_nodes, blocks)
                batch_labels = labels[output_nodes]
                batch_loss = F.binary_cross_entropy(logits, batch_labels)
                valid_loss += batch_loss.detach().item()
                preds = np.append(preds, logits.detach().cpu().numpy())
            valid_loss /= valid_steps
                
    with th.no_grad():
        test_steps = int(math.ceil(len(test_nids) / batch_size))
        test_loss = 0
        preds = []
        with ck.progressbar(length=test_steps, show_pos=True) as bar:
            for input_nodes, output_nodes, blocks in test_dataloader:
                bar.update(1)
                logits = net(input_nodes, output_nodes, blocks)
                batch_labels = labels[output_nodes]
                batch_loss = F.binary_cross_entropy(logits, batch_labels)
                test_loss += batch_loss.detach().cpu().item()
                preds = np.append(preds, logits.detach().cpu().numpy())
            test_loss /= test_steps
        preds = preds.reshape(-1, n_terms)
        roc_auc = compute_roc(test_labels, preds)
    print(f'Valid Loss - {valid_loss}, Test Loss - {test_loss}, AUC - {roc_auc}')
    
    preds = list(preds)
    # Propagate scores using ontology structure
    for i in range(len(preds)):
        prop_annots = {}
        for go_id, j in terms_dict.items():
            score = preds[i][j]
            for sup_go in go.get_anchestors(go_id):
                if sup_go in prop_annots:
                    prop_annots[sup_go] = max(prop_annots[sup_go], score)
                else:
                    prop_annots[sup_go] = score
        for go_id, score in prop_annots.items():
            if go_id in terms_dict:
                preds[i][terms_dict[go_id]] = score

    test_df['preds'] = preds

    test_df.to_pickle(out_file)

    
def compute_roc(labels, preds):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)

    return roc_auc


def load_normal_forms(go_file, terms_dict):
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


class Residual(nn.Module):

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return x + self.fn(x)
    
        
class MLPBlock(nn.Module):

    def __init__(self, in_features, out_features, bias=True, layer_norm=False, dropout=0.1, activation=nn.ReLU):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        self.activation = activation()
        self.layer_norm = nn.LayerNorm(out_features) if layer_norm else None
        self.dropout = nn.Dropout(dropout) if dropout else None

    def forward(self, x):
        x = self.activation(self.linear(x))
        if self.layer_norm:
            x = self.layer_norm(x)
        if self.dropout:
            x = self.dropout(x)
        return x


class DeepGO2Model(nn.Module):

    def __init__(self, input_length, nb_gos, nb_rels, nb_zero_gos, device, hidden_dim=1024, embed_dim=1024, margin=0.1):
        super().__init__()
        self.nb_gos = nb_gos
        self.net1 = MLPBlock(input_length, hidden_dim)
        self.conv1 = GATConv(hidden_dim, hidden_dim, num_heads=1)
        input_length = hidden_dim
        self.net2 = nn.Sequential(
            nn.Linear(hidden_dim, nb_gos),
            nn.Sigmoid())

                # ELEmbeddings
        self.embed_dim = embed_dim
        self.hasFuncIndex = th.LongTensor([nb_rels]).to(device)
        self.go_embed = nn.Embedding(nb_gos + nb_zero_gos, embed_dim)
        self.go_norm = nn.BatchNorm1d(embed_dim)
        k = math.sqrt(1 / embed_dim)
        nn.init.uniform_(self.go_embed.weight, -k, k)
        self.go_rad = nn.Embedding(nb_gos + nb_zero_gos, 1)
        nn.init.uniform_(self.go_rad.weight, -k, k)
        # self.go_embed.weight.requires_grad = False
        # self.go_rad.weight.requires_grad = False
        
        self.rel_embed = nn.Embedding(nb_rels + 1, embed_dim)
        nn.init.uniform_(self.rel_embed.weight, -k, k)
        self.all_gos = th.arange(self.nb_gos).to(device)
        self.margin = margin


        
    def forward(self, input_nodes, output_nodes, blocks, residual=True):
        g1 = blocks[0]
        # g2 = blocks[1]
        features = g1.ndata['feat']['_N']
        x = self.net1(features)
        x = self.conv1(g1, x).squeeze(dim=1)

        go_embed = self.go_embed(self.all_gos)
        hasFunc = self.rel_embed(self.hasFuncIndex)
        hasFuncGO = go_embed + hasFunc
        go_rad = th.abs(self.go_rad(self.all_gos).view(1, -1))
        x = th.matmul(x, hasFuncGO.T) + go_rad
        logits = th.sigmoid(x)
        return logits

        
    def el_loss(self, go_normal_forms):
        nf1, nf2, nf3, nf4 = go_normal_forms
        nf1_loss = self.nf1_loss(nf1)
        nf2_loss = self.nf2_loss(nf2)
        nf3_loss = self.nf3_loss(nf3)
        nf4_loss = self.nf4_loss(nf4)
        # print()
        # print(nf1_loss.detach().item(),
        #       nf2_loss.detach().item(),
        #       nf3_loss.detach().item(),
        #       nf4_loss.detach().item())
        return nf1_loss + nf3_loss + nf4_loss + nf2_loss

    def class_dist(self, data):
        c = self.go_norm(self.go_embed(data[:, 0]))
        d = self.go_norm(self.go_embed(data[:, 1]))
        rc = th.abs(self.go_rad(data[:, 0]))
        rd = th.abs(self.go_rad(data[:, 1]))
        dist = th.linalg.norm(c - d, dim=1, keepdim=True) + rc - rd
        return dist
        
    def nf1_loss(self, data):
        pos_dist = self.class_dist(data)
        loss = th.mean(th.relu(pos_dist - self.margin))
        return loss

    def nf2_loss(self, data):
        c = self.go_norm(self.go_embed(data[:, 0]))
        d = self.go_norm(self.go_embed(data[:, 1]))
        e = self.go_norm(self.go_embed(data[:, 2]))
        rc = th.abs(self.go_rad(data[:, 0]))
        rd = th.abs(self.go_rad(data[:, 1]))
        re = th.abs(self.go_rad(data[:, 2]))
        
        sr = rc + rd
        dst = th.linalg.norm(c - d, dim=1, keepdim=True)
        dst2 = th.linalg.norm(e - c, dim=1, keepdim=True)
        dst3 = th.linalg.norm(e - d, dim=1, keepdim=True)
        loss = th.mean(th.relu(dst - sr - self.margin)
                    + th.relu(dst2 - rc - self.margin)
                    + th.relu(dst3 - rd - self.margin))

        return loss

    def nf3_loss(self, data):
        # R some C subClassOf D
        n = data.shape[0]
        # rS = self.rel_space(data[:, 0])
        # rS = rS.reshape(-1, self.embed_dim, self.embed_dim)
        rE = self.rel_embed(data[:, 0])
        c = self.go_norm(self.go_embed(data[:, 1]))
        d = self.go_norm(self.go_embed(data[:, 2]))
        # c = th.matmul(c, rS).reshape(n, -1)
        # d = th.matmul(d, rS).reshape(n, -1)
        rc = th.abs(self.go_rad(data[:, 1]))
        rd = th.abs(self.go_rad(data[:, 2]))
        
        rSomeC = c + rE
        euc = th.linalg.norm(rSomeC - d, dim=1, keepdim=True)
        loss = th.mean(th.relu(euc + rc - rd - self.margin))
        return loss


    def nf4_loss(self, data):
        # C subClassOf R some D
        n = data.shape[0]
        c = self.go_norm(self.go_embed(data[:, 0]))
        rE = self.rel_embed(data[:, 1])
        d = self.go_norm(self.go_embed(data[:, 2]))
        
        rc = th.abs(self.go_rad(data[:, 1]))
        rd = th.abs(self.go_rad(data[:, 2]))
        sr = rc + rd
        # c should intersect with d + r
        rSomeD = d + rE
        dst = th.linalg.norm(c - rSomeD, dim=1, keepdim=True)
        loss = th.mean(th.relu(dst - sr - self.margin))
        return loss

    
def load_data(data_root, ont):
    terms_df = pd.read_pickle(f'{data_root}/{ont}/terms.pkl')
    terms = terms_df['gos'].values.flatten()
    terms_dict = {v: i for i, v in enumerate(terms)}
    print('Terms', len(terms))
    
    mf_df = pd.read_pickle(f'{data_root}/mf/terms.pkl')
    mfs = mf_df['gos'].values
    mfs_dict = {v:k for k, v in enumerate(mfs)}

    train_df = pd.read_pickle(f'{data_root}/{ont}/train_data.pkl')
    valid_df = pd.read_pickle(f'{data_root}/{ont}/valid_data.pkl')
    test_df = pd.read_pickle(f'{data_root}/{ont}/nextprot_data.pkl')

    df = pd.concat([train_df, valid_df, test_df])
    graphs, nids = dgl.load_graphs(f'{data_root}/{ont}/ppi_nextprot.bin')

    data, labels = get_data(df, terms_dict)
    graph = graphs[0]
    graph.ndata['feat'] = data
    graph.ndata['labels'] = labels
    train_nids, valid_nids, test_nids = nids['train_nids'], nids['valid_nids'], nids['test_nids']
    return terms_dict, graph, train_nids, valid_nids, test_nids, data, labels, test_df

def get_data(df, terms_dict):
    data = th.zeros((len(df), 2560), dtype=th.float32)
    labels = th.zeros((len(df), len(terms_dict)), dtype=th.float32)
    for i, row in enumerate(df.itertuples()):
        data[i, :] = th.FloatTensor(row.esm2)
        for go_id in row.prop_annotations:
            if go_id in terms_dict:
                g_id = terms_dict[go_id]
                labels[i, g_id] = 1
    return data, labels

if __name__ == '__main__':
    main()
