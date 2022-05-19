import os
import click as ck
import pandas as pd
from utils import Ontology
import torch as th
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.metrics import roc_curve, auc, matthews_corrcoef
import copy
from torch.utils.data import DataLoader, Dataset, TensorDataset
from itertools import cycle
import math
from aminoacids import to_onehot, MAXLEN, AAINDEX
from dgl.nn import GraphConv, GATConv
import dgl
from dgl.dataloading import GraphDataLoader
from torch_utils import FastTensorDataLoader


@ck.command()
@ck.option(
    '--data-root', '-dr', default='data',
    help='Prediction model')
@ck.option(
    '--ont', '-ont', default='mf',
    help='Prediction model')
@ck.option(
    '--batch-size', '-bs', default=37,
    help='Batch size for training')
@ck.option(
    '--epochs', '-ep', default=32,
    help='Training epochs')
@ck.option(
    '--load', '-ld', is_flag=True, help='Load Model?')
@ck.option(
    '--device', '-d', default='cuda:1',
    help='Device')
def main(data_root, ont, batch_size, epochs, load, device):
    go_file = f'{data_root}/go.norm'
    model_file = f'{data_root}/{ont}/deepgo2_alpha.th'
    terms_file = f'{data_root}/{ont}/terms.pkl'
    out_file = f'{data_root}/{ont}/predictions_deepgo2_alpha.pkl'

    go = Ontology(f'{data_root}/go.obo', with_rels=True)
    loss_func = nn.BCELoss()
    terms_dict, train_data, valid_data, test_data, test_df = load_data(data_root, ont, terms_file)
    n_terms = len(terms_dict)
    
    net = DGAlphaModel(1280, n_terms, device).to(device)
    print(net)
    train_data, train_graphs, train_labels = train_data
    valid_data, valid_graphs, valid_labels = valid_data
    test_data, test_graphs, test_labels = test_data
    
    train_dataset = MyDataset(train_data, train_graphs, train_labels)
    valid_dataset = MyDataset(valid_data, valid_graphs, valid_labels)
    test_dataset = MyDataset(test_data, test_graphs, test_labels)
    
    train_loader = GraphDataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = GraphDataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = GraphDataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)
    
    valid_labels = valid_labels.detach().cpu().numpy()
    test_labels = test_labels.detach().cpu().numpy()
    
    optimizer = th.optim.Adam(net.parameters(), lr=3e-4)
    scheduler = MultiStepLR(optimizer, milestones=[5,], gamma=0.1)

    best_loss = 10000.0
    if not load:
        print('Training the model')
        for epoch in range(epochs):
            net.train()
            train_loss = 0
            lmbda = 0.1
            train_steps = int(math.ceil(len(train_labels) / batch_size))
            with ck.progressbar(length=train_steps, show_pos=True) as bar:
                for batch_features, batch_graphs, batch_labels in train_loader:
                    bar.update(1)
                    batch_features = batch_features.to(device)
                    batch_graphs = batch_graphs.to(device)
                    batch_labels = batch_labels.to(device)
                    logits = net(batch_features, batch_graphs)
                    loss = F.binary_cross_entropy(logits, batch_labels)
                    total_loss = loss
                    train_loss += loss.detach().item()
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()
                    
            train_loss /= train_steps
            
            print('Validation')
            net.eval()
            with th.no_grad():
                valid_steps = int(math.ceil(len(valid_labels) / batch_size))
                valid_loss = 0
                preds = []
                with ck.progressbar(length=valid_steps, show_pos=True) as bar:
                    for batch_features, batch_graphs, batch_labels in valid_loader:
                        bar.update(1)
                        batch_features = batch_features.to(device)
                        batch_graphs = batch_graphs.to(device)
                        batch_labels = batch_labels.to(device)
                        logits = net(batch_features, batch_graphs)
                        batch_loss = F.binary_cross_entropy(logits, batch_labels)
                        valid_loss += batch_loss.detach().item()
                        preds = np.append(preds, logits.detach().cpu().numpy())
                valid_loss /= valid_steps
                roc_auc = compute_roc(valid_labels, preds)
                print(f'Epoch {epoch}: Loss - {train_loss}, Valid loss - {valid_loss}, AUC - {roc_auc}')

            if valid_loss < best_loss:
                best_loss = valid_loss
                print('Saving model')
                th.save(net.state_dict(), model_file)

            #scheduler.step()
            

    # Loading best model
    print('Loading the best model')
    net.load_state_dict(th.load(model_file))
    net.eval()
    with th.no_grad():
        test_steps = int(math.ceil(len(test_labels) / batch_size))
        test_loss = 0
        preds = []
        with ck.progressbar(length=test_steps, show_pos=True) as bar:
            for batch_features, batch_graphs, batch_labels in test_loader:
                bar.update(1)
                batch_features = batch_features.to(device)
                batch_graphs = batch_graphs.to(device)
                batch_labels = batch_labels.to(device)
                logits = net(batch_features, batch_graphs)
                batch_loss = F.binary_cross_entropy(logits, batch_labels)
                test_loss += batch_loss.detach().cpu().item()
                preds = np.append(preds, logits.detach().cpu().numpy())
            test_loss /= test_steps
        preds = preds.reshape(-1, n_terms)
        roc_auc = compute_roc(test_labels, preds)
        print(f'Test Loss - {test_loss}, AUC - {roc_auc}')

        
    preds = list(preds)
    # Propagate scores using ontology structure
    for i, scores in enumerate(preds):
        prop_annots = {}
        for go_id, j in terms_dict.items():
            score = scores[j]
            for sup_go in go.get_anchestors(go_id):
                if sup_go in prop_annots:
                    prop_annots[sup_go] = max(prop_annots[sup_go], score)
                else:
                    prop_annots[sup_go] = score
        for go_id, score in prop_annots.items():
            if go_id in terms_dict:
                scores[terms_dict[go_id]] = score

    test_df['preds'] = preds

    test_df.to_pickle(out_file)

    
def compute_roc(labels, preds):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)

    return roc_auc

class Residual(nn.Module):

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return x + self.fn(x)
    
        
class MLPBlock(nn.Module):

    def __init__(self, in_features, out_features, bias=True, layer_norm=True, dropout=0.1, activation=nn.ReLU):
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


class DGAlphaModel(nn.Module):

    def __init__(self, input_length, nb_gos, device, nodes=[1280 + 21,]):
        super().__init__()
        self.nb_gos = nb_gos

        self.gcn1 = GraphConv(21, 21)
        # self.gcn2 = GraphConv(256, 256)
        self.sag_pool = SAGPool(21)
        
        net = []
        for hidden_dim in nodes:
            net.append(MLPBlock(hidden_dim, hidden_dim))
            net.append(Residual(MLPBlock(hidden_dim, hidden_dim)))
            input_length = hidden_dim
        net.append(nn.Linear(hidden_dim, nb_gos))
        net.append(nn.Sigmoid())
        self.net = nn.Sequential(*net)
        
    def forward(self, features, graphs):
        x = self.gcn1(graphs, graphs.ndata['h'])
        # x = self.gcn2(graphs, x)
        graphs, x, perm = self.sag_pool(graphs, x)
        graphs.ndata['h'] = x
        x = dgl.mean_nodes(graphs, 'h')
        x = th.cat([features, x], dim=1)
        return self.net(x)
    
    
def load_data(data_root, ont, terms_file):
    terms_df = pd.read_pickle(terms_file)
    terms = terms_df['gos'].values.flatten()
    terms_dict = {v: i for i, v in enumerate(terms)}
    print('Terms', len(terms))
    
    train_df = pd.read_pickle(f'{data_root}/{ont}/train_data.pkl')#[:1000]
    valid_df = pd.read_pickle(f'{data_root}/{ont}/valid_data.pkl')#[:1000]
    test_df = pd.read_pickle(f'{data_root}/{ont}/test_data.pkl')#[:1000]

    train_data = get_data(train_df, terms_dict)
    valid_data = get_data(valid_df, terms_dict)
    test_data = get_data(test_df, terms_dict)

    return terms_dict, train_data, valid_data, test_data, test_df

def get_data(df, terms_dict):
    graphs = []
    data = []
    labels = th.zeros((len(df), len(terms_dict)), dtype=th.float32)
    index = []
    with ck.progressbar(length=len(df), show_pos=True) as bar:
        for i, row in enumerate(df.itertuples()):
            graph_path = 'data/graphs/' + row.proteins + '.bin'
            if not os.path.exists(graph_path):
                continue
            gs, _ = dgl.load_graphs(graph_path)
            g = gs[0]
            seq_len = min(1000, len(row.sequences))
            onehot = th.zeros((seq_len, 21), dtype=th.float32)
            for i in range(seq_len):
                onehot[i, AAINDEX.get(row.sequences[i], 0)] = 1
            g.ndata['h'] = onehot
            graphs.append(g)
            data.append(row.esm)
            index.append(i)
            for go_id in row.prop_annotations: # prop_annotations for full model
                if go_id in terms_dict:
                    g_id = terms_dict[go_id]
                    labels[i, g_id] = 1
            bar.update(1)
    labels = labels[index, :]
    return data, graphs, labels


class SAGPool(torch.nn.Module):
    """The Self-Attention Pooling layer in paper 
    `Self Attention Graph Pooling <https://arxiv.org/pdf/1904.08082.pdf>`
    Args:
        in_dim (int): The dimension of node feature.
        ratio (float, optional): The pool ratio which determines the amount of nodes
            remain after pooling. (default: :obj:`0.5`)
        conv_op (torch.nn.Module, optional): The graph convolution layer in dgl used to
        compute scale for each node. (default: :obj:`dgl.nn.GraphConv`)
        non_linearity (Callable, optional): The non-linearity function, a pytorch function.
            (default: :obj:`torch.tanh`)
    """
    def __init__(self, in_dim:int, ratio=0.05, conv_op=GraphConv, non_linearity=torch.tanh):
        super(SAGPool, self).__init__()
        self.in_dim = in_dim
        self.ratio = ratio
        self.score_layer = conv_op(in_dim, 1)
        self.non_linearity = non_linearity
    
    def forward(self, graph:dgl.DGLGraph, feature:torch.Tensor):
        score = self.score_layer(graph, feature).squeeze()
        perm, next_batch_num_nodes = topk(score, self.ratio, get_batch_id(graph.batch_num_nodes()), graph.batch_num_nodes())
        feature = feature[perm] * self.non_linearity(score[perm]).view(-1, 1)
        graph = dgl.node_subgraph(graph, perm)

        # node_subgraph currently does not support batch-graph,
        # the 'batch_num_nodes' of the result subgraph is None.
        # So we manually set the 'batch_num_nodes' here.
        # Since global pooling has nothing to do with 'batch_num_edges',
        # we can leave it to be None or unchanged.
        graph.set_batch_num_nodes(next_batch_num_nodes)
        
        return graph, feature, perm

def get_batch_id(num_nodes:torch.Tensor):
    """Convert the num_nodes array obtained from batch graph to batch_id array
    for each node.
    Args:
        num_nodes (torch.Tensor): The tensor whose element is the number of nodes
            in each graph in the batch graph.
    """
    batch_size = num_nodes.size(0)
    batch_ids = []
    for i in range(batch_size):
        item = torch.full((num_nodes[i],), i, dtype=torch.long, device=num_nodes.device)
        batch_ids.append(item)
    return torch.cat(batch_ids)

def topk(x:torch.Tensor, ratio:float, batch_id:torch.Tensor, num_nodes:torch.Tensor):
    """The top-k pooling method. Given a graph batch, this method will pool out some
    nodes from input node feature tensor for each graph according to the given ratio.
    Args:
        x (torch.Tensor): The input node feature batch-tensor to be pooled.
        ratio (float): the pool ratio. For example if :obj:`ratio=0.5` then half of the input
            tensor will be pooled out.
        batch_id (torch.Tensor): The batch_id of each element in the input tensor.
        num_nodes (torch.Tensor): The number of nodes of each graph in batch.
    
    Returns:
        perm (torch.Tensor): The index in batch to be kept.
        k (torch.Tensor): The remaining number of nodes for each graph.
    """
    batch_size, max_num_nodes = num_nodes.size(0), num_nodes.max().item()
    
    cum_num_nodes = torch.cat(
        [num_nodes.new_zeros(1),
         num_nodes.cumsum(dim=0)[:-1]], dim=0)
    
    index = torch.arange(batch_id.size(0), dtype=torch.long, device=x.device)
    index = (index - cum_num_nodes[batch_id]) + (batch_id * max_num_nodes)

    dense_x = x.new_full((batch_size * max_num_nodes, ), torch.finfo(x.dtype).min)
    dense_x[index] = x
    dense_x = dense_x.view(batch_size, max_num_nodes)

    _, perm = dense_x.sort(dim=-1, descending=True)
    perm = perm + cum_num_nodes.view(-1, 1)
    perm = perm.view(-1)

    k = (ratio * num_nodes.to(torch.float)).ceil().to(torch.long)
    mask = [
        torch.arange(k[i], dtype=torch.long, device=x.device) + 
        i * max_num_nodes for i in range(batch_size)]

    mask = torch.cat(mask, dim=0)
    perm = perm[mask]

    return perm, k


class MyDataset(Dataset):

    def __init__(self, data, graphs, labels):
        self.data = data
        self.graphs = graphs
        self.labels = labels
        
    def __getitem__(self, idx):
        return self.data[idx], self.graphs[idx], self.labels[idx]

    def __len__(self):
        return len(self.graphs)


if __name__ == '__main__':
    main()
