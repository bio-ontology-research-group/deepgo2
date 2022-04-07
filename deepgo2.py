import click as ck
import pandas as pd
from utils import Ontology
import torch as th
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.metrics import roc_curve, auc, matthews_corrcoef
import copy
from torch.utils.data import DataLoader, IterableDataset, TensorDataset
from itertools import cycle
import math
from aminoacids import to_onehot, MAXLEN
from dgl.nn import GraphConv, GATConv
import dgl
from torch_utils import FastTensorDataLoader
from collections import Counter


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
    '--epochs', '-ep', default=512,
    help='Training epochs')
@ck.option(
    '--load', '-ld', is_flag=True, help='Load Model?')
@ck.option(
    '--device', '-d', default='cuda:1',
    help='Device')
def main(data_root, ont, batch_size, epochs, load, device):
    go_file = f'{data_root}/go.norm'
    model_file = f'{data_root}/{ont}/deepgo2.th'
    terms_file = f'{data_root}/{ont}/terms.pkl'
    out_file = f'{data_root}/{ont}/predictions_deepgo2.pkl'

    go = Ontology(f'{data_root}/go.obo', with_rels=True)
    
    loss_func = nn.BCELoss()
    iprs_dict, terms_dict, train_data, valid_data, test_data, test_df = load_data(data_root, ont, terms_file, go)
    n_terms = len(terms_dict)
    n_iprs = len(iprs_dict)
    
    nf1, nf2, nf3, nf4, relations, zero_classes = load_normal_forms(
        go_file, terms_dict)
    n_rels = len(relations)
    n_zeros = len(zero_classes)

    
    normal_forms = nf1, nf2, nf3, nf4
    nf1 = th.LongTensor(nf1).to(device)
    nf2 = th.LongTensor(nf2).to(device)
    nf3 = th.LongTensor(nf3).to(device)
    nf4 = th.LongTensor(nf4).to(device)
    normal_forms = nf1, nf2, nf3, nf4


    train_iprs, train_esm, train_diam, train_seqs, train_dl2vec, train_ics, train_labels = train_data
    valid_iprs, valid_esm, valid_diam, valid_seqs, valid_dl2vec, valid_ics, valid_labels = valid_data
    test_iprs, test_esm, test_diam, test_seqs, test_dl2vec, test_ics, test_labels = test_data
    
    train_loader = FastTensorDataLoader(
        *train_data, batch_size=batch_size, shuffle=True)
    valid_loader = FastTensorDataLoader(
        *valid_data, batch_size=batch_size, shuffle=False)
    test_loader = FastTensorDataLoader(
        *test_data, batch_size=batch_size, shuffle=False)
    
    valid_labels = valid_labels.detach().cpu().numpy()
    test_labels = test_labels.detach().cpu().numpy()
    
    cnn_params = {'nb_filters': 256, 'max_kernel': 129}
    go_graph = load_go_graph(terms_dict).to(device)
    graphs = [go_graph] * batch_size
    rem_batch = len(train_labels) % batch_size
    rem_graphs = [go_graph] * rem_batch
    rem_batch_val = len(valid_labels) % batch_size
    rem_graphs_val = [go_graph] * rem_batch_val
    rem_batch_test = len(test_labels) % batch_size
    rem_graphs_test = [go_graph] * rem_batch_test
    graphs = {
        batch_size: dgl.batch(graphs),
        rem_batch: dgl.batch(rem_graphs),
        rem_batch_val: dgl.batch(rem_graphs_val),
        rem_batch_test: dgl.batch(rem_graphs_test)}
    
    print(go_graph)
    
    mlp = MLPModel(n_iprs, n_terms, device).to(device)
    dgesm = DGESMModel(1280, n_terms, device).to(device)
    dgdl2v = DGDL2VModel(200, n_terms, device).to(device)
    dgcnn = DGCNNModel(n_terms, device).to(device)
    
    net = DG2Model(graphs, n_terms, mlp, dgesm, dgcnn, dgdl2v, device).to(device)
    
    print(net)

    
    optimizer = th.optim.Adam(net.parameters(), lr=3e-4)
    scheduler = MultiStepLR(optimizer, milestones=[1], gamma=0.1)

    best_loss = 10000.0
    if not load:
        print('Training the model')
        for epoch in range(epochs):
            net.train()
            train_loss = 0
            train_elloss = 0
            lmbda = 0.1
            train_steps = int(math.ceil(len(train_labels) / batch_size))
            with ck.progressbar(length=train_steps, show_pos=True) as bar:
                for batch_iprs, batch_esm, batch_diam, batch_seqs, batch_dl2vec, batch_ics, batch_labels in train_loader:
                    bar.update(1)
                    batch_iprs = batch_iprs.to(device)
                    batch_esm = batch_esm.to(device)
                    batch_diam = batch_diam.to(device)
                    batch_dl2vec = batch_dl2vec.to(device)
                    batch_seqs = batch_seqs.to(device)
                    batch_ics = batch_ics.to(device)
                    batch_labels = batch_labels.to(device)
                    logits = net(batch_iprs, batch_esm, batch_diam, batch_seqs, batch_dl2vec, batch_ics)
                    loss = F.binary_cross_entropy(logits, batch_labels)
                    # el_loss = net.el_loss(normal_forms)
                    total_loss = loss #+ el_loss
                    train_loss += loss.detach().item()
                    # train_elloss = el_loss.detach().item()
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
                    for batch_iprs, batch_esm, batch_diam, batch_seqs, batch_dl2vec, batch_ics, batch_labels in valid_loader:
                        bar.update(1)
                        batch_iprs = batch_iprs.to(device)
                        batch_esm = batch_esm.to(device)
                        batch_diam = batch_diam.to(device)
                        batch_seqs = batch_seqs.to(device)
                        batch_labels = batch_labels.to(device)
                        batch_dl2vec = batch_dl2vec.to(device)
                        batch_ics = batch_ics.to(device)
                        logits = net(batch_iprs, batch_esm, batch_diam, batch_seqs, batch_dl2vec, batch_ics)
                        batch_loss = F.binary_cross_entropy(logits, batch_labels)
                        valid_loss += batch_loss.detach().item()
                        preds = np.append(preds, logits.detach().cpu().numpy())
                valid_loss /= valid_steps
                roc_auc = compute_roc(valid_labels, preds)
                print(f'Epoch {epoch}: Loss - {train_loss}, EL Loss: {train_elloss}, Valid loss - {valid_loss}, AUC - {roc_auc}')

            print('EL Loss', train_elloss)
            if valid_loss < best_loss:
                best_loss = valid_loss
                print('Saving model')
                th.save(net.state_dict(), model_file)

            scheduler.step()
            

    # Loading best model
    print('Loading the best model')
    net.load_state_dict(th.load(model_file))
    net.eval()
    with th.no_grad():
        test_steps = int(math.ceil(len(test_labels) / batch_size))
        test_loss = 0
        preds = []
        with ck.progressbar(length=test_steps, show_pos=True) as bar:
            for batch_iprs, batch_esm, batch_diam, batch_seqs, batch_dl2vec, batch_ics, batch_labels in test_loader:
                bar.update(1)
                batch_iprs = batch_iprs.to(device)
                batch_esm = batch_esm.to(device)
                batch_diam = batch_diam.to(device)
                batch_seqs = batch_seqs.to(device)
                batch_labels = batch_labels.to(device)
                batch_dl2vec = batch_dl2vec.to(device)
                batch_ics = batch_ics.to(device)
                logits = net(batch_iprs, batch_esm, batch_diam, batch_seqs, batch_dl2vec, batch_ics)
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


def load_go_graph(terms_dict):
    go = Ontology('data/go.obo', with_rels=True)
    src = []
    dst = []
    for go_id, s in terms_dict.items():
        for g_id in go.get_parents(go_id):
            if g_id in terms_dict:
                src.append(s)
                dst.append(terms_dict[g_id])
    src = th.tensor(src)
    dst = th.tensor(dst)
    graph = dgl.graph((src, dst), num_nodes=len(terms_dict))
    graph = graph.add_self_loop()
    # dgl.save_graphs('data/go_mf.bin', graph)
    return graph


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


class MLPModel(nn.Module):

    def __init__(self, nb_iprs, nb_gos, device, nodes=[1024,]):
        super().__init__()
        self.nb_gos = nb_gos
        input_length = nb_iprs
        net = []
        for hidden_dim in nodes:
            net.append(MLPBlock(input_length, hidden_dim))
            net.append(Residual(MLPBlock(hidden_dim, hidden_dim)))
            input_length = hidden_dim
        net.append(nn.Linear(input_length, nb_gos))
        # net.append(nn.ReLU())
        self.net = nn.Sequential(*net)
        
    def forward(self, features):
        return self.net(features)


class DGESMModel(nn.Module):

    def __init__(self, input_length, nb_gos, device, nodes=[1024,]):
        super().__init__()
        self.nb_gos = nb_gos
        net = []
        for hidden_dim in nodes:
            net.append(MLPBlock(input_length, hidden_dim))
            net.append(Residual(MLPBlock(hidden_dim, hidden_dim)))
            input_length = hidden_dim
        net.append(nn.Linear(input_length, nb_gos))
        # net.append(nn.ReLU())
        self.net = nn.Sequential(*net)
        
    def forward(self, features):
        return self.net(features)


class DGCNNModel(nn.Module):

    def __init__(self, nb_gos, device, nb_filters=64, max_kernel=257, hidden_dim=512):
        super().__init__()
        self.nb_gos = nb_gos
        # DeepGOCNN
        kernels = range(8, max_kernel, 8)
        convs = []
        for kernel in kernels:
            convs.append(
                nn.Sequential(
                    nn.Conv1d(21, nb_filters, kernel, device=device),
                    nn.MaxPool1d(MAXLEN - kernel + 1)
                ))
        self.convs = nn.ModuleList(convs)
        self.fc1 = MLPBlock(len(kernels) * nb_filters, hidden_dim)
        self.fc2 = MLPBlock(hidden_dim, nb_gos)
        
    def deepgocnn(self, proteins):
        n = proteins.shape[0]
        output = []
        for conv in self.convs:
            output.append(conv(proteins))
        output = th.cat(output, dim=1)
        output = self.fc1(output.view(n, -1))
        output = self.fc2(output)
        return output
    
    def forward(self, proteins):
        return self.deepgocnn(proteins)



class DGDL2VModel(nn.Module):

    def __init__(self, input_length, nb_gos, device, nodes=[1024,]):
        super().__init__()
        self.nb_gos = nb_gos
        net = []
        for hidden_dim in nodes:
            net.append(MLPBlock(input_length, hidden_dim))
            net.append(Residual(MLPBlock(hidden_dim, hidden_dim)))
            input_length = hidden_dim
        net.append(nn.Linear(input_length, nb_gos))
        # net.append(nn.ReLU())
        self.net = nn.Sequential(*net)
        
    def forward(self, features):
        return self.net(features)

    
class DG2Model(nn.Module):

    def __init__(
            self, go_graph, nb_gos, mlp, dgesm, dgcnn, dgdl2v, device,):
        super().__init__()
        self.nb_gos = nb_gos
        self.go_graph = go_graph
        self.mlp = mlp
        # self.mlp.requires_grad = False
        self.dgcnn = dgcnn
        # self.dgcnn.requires_grad = False
        self.dgesm = dgesm
        # self.dgesm.requires_grad = False
        self.dgdl2v = dgdl2v
        self.gcn1 = GATConv(5, 5, num_heads=1)
        self.gcn2 = GATConv(5, 1, num_heads=1)
        self.linear = nn.Linear(4, 1)

    def forward(self, iprs, esm, diam, seqs, dl2vec, ics):
        batch_size = iprs.shape[0]
        # dgzero = self.dgzero(iprs).reshape(-1, self.nb_gos, 1)
        mlp = self.mlp(iprs).reshape(-1, self.nb_gos, 1)
        dgcnn = self.dgcnn(seqs).reshape(-1, self.nb_gos, 1)
        esm = self.dgesm(esm).reshape(-1, self.nb_gos, 1)
        # diam = diam.reshape(-1, self.nb_gos, 1)
        # ics = ics.reshape(-1, self.nb_gos, 1)
        dl2vec = self.dgdl2v(dl2vec).reshape(-1, self.nb_gos, 1)
        x = th.cat((mlp, dgcnn, dl2vec, esm), dim=2)
        # x = x.reshape(batch_size * self.nb_gos, -1)
        # x = self.gcn1(self.go_graph[batch_size], x)
        # x = self.gcn2(self.go_graph[batch_size], x).reshape(-1, self.nb_gos)
        # logits = th.sigmoid(esm)
        logits = th.sigmoid(self.linear(x).reshape(-1, self.nb_gos))
        return logits
            
    
def load_data(data_root, ont, terms_file, go):
    terms_df = pd.read_pickle(terms_file)
    terms = terms_df['gos'].values.flatten()
    terms_dict = {v: i for i, v in enumerate(terms)}
    terms_set = set(terms_dict)
    print('Terms', len(terms))
    
    ipr_df = pd.read_pickle(f'{data_root}/{ont}/interpros.pkl')
    iprs = ipr_df['interpros'].values
    iprs_dict = {v:k for k, v in enumerate(iprs)}

    train_df = pd.read_pickle(f'{data_root}/{ont}/train_data.pkl')
    valid_df = pd.read_pickle(f'{data_root}/{ont}/valid_data.pkl')
    test_df = pd.read_pickle(f'{data_root}/{ont}/test_data.pkl')

    annots_df = pd.concat([train_df, valid_df])
    annotations = annots_df['prop_annotations'].values
    annotations = list(map(lambda x: set(x), annotations))
    cnt = Counter()
    max_n = 0
    for x in annotations:
        cnt.update(x & terms_set)
        
    max_n = cnt.most_common(1)[0][1]
    
    scores = {}
    for go_id, n in cnt.items():
        score = n / max_n
        scores[go_id] = score

    train_data = get_data(train_df, iprs_dict, terms_dict, ont, scores)
    valid_data = get_data(valid_df, iprs_dict, terms_dict, ont, scores)
    test_data = get_data(test_df, iprs_dict, terms_dict, ont, scores)

    return iprs_dict, terms_dict, train_data, valid_data, test_data, test_df

def get_data(df, iprs_dict, terms_dict, ont, go_scores):
    iprs = th.zeros((len(df), len(iprs_dict)), dtype=th.float32)
    esm = th.zeros((len(df), 1280), dtype=th.float32)
    dl2vec = th.zeros((len(df), 200), dtype=th.float32)
    diam = th.zeros((len(df), len(terms_dict)), dtype=th.float32)
    ics = th.zeros((len(df), len(terms_dict)), dtype=th.float32)
    seqs = th.zeros((len(df), 21, MAXLEN), dtype=th.float32)
    ic = th.zeros((len(terms_dict),), dtype=th.float32)
    for go_id, go_ind in  terms_dict.items():
        if go_id in go_scores:
            ic[go_ind] = go_scores[go_id]
    
    labels = th.zeros((len(df), len(terms_dict)), dtype=th.float32)
    for i, row in enumerate(df.itertuples()):
        for ipr in row.interpros:
            if ipr in iprs_dict:
                iprs[i, iprs_dict[ipr]] = 1
        esm[i, :] = th.FloatTensor(row.esm)
        dl2vec[i, :] = th.FloatTensor(getattr(row, f'{ont}_dl2vec'))
        # for go_id, score in row.diam_preds.items():
        #     if go_id in terms_dict:
        #         diam[i, terms_dict[go_id]] = float(score)
        seq = row.sequences
        seq = th.FloatTensor(to_onehot(seq))
        seqs[i, :, :] = seq
        ics[i, :] = ic
        for go_id in row.prop_annotations: # prop_annotations for full model
            if go_id in terms_dict:
                g_id = terms_dict[go_id]
                labels[i, g_id] = 1
    return iprs, esm, diam, seqs, dl2vec, ics, labels

if __name__ == '__main__':
    main()
