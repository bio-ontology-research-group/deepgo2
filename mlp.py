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
    '--device', '-d', default='cuda:0',
    help='Device')
def main(data_root, ont, batch_size, epochs, load, device):
    go_file = f'{data_root}/go.norm'
    model_file = f'{data_root}/{ont}/mlp.th'
    terms_file = f'{data_root}/{ont}/terms.pkl'
    out_file = f'{data_root}/{ont}/predictions_mlp.pkl'

    go = Ontology(f'{data_root}/go.obo', with_rels=True)
    loss_func = nn.BCELoss()
    iprs_dict, terms_dict, train_data, valid_data, test_data, test_df = load_data(data_root, ont, terms_file)
    n_terms = len(terms_dict)
    n_iprs = len(iprs_dict)
    
    net = MLPModel(n_iprs, 1280, n_terms, device).to(device)
    print(net)
    train_iprs, train_esm, train_diam, train_labels = train_data
    valid_iprs, valid_esm, valid_diam, valid_labels = valid_data
    test_iprs, test_esm, test_diam, test_labels = test_data
    
    train_loader = FastTensorDataLoader(
        *train_data, batch_size=batch_size, shuffle=True)
    valid_loader = FastTensorDataLoader(
        *valid_data, batch_size=batch_size, shuffle=False)
    test_loader = FastTensorDataLoader(
        *test_data, batch_size=batch_size, shuffle=False)
    
    valid_labels = valid_labels.detach().cpu().numpy()
    test_labels = test_labels.detach().cpu().numpy()
    
    optimizer = th.optim.Adam(net.parameters(), lr=1e-3)
    scheduler = MultiStepLR(optimizer, milestones=[5,], gamma=0.1)

    best_loss = 10000.0
    if not load:
        print('Training the model')
        for epoch in range(epochs):
            net.train()
            train_loss = 0
            train_steps = int(math.ceil(len(train_labels) / batch_size))
            with ck.progressbar(length=train_steps, show_pos=True) as bar:
                for batch_iprs, batch_esm, batch_diam, batch_labels in train_loader:
                    bar.update(1)
                    batch_iprs = batch_iprs.to(device)
                    batch_esm = batch_esm.to(device)
                    batch_diam = batch_diam.to(device)
                    batch_labels = batch_labels.to(device)
                    logits = net(batch_iprs, batch_esm, batch_diam)
                    loss = F.binary_cross_entropy(logits, batch_labels)
                    train_loss += loss.detach().item()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
            train_loss /= train_steps
            
            print('Validation')
            net.eval()
            with th.no_grad():
                valid_steps = int(math.ceil(len(valid_labels) / batch_size))
                valid_loss = 0
                preds = []
                with ck.progressbar(length=valid_steps, show_pos=True) as bar:
                    for batch_iprs, batch_esm, batch_diam, batch_labels in valid_loader:
                        bar.update(1)
                        batch_iprs = batch_iprs.to(device)
                        batch_esm = batch_esm.to(device)
                        batch_diam = batch_diam.to(device)
                        batch_labels = batch_labels.to(device)
                        logits = net(batch_iprs, batch_esm, batch_diam)
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
            for batch_iprs, batch_esm, batch_diam, batch_labels in test_loader:
                bar.update(1)
                batch_iprs = batch_iprs.to(device)
                batch_esm = batch_esm.to(device)
                batch_diam = batch_diam.to(device)
                batch_labels = batch_labels.to(device)
                logits = net(batch_iprs, batch_esm, batch_diam)
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


class MLPModel(nn.Module):

    def __init__(self, nb_iprs, esm_length, nb_gos, device, hidden_dim=512, embed_dim=1024, margin=0.1):
        super().__init__()
        self.nb_gos = nb_gos
        iprs_net = []
        iprs_net.append(MLPBlock(nb_iprs, hidden_dim))
        iprs_net.append(Residual(MLPBlock(hidden_dim, hidden_dim)))
        iprs_net.append(nn.Linear(hidden_dim, nb_gos))
        iprs_net.append(nn.ReLU())
        self.iprs_net = nn.Sequential(*iprs_net)

        esm_net = []
        esm_net.append(MLPBlock(esm_length, hidden_dim))
        esm_net.append(Residual(MLPBlock(hidden_dim, hidden_dim)))
        esm_net.append(nn.Linear(hidden_dim, nb_gos))
        esm_net.append(nn.ReLU())
        self.esm_net = nn.Sequential(*esm_net)

        self.linear = nn.Linear(3, 1)

        
    def forward(self, iprs, esm, diam):
        batch_size = iprs.shape[0]
        iprs_out = self.iprs_net(iprs).reshape(-1, self.nb_gos, 1)
        esm_out = self.esm_net(esm).reshape(-1, self.nb_gos, 1)
        diam = diam.reshape(-1, self.nb_gos, 1)
        x = th.cat((iprs_out, esm_out, diam), dim=2)
        logits = th.sigmoid(self.linear(x)).reshape(-1, self.nb_gos)
        return logits

    
def load_data(data_root, ont, terms_file):
    terms_df = pd.read_pickle(terms_file)
    terms = terms_df['gos'].values.flatten()
    terms_dict = {v: i for i, v in enumerate(terms)}
    print('Terms', len(terms))
    
    ipr_df = pd.read_pickle(f'{data_root}/{ont}/interpros.pkl')
    iprs = ipr_df['interpros'].values
    iprs_dict = {v:k for k, v in enumerate(iprs)}

    train_df = pd.read_pickle(f'{data_root}/{ont}/train_data_diam.pkl')
    valid_df = pd.read_pickle(f'{data_root}/{ont}/valid_data_diam.pkl')
    test_df = pd.read_pickle(f'{data_root}/{ont}/test_data_diam.pkl')

    train_data = get_data(train_df, iprs_dict, terms_dict)
    valid_data = get_data(valid_df, iprs_dict, terms_dict)
    test_data = get_data(test_df, iprs_dict, terms_dict)

    return iprs_dict, terms_dict, train_data, valid_data, test_data, test_df

def get_data(df, iprs_dict, terms_dict):
    iprs = th.zeros((len(df), len(iprs_dict)), dtype=th.float32)
    esm = th.zeros((len(df), 1280), dtype=th.float32)
    diam = th.zeros((len(df), len(terms_dict)), dtype=th.float32)
    
    labels = th.zeros((len(df), len(terms_dict)), dtype=th.float32)
    for i, row in enumerate(df.itertuples()):
        for ipr in row.interpros:
            if ipr in iprs_dict:
                iprs[i, iprs_dict[ipr]] = 1
        esm[i, :] = th.FloatTensor(row.esm)
        for go_id, score in row.diam_preds.items():
            if go_id in terms_dict:
                diam[i, terms_dict[go_id]] = float(score)
        for go_id in row.prop_annotations: # prop_annotations for full model
            if go_id in terms_dict:
                g_id = terms_dict[go_id]
                labels[i, g_id] = 1
    return iprs, esm, diam, labels

if __name__ == '__main__':
    main()
