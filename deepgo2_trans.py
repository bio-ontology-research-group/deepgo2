import os
import click as ck
import pandas as pd
from utils import Ontology
import torch
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
from aminoacids import to_tokens, MAXLEN, AAINDEX
from dgl.nn import GraphConv, GATConv, GATv2Conv
import dgl
from dgl.dataloading import GraphDataLoader
from torch_utils import FastTensorDataLoader
from transformers import T5EncoderModel

@ck.command()
@ck.option(
    '--data-root', '-dr', default='data',
    help='Prediction model')
@ck.option(
    '--ont', '-ont', default='mf',
    help='Prediction model')
@ck.option(
    '--batch-size', '-bs', default=4,
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
    go_file = f'{data_root}/go.obo'
    model_file = f'{data_root}/{ont}/deepgo2_transa.th'
    terms_file = f'{data_root}/{ont}/terms.pkl'
    out_file = f'{data_root}/{ont}/predictions_deepgo2_trans.pkl'

    go = Ontology(f'{data_root}/go.obo', with_rels=True)
    loss_func = nn.BCELoss()
    terms_dict, train_data, valid_data, test_data, test_df = load_data(data_root, ont, terms_file)
    n_terms = len(terms_dict)
    
    net = DGTransModel(n_terms, device).to(device)
    print(net)
    # train_data, train_graphs, train_labels = train_data
    # valid_data, valid_graphs, valid_labels = valid_data
    # test_data, test_graphs, test_labels = test_data
    
    train_loader = FastTensorDataLoader(
        *train_data, batch_size=batch_size, shuffle=True)
    valid_loader = FastTensorDataLoader(
        *valid_data, batch_size=batch_size, shuffle=False)
    test_loader = FastTensorDataLoader(
        *test_data, batch_size=batch_size, shuffle=False)
    
    train_labels = train_data[2].detach().cpu().numpy()
    valid_labels = valid_data[2].detach().cpu().numpy()
    test_labels = test_data[2].detach().cpu().numpy()
    
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-4)
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
                    logits = net(batch_features)
                    loss = F.binary_cross_entropy(logits, batch_labels)
                    total_loss = loss
                    train_loss += loss.detach().item()
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()
                    
            train_loss /= train_steps
            
            print('Validation')
            net.eval()
            with torch.no_grad():
                valid_steps = int(math.ceil(len(valid_labels) / batch_size))
                valid_loss = 0
                preds = []
                with ck.progressbar(length=valid_steps, show_pos=True) as bar:
                    for batch_features, batch_graphs, batch_labels in valid_loader:
                        bar.update(1)
                        batch_features = batch_features.to(device)
                        batch_graphs = batch_graphs.to(device)
                        batch_labels = batch_labels.to(device)
                        logits = net(batch_features)
                        batch_loss = F.binary_cross_entropy(logits, batch_labels)
                        valid_loss += batch_loss.detach().item()
                        preds = np.append(preds, logits.detach().cpu().numpy())
                valid_loss /= valid_steps
                roc_auc = compute_roc(valid_labels, preds)
                print(f'Epoch {epoch}: Loss - {train_loss}, Valid loss - {valid_loss}, AUC - {roc_auc}')

            if valid_loss < best_loss:
                best_loss = valid_loss
                print('Saving model')
                torch.save(net.state_dict(), model_file)

            #scheduler.step()
            

    # Loading best model
    print('Loading the best model')
    net.load_state_dict(torch.load(model_file))
    net.eval()
    with torch.no_grad():
        test_steps = int(math.ceil(len(test_labels) / batch_size))
        test_loss = 0
        preds = []
        with ck.progressbar(length=test_steps, show_pos=True) as bar:
            for batch_features, batch_graphs, batch_labels in test_loader:
                bar.update(1)
                batch_features = batch_features.to(device)
                batch_graphs = batch_graphs.to(device)
                batch_labels = batch_labels.to(device)
                logits = net(batch_features)
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


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class DGTransModel(nn.Module):

    def __init__(self, nb_gos, device, ninp=1024, ntoken=22, nlayers=6, nhead=1, nhid=1024, dropout=0.5):
        super().__init__()
        self.nb_gos = nb_gos

        net = []
        net.append(MLPBlock(ninp, nhid))
        net.append(Residual(MLPBlock(nhid, nhid)))
        net.append(nn.Linear(nhid, nb_gos))
        net.append(nn.Sigmoid())
        self.net = nn.Sequential(*net)

        self.transformer = T5EncoderModel.from_pretrained('Rostlab/prot_t5_xl_uniref50')
        # self.src_mask = None
        # self.pos_encoder = PositionalEncoding(ninp, dropout)
        # encoder_layers = nn.TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        # self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        # self.encoder = nn.Embedding(ntoken, ninp)
        # self.ninp = ninp
        # self.decoder = nn.Linear(ninp, ntoken)

    #     self.init_weights()

    # def _generate_square_subsequent_mask(self, sz):
    #     mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    #     mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    #     return mask

    # def init_weights(self):
    #     initrange = 0.1
    #     nn.init.uniform_(self.encoder.weight, -initrange, initrange)
    #     nn.init.zeros_(self.decoder.bias)
    #     nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, has_mask=True):
        # if has_mask:
        #     device = src.device
        #     if self.src_mask is None or self.src_mask.size(0) != len(src):
        #         mask = self._generate_square_subsequent_mask(len(src)).to(device)
        #         self.src_mask = mask
        # else:
        #     self.src_mask = None

        # src = self.encoder(src) * math.sqrt(self.ninp)
        # src = self.pos_encoder(src)
        # output = self.transformer_encoder(src, self.src_mask)
        # output = torch.mean(output, axis=1)
        # output = self.decoder(output)
        ouput = self.transformer(src)
        return self.net(output)
    
    
def load_data(data_root, ont, terms_file):
    terms_df = pd.read_pickle(terms_file)
    terms = terms_df['gos'].values.flatten()
    terms_dict = {v: i for i, v in enumerate(terms)}
    print('Terms', len(terms))
    
    train_df = pd.read_pickle(f'{data_root}/{ont}/train_data.pkl')
    valid_df = pd.read_pickle(f'{data_root}/{ont}/valid_data.pkl')
    test_df = pd.read_pickle(f'{data_root}/{ont}/test_data.pkl')

    train_data = get_data(train_df, terms_dict)
    valid_data = get_data(valid_df, terms_dict)
    test_data = get_data(test_df, terms_dict)

    return terms_dict, train_data, valid_data, test_data, test_df

def get_data(df, terms_dict):
    graphs = torch.zeros((len(df), 1, 1), dtype=torch.float32)
    data = torch.zeros((len(df), MAXLEN), dtype=torch.int32)
    labels = torch.zeros((len(df), len(terms_dict)), dtype=torch.float32)
    index = []
    with ck.progressbar(length=len(df), show_pos=True) as bar:
        for i, row in enumerate(df.itertuples()):
            # graph_path = 'data/graphs/' + row.proteins + '.bin'
            # if not os.path.exists(graph_path):
            #     continue
            # gs, _ = dgl.load_graphs(graph_path)
            # g = gs[0]
            # nnodes = g.num_nodes()
            # graphs[i, :nnodes, :nnodes] = g.adj().to_dense()
            tokens = to_tokens(row.sequences)
            data[i, :] = torch.IntTensor(tokens)
            index.append(i)
            for go_id in row.prop_annotations: # prop_annotations for full model
                if go_id in terms_dict:
                    g_id = terms_dict[go_id]
                    labels[i, g_id] = 1
            bar.update(1)
    labels = labels[index, :]
    return data, graphs, labels



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
