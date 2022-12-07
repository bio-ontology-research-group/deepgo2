import click as ck
import pandas as pd
from utils import Ontology
import torch as th
import torch
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
from torch_utils import FastTensorDataLoader
import mowl
mowl.init_jvm("10g")

import java
from mowl.datasets import Dataset as MOWLDataset
from mowl.base_models.alcmodel import EmbeddingALCModel
from mowl.models.falcon.module import FALCONModule 
from mowl.owlapi import OWLAPIAdapter
from org.semanticweb.owlapi.model import AxiomType


owlapi = OWLAPIAdapter()

@ck.command()
@ck.option(
    '--data-root', '-dr', default='data',
    help='Prediction model')
@ck.option(
    '--ont', '-ont', default='mf',
    help='Prediction model')
@ck.option(
    '--batch-size', '-bs', default=64,
    help='Batch size for training')
@ck.option(
    '--epochs', '-ep', default=128,
    help='Training epochs')
@ck.option(
    '--model-name', '-m', default='deepgo2_falcon', help='Model name')
@ck.option(
    '--load', '-ld', is_flag=True, help='Load Model?')
@ck.option(
    '--device', '-d', default='cuda:1',
    help='Device')
def main(data_root, ont, batch_size, epochs, model_name, load, device):
    go_file = f'{data_root}/go.owl'
    model_file = f'{data_root}/{ont}/{model_name}.th'
    terms_file = f'{data_root}/{ont}/terms.pkl'
    out_file = f'{data_root}/{ont}/predictions_{model_name}.pkl'

    go = Ontology(f'{data_root}/go.obo', with_rels=True)
    terms_dict, terms_index, train_data, valid_data, test_data, test_df, owl_dataset = load_data(data_root, ont, terms_file)
    n_terms = len(terms_dict)
    train_features, train_labels = train_data
    valid_features, valid_labels = valid_data
    test_features, test_labels = test_data
    
    model = DGFalconModel(
        owl_dataset, batch_size, model_file, 2560, n_terms,
        epochs=epochs, device=device)
    if not load:
        model.train(train_data, valid_data, terms_index, batch_size)
    preds = model.evaluate(test_data, terms_index, batch_size)

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


class DGFALCONModule(FALCONModule):

    def __init__(self, ninp, nclasses, nentities, nrelations,
                 anon_e, nb_gos, embed_dim=128, device='cpu'):
        heads_dict = {}
        tails_dict = {}
        super().__init__(
            nclasses,
            nentities,
            nrelations + 1,
            heads_dict,
            tails_dict,
            embed_dim=embed_dim,
            anon_e=anon_e,)
        self.embed_dim = embed_dim
        self.c_bias = nn.Embedding(nclasses, 1)
        self.ci_embedding = nn.Embedding(nclasses, embed_dim)
        # self.r_embedding = nn.Embedding(nrelations + 1, embed_dim * embed_dim)
        k = math.sqrt(1 / embed_dim)
        nn.init.uniform_(self.c_embedding.weight, -k, k)
        nn.init.uniform_(self.ci_embedding.weight, -k, k)
        nn.init.uniform_(self.c_bias.weight, -k, k)
        
        self.project = nn.Sequential(
            MLPBlock(ninp, embed_dim),
            MLPBlock(embed_dim, embed_dim)
        )
        self.mem_net = nn.Sequential(
            nn.Linear(embed_dim * 2, 1),
            nn.Sigmoid()
        )
        self.hasFuncIndex = th.LongTensor([nrelations,]).to(device)

    def _get_c_fs_batch(self, c_emb, e_emb):
        e_emb = e_emb.unsqueeze(
            dim=0).repeat(c_emb.size()[0], 1, 1)
        c_emb = c_emb.unsqueeze(dim=1).expand_as(e_emb)
        return self._mem(c_emb, e_emb).squeeze(dim=-1)

    def _get_r_fs_batch(self, r_emb, e_emb):
        e_emb = e_emb.unsqueeze(
            dim=0).repeat(r_emb.size()[0], 1, 1)
        r_emb = r_emb.unsqueeze(dim=1).expand_as(e_emb)
        c_emb = e_emb + r_emb
        return self._mem(c_emb, e_emb).squeeze(dim=-1)

    def _mem(self, c_emb, e_emb):
        # emb = th.cat([c_emb, e_emb], dim=-1)
        # return self.mem_net(emb)
        return th.sigmoid(th.sum(c_emb * e_emb, dim=-1, keepdims=True))

    def function_predict(self, features, terms_index):
        x = self.project(features)
        go_embed = self.ci_embedding(terms_index)
        hasFunc = self.r_embedding(self.hasFuncIndex)
        # go_bias = self.c_bias(terms_index).view(1, -1)
        x = x + hasFunc
        x = th.matmul(x, go_embed.T)
        logits = th.sigmoid(x)
        return logits

        #go_embed = go_embed.unsqueeze(
        #    dim=0).repeat(x.size()[0], 1, 1)
        #x = x.unsqueeze(dim=1).expand_as(go_embed)
        #return self._mem(x, go_embed).squeeze(dim=-1)
        
        
    def forward(self, axiom, x, e_emb):
        # e_emb = self.project(e_emb)
        return super().forward(axiom, x, e_emb)
        
    
class DGFalconModel(EmbeddingALCModel):

    def __init__(self, owl_dataset, batch_size, model_file, input_length,
                 nb_gos, embed_dim=1024, epochs=12, device='cpu'):
        super().__init__(
            owl_dataset, batch_size, model_filepath=model_file)
        self.nb_gos = nb_gos
        self.embed_dim = embed_dim
        self.anon_e = 4
        self.device = device
        self.epochs = epochs
        self.nclasses = len(self.dataset.classes)
        self.nentities = len(self.dataset.individuals)
        self.nrelations = len(self.dataset.object_properties)
        
        self.model = DGFALCONModule(
            input_length,
            self.nclasses,
            self.nentities,
            self.nrelations,
            self.anon_e,
            nb_gos,
            embed_dim=embed_dim,
            device=device,
        ).to(device)
        
    def train(self, train_data, valid_data, terms_index, batch_size=64):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.003)
        best_loss = float('inf')

        train_features, train_labels = train_data
        valid_features, valid_labels = valid_data

        train_loader = FastTensorDataLoader(
            *train_data, batch_size=batch_size, shuffle=True)
        valid_loader = FastTensorDataLoader(
            *valid_data, batch_size=batch_size, shuffle=False)
    

        terms_index = terms_index.to(self.device)
        print('Training the model')
        for epoch in range(self.epochs):
            self.model.train()

            # e_emb_1 = self.model.e_embedding.weight.detach()[:self.anon_e // 2] \
            #     + torch.normal(0, 0.1, size=(self.anon_e // 2, self.embed_dim)).to(self.device)
            # e_emb_2 = torch.rand(self.anon_e // 2, self.embed_dim).to(self.device)
            # torch.nn.init.xavier_uniform_(e_emb_2)
            # e_emb = torch.cat([anon_e_emb_1, anon_e_emb_2], dim=0)

            train_loss = 0
            train_falcon_loss = 0
            train_falcon_loss2 = 0
            train_steps = int(math.ceil(len(train_labels) / batch_size))
            with ck.progressbar(length=train_steps, show_pos=True) as bar:
                for batch_features, batch_labels in train_loader:
                    batch_features = batch_features.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    logits = self.model.function_predict(batch_features, terms_index)
                    loss = F.binary_cross_entropy(logits, batch_labels)

                    c_inds = th.randint(self.nclasses, (batch_size,)).to(self.device)
                    c_label = th.ones((batch_size, 1), dtype=th.float32).to(self.device)
                    c_emb = self.model.c_embedding(c_inds)
                    ci_emb = self.model.ci_embedding(c_inds)
                    ci_loss = F.binary_cross_entropy(
                        self.model._mem(c_emb, ci_emb), c_label)
                    
                    train_loss += loss.detach().item()
                    axiom_loss = 0
                    p_emb = self.model.project(batch_features)
                    emb = th.cat([ci_emb, p_emb], dim=0)

                    for axiom, dataloader in self.training_dataloaders.items():
                        dataloader = iter(dataloader)
                        batch_data = next(dataloader)
                        x = batch_data[0]
                        xx = batch_data[0].to(self.device)
                        label = th.zeros((x.shape[0], 1), dtype=th.float32)
                        axiom_loss += self.model(axiom, xx, emb)
                    optimizer.zero_grad()
                    (loss + ci_loss + axiom_loss).backward()
                    optimizer.step()
                    train_falcon_loss += axiom_loss.detach().item()
                    train_falcon_loss2 += ci_loss.detach().item()
                    
                    bar.update(1)
                train_loss /= train_steps
                train_falcon_loss /= train_steps * len(self.training_dataloaders)
                train_falcon_loss2 /= train_steps
                

            loss = 0
            with torch.no_grad():
                self.model.eval()
                valid_loss = 0
                valid_steps = int(math.ceil(len(valid_labels) / batch_size))
                with ck.progressbar(length=valid_steps, show_pos=True) as bar:
                    for batch_features, batch_labels in valid_loader:
                        batch_features = batch_features.to(self.device)
                        batch_labels = batch_labels.to(self.device)
                        logits = self.model.function_predict(batch_features, terms_index)
                        loss = F.binary_cross_entropy(logits, batch_labels)
                        valid_loss += loss.detach().item()
                        bar.update(1)
                    valid_loss /= valid_steps
                    
            if best_loss > valid_loss:
                best_loss = valid_loss
                torch.save(self.model.state_dict(), self.model_filepath)
                print('Saved model', self.model_filepath)
            print(f'Epoch {epoch}: Train loss: {train_loss}, {train_falcon_loss}, {train_falcon_loss2} Valid loss: {valid_loss}')


    def evaluate(self, test_data, terms_index, batch_size=64):

        test_features, test_labels = test_data

        test_loader = FastTensorDataLoader(
            *test_data, batch_size=batch_size, shuffle=False)

        terms_index = terms_index.to(self.device)
        # Loading best model
        print('Loading the best model')
        self.model.load_state_dict(th.load(self.model_filepath))
        self.model.eval()
        with th.no_grad():
            test_steps = int(math.ceil(len(test_labels) / batch_size))
            test_loss = 0
            preds = []
            with ck.progressbar(length=test_steps, show_pos=True) as bar:
                for batch_features, batch_labels in test_loader:
                    bar.update(1)
                    batch_features = batch_features.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    logits = self.model.function_predict(batch_features, terms_index)
                    batch_loss = F.binary_cross_entropy(logits, batch_labels)
                    test_loss += batch_loss.detach().cpu().item()
                    preds = np.append(preds, logits.detach().cpu().numpy())
                test_loss /= test_steps
            preds = preds.reshape(-1, len(terms_index))
            roc_auc = compute_roc(test_labels, preds)
            print(f'Test Loss - {test_loss}, AUC - {roc_auc}')

        return preds
    

def load_data(data_root, ont, terms_file):
    terms_df = pd.read_pickle(terms_file)
    terms = terms_df['gos'].values.flatten()
    
    train_ont = owlapi.owl_manager.loadOntologyFromOntologyDocument(
        java.io.File(f'{data_root}/{ont}/train_data.owl'))
    # valid_ont = owlapi.owl_manager.loadOntologyFromOntologyDocument(
    #     java.io.File(f'{data_root}/{ont}/valid_data.owl'))
    # test_ont = owlapi.owl_manager.loadOntologyFromOntologyDocument(
    #     java.io.File(f'{data_root}/{ont}/test_data.owl'))
    owl_dataset = MOWLDataset(train_ont)

    terms_index = []
    new_terms = []
    for go_id in terms:
        go_cls = go_id.replace('GO:', 'http://purl.obolibrary.org/obo/GO_')
        go_cls = owlapi.create_class(go_cls)
        if go_cls in owl_dataset.class_to_id:
            terms_index.append(owl_dataset.class_to_id[go_cls])
            new_terms.append(go_id)
    
    terms_dict = {v: i for i, v in enumerate(new_terms)}
    print('Terms', len(terms_dict))
    terms_df = pd.DataFrame({'gos': new_terms})
    terms_df.to_pickle(terms_file)

    terms_index = torch.tensor(terms_index)
    train_df = pd.read_pickle(f'{data_root}/{ont}/train_data.pkl')
    valid_df = pd.read_pickle(f'{data_root}/{ont}/valid_data.pkl')
    test_df = pd.read_pickle(f'{data_root}/{ont}/test_data.pkl')

    print('Training data', len(train_df))
    print('Validation data', len(valid_df))
    print('Testing data', len(test_df))
    
    train_data = get_data(train_df, terms_dict)
    valid_data = get_data(valid_df, terms_dict)
    test_data = get_data(test_df, terms_dict)


    return terms_dict, terms_index, train_data, valid_data, test_data, test_df, owl_dataset

def get_data(df, terms_dict):
    data = th.zeros((len(df), 2560), dtype=th.float32)
    labels = th.zeros((len(df), len(terms_dict)), dtype=th.float32)
    for i, row in enumerate(df.itertuples()):
        data[i, :] = th.FloatTensor(row.esm2)
        for go_id in row.prop_annotations: # prop_annotations for full model
            if go_id in terms_dict:
                g_id = terms_dict[go_id]
                labels[i, g_id] = 1
    return data, labels


    
if __name__ == '__main__':
    main()
