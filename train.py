import click as ck
import pandas as pd
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
from deepgo.torch_utils import FastTensorDataLoader
from deepgo.utils import Ontology, propagate_annots
from multiprocessing import Pool
from functools import partial
from deepgo.data import load_data, load_normal_forms
from deepgo.models import DeepGOModel

@ck.command()
@ck.option(
    '--data-root', '-dr', default='data',
    help='Prediction model')
@ck.option(
    '--ont', '-ont', default='mf',
    help='Prediction model')
@ck.option(
    '--model-name', '-m', default='deepgozero_esm',
    help='Prediction model. Choices: deepgozero_esm, deepgozero_esm_plus,\
    deepgozero, deepgozero_plus')
@ck.option(
    '--test-data-name', '-td', default='test',
    help='Test data set. Choices: test, nextprot')
@ck.option(
    '--batch-size', '-bs', default=37,
    help='Batch size for training')
@ck.option(
    '--epochs', '-ep', default=128,
    help='Training epochs')
@ck.option(
    '--load', '-ld', is_flag=True, help='Load Model?')
@ck.option(
    '--device', '-d', default='cuda:1',
    help='Device')
def main(data_root, ont, model_name, test_data_name, batch_size, epochs, load, device):
    """
    This script is used to train DeepGO models
    """
    if model_name.find('plus') != -1:
        go_norm_file = f'{data_root}/go-plus.norm'
        go_file = f'{data_root}/go-plus.obo'
    else:
        go_norm_file = f'{data_root}/go.norm'
        go_file = f'{data_root}/go.obo'
        
    model_file = f'{data_root}/{ont}/{model_name}.th'
    terms_file = f'{data_root}/{ont}/terms.pkl'
    out_file = f'{data_root}/{ont}/{test_data_name}_predictions_{model_name}.pkl'

    # Load Gene Ontology and Normalized axioms
    go = Ontology(go_file, with_rels=True)

    # Load the datasets
    if model_name.find('esm') != -1:
        features_length = 2560
        features_column = 'esm2'
    else:
        features_length = None # Optional in this case
        features_column = 'interpros'
    test_data_file = f'{test_data_name}_data.pkl'
    iprs_dict, terms_dict, train_data, valid_data, test_data, test_df = load_data(
        data_root, ont, terms_file, features_length, features_column, test_data_file)
    n_terms = len(terms_dict)
    if features_column == 'interpros':
        features_length = len(iprs_dict)
    train_features, train_labels = train_data
    valid_features, valid_labels = valid_data
    test_features, test_labels = test_data
    valid_labels = valid_labels.detach().cpu().numpy()
    test_labels = test_labels.detach().cpu().numpy()

    # Load normal forms
    nf1, nf2, nf3, nf4, relations, zero_classes = load_normal_forms(
        go_norm_file, terms_dict)
    n_rels = len(relations)
    n_zeros = len(zero_classes)

    normal_forms = nf1, nf2, nf3, nf4
    nf1 = th.LongTensor(nf1).to(device)
    nf2 = th.LongTensor(nf2).to(device)
    nf3 = th.LongTensor(nf3).to(device)
    nf4 = th.LongTensor(nf4).to(device)
    normal_forms = nf1, nf2, nf3, nf4

    
    # Create DataLoaders
    train_loader = FastTensorDataLoader(
        *train_data, batch_size=batch_size, shuffle=True)
    valid_loader = FastTensorDataLoader(
        *valid_data, batch_size=batch_size, shuffle=False)
    test_loader = FastTensorDataLoader(
        *test_data, batch_size=batch_size, shuffle=False)
    

    loss_func = nn.BCELoss()
    net = DeepGOModel(features_length, n_terms, n_zeros, n_rels, device).to(device)
    print(net)

    optimizer = th.optim.Adam(net.parameters(), lr=5e-4)
    scheduler = MultiStepLR(optimizer, milestones=[5, 20], gamma=0.1)

    best_loss = 10000.0
    if not load:
        print('Training the model')
        for epoch in range(epochs):
            net.train()
            train_loss = 0
            train_elloss = 0
            train_steps = int(math.ceil(len(train_labels) / batch_size))
            with ck.progressbar(length=train_steps, show_pos=True) as bar:
                for batch_features, batch_labels in train_loader:
                    bar.update(1)
                    batch_features = batch_features.to(device)
                    batch_labels = batch_labels.to(device)
                    logits = net(batch_features)
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
                valid_steps = int(math.ceil(len(valid_labels) / batch_size))
                valid_loss = 0
                preds = []
                with ck.progressbar(length=valid_steps, show_pos=True) as bar:
                    for batch_features, batch_labels in valid_loader:
                        bar.update(1)
                        batch_features = batch_features.to(device)
                        batch_labels = batch_labels.to(device)
                        logits = net(batch_features)
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
        valid_steps = int(math.ceil(len(valid_labels) / batch_size))
        valid_loss = 0
        preds = []
        with ck.progressbar(length=valid_steps, show_pos=True) as bar:
            for batch_features, batch_labels in valid_loader:
                bar.update(1)
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                logits = net(batch_features)
                batch_loss = F.binary_cross_entropy(logits, batch_labels)
                valid_loss += batch_loss.detach().item()
                preds = np.append(preds, logits.detach().cpu().numpy())
        valid_loss /= valid_steps

    with th.no_grad():
        test_steps = int(math.ceil(len(test_labels) / batch_size))
        test_loss = 0
        preds = []
        with ck.progressbar(length=test_steps, show_pos=True) as bar:
            for batch_features, batch_labels in test_loader:
                bar.update(1)
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                logits = net(batch_features)
                batch_loss = F.binary_cross_entropy(logits, batch_labels)
                test_loss += batch_loss.detach().cpu().item()
                preds.append(logits.detach().cpu().numpy())
            test_loss /= test_steps
        preds = np.concatenate(preds)
        roc_auc = compute_roc(test_labels, preds)
        print(f'Valid Loss - {valid_loss}, Test Loss - {test_loss}, Test AUC - {roc_auc}')

    # return
    preds = list(preds)
    # Propagate scores using ontology structure
    with Pool(32) as p:
        preds = p.map(partial(propagate_annots, go=go, terms_dict=terms_dict), preds)

    test_df['preds'] = preds

    test_df.to_pickle(out_file)




if __name__ == '__main__':
    main()
