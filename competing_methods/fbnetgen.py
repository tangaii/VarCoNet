parent_path = r'/.../baselines'
import sys
sys.path.append(parent_path)
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim import Adam
from utils import FBNETGENLoader
from FBNETGEN.model.model import FBNETGEN
from FBNETGEN.train import BasicTrain
import os
import pickle
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold
import copy
import argparse
from pathlib import Path
    
def train(x, y, encoder_model, optimizer, loss_func, num_classes):
    encoder_model.train()
    optimizer.zero_grad()
    z = encoder_model(x, x.shape[1])
    loss = loss_func(z, F.one_hot(y, num_classes=num_classes).float())
    loss.backward()
    optimizer.step()
    z = z[:,-1].detach().cpu().numpy()
    y = y.to(torch.device("cpu")).numpy()
    auc_score = roc_auc_score(y, z)
    return loss.item(),auc_score


def test(encoder_model, test_data_loader, batch_size, loss_func, num_classes, device):
    encoder_model.eval()
    with torch.no_grad():
        zs = []
        ys = []
        for (x,y) in test_data_loader:
            zs.append(encoder_model(x.to(device)))
            ys.append(y)
        z = torch.cat(zs,dim=0)
        y = torch.cat(ys,dim=0)
        loss = loss_func(z, F.one_hot(y, num_classes=num_classes).float().to(device))
        z = z[:,-1].cpu().numpy()
        y = y.numpy()
        auc_score = roc_auc_score(y, z)
                   
    return loss.item(), auc_score, z, y


def main(config):
    path = config['path_data']
           
    names = []
    with open(os.path.join(path,'ABIDEI_nilearn_names.txt'), 'r') as f:
        for line in f:
            names.append(line.strip())
    
    data_list = np.load(os.path.join(path,'ABIDEI_nilearn_' + config['atlas'] + '.npz'))
    data = []
    for key in data_list:
        data.append(data_list[key][:config['length'],:].T)
        
    
    names_unique, counts = np.unique(names, return_counts=True)
    names_dupl = names_unique[counts > 1]
    pos_duplicates = []
    names_duplicate = []
    for name in names_dupl:
        temp = np.where(np.array(names) == name)[0]
        for t in temp:
            pos_duplicates.append(t)
            names_duplicate.append(name)
    names_unique = names_unique[counts == 1]
    pos_unique = []
    for name in names_unique:
        pos_unique.append(np.where(np.array(names) == name)[0][0])
    train_DATA = [data[i] for i in pos_duplicates]
    data = [data[i] for i in pos_unique]
    y = np.load(os.path.join(path,'ABIDEI_nilearn_classes.npy'))   
    Y_train = y[pos_duplicates]
    y = y[pos_unique]
    
    ext_test = list(range(51456,51494))
    names_ext_test = []
    for name in ext_test:      
        if 'sub-00'+str(name) in names_unique:
            names_ext_test.append('sub-00'+str(name))
    names = []
    for name in names_unique:
        if name not in names_ext_test:
            names.append(name)
            
    pos_ext_test = []
    for name in names_ext_test:
        pos_ext_test.append(np.where(np.array(names_unique) == name)[0][0])
    ext_test_data = [data[i] for i in pos_ext_test]
    y_ext_test = y[pos_ext_test]
    
    pos = []
    for name in names:
        pos.append(np.where(np.array(names_unique) == name)[0][0])
    data = [data[i] for i in pos]
    y = y[pos]
    
     
    device = torch.device(config['device']) if torch.cuda.is_available() else torch.device("cpu") 
    
    model_config = config['model_config'] 
    
    '''------------------------------------KFold CV------------------------------------'''
    '''
    names_train_all = []
    names_val_all = []
    names_test_all = []
    for i in range(10):
        skf = StratifiedKFold(n_splits=10, shuffle = True, random_state=42+i)
        for j, (train_index, test_index) in enumerate(skf.split(data, y)):
            train_data = [data[i] for i in train_index]
            test_data = [data[n] for n in test_index]
            y_train = y[train_index]
            y_test = y[test_index]
            names_train = [names[n] for n in train_index]
            names_test = [names[n] for n in test_index]
            train_data, val_data, y_train, y_val, train_idx, val_idx = train_test_split(train_data,
                                                                                        y_train,
                                                                                        np.arange(len(train_data)),
                                                                                        test_size=0.15, 
                                                                                        random_state=42,
                                                                                        stratify=y_train)
            names_val = [names_train[n] for n in val_idx]
            names_train = [names_train[n] for n in train_idx]
            train_data = train_DATA + train_data
            y_train = np.concatenate((Y_train, y_train))
            names_train = names_duplicate + names_train
            train_loader, node_size, node_feature_size, timeseries = FBNETGENLoader(train_data,
                                                                                    y_train,
                                                                                    config['batch_size'])
            val_loader,_,_,_ = FBNETGENLoader(val_data, y_val, config['batch_size'])
            test_loader,_,_,_  = FBNETGENLoader(test_data, y_test, config['batch_size']) 
            names_train_all.append(names_train)
            names_val_all.append(names_val)
            names_test_all.append(names_test)
            dataloaders = (train_loader, val_loader, test_loader)
            config["seq_len"] = timeseries
            config["node_size"] = node_size
        
            model = FBNETGEN(model_config, node_size, node_feature_size, config['length']).to(device)
            optimizer = Adam(model.parameters(), lr=config['lr'],weight_decay=config['weight_decay'])
            opts = (optimizer,)
            
            loss_name = 'loss'
            if config["group_loss"]:
                loss_name = f"{loss_name}_group_loss"
            if config["sparsity_loss"]:
                loss_name = f"{loss_name}_sparsity_loss"
            
            folder_suffix = 'rs' + str(i) + '_fold' + str(j)
            save_folder_name = Path(os.path.join(config['path_save'], 'results_ABIDEI', 'FBNETGEN', config['atlas'])) / folder_suffix
            train_process = BasicTrain(config, model, opts, dataloaders, save_folder_name)
            
            train_process.train()
    '''       
    
    '''------------------------------------Ext. test------------------------------------'''
    names_train_ext_all = []
    names_val_ext_all = []
    for i in range(10):
        train_data, val_data, y_train, y_val, train_idx, val_idx = train_test_split(data,
                                                                                    y, 
                                                                                    np.arange(len(data)), 
                                                                                    test_size=0.1, 
                                                                                    random_state=42+i,
                                                                                    stratify=y)
        names_val = [names[n] for n in val_idx]
        names_train = [names[n] for n in train_idx]
        train_data = train_DATA + train_data
        y_train = np.concatenate((Y_train, y_train))
        names_train = names_duplicate + names_train
        train_loader, node_size, node_feature_size, timeseries = FBNETGENLoader(train_data,
                                                                                y_train,
                                                                                config['batch_size'])
        val_loader,_,_,_ = FBNETGENLoader(val_data, y_val, config['batch_size'])
        test_loader,_,_,_  = FBNETGENLoader(ext_test_data, y_ext_test, config['batch_size']) 
        names_train_ext_all.append(names_train)
        names_val_ext_all.append(names_val)
        dataloaders = (train_loader, val_loader, test_loader)
        config["seq_len"] = timeseries
        config["node_size"] = node_size
        
        model = FBNETGEN(model_config, node_size, node_feature_size, config['length']).to(device)
        optimizer = Adam(model.parameters(), lr=config['lr'],weight_decay=config['weight_decay'])
        opts = (optimizer,)
        
        loss_name = 'loss'
        if config["group_loss"]:
            loss_name = f"{loss_name}_group_loss"
        if config["sparsity_loss"]:
            loss_name = f"{loss_name}_sparsity_loss"
        
        folder_suffix = 'rs' + str(i)
        save_folder_name = Path(os.path.join(config['path_save'], 'results_ABIDEI', 'FBNETGEN', config['atlas'])) / folder_suffix
        train_process = BasicTrain(config, model, opts, dataloaders, save_folder_name)
        
        train_process.train()
        

if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description='Run FBNETGEN on ABIDE I for ASD classification')

    parser.add_argument('--path_data', type=str,
                        help='Path to the dataset')
    parser.add_argument('--path_save', type=str,
                        help='Path to save results')
    parser.add_argument('--atlas', type=str, choices=['AICHA', 'AAL'], default='AICHA',
                        help='Atlas type to use')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use for training')
    parser.add_argument('--length', type=int, default=120,
                        help='Length of input signals')
    parser.add_argument('--extractor_type', type=str, default='cnn',
                        help='Sequence model to process fMRI data (cnn or gru)')
    parser.add_argument('--embedding_size', type=int, default=8,
                        help='Embedding size for the sequence model')
    parser.add_argument('--window_size', type=int, default=4,
                        help='Kernel size for the 1D CNN')
    parser.add_argument('--graph_generation', type=str, default='product',
                        help='Method for generating th graph')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--group_loss', action='store_true')
    parser.add_argument('--sparsity_loss', action='store_true')
    parser.add_argument('--sparsity_loss_weight', type=float, default=1e-4)
    parser.add_argument('--pure_gnn_graph', type=str, default='pearson')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='Number of classes for the classification')
    parser.add_argument('--save_models', action='store_true',
                        help='Flag to save trained models')
    parser.add_argument('--save_results', action='store_true',
                        help='Flag to save results')

    args = parser.parse_args()

    config = {
        'path_data': args.path_data,
        'path_save': args.path_save,
        'atlas': args.atlas,
        'shuffle': True,
        'length': args.length,
        'epochs': args.epochs,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'group_loss': args.group_loss,
        'sparsity_loss': args.sparsity_loss,
        'sparsity_loss_weight': args.sparsity_loss_weight,
        'pure_gnn_graph': args.pure_gnn_graph,
        'batch_size': args.batch_size,
        'num_classes': args.num_classes,
        'save_models': args.save_models,
        'save_results': args.save_results,
        'device': args.device,
        'model_config': {}
    }
    
    config['model_config']['extractor_type'] = args.extractor_type
    config['model_config']['embedding_size'] = args.embedding_size
    config['model_config']['window_size'] = args.window_size
    config['model_config']['graph_generation'] = args.graph_generation

    results = main(config)

