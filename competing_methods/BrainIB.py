parent_path = r'/.../baselines'
import sys
sys.path.append(parent_path)
from torch_geometric.loader import DataLoader
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utils import BrainIBDataset
from BRAINIB.BrainIB_V2.SGSIB.GNN import GNN
from BRAINIB.BrainIB_V2.SGSIB.utils import train
from BRAINIB.BrainIB_V2.SGSIB.sub_graph_generator import MLP_subgraph
import os
import pickle
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold
import random
import copy
import argparse


def test(model, test_dataset, batch_size):
    with torch.no_grad():
        zs = []
        ys = []
        indices = range(0, len(test_dataset), batch_size)
        for i in indices:
            model.eval()
            graphs = test_dataset[i : i + config['batch_size']]
            batch_graph = next(iter(DataLoader(graphs, batch_size=len(graphs))))
            _, original_output = model(batch_graph)
            zs.append(original_output)
            ys.append(batch_graph.y)
        z = torch.cat(zs,dim=0)
        y = torch.cat(ys,dim=0)
        loss_func = nn.BCELoss()
        z = nn.Softmax()(z)
        loss = loss_func(z, F.one_hot(y, num_classes=2).float())
        z = z[:,-1].detach().cpu().numpy()
        y = y.cpu().numpy()
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
        data.append(data_list[key])
    
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
    
    
    '''------------------------------------KFold CV------------------------------------'''
    test_losses_all = []
    test_aucs_all = []
    train_losses = []
    val_losses_all = []
    val_aucs_all = []
    val_probs_all = []
    y_val_all = []
    y_test_all = []
    test_probs_all = []
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
            train_dataset = BrainIBDataset(train_data, y_train)
            train_dataset = random.sample(train_dataset, len(train_dataset))
            val_dataset = BrainIBDataset(val_data, y_val)
            test_dataset = BrainIBDataset(test_data, y_test) 
            names_train_all.append(names_train)
            names_val_all.append(names_val)
            names_test_all.append(names_test)
            
            roi_num = test_data[0].shape[1]
            # Instantiate the backbone network
            model = GNN(num_of_features=roi_num, device=device).to(device)
            # Instantiate the subgraph generator
            SG_model = MLP_subgraph(node_features_num=roi_num, edge_features_num=1, device=device)
    
            optimizer = torch.optim.Adam([
                {'params': model.parameters(), 'lr': config['model_learning_rate']},
                {'params': SG_model.parameters(), 'lr': config['SGmodel_learning_rate']}
                ])
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
            
            min_val_loss = 1000
            losses = []
            val_losses = []
            test_losses = []
            aucs = []
            val_aucs = []
            test_aucs = []
            val_probs = []
            test_probs = []
            y_vals = []
            y_tests = []
            with tqdm(total=config['epochs'], desc='(T)') as pbar:
                for epoch in range(1,config['epochs']+1):
                    avg_loss, mi_loss, avg_auc = train(config, model, train_dataset, optimizer, epoch, SG_model, device)
                    scheduler.step()
                    val_loss,val_auc,val_prob,y_val = test(model, val_dataset,config['batch_size'])
                    losses.append(avg_loss)
                    val_losses.append(val_loss)
                    aucs.append(avg_auc)
                    val_aucs.append(val_auc)
                    val_probs.append(val_prob)
                    y_vals.append(y_val)
                    pbar.set_postfix({
                        'loss': avg_loss, 
                        'auc': avg_auc,
                        'val_loss': val_loss, 
                        'val_auc': val_auc
                    })
                    pbar.update()  
                    if val_loss < min_val_loss:
                        min_val_loss_model = copy.deepcopy(model.state_dict())
                    test_loss,test_auc,test_prob,y_test = test(model, test_dataset,config['batch_size'])
                    test_losses.append(test_loss)
                    test_aucs.append(test_auc)
                    test_probs.append(test_prob)
                    y_tests.append(y_test)
            test_losses_all.append(test_losses)
            test_aucs_all.append(test_aucs)
            train_losses.append(losses)
            val_losses_all.append(val_losses)
            val_aucs_all.append(val_aucs)
            val_probs_all.append(val_probs)
            test_probs_all.append(test_probs)
            y_val_all.append(y_vals)
            y_test_all.append(y_tests)        
            
            if config['save_models']:
                if not os.path.exists(os.path.join(config['path_save'],'models_ABIDEI',config['atlas'],'BrainIB')):
                    os.makedirs(os.path.join(config['path_save'],'models_ABIDEI',config['atlas'],'BrainIB'),exist_ok=True)
                torch.save(min_val_loss_model, os.path.join(config['path_save'],'models_ABIDEI',config['atlas'],'BrainIB','min_val_loss_model_rs' + str(i) + '_fold' + str(j) + '.pth'))
    
    '''------------------------------------Ext. test------------------------------------'''
    ext_test_losses_all = []
    ext_test_aucs_all = []
    train_losses_ext = []
    val_losses_all_ext = []
    val_aucs_all_ext = []
    val_probs_all_ext = []
    y_val_all_ext = []
    ext_test_probs_all = []
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
        train_dataset = BrainIBDataset(train_data, y_train)
        train_dataset = random.sample(train_dataset, len(train_dataset))
        val_dataset = BrainIBDataset(val_data, y_val)
        test_dataset = BrainIBDataset(ext_test_data, y_ext_test) 
        names_train_ext_all.append(names_train)
        names_val_ext_all.append(names_val)
        
        roi_num = test_data[0].shape[1]
        # Instantiate the backbone network
        model = GNN(num_of_features=roi_num, device=device).to(device)
        # Instantiate the subgraph generator
        SG_model = MLP_subgraph(node_features_num=roi_num, edge_features_num=1, device=device)

        optimizer = torch.optim.Adam([
            {'params': model.parameters(), 'lr': config['model_learning_rate']},
            {'params': SG_model.parameters(), 'lr': config['SGmodel_learning_rate']}
            ])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        
        min_val_loss = 1000
        losses = []
        val_losses = []
        test_losses = []
        aucs = []
        val_aucs = []
        test_aucs = []
        val_probs = []
        test_probs = []
        y_vals = []
        y_tests = []
        with tqdm(total=config['epochs'], desc='(T)') as pbar:
            for epoch in range(1,config['epochs']+1):
                avg_loss, mi_loss, avg_auc = train(config, model, train_dataset, optimizer, epoch, SG_model, device)
                scheduler.step()
                val_loss,val_auc,val_prob,y_val = test(model, val_dataset,config['batch_size'])
                losses.append(avg_loss)
                val_losses.append(val_loss)
                aucs.append(avg_auc)
                val_aucs.append(val_auc)
                val_probs.append(val_prob)
                y_vals.append(y_val)
                pbar.set_postfix({
                    'loss': avg_loss, 
                    'auc': avg_auc,
                    'val_loss': val_loss, 
                    'val_auc': val_auc
                })
                pbar.update()  
                if val_loss < min_val_loss:
                    min_val_loss_model = copy.deepcopy(model.state_dict())
                test_loss,test_auc,test_prob,y_ext_test = test(model, test_dataset, config['batch_size'])
                test_losses.append(test_loss)
                test_aucs.append(test_auc)
                test_probs.append(test_prob)
                y_tests.append(y_test)
        
        ext_test_losses_all.append(test_losses)
        ext_test_aucs_all.append(test_aucs)
        train_losses_ext.append(losses)
        val_losses_all_ext.append(val_losses)
        val_aucs_all_ext.append(val_aucs)
        val_probs_all_ext.append(val_probs)
        ext_test_probs_all.append(test_probs)
        y_val_all_ext.append(y_vals)
        
        if config['save_models']:
            if not os.path.exists(os.path.join(config['path_save'],'models_ABIDEI',config['atlas'],'BrainIB')):
                os.makedirs(os.path.join(config['path_save'],'models_ABIDEI',config['atlas'],'BrainIB'), exist_ok=True)
            torch.save(min_val_loss_model, os.path.join(config['path_save'],'models_ABIDEI',config['atlas'],'BrainIB','min_val_loss_model_rs' + str(i) + '.pth'))
    
    results = {}
    results['losses'] = train_losses
    results['val_losses'] = val_losses_all
    results['test_losses'] = test_losses_all
    results['test_aucs'] = test_aucs_all
    results['val_aucs'] = val_aucs_all
    results['val_probs'] = val_probs_all
    results['test_probs'] = test_probs_all
    results['y_val'] = y_val_all
    results['y_test'] = y_test_all
    results['names_train'] = names_train_all
    results['names_val'] = names_val_all
    results['names_test'] = names_test_all
    results['losses_ext'] = train_losses_ext
    results['val_losses_ext'] = val_losses_all_ext
    results['ext_test_losses'] = ext_test_losses_all
    results['ext_test_aucs'] = ext_test_aucs_all
    results['val_aucs_ext'] = val_aucs_all_ext
    results['val_probs_ext'] = val_probs_all_ext
    results['ext_test_probs'] = ext_test_probs_all
    results['y_val_ext'] = y_val_all_ext
    results['y_ext_test'] = y_ext_test
    results['names_val_ext'] = names_val_ext_all
    results['names_ext_test'] = names_ext_test
    if config['save_results']:            
        if not os.path.exists(os.path.join(config['path_save'],'results_ABIDEI',config['atlas'])):
            os.makedirs(os.path.join(config['path_save'],'results_ABIDE',config['atlas']),exist_ok=True)
        with open(os.path.join(config['path_save'],'results_ABIDEI',config['atlas'],'ABIDEI_BrainIB_results.pkl'), "wb") as pickle_file:
            pickle.dump(results, pickle_file)
    return results

if __name__ == '__main__':   
    parser = argparse.ArgumentParser(description='Run BrainIB on ABIDE I for ASD classification')

    parser.add_argument('--path_data', type=str,
                        help='Path to the dataset')
    parser.add_argument('--path_save', type=str,
                        help='Path to save results')
    parser.add_argument('--atlas', type=str, choices=['AICHA', 'AAL'], default='AICHA',
                        help='Atlas type to use')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use for training')
    parser.add_argument('--iters_per_epoch', type=int, default=1,
                        help='number of iterations per each epoch (default: 1)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for splitting the dataset into 10 (default: 0)')
    parser.add_argument("--mi_weight", type=float, default=0.001,
                        help="weight of mutual information loss (default: 0.001)")
    parser.add_argument("--pos_weight", type=float, default= 0.001,
                        help="weight of mutual information loss (default: 0.001)")
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--model_learning_rate', type=float, default=0.0005,
                        help='learning rate of graph model (default: 0.0005)')
    parser.add_argument('--SGmodel_learning_rate', type=float, default=0.001,
                        help='learning rate of subgraph model (default: 0.0005)')
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
        'iters_per_epoch': args.iters_per_epoch,
        'epochs': args.epochs,
        'model_learning_rate': args.model_learning_rate,
        'SGmodel_learning_rate': args.SGmodel_learning_rate,
        'batch_size': args.batch_size,
        'mi_weight': args.mi_weight,
        'pos_weight': args.pos_weight,
        'seed': args.seed,
        'num_classes': args.num_classes,
        'save_models': args.save_models,
        'save_results': args.save_results,
        'device': args.device,
    }

    results = main(config)

