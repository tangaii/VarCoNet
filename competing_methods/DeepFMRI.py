from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim import Adam
from utils import ABIDEDataset
from model_scripts.competing_models import DeepFMRI
import os
import pickle
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold
import copy
import argparse


def train(x, y, encoder_model, optimizer, loss_func, num_classes):
    encoder_model.train()
    optimizer.zero_grad()
    z = encoder_model(x)
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
        data.append((data_list[key][:config['length'],:] - np.mean(data_list[key][:config['length'],:],axis=0,keepdims=True))/np.std(data_list[key][:config['length'],:],axis=0,keepdims=True))
    
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
            train_dataset = ABIDEDataset(train_data, y_train)
            train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=config['shuffle'])
            val_dataset = ABIDEDataset(val_data, y_val)
            val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
            test_dataset = ABIDEDataset(test_data, y_test)
            test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])  
            names_train_all.append(names_train)
            names_val_all.append(names_val)
            names_test_all.append(names_test)
            
            roi_num = test_data[0].shape[1]
            encoder_model = DeepFMRI(roi_num).to(device)
            loss_func = nn.BCELoss()
            optimizer = Adam(encoder_model.parameters(), lr=config['lr'])
            
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
                    total_loss = 0.0
                    total_auc = 0.0
                    batch_count = 0                          
                    for batch_idx, (batch_data, batch_labels) in enumerate(train_loader):            
                        loss,auc = train(batch_data.to(device), batch_labels.to(device), 
                                         encoder_model, optimizer, loss_func, config['num_classes'])
                        total_loss += loss
                        total_auc += auc
                        batch_count += 1
                    val_loss,val_auc,val_prob,y_val = test(encoder_model, val_loader,
                                                           config['batch_size'], loss_func,
                                                           config['num_classes'], device)
                            
                    average_loss = total_loss / batch_count if batch_count > 0 else float('nan')   
                    average_auc = total_auc / batch_count if batch_count > 0 else float('nan') 
                    losses.append(average_loss)
                    val_losses.append(val_loss)
                    aucs.append(average_auc)
                    val_aucs.append(val_auc)
                    val_probs.append(val_prob)
                    y_vals.append(y_val)
                    pbar.set_postfix({
                        'loss': average_loss, 
                        'auc': average_auc,
                        'val_loss': val_loss, 
                        'val_auc': val_auc
                    })
                    pbar.update()  
                    if val_loss < min_val_loss:
                        min_val_loss_model = copy.deepcopy(encoder_model.state_dict())
                    test_loss,test_auc,test_prob,y_test = test(encoder_model, test_loader,
                                                               config['batch_size'], loss_func, 
                                                               config['num_classes'], device)
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
                if not os.path.exists(os.path.join(config['path_save'],'models_ABIDEI',config['atlas'],'DeepFMRI')):
                    os.makedirs(os.path.join(config['path_save'],'models_ABIDEI',config['atlas'],'DeepFMRI'),exist_ok=True)
                torch.save(min_val_loss_model, os.path.join(config['path_save'],'models_ABIDEI',config['atlas'],'DeepFMRI','min_val_loss_model_rs' + str(i) + '_fold' + str(j) + '.pth'))
    
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
        train_dataset = ABIDEDataset(train_data, y_train)
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=config['shuffle'])
        val_dataset = ABIDEDataset(val_data, y_val)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
        test_dataset = ABIDEDataset(ext_test_data, y_ext_test)
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])  
        names_train_ext_all.append(names_train)
        names_val_ext_all.append(names_val)
        
        roi_num = test_data[0].shape[1]
        encoder_model = DeepFMRI(roi_num).to(device)
        loss_func = nn.BCELoss()
        optimizer = Adam(encoder_model.parameters(), lr=config['lr'])
        
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
                total_loss = 0.0
                total_auc = 0.0
                batch_count = 0                          
                for batch_idx, (batch_data, batch_labels) in enumerate(train_loader):            
                    loss,auc = train(batch_data.to(device), batch_labels.to(device), 
                                     encoder_model, optimizer, loss_func, config['num_classes'])
                    total_loss += loss
                    total_auc += auc
                    batch_count += 1
                val_loss,val_auc,val_prob,y_val = test(encoder_model, val_loader,
                                                       config['batch_size'], loss_func,
                                                       config['num_classes'], device)
                        
                average_loss = total_loss / batch_count if batch_count > 0 else float('nan')   
                average_auc = total_auc / batch_count if batch_count > 0 else float('nan') 
                losses.append(average_loss)
                val_losses.append(val_loss)
                aucs.append(average_auc)
                val_aucs.append(val_auc)
                val_probs.append(val_prob)
                y_vals.append(y_val)
                pbar.set_postfix({
                    'loss': average_loss, 
                    'auc': average_auc,
                    'val_loss': val_loss, 
                    'val_auc': val_auc
                })
                pbar.update()  
                if val_loss < min_val_loss:
                    min_val_loss_model = copy.deepcopy(encoder_model.state_dict())
                test_loss,test_auc,test_prob,y_ext_test = test(encoder_model, test_loader,
                                                               config['batch_size'], loss_func,
                                                               config['num_classes'], device)
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
            if not os.path.exists(os.path.join(config['path_save'],'models_ABIDEI',config['atlas'],'DeepFMRI')):
                os.makedirs(os.path.join(config['path_save'],'models_ABIDEI',config['atlas'],'DeepFMRI'), exist_ok=True)
            torch.save(min_val_loss_model, os.path.join(config['path_save'],'models_ABIDEI',config['atlas'],'DeepFMRI','min_val_loss_model_rs' + str(i) + '.pth'))
    
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
        with open(os.path.join(config['path_save'],'results_ABIDEI',config['atlas'],'ABIDEI_DeepFMRI_results.pkl'), "wb") as pickle_file:
            pickle.dump(results, pickle_file)
    return results

if __name__ == '__main__':   
    parser = argparse.ArgumentParser(description='Run DeepFMRI on ABIDE I for ASD classification')

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
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Learning rate')
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
        'batch_size': args.batch_size,
        'num_classes': args.num_classes,
        'save_models': args.save_models,
        'save_results': args.save_results,
        'device': args.device,
    }

    results = main(config)

