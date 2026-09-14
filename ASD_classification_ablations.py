from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.nn as nn
from utils import InfoNCE
from tqdm import tqdm
import torch.nn.functional as F
from torch.optim import Adam
from utils import DualBranchContrast
from pl_bolts.optimizers import LinearWarmupCosineAnnealingLR
from model_scripts.VarCoNet import VarCoNet_noSSL, VarCoNet_noCNN, VarCoNet_noTransformer, VarCoNet
from utils import ABIDEDataset
import os
from model_scripts.classifier import LREvaluator
import pickle
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from utils import augment, removeDuplicates, test_augment_overlap, other_augment1, other_augment2
import argparse


def train(x, encoder_model, contrast_model, optimizer):
    encoder_model.train()
    optimizer.zero_grad()
    if len(x) == 2:
        z1 = encoder_model(x[0])
        z2 = encoder_model(x[1])
        loss = contrast_model(z1, z2)
    elif len(x) == 4:
        z1 = encoder_model(x[0])
        z2 = encoder_model(x[1])
        z3 = encoder_model(x[2])
        z4 = encoder_model(x[3])
        loss = contrast_model(z1, z2) + contrast_model(z3, z4)
    loss.backward()
    optimizer.step()
    return loss.item(), z1.shape[1]

def train_supervised(x, y, encoder_model, optimizer, loss_func, num_classes):
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


def test(encoder_model, train_loader, val_loader, test_loader,
         min_length, max_length, num_classes, device, num_epochs, lr):
    encoder_model.eval()
    with torch.no_grad():
        #outputs_train = []
        #y_train = []
        #for (x,y) in train_loader:
        #    xs = []
        #    ys = []
        #    for (xi,yi) in zip(x,y):
        #        xs.append(xi.unsqueeze(0).to(device))
        #        ys.append(yi)
        #        xi = test_augment_overlap(xi,[4,3,2,1],0.5,min_length,max_length,device)
        #        xs.append(xi)
        #        for i in range(xi.shape[0]):
        #            ys.append(yi)
        #    xs = torch.cat(xs)
        #    xs = xs.to(device)
        #    y_train.append(torch.tensor(ys))
        #    outputs_train.append(encoder_model(xs))
        #outputs_train = torch.cat(outputs_train, dim=0).clone().detach()
        #y_train = torch.cat(y_train,dim=0).to(device)
        
        outputs_train = []
        y_train = []
        for (x,y) in train_loader:
            x = x.to(device)
            y_train.append(y)
            outputs_train.append(encoder_model(x))
        outputs_train = torch.cat(outputs_train, dim=0).clone().detach()
        y_train = torch.cat(y_train,dim=0).to(device)
    
        
        outputs_val = []
        y_val = []
        for (x,y) in val_loader:
            x = x.to(device)
            y_val.append(y)
            outputs_val.append(encoder_model(x))
        outputs_val = torch.cat(outputs_val, dim=0).clone().detach()
        y_val = torch.cat(y_val,dim=0).to(device)
        
        outputs_test= []
        y_test = []
        for (x,y) in test_loader:
            x = x.to(device)
            y_test.append(y)
            outputs_test.append(encoder_model(x))
        outputs_test = torch.cat(outputs_test, dim=0).clone().detach()
        y_test = torch.cat(y_test,dim=0).to(device)
        
    result,linear_state_dict = LREvaluator(num_epochs=num_epochs,learning_rate=lr).evaluate(encoder_model, outputs_train, y_train, outputs_val, y_val, outputs_test, y_test, num_classes, device)
                
    return result,linear_state_dict


def test_supervised(encoder_model, test_data_loader, batch_size, loss_func, num_classes, device):
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
    results = {}
    results['no_SSL'] = {}
    results['no_CNN'] = {}
    results['no_Transformer'] = {}
    results['other_augmentations'] = {}
    results['other_augmentations']['augmentation1'] = {}
    results['other_augmentations']['augmentation2'] = {}
    for atlas in ['AAL', 'AICHA']:
        with open(f'best_params_VarCoNet_{atlas}.pkl', 'rb') as f:
            best_params = pickle.load(f)

        config['batch_size'] = best_params['batch_size']
        config['tau'] = best_params['tau']
        config['lr'] = best_params['lr']
        config['model_config']['layers'] = best_params['layers']
        config['model_config']['n_heads'] = best_params['n_heads']
        config['model_config']['dim_feedforward'] = best_params['dim_feedforward']
        
        path = config['path_data']
        
        names = []
        with open(os.path.join(path,'ABIDEI_nilearn_names.txt'), 'r') as f:
            for line in f:
                names.append(line.strip())
        
        data_list = np.load(os.path.join(path,'ABIDEI_nilearn_' + atlas + '.npz'))
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
        
        pos = []
        for name in names:
            pos.append(np.where(np.array(names_unique) == name)[0][0])
        data = [data[i] for i in pos]
        y = y[pos]
        
        device = torch.device(config['device']) if torch.cuda.is_available() else torch.device("cpu")
        max_length = data[0].shape[0]
        train_length_limits = [config['min_length'], max_length]
        eval_epochs = list(range(1, config['epochs']+1)) 
        
        model_config = config['model_config'] 
        model_config['max_length'] = max_length
        
        '''---------------------------------no Transformer---------------------------------'''
        '''------------------------------------KFold CV------------------------------------'''
        
        losses_all = []
        test_result_all = []
        min_val_loss_epochs = []
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
                train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],shuffle=config['shuffle'])
                val_dataset = ABIDEDataset(val_data, y_val)
                val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
                test_dataset = ABIDEDataset(test_data, y_test)
                test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])  
                names_train_all.append(names_train)
                names_val_all.append(names_val)
                names_test_all.append(names_test)
                
                roi_num = test_data[0].shape[1]
                max_length = test_data[0].shape[0]
                model_config['max_length'] = max_length
                encoder_model = VarCoNet_noTransformer(model_config, roi_num).to(device)
                contrast_model = DualBranchContrast(loss=InfoNCE(tau=config['tau']),mode='L2L').to(device)
                optimizer = Adam(encoder_model.parameters(), lr=config['lr'])
                scheduler = LinearWarmupCosineAnnealingLR(
                    optimizer=optimizer,
                    warmup_start_lr = 1e-5,
                    warmup_epochs=config['warm_up_epochs'],
                    max_epochs=config['epochs'])
                      
                min_val_loss = 1000
                test_result = []
                losses = []
                with tqdm(total=config['epochs'], desc='(T)') as pbar:
                    for epoch in range(1,config['epochs']+1):
                        total_loss = 0.0
                        batch_count = 0                
                        for batch_idx, sample_inds in enumerate(train_loader.batch_sampler):
                            sample_inds = removeDuplicates(names_train,sample_inds)
                            batch_list = [train_data[i] for i in sample_inds]
                            batch_loader = DataLoader(batch_list, batch_size=len(batch_list))
                            batch_data = next(iter(batch_loader))
                            batch_data = augment(batch_data,train_length_limits,device)
                            loss,input_dim = train(batch_data,encoder_model,contrast_model,optimizer)
                            total_loss += loss
                            batch_count += 1
                        scheduler.step()
                                
                        average_loss = total_loss / batch_count if batch_count > 0 else float('nan')   
                        losses.append(average_loss)
                        pbar.set_postfix({'loss': average_loss})
                        pbar.update()        
                        
                        if epoch in eval_epochs:
                            res,linear_state_dict = test(encoder_model,train_loader,val_loader,test_loader,
                                       config['min_length'],max_length,config['num_classes'],
                                       device,config['epochs_cls'],config['lr_cls'])
                            test_result.append(res) 
                            if res['best_val_loss'] < min_val_loss:
                                min_val_loss = res['best_val_loss']
                                min_val_loss_epoch = epoch
                losses_all.append(losses)
                test_result_all.append(test_result)
                min_val_loss_epochs.append(min_val_loss_epoch)
        
        results['no_Transformer'][atlas] = {}
        results['no_Transformer'][atlas]['losses'] = losses_all
        results['no_Transformer'][atlas]['epoch_results'] = test_result_all
        results['no_Transformer'][atlas]['min_val_loss_epoch'] = min_val_loss_epochs
        results['no_Transformer'][atlas]['names_train'] = names_train_all
        results['no_Transformer'][atlas]['names_val'] = names_val_all
        results['no_Transformer'][atlas]['names_test'] = names_test_all
        
        '''-------------------------------------no CNN-------------------------------------'''
        '''------------------------------------KFold CV------------------------------------'''
        
        losses_all = []
        test_result_all = []
        min_val_loss_epochs = []
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
                train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],shuffle=config['shuffle'])
                val_dataset = ABIDEDataset(val_data, y_val)
                val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
                test_dataset = ABIDEDataset(test_data, y_test)
                test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])  
                names_train_all.append(names_train)
                names_val_all.append(names_val)
                names_test_all.append(names_test)
                
                roi_num = test_data[0].shape[1]
                encoder_model = VarCoNet_noCNN(model_config, roi_num).to(device)
                contrast_model = DualBranchContrast(loss=InfoNCE(tau=config['tau']),mode='L2L').to(device)
                optimizer = Adam(encoder_model.parameters(), lr=config['lr'])
                scheduler = LinearWarmupCosineAnnealingLR(
                    optimizer=optimizer,
                    warmup_start_lr = 1e-5,
                    warmup_epochs=config['warm_up_epochs'],
                    max_epochs=config['epochs'])
                      
                min_val_loss = 1000
                test_result = []
                losses = []
                with tqdm(total=config['epochs'], desc='(T)') as pbar:
                    for epoch in range(1,config['epochs']+1):
                        total_loss = 0.0
                        batch_count = 0                
                        for batch_idx, sample_inds in enumerate(train_loader.batch_sampler):
                            sample_inds = removeDuplicates(names_train,sample_inds)
                            batch_list = [train_data[i] for i in sample_inds]
                            batch_loader = DataLoader(batch_list, batch_size=len(batch_list))
                            batch_data = next(iter(batch_loader))
                            batch_data = augment(batch_data,train_length_limits,device)
                            loss,input_dim = train(batch_data,encoder_model,contrast_model,optimizer)
                            total_loss += loss
                            batch_count += 1
                        scheduler.step()
                                
                        average_loss = total_loss / batch_count if batch_count > 0 else float('nan')   
                        losses.append(average_loss)
                        pbar.set_postfix({'loss': average_loss})
                        pbar.update()        
                        
                        if epoch in eval_epochs:
                            res,linear_state_dict = test(encoder_model,train_loader,val_loader,test_loader,
                                       config['min_length'],max_length,config['num_classes'],
                                       device,config['epochs_cls'],config['lr_cls'])
                            test_result.append(res) 
                            if res['best_val_loss'] < min_val_loss:
                                min_val_loss = res['best_val_loss']
                                min_val_loss_epoch = epoch
                losses_all.append(losses)
                test_result_all.append(test_result)
                min_val_loss_epochs.append(min_val_loss_epoch)
        
        results['no_CNN'][atlas] = {}
        results['no_CNN'][atlas]['losses'] = losses_all
        results['no_CNN'][atlas]['epoch_results'] = test_result_all
        results['no_CNN'][atlas]['min_val_loss_epoch'] = min_val_loss_epochs
        results['no_CNN'][atlas]['names_train'] = names_train_all
        results['no_CNN'][atlas]['names_val'] = names_val_all
        results['no_CNN'][atlas]['names_test'] = names_test_all
        
        
        '''-------------------------------------no SSL-------------------------------------'''
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
                train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                                          shuffle=config['shuffle'],
                                          drop_last=True)
                val_dataset = ABIDEDataset(val_data, y_val)
                val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
                test_dataset = ABIDEDataset(test_data, y_test)
                test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])  
                names_train_all.append(names_train)
                names_val_all.append(names_val)
                names_test_all.append(names_test)
                roi_num = test_data[0].shape[1]
                
                encoder_model = VarCoNet_noSSL(model_config, roi_num, config['num_classes']).to(device)
                
                loss_func = nn.BCELoss()
                optimizer = Adam(encoder_model.parameters(), lr=config['lr'])
                scheduler = LinearWarmupCosineAnnealingLR(
                    optimizer=optimizer,
                    warmup_start_lr = 1e-5,
                    warmup_epochs=config['warm_up_epochs'],
                    max_epochs=config['epochs'])
                
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
                            loss,auc = train_supervised(batch_data.to(device),batch_labels.to(device),
                                             encoder_model,optimizer,loss_func,config['num_classes'])
                            total_loss += loss
                            total_auc += auc
                            batch_count += 1
                        scheduler.step()
                        val_loss,val_auc,val_prob,y_val = test_supervised(encoder_model,val_loader,
                                                               config['batch_size'],loss_func,
                                                               config['num_classes'],device)
                                
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

                        test_loss,test_auc,test_prob,y_test = test_supervised(encoder_model,test_loader,
                                                                   config['batch_size'],loss_func,
                                                                   config['num_classes'],device)
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
        results['no_SSL'][atlas] = {}
        results['no_SSL'][atlas]['losses'] = train_losses
        results['no_SSL'][atlas]['val_losses'] = val_losses_all
        results['no_SSL'][atlas]['test_losses'] = test_losses_all
        results['no_SSL'][atlas]['test_aucs'] = test_aucs_all
        results['no_SSL'][atlas]['val_aucs'] = val_aucs_all
        results['no_SSL'][atlas]['val_probs'] = val_probs_all
        results['no_SSL'][atlas]['test_probs'] = test_probs_all
        results['no_SSL'][atlas]['y_val'] = y_val_all
        results['no_SSL'][atlas]['y_test'] = y_test_all
        results['no_SSL'][atlas]['names_train'] = names_train_all
        results['no_SSL'][atlas]['names_val'] = names_val_all
        
        '''-----------------------Other augmentations - augmentation 1---------------------'''
        '''------------------------------------KFold CV------------------------------------'''
        
        losses_all = []
        test_result_all = []
        min_val_loss_epochs = []
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
                train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],shuffle=config['shuffle'])
                val_dataset = ABIDEDataset(val_data, y_val)
                val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
                test_dataset = ABIDEDataset(test_data, y_test)
                test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])  
                names_train_all.append(names_train)
                names_val_all.append(names_val)
                names_test_all.append(names_test)
                
                roi_num = test_data[0].shape[1]
                encoder_model = VarCoNet(model_config, roi_num).to(device)
                contrast_model = DualBranchContrast(loss=InfoNCE(tau=config['tau']),mode='L2L').to(device)
                optimizer = Adam(encoder_model.parameters(), lr=config['lr'])
                scheduler = LinearWarmupCosineAnnealingLR(
                    optimizer=optimizer,
                    warmup_start_lr = 1e-5,
                    warmup_epochs=config['warm_up_epochs'],
                    max_epochs=config['epochs'])
                      
                min_val_loss = 1000
                test_result = []
                losses = []
                with tqdm(total=config['epochs'], desc='(T)') as pbar:
                    for epoch in range(1,config['epochs']+1):
                        total_loss = 0.0
                        batch_count = 0                
                        for batch_idx, sample_inds in enumerate(train_loader.batch_sampler):
                            sample_inds = removeDuplicates(names_train,sample_inds)
                            batch_list = [train_data[i] for i in sample_inds]
                            batch_loader = DataLoader(batch_list, batch_size=len(batch_list))
                            batch_data = next(iter(batch_loader))
                            batch_data = other_augment1(batch_data,train_length_limits,device)
                            loss,input_dim = train(batch_data,encoder_model,contrast_model,optimizer)
                            total_loss += loss
                            batch_count += 1
                        scheduler.step()
                                
                        average_loss = total_loss / batch_count if batch_count > 0 else float('nan')   
                        losses.append(average_loss)
                        pbar.set_postfix({'loss': average_loss})
                        pbar.update()        
                        
                        if epoch in eval_epochs:
                            res,linear_state_dict = test(encoder_model,train_loader,val_loader,test_loader,
                                       config['min_length'],max_length,config['num_classes'],
                                       device,config['epochs_cls'],config['lr_cls'])
                            test_result.append(res) 
                            if res['best_val_loss'] < min_val_loss:
                                min_val_loss = res['best_val_loss']
                                min_val_loss_epoch = epoch
                losses_all.append(losses)
                test_result_all.append(test_result)
                min_val_loss_epochs.append(min_val_loss_epoch)
        
        results['other_augmentations']['augmentation1'][atlas] = {}
        results['other_augmentations']['augmentation1'][atlas]['losses'] = losses_all
        results['other_augmentations']['augmentation1'][atlas]['epoch_results'] = test_result_all
        results['other_augmentations']['augmentation1'][atlas]['min_val_loss_epoch'] = min_val_loss_epochs
        results['other_augmentations']['augmentation1'][atlas]['names_train'] = names_train_all
        results['other_augmentations']['augmentation1'][atlas]['names_val'] = names_val_all
        results['other_augmentations']['augmentation1'][atlas]['names_test'] = names_test_all
        
        '''-----------------------Other augmentations - augmentation 2---------------------'''
        '''------------------------------------KFold CV------------------------------------'''
        
        losses_all = []
        test_result_all = []
        min_val_loss_epochs = []
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
                train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],shuffle=config['shuffle'])
                val_dataset = ABIDEDataset(val_data, y_val)
                val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
                test_dataset = ABIDEDataset(test_data, y_test)
                test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])  
                names_train_all.append(names_train)
                names_val_all.append(names_val)
                names_test_all.append(names_test)
                
                roi_num = test_data[0].shape[1]
                encoder_model = VarCoNet(model_config, roi_num).to(device)
                contrast_model = DualBranchContrast(loss=InfoNCE(tau=config['tau']),mode='L2L').to(device)
                optimizer = Adam(encoder_model.parameters(), lr=config['lr'])
                scheduler = LinearWarmupCosineAnnealingLR(
                    optimizer=optimizer,
                    warmup_start_lr = 1e-5,
                    warmup_epochs=config['warm_up_epochs'],
                    max_epochs=config['epochs'])
                      
                min_val_loss = 1000
                test_result = []
                losses = []
                with tqdm(total=config['epochs'], desc='(T)') as pbar:
                    for epoch in range(1,config['epochs']+1):
                        total_loss = 0.0
                        batch_count = 0                
                        for batch_idx, sample_inds in enumerate(train_loader.batch_sampler):
                            sample_inds = removeDuplicates(names_train,sample_inds)
                            batch_list = [train_data[i] for i in sample_inds]
                            batch_loader = DataLoader(batch_list, batch_size=len(batch_list))
                            batch_data = next(iter(batch_loader))
                            batch_data = other_augment2(batch_data,train_length_limits,device)
                            loss,input_dim = train(batch_data,encoder_model,contrast_model,optimizer)
                            total_loss += loss
                            batch_count += 1
                        scheduler.step()
                                
                        average_loss = total_loss / batch_count if batch_count > 0 else float('nan')   
                        losses.append(average_loss)
                        pbar.set_postfix({'loss': average_loss})
                        pbar.update()        
                        
                        if epoch in eval_epochs:
                            res,linear_state_dict = test(encoder_model,train_loader,val_loader,test_loader,
                                       config['min_length'],max_length,config['num_classes'],
                                       device,config['epochs_cls'],config['lr_cls'])
                            test_result.append(res) 
                            if res['best_val_loss'] < min_val_loss:
                                min_val_loss = res['best_val_loss']
                                min_val_loss_epoch = epoch
                losses_all.append(losses)
                test_result_all.append(test_result)
                min_val_loss_epochs.append(min_val_loss_epoch)
        
        results['other_augmentations']['augmentation2'][atlas] = {}
        results['other_augmentations']['augmentation2'][atlas]['losses'] = losses_all
        results['other_augmentations']['augmentation2'][atlas]['epoch_results'] = test_result_all
        results['other_augmentations']['augmentation2'][atlas]['min_val_loss_epoch'] = min_val_loss_epochs
        results['other_augmentations']['augmentation2'][atlas]['names_train'] = names_train_all
        results['other_augmentations']['augmentation2'][atlas]['names_val'] = names_val_all
        results['other_augmentations']['augmentation2'][atlas]['names_test'] = names_test_all
        
        if config['save_results']:          
            if not os.path.exists(os.path.join(config['path_save'],'ablations','results_ABIDEI')):
                os.makedirs(os.path.join(config['path_save'],'ablations','results_ABIDEI'),exist_ok=True)
            with open(os.path.join(config['path_save'],'ablations','results_ABIDEI','ABIDEI_VarCoNet_ablations.pkl'), 'wb') as f:
                pickle.dump(results,f)
    return results


if __name__ == '__main__':   
    
    parser = argparse.ArgumentParser(description='Run VarCoNet ablations on ABIDE I for ASD classification')

    parser.add_argument('--path_data', type=str,
                        help='Path to the dataset')
    parser.add_argument('--path_save', type=str,
                        help='Path to save results')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use for training')
    parser.add_argument('--min_length', type=int, default=80,
                        help='Minimum length for augmentation')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--warm_up_epochs', type=int, default=10,
                        help='Number of warm up epochs for the lr scheduler')
    parser.add_argument('--epochs_cls', type=int, default=150,
                        help='Number of epochs for the linear classification layer')
    parser.add_argument('--lr_cls', type=float, default=5e-5,
                        help='Learning rate for the linear classification layer')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='Number of classes for the classification')
    parser.add_argument('--save_results', action='store_true',
                        help='Flag to save results')

    args = parser.parse_args()
    
    config = {
        'path_data': args.path_data,
        'path_save': args.path_save,
        'min_length': args.min_length,
        'shuffle': True,
        'epochs': args.epochs,
        'warm_up_epochs': args.warm_up_epochs,
        'epochs_cls': args.epochs_cls,
        'lr_cls': args.lr_cls,
        'num_classes': args.num_classes,
        'save_results': args.save_results,
        'device': args.device,
        'model_config': {}
    }

    results = main(config)
