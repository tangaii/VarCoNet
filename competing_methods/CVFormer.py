from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim import Adam
from pl_bolts.optimizers import LinearWarmupCosineAnnealingLR
from model_scripts.competing_models import CrossViewModel
from utils import ABIDEDataset
import os
import pickle
from sklearn.metrics import roc_auc_score
from utils import InfoNCE
from utils import DualBranchContrast
from sklearn.model_selection import train_test_split, StratifiedKFold
import copy
import argparse


def train(x, y, encoder_model, contrast_model, optimizer, loss_func, epoch, contrastive_epochs, window_size, num_classes):
    encoder_model.train()
    optimizer.zero_grad()
    z,emb1,emb2 = encoder_model(x[0], x[1], window_size)
    if epoch <= contrastive_epochs:
        loss = contrast_model(emb1, emb2)
    else:
        contrast_loss = contrast_model(emb1, emb2)
        cls_loss = loss_func(z, F.one_hot(y, num_classes=num_classes).float())
        loss = cls_loss + 0.1*contrast_loss
    loss.backward()
    optimizer.step()
    z = z[:,-1].detach().cpu().numpy()
    y = y.to(torch.device("cpu")).numpy()
    auc_score = roc_auc_score(y, z)
    return loss.item(),auc_score


def test(encoder_model, test_data_loader, test_data_T, test_data_C, y_test, loss_func, window_size, num_classes, device):
    encoder_model.eval()
    with torch.no_grad():
        zs = []
        ys = []
        for batch_inds in test_data_loader.batch_sampler:
            batch_list_T = [test_data_T[i] for i in batch_inds]
            batch_list_C = [test_data_C[i] for i in batch_inds]
            batch_labels = y_test[batch_inds]
            batch_loader_T = DataLoader(batch_list_T, batch_size=len(batch_list_T))
            batch_loader_C = DataLoader(batch_list_C, batch_size=len(batch_list_C))
            batch_data_T = next(iter(batch_loader_T)).float()
            batch_data_C = next(iter(batch_loader_C)).float()
            y = torch.from_numpy(batch_labels)
            z,_,_ = encoder_model(batch_data_T.to(device),batch_data_C.to(device), window_size)
            zs.append(z)
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
    data_T = []
    data_C = []
    for key in data_list:
        dat = (data_list[key][:config['length'],:] - np.mean(data_list[key][:config['length'],:],axis=0,keepdims=True))/np.std(data_list[key][:config['length'],:],axis=0,keepdims=True)
        zero_rows = np.all(dat == 0, axis=1)
        zero_row_indices = np.where(zero_rows)[0]
        if config['atlas'] == 'AAL':
            dat = np.concatenate((dat,dat[:,0:2]),axis=1)
        if len(zero_row_indices) > 0:
            dat1 = dat[:np.min(zero_row_indices),:]
        else:
            dat1 = dat
        data_T.append(dat)
        corr = np.corrcoef(dat1.T)
        corr[np.abs(corr) < np.quantile(np.abs(corr), 0.7)] = 0
        data_C.append(corr)
    
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
    train_DATA_C = [data_C[i] for i in pos_duplicates]
    train_DATA_T = [data_T[i] for i in pos_duplicates]
    data_C = [data_C[i] for i in pos_unique]
    data_T = [data_T[i] for i in pos_unique]
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
    ext_test_data_C = [data_C[i] for i in pos_ext_test]
    ext_test_data_T = [data_T[i] for i in pos_ext_test]
    y_ext_test = y[pos_ext_test]
    
    pos = []
    for name in names:
        pos.append(np.where(np.array(names_unique) == name)[0][0])
    data_C = [data_C[i] for i in pos]
    data_T = [data_T[i] for i in pos]
    y = y[pos]
        
     
    device = torch.device(config['device']) if torch.cuda.is_available() else torch.device("cpu")  
    
    model_config = config['model_config'] 
    
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
        for j, (train_index, test_index) in enumerate(skf.split(data_T, y)):
            train_data_T = [data_T[i] for i in train_index]
            train_data_C = [data_C[i] for i in train_index]
            test_data_T = [data_T[n] for n in test_index]
            test_data_C = [data_C[n] for n in test_index]
            y_train = y[train_index]
            y_test = y[test_index]
            names_train = [names[n] for n in train_index]
            names_test = [names[n] for n in test_index]
            _, _, y_train, y_val, train_idx, val_idx = train_test_split(train_data_T,
                                                                        y_train,
                                                                        np.arange(len(train_data_T)), 
                                                                        test_size=0.15, 
                                                                        random_state=42, 
                                                                        stratify=y_train)
            val_data_T = [train_data_T[i] for i in val_idx]
            val_data_C = [train_data_C[i] for i in val_idx]
            train_data_T = [train_data_T[i] for i in train_idx]
            train_data_C = [train_data_C[i] for i in train_idx]
            names_val = [names_train[n] for n in val_idx]
            names_train = [names_train[n] for n in train_idx]
            train_data_T = train_DATA_T + train_data_T
            train_data_C = train_DATA_C + train_data_C
            y_train = np.concatenate((Y_train, y_train))
            names_train = names_duplicate + names_train
            train_dataset_T = ABIDEDataset(train_data_T, y_train)
            train_loader = DataLoader(train_dataset_T, batch_size=config['batch_size'],
                                      shuffle=config['shuffle'], drop_last=True)
            val_dataset_T = ABIDEDataset(val_data_T, y_val)
            val_loader = DataLoader(val_dataset_T, batch_size=config['batch_size'])
            test_dataset_T = ABIDEDataset(test_data_T, y_test)
            test_loader = DataLoader(test_dataset_T, batch_size=config['batch_size'])
            names_train_all.append(names_train)
            names_val_all.append(names_val)
            names_test_all.append(names_test)
    
            roi_num = test_data_T[0].shape[1]
            model_config['d_model1'] = roi_num
            model_config['d_model2'] = int((roi_num/model_config['window_size'])**2)
            encoder_model = CrossViewModel(model_config).to(device)
            contrast_model = DualBranchContrast(loss=InfoNCE(tau=0.07),mode='L2L').to(device)
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
                    for batch_idx, batch_inds in enumerate(train_loader.batch_sampler):   
                        batch_list_T = [train_data_T[i] for i in batch_inds]
                        batch_list_C = [train_data_C[i] for i in batch_inds]
                        batch_labels = y_train[batch_inds]
                        batch_loader_T = DataLoader(batch_list_T, batch_size=len(batch_list_T))
                        batch_loader_C = DataLoader(batch_list_C, batch_size=len(batch_list_C))
                        batch_data_T = next(iter(batch_loader_T)).float()
                        batch_data_C = next(iter(batch_loader_C)).float()
                        batch_labels = torch.from_numpy(batch_labels)
                        loss,auc = train([batch_data_T.to(device),batch_data_C.to(device)],
                                         batch_labels.to(device), encoder_model, contrast_model,
                                         optimizer, loss_func, epoch, config['contrastive_epochs'],
                                         model_config['window_size'],config['num_classes'])
                        total_loss += loss
                        total_auc += auc
                        batch_count += 1
                    scheduler.step()
                    val_loss,val_auc,val_prob,y_val = test(encoder_model,val_loader,
                                                           val_data_T,val_data_C,
                                                           y_val,loss_func,
                                                           model_config['window_size'],
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
                    if epoch > config['contrastive_epochs']:
                        if val_loss < min_val_loss:
                            min_val_loss_model = copy.deepcopy(encoder_model.state_dict())
                    test_loss,test_auc,test_prob,y_test = test(encoder_model,test_loader,
                                                               test_data_T,test_data_C,
                                                               y_test,loss_func,model_config['window_size'],
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
            
            if config['save_models']:
                if not os.path.exists(os.path.join(config['path_save'],'models_ABIDEI',config['atlas'],'CVFormer')):
                    os.makedirs(os.path.join(config['path_save'],'models_ABIDEI',config['atlas'],'CVFormer'),exist_ok=True)
                torch.save(min_val_loss_model, os.path.join(config['path_save'],'models_ABIDEI',config['atlas'],'CVFormer','min_val_loss_model_rs' + str(i) + '_fold_' + str(j) + '.pth'))
    
    
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
        _, _, y_train, y_val, train_idx, val_idx = train_test_split(data_T,
                                                                    y, 
                                                                    np.arange(len(data_T)),
                                                                    test_size=0.1,
                                                                    random_state=42+i,
                                                                    stratify=y)
        val_data_T = [data_T[i] for i in val_idx]
        val_data_C = [data_C[i] for i in val_idx]
        train_data_T = [data_T[i] for i in train_idx]
        train_data_C = [data_C[i] for i in train_idx]
        names_val = [names[n] for n in val_idx]
        names_train = [names[n] for n in train_idx]
        train_data_T = train_DATA_T + train_data_T
        train_data_C = train_DATA_C + train_data_C
        y_train = np.concatenate((Y_train, y_train))
        names_train = names_duplicate + names_train
        train_dataset_T = ABIDEDataset(train_data_T, y_train)
        train_loader = DataLoader(train_dataset_T, batch_size=config['batch_size'], 
                                  shuffle=config['shuffle'], drop_last=True)
        val_dataset_T = ABIDEDataset(val_data_T, y_val)
        val_loader = DataLoader(val_dataset_T, batch_size=config['batch_size'])
        test_dataset_T = ABIDEDataset(ext_test_data_T, y_ext_test)
        test_loader = DataLoader(test_dataset_T, batch_size=config['batch_size'])
        names_train_ext_all.append(names_train)
        names_val_ext_all.append(names_val)

        roi_num = test_data_T[0].shape[1]
        model_config['d_model1'] = roi_num
        model_config['d_model2'] = int((roi_num/model_config['window_size'])**2)
        encoder_model = CrossViewModel(model_config).to(device)
        contrast_model = DualBranchContrast(loss=InfoNCE(tau=0.07),mode='L2L').to(device)
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
                for batch_idx, batch_inds in enumerate(train_loader.batch_sampler):   
                    batch_list_T = [train_data_T[i] for i in batch_inds]
                    batch_list_C = [train_data_C[i] for i in batch_inds]
                    batch_labels = y_train[batch_inds]
                    batch_loader_T = DataLoader(batch_list_T, batch_size=len(batch_list_T))
                    batch_loader_C = DataLoader(batch_list_C, batch_size=len(batch_list_C))
                    batch_data_T = next(iter(batch_loader_T)).float()
                    batch_data_C = next(iter(batch_loader_C)).float()
                    batch_labels = torch.from_numpy(batch_labels)
                    loss,auc = train([batch_data_T.to(device),batch_data_C.to(device)],
                                     batch_labels.to(device), encoder_model, contrast_model,
                                     optimizer, loss_func, epoch, config['contrastive_epochs'],
                                     model_config['window_size'], config['num_classes'])
                    total_loss += loss
                    total_auc += auc
                    batch_count += 1
                scheduler.step()
                val_loss,val_auc,val_prob,y_val = test(encoder_model,val_loader,
                                                       val_data_T,val_data_C,y_val,
                                                       loss_func,model_config['window_size'],
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
                if epoch > config['contrastive_epochs']:
                    if val_loss < min_val_loss:
                        min_val_loss_model = copy.deepcopy(encoder_model.state_dict())
                test_loss,test_auc,test_prob,y_ext_test = test(encoder_model,test_loader,
                                                               ext_test_data_T,ext_test_data_C,y_test,
                                                               loss_func,model_config['window_size'],
                                                               config['num_classes'],device)
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
            if not os.path.exists(os.path.join(config['path_save'],'models_ABIDEI',config['atlas'],'CVFormer')):
                os.makedirs(os.path.join(config['path_save'],'models_ABIDEI',config['atlas'],'CVFormer'),exist_ok=True)
            torch.save(min_val_loss_model, os.path.join(config['path_save'],'models_ABIDEI',config['atlas'],'CVFormer','min_val_loss_model_rs' + str(i) + '.pth'))
    
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
            os.makedirs(os.path.join(config['path_save'],'results_ABIDEI',config['atlas']), exist_ok=True)
        with open(os.path.join(config['path_save'],'results_ABIDEI',config['atlas'],'ABIDEI_CVFormer_results.pkl'), "wb") as pickle_file:
            pickle.dump(results, pickle_file)
    return results

if __name__ == '__main__':   
    parser = argparse.ArgumentParser(description='Run CvFormer on ABIDE I for ASD classification')

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
    parser.add_argument('--warm_up_epochs', type=int, default=10,
                        help='Number of warm up epochs for the lr scheduler')
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
        'batch_size': args.batch_size,
        'warm_up_epochs': args.warm_up_epochs,
        'num_classes': args.num_classes,
        'save_models': args.save_models,
        'save_results': args.save_results,
        'device': args.device,
        'model_config': {},
    }
    
    config['model_config']['num_layers1'] = 4
    config['model_config']['num_layers2'] = 3
    config['model_config']['num_layers'] = 4
    config['model_config']['num_classes'] = 2
    config['model_config']['patch_proj'] = config['length']
    config['model_config']['length'] = config['length']
    if config['atlas'] == 'AAL':
        config['lr'] = 4e-5
        config['contrastive_epochs'] = 5
        config['model_config']['window_size'] = 12
    elif config['atlas'] == 'AICHA':
        config['lr'] = 4e-5
        config['contrastive_epochs'] = 5
        config['model_config']['window_size'] = 24

    results = main(config)

