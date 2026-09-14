parent_path = r'/.../baselines'
import sys
sys.path.append(parent_path)
import numpy as np
import torch
from torch_geometric.transforms import Compose
from torch_geometric.loader import DataLoader
from A_GCL.agcl_ABIDE_queue import run
from A_GCL.unsupervised.utils import set_tu_dataset_y_shape
from utils import BatchSampler, AGCLDataset, AGCLDataset2
import os
import pickle
from sklearn.model_selection import train_test_split, StratifiedKFold
import joblib
import shutil
import argparse


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
    
    config['device'] = torch.device(config['device']) if torch.cuda.is_available() else torch.device("cpu")
    config['eval_epochs'] = list(range(1, config['epochs']+1)) 
    
    '''------------------------------------KFold CV------------------------------------'''
    losses_all = []
    test_result_all = []
    min_val_loss_epochs = []
    min_loss_epochs = []
    names_train_all = []
    names_val_all = []
    names_test_all = []
    for i in range(10):
        skf = StratifiedKFold(n_splits=10, shuffle = True, random_state=42+i)
        for j, (train_index, test_index) in enumerate(skf.split(data, y)):
            train_data = [data[n] for n in train_index]
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
            my_transforms = Compose([set_tu_dataset_y_shape]) 
            train_dataset = AGCLDataset(os.path.join(config['path_data'],'A_GCL'), train_data, y_train, config['atlas'], transform=my_transforms) 
            batch_sampler = BatchSampler(names_train, batch_size=config['batch_size'],
                                         shuffle=config['shuffle'],drop_last=True)
            train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler)
            val_dataset = AGCLDataset2(val_data, y_val)
            val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
            test_dataset = AGCLDataset2(test_data, y_test)
            test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])  
            names_train_all.append(names_train)
            names_val_all.append(names_val)
            names_test_all.append(names_test)
            
            min_val_loss_model, min_val_loss_classifier, test_result, min_val_loss_epoch, losses = run(config,
                                                                                                       train_loader,
                                                                                                       val_loader,
                                                                                                       test_loader)
            temp_path = os.path.join(config['path_data'],'A_GCL', 'processed')
            shutil.rmtree(temp_path)
            losses_all.append(losses)
            test_result_all.append(test_result)
            min_val_loss_epochs.append(min_val_loss_epoch)
            
            if config['save_models']:
                if not os.path.exists(os.path.join(config['path_save'],'models_ABIDEI',config['atlas'],'A_GCL')):
                    os.makedirs(os.path.join(config['path_save'],'models_ABIDEI',config['atlas'],'A_GCL'),exist_ok=True)
                torch.save(min_val_loss_model, os.path.join(config['path_save'],'models_ABIDEI',config['atlas'],'A_GCL','min_val_loss_model_rs' + str(i) + '_fold' + str(j) + '.pth'))
                joblib.dump(min_val_loss_classifier, os.path.join(config['path_save'],'models_ABIDEI',config['atlas'],'A_GCL','min_val_loss_classifier_rs' + str(i) + '_fold' + str(j) + '.pkl'))
                
    '''------------------------------------Ext. test------------------------------------'''
    losses_all_ext = []
    ext_test_result_all = []
    min_val_loss_epochs_ext = []
    min_loss_epochs_ext = []
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
        train_dataset = AGCLDataset(os.path.join(config['path_data'],'A_GCL'), train_data, y_train, config['atlas'], transform=my_transforms) 
        batch_sampler = BatchSampler(names_train, batch_size=config['batch_size'],
                                     shuffle=config['shuffle'],drop_last=True)
        train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler)
        val_dataset = AGCLDataset2(val_data, y_val)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
        ext_test_dataset = AGCLDataset2(ext_test_data, y_ext_test)
        ext_test_loader = DataLoader(ext_test_dataset, batch_size=config['batch_size'])  
        names_train_ext_all.append(names_train)
        names_val_ext_all.append(names_val)
        
        min_val_loss_model, min_val_loss_classifier, test_result, min_val_loss_epoch, losses = run(config,
                                                                                                   train_loader,
                                                                                                   val_loader,
                                                                                                   ext_test_loader)
        temp_path = os.path.join(config['path_data'],'A_GCL', 'processed')
        shutil.rmtree(temp_path)
        losses_all_ext.append(losses)
        ext_test_result_all.append(test_result)
        min_val_loss_epochs_ext.append(min_val_loss_epoch)
        
        if config['save_models']:
            if not os.path.exists(os.path.join(config['path_save'],'models_ABIDEI',config['atlas'],'A_GCL')):
                os.makedirs(os.path.join(config['path_save'],'models_ABIDEI',config['atlas'],'A_GCL'),exist_ok=True)
            torch.save(min_val_loss_model, os.path.join(config['path_save'],'models_ABIDEI',config['atlas'],'A_GCL','min_val_loss_model_rs' + str(i) + '.pth'))
            joblib.dump(min_val_loss_classifier, os.path.join(config['path_save'],'models_ABIDEI',config['atlas'],'A_GCL','min_val_loss_classifier_rs' + str(i) + '.pkl'))
            
    results = {}
    results['losses'] = losses_all
    results['epoch_results'] = test_result_all
    results['min_val_loss_epoch'] = min_val_loss_epochs
    results['min_loss_epoch'] = min_loss_epochs
    results['losses_ext'] = losses_all_ext
    results['epoch_results_ext'] = ext_test_result_all
    results['min_val_loss_epoch_ext'] = min_val_loss_epochs_ext
    results['min_loss_epoch_ext'] = min_loss_epochs_ext
    results['names_train'] = names_train_all
    results['names_val'] = names_val_all
    results['names_test'] = names_test_all
    results['names_train_ext'] = names_train_ext_all
    results['names_val_ext'] = names_val_ext_all
    results['names_ext_test'] = names_ext_test
    
    if config['save_results']:          
        if not os.path.exists(os.path.join(config['path_save'],'results_ABIDEI',config['atlas'])):
            os.makedirs(os.path.join(config['path_save'],'results_ABIDEI',config['atlas']),exist_ok=True)
        with open(os.path.join(config['path_save'],'results_ABIDEI',config['atlas'],'ABIDEI_A_GCL_results.pkl'), 'wb') as f:
            pickle.dump(results,f)
    return results


if __name__ == '__main__':   
    
    parser = argparse.ArgumentParser(description='Run A-GCL on ABIDE I for ASD classification')
    parser.add_argument('--path_data', type=str,
                        help='Path to the dataset')
    parser.add_argument('--path_save', type=str,
                        help='Path to save results')
    parser.add_argument('--atlas', type=str, choices=['AICHA', 'AAL'], default='AICHA',
                        help='Atlas type to use')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use for training')
    parser.add_argument('--model_lr', type=float, default=0.0005,
                        help='Model Learning rate.')
    parser.add_argument('--view_lr', type=float, default=0.0005,
                        help='View Learning rate.')
    parser.add_argument('--num_gc_layers', type=int, default=2,
                        help='Number of GNN layers before pooling')
    parser.add_argument('--pooling_type', type=str, default='standard',
                        help='GNN Pooling Type Standard/Layerwise')
    parser.add_argument('--emb_dim', type=int, default=32,
                        help='embedding dimension')
    parser.add_argument('--mlp_edge_model_dim', type=int, default=64,
                        help='embedding dimension')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--drop_ratio', type=float, default=0.3,
                        help='Dropout Ratio / Probability')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Train Epochs')
    parser.add_argument('--reg_lambda', type=float, default=2.0,
                        help='View Learner Edge Perturb Regularization Strength')
    parser.add_argument('--eval_interval', type=int, default=5, 
                        help="eval epochs interval")
    parser.add_argument('--max_length', type=int, default=256,
                        help='max length of memory bank')
    parser.add_argument('--cr_lambda', type=float, default=0.4,
                        help='Regularization coefficients for loss of cross-batch memory bank')
    parser.add_argument('--memory_type', type=str, default='queue', choices=['momentum', 'queue'],
                        help="type of memory bank")
    parser.add_argument('--feature_type', type=str, default='instance', choices=['mean', 'instance'],
                        help="type of feature in memory bank")
    parser.add_argument('--seed', type=int, default=42)
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
        'model_lr': args.epochs,
        'view_lr': args.view_lr,
        'num_gc_layers': args.num_gc_layers,
        'pooling_type': args.pooling_type,
        'emb_dim': args.emb_dim,
        'mlp_edge_model_dim': args.mlp_edge_model_dim,
        'batch_size': args.batch_size,
        'drop_ratio': args.drop_ratio,
        'epochs': args.epochs,
        'reg_lambda': args.reg_lambda,
        'max_length': args.max_length,
        'cr_lambda': args.cr_lambda,
        'memory_type': args.memory_type,
        'feature_type': args.feature_type,
        'seed': args.seed,
        'num_classes': args.num_classes,
        'save_models': args.save_models,
        'save_results': args.save_results,
        'device': args.device,
        'model_config': {}
    }

    results = main(config)
