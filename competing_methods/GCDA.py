parent_path = r'/.../baselines'
import sys
sys.path.append(parent_path)
import numpy as np
import torch
from GCDA_code.pretraining import train_new
from torch.utils.data import DataLoader
from utils import GCDADataset, GCDADataset2, BatchSampler, GCDA_dims
import os
import pickle
from sklearn.model_selection import train_test_split, StratifiedKFold
import argparse
import json


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
    config['path_model'] = os.path.join(config['path_save'], 'models_HCP', config['atlas'], 'GCDA', 'best_epoch_model.pth')
    
    '''------------------------------------KFold CV------------------------------------'''
    test_result_all = []
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
            train_dataset = GCDADataset(train_data, y_train) 
            batch_sampler = BatchSampler(names_train, batch_size=config['batch_size'],shuffle=config['shuffle'])
            train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler)
            train_dataset2 = GCDADataset2(train_data, y_train)
            train_loader2 = DataLoader(train_dataset2, batch_size=config['batch_size'])
            val_dataset = GCDADataset2(val_data, y_val)
            val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
            test_dataset = GCDADataset2(test_data, y_test)
            test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])  
            names_train_all.append(names_train)
            names_val_all.append(names_val)
            names_test_all.append(names_test)
            
            #X,E,y = GCDA_dims(config['atlas'])
            #config['X'] = X
            #config['E'] = E
            min_val_loss_model, test_result = train_new(config,
                                                        train_loader,
                                                        train_loader2,
                                                        val_loader,
                                                        test_loader)
            
            test_result_all.append(test_result)
            
            if config['save_models']:
                if not os.path.exists(os.path.join(config['path_save'],'models_ABIDEI',config['atlas'],'GCDA')):
                    os.makedirs(os.path.join(config['path_save'],'models_ABIDEI',config['atlas'],'GCDA'),exist_ok=True)
                torch.save(min_val_loss_model, os.path.join(config['path_save'],'models_ABIDEI',config['atlas'],'GCDA','min_val_loss_model_rs' + str(i) + '_fold' + str(j) + '.pth'))
    
    '''------------------------------------Ext. test------------------------------------'''
    
    ext_test_result_all = []
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
        train_dataset = GCDADataset(train_data, y_train) 
        batch_sampler = BatchSampler(names_train, batch_size=config['batch_size'],shuffle=config['shuffle'])
        train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler)
        train_dataset2 = GCDADataset2(train_data, y_train)
        train_loader2 = DataLoader(train_dataset2, batch_size=config['batch_size'])
        val_dataset = GCDADataset2(val_data, y_val)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
        ext_test_dataset = GCDADataset2(ext_test_data, y_ext_test)
        ext_test_loader = DataLoader(ext_test_dataset, batch_size=config['batch_size'])  
        names_train_ext_all.append(names_train)
        names_val_ext_all.append(names_val)
        
        #X,E,y = GCDA_dims(config['atlas'])
        #config['X'] = X
        #config['E'] = E
        min_val_loss_model, test_result = train_new(config,
                                                    train_loader,
                                                    train_loader2,
                                                    val_loader,
                                                    ext_test_loader)
        ext_test_result_all.append(test_result)
        
        if config['save_models']:
            if not os.path.exists(os.path.join(config['path_save'],'models_ABIDEI',config['atlas'],'GCDA')):
                os.makedirs(os.path.join(config['path_save'],'models_ABIDEI',config['atlas'],'GCDA'),exist_ok=True)
            torch.save(min_val_loss_model, os.path.join(config['path_save'],'models_ABIDEI',config['atlas'],'GCDA','min_val_loss_model_rs' + str(i) + '.pth'))
        
    results = {}
    results['epoch_results'] = test_result_all
    results['epoch_results_ext'] = ext_test_result_all
    results['names_train'] = names_train_all
    results['names_val'] = names_val_all
    results['names_test'] = names_test_all
    results['names_train_ext'] = names_train_ext_all
    results['names_val_ext'] = names_val_ext_all
    results['names_ext_test'] = names_ext_test
    
    if config['save_results']:          
        if not os.path.exists(os.path.join(config['path_save'],'results_ABIDEI',config['atlas'])):
            os.makedirs(os.path.join(config['path_save'],'results_ABIDEI',config['atlas']),exist_ok=True)
        with open(os.path.join(config['path_save'],'results_ABIDEI',config['atlas'],'ABIDEI_GCDA_results.pkl'), 'wb') as f:
            pickle.dump(results,f)
    return results


if __name__ == '__main__':   
    
    parser = argparse.ArgumentParser(description='Run GCDA on ABIDE I for ASD classification')

    parser.add_argument('--path_data', type=str,
                        help='Path to the dataset')
    parser.add_argument('--path_save', type=str,
                        help='Path to save results')
    parser.add_argument('--atlas', type=str, choices=['AICHA', 'AAL'], default='AICHA',
                        help='Atlas type to use')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use for training')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.02,
                        help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum of optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--T', type=int, default=1000)
    parser.add_argument('--diffusion_hidden_mlp_dims', type=json.loads, default='{"X": 64, "E": 4, "y": 16}',
                        help='Hidden MLP dimensions as JSON dict')
    parser.add_argument('--diffusion_hidden_dims', type=json.loads, 
                        default='{"dx":64,"de":8,"dy":8,"n_head":2,"dim_ffX":128,"dim_ffE":16,"dim_ffy":16}',
                        help='Hidden dimensions for difusion as JSON dict')
    parser.add_argument("--diffusion_num_layers", type=int, default=1,
                        help='Number of difusion layers')
    parser.add_argument("--GIN_hidden_dim", type=int, default=64,
                        help='GNN embedding dimension')
    parser.add_argument("--GIN_num_layers", type=int, default=2,
                        help='Number of GNN layers')
    parser.add_argument("--projector_input_dim", type=int, default=64,
                        help='Projector input dimension')
    parser.add_argument("--projector_hidden_dim", type=int, default=32,
                        help='Projector embedding dimension')
    parser.add_argument('--epochs_cls', type=int, default=30,
                        help='Number of epochs for the linear classification layer')
    parser.add_argument('--lr_cls', type=float, default=0.02,
                        help='Learning rate for the linear classification layer')
    parser.add_argument('--momentum_cls', type=float, default=0.9,
                        help='Momentum of optimizer for fine tuning')
    parser.add_argument('--weight_decay_cls', type=float, default=1e-4,
                        help='Weight decay for fine tuning')
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
        'epochs':  args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'momentum': args.momentum,
        'weight_decay': args.weight_decay,
        'T': args.T,
        'diffusion_hidden_mlp_dims': args.diffusion_hidden_mlp_dims,
        'diffusion_hidden_dims': args.diffusion_hidden_dims,
        'diffusion_num_layers': args.diffusion_num_layers,
        'GIN_hidden_dim': args.GIN_hidden_dim,
        'GIN_num_layers': args.GIN_num_layers,
        'projector_input_dim': args.projector_input_dim,
        'projector_hidden_dim': args.projector_hidden_dim,
        'epochs_cls': args.epochs_cls,
        'lr_cls': args.lr_cls,
        'momentum_cls': args.momentum_cls,
        'weight_decay_cls': args.weight_decay_cls,
        'num_classes': args.num_classes,
        'save_models': args.save_models,
        'save_results': args.save_results,
        'device': args.device,
        'model_config': {}
    }
    if config['atlas'] == 'AICHA':
        config['GIN_input_dim'] = 384
    elif config['atlas'] == 'AAL':
        config['GIN_input_dim'] = 166

    results = main(config)
