parent_path = r'/.../baselines'
import sys
sys.path.append(parent_path)
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from utils import ABIDEDataset_BAnD
import os
import pickle
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold
import copy
from BAnD.band.models.models import S3ConvXTransFC
from BAnD.band.models.TransformerFM import BertConfig
import argparse  

    
def train(x, y, encoder_model, optimizer, loss_func, num_classes):
    encoder_model.train()
    optimizer.zero_grad()
    x = x.permute(0,4,1,2,3)
    z = encoder_model(x)
    soft = nn.Softmax(dim=-1)
    z = soft(z)
    loss = loss_func(z, F.one_hot(y, num_classes=num_classes).float())
    loss.backward()
    optimizer.step()
    return loss.item()


def test(encoder_model, test_data_loader, loss_func, window_size, num_classes, device):
    encoder_model.eval()
    soft = nn.Softmax(dim=-1)
    with torch.no_grad():
        zs = []
        ys = []
        for (x,y) in test_data_loader:
            last_dim = x.shape[-1]
            for i,start in enumerate(range(0, last_dim - window_size + 1, window_size)):
                end = start + window_size
                x_temp = x[:,:,:,:,start:end]
                if i == 0:
                    logits = encoder_model(x_temp.to(device))
                else:
                    logits += encoder_model(x_temp.to(device))
            logits = logits/(i+1)
            zs.append(logits)
            ys.append(y)
        z = torch.cat(zs,dim=0)
        y = torch.cat(ys,dim=0)
        loss = loss_func(soft(z), F.one_hot(y, num_classes=num_classes).float().to(device))
        z = z[:,-1].cpu().numpy()
        y = y.numpy()
        auc_score = roc_auc_score(y, z)
                   
    return loss.item(), auc_score, z, y


def main(config):
    path = config['path_data']
    names = []
    y = []
    with open(os.path.join(path,'labels.txt'), 'r') as f:
        lines = f.readlines()
        for line in lines:
            fields = line.split(' ')
            name = fields[0]
            pos = name.find('_run')
            names.append(name[:pos])
            y.append(fields[-1])
    data = os.listdir(os.path.join(path,'data'))
    names_new = []
    y_new = []
    for file in data:
        pos = name.find('_run')
        file = file[:pos]
        names_new.append(file)
        pos_label = np.where(np.array(names) == file)
        y_new.append(int(y[pos_label[0][0]]))
    names = names_new
    y = np.array(y_new)
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
    
    bert_config = BertConfig(vocab_size_or_config_json_file=-1,
                             hidden_size=512,
                             num_hidden_layers=1,
                             num_attention_heads=8,
                             intermediate_size=512,
                             hidden_act="relu",
                             hidden_dropout_prob=0.2,
                             attention_probs_dropout_prob=0.2,
                             max_position_embeddings=128,
                             type_vocab_size=None,
                             initializer_range=0.02,
                             layer_norm_eps=1e-12
                             )
    
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
    for i in range(1):
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
            train_dataset = ABIDEDataset_BAnD(train_data, y_train, os.path.join(path,'data'), config['max_length'], 'train')
            train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=config['shuffle'])
            val_dataset = ABIDEDataset_BAnD(val_data, y_val, os.path.join(path,'data'), config['max_length'], 'test')
            val_loader = DataLoader(val_dataset, batch_size=1)
            test_dataset = ABIDEDataset_BAnD(test_data, y_test, os.path.join(path,'data'), config['max_length'], 'test')
            test_loader = DataLoader(test_dataset, batch_size=1)  
            names_train_all.append(names_train)
            names_val_all.append(names_val)
            names_test_all.append(names_test)
            
            encoder_model = S3ConvXTransFC(1, 2, bert_config, 512).to(device)
            loss_func = nn.BCELoss()
            optimizer = SGD(
                encoder_model.parameters(),
                lr=config['lr'],
                momentum=config['momentum'],
                weight_decay=config['weight_decay']
            )
            scheduler = MultiStepLR(optimizer, milestones=config['milestones'], gamma=config['gamma'])
            
            min_val_loss = 1000
            losses = []
            val_losses = []
            test_losses = []
            val_aucs = []
            test_aucs = []
            val_probs = []
            test_probs = []
            y_vals = []
            y_tests = []
            with tqdm(total=config['epochs'], desc='(T)') as pbar:
                for epoch in range(1,config['epochs']+1):
                    total_loss = 0.0
                    batch_count = 0                          
                    for batch_idx, (batch_data, batch_labels) in enumerate(train_loader):            
                        loss = train(batch_data.to(device), batch_labels.to(device), 
                                     encoder_model, optimizer, loss_func, config['num_classes'])
                        total_loss += loss
                        batch_count += 1
                    scheduler.step()
                    average_loss = total_loss / batch_count if batch_count > 0 else float('nan')   
                    losses.append(average_loss)
                    if epoch % 5 != 0:
                        pbar.set_postfix({
                            'loss': average_loss
                        })
                    else:
                        val_loss,val_auc,val_prob,y_val = test(encoder_model,val_loader,
                                                               loss_func,config['max_length'],
                                                               config['num_classes'],device)
                        val_losses.append(val_loss)
                        val_aucs.append(val_auc)
                        val_probs.append(val_prob)
                        y_vals.append(y_val)
                        if val_loss < min_val_loss:
                            min_val_loss_model = copy.deepcopy(encoder_model.state_dict())
                        test_loss,test_auc,test_prob,y_test = test(encoder_model,test_loader,
                                                                   loss_func,config['max_length'],
                                                                   config['num_classes'],device)
                        test_losses.append(test_loss)
                        test_aucs.append(test_auc)
                        test_probs.append(test_prob)
                        y_tests.append(y_test)
                        pbar.set_postfix({
                            'loss': average_loss, 
                            'val_loss': val_loss, 
                            'val_auc': val_auc
                        })
                    pbar.update()  
            
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
                if not os.path.exists(os.path.join(config['path_save'],'models_ABIDEI','BAnD')):
                    os.makedirs(os.path.join(config['path_save'],'models_ABIDEI','BAnD'), exist_ok=True)
                torch.save(min_val_loss_model, os.path.join(config['path_save'],'models_ABIDEI','BAnD','min_val_loss_model_rs' + str(i) + '_fold' + str(j) + '.pth'))
    
    
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
        train_dataset = ABIDEDataset_BAnD(train_data, y_train, os.path.join(path,'data'), config['max_length'], 'train')
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=config['shuffle'])
        val_dataset = ABIDEDataset_BAnD(val_data, y_val, os.path.join(path,'data'), config['max_length'], 'test')
        val_loader = DataLoader(val_dataset, batch_size=1)
        test_dataset = ABIDEDataset_BAnD(ext_test_data, y_ext_test, os.path.join(path,'data'), config['max_length'], 'test')
        test_loader = DataLoader(test_dataset, batch_size=1)
        names_train_ext_all.append(names_train)
        names_val_ext_all.append(names_val)
        
        encoder_model = S3ConvXTransFC(1, config['num_classes'], bert_config, 512).to(device)
        loss_func = nn.BCELoss()
        optimizer = SGD(
            encoder_model.parameters(),
            lr=config['lr'],
            momentum=config['momentum'],
            weight_decay=config['weight_decay']
        )
        scheduler = MultiStepLR(optimizer, milestones=config['milestones'], gamma=config['gamma'])
        
        min_val_loss = 1000
        losses = []
        val_losses = []
        test_losses = []
        val_aucs = []
        test_aucs = []
        val_probs = []
        test_probs = []
        y_vals = []
        y_tests = []
        with tqdm(total=config['epochs'], desc='(T)') as pbar:
            for epoch in range(1,config['epochs']+1):
                total_loss = 0.0
                batch_count = 0                          
                for batch_idx, (batch_data, batch_labels) in enumerate(train_loader):            
                    loss = train(batch_data.to(device), batch_labels.to(device), encoder_model,
                                 optimizer, loss_func, config['num_classes'])
                    total_loss += loss
                    batch_count += 1
                scheduler.step()
                average_loss = total_loss / batch_count if batch_count > 0 else float('nan')   
                losses.append(average_loss)
                if epoch % 5 != 0:
                    pbar.set_postfix({
                        'loss': average_loss
                    })
                else:
                    val_loss,val_auc,val_prob,y_val = test(encoder_model,val_loader,
                                                           loss_func,config['max_length'],
                                                           config['num_classes'],device)
                    val_losses.append(val_loss)
                    val_aucs.append(val_auc)
                    val_probs.append(val_prob)
                    y_vals.append(y_val)
                    pbar.set_postfix({
                        'loss': average_loss, 
                        'val_loss': val_loss, 
                        'val_auc': val_auc
                    })
                    if val_loss < min_val_loss:
                        min_val_loss_model = copy.deepcopy(encoder_model.state_dict())
                    test_loss,test_auc,test_prob,y_ext_test = test(encoder_model,test_loader,
                                                                   loss_func,config['max_length'],
                                                                   config['num_classes'],device)
                    test_losses.append(test_loss)
                    test_aucs.append(test_auc)
                    test_probs.append(test_prob)
                    y_tests.append(y_test)
                pbar.update()  
        
        ext_test_losses_all.append(test_losses)
        ext_test_aucs_all.append(test_aucs)
        train_losses_ext.append(losses)
        val_losses_all_ext.append(val_losses)
        val_aucs_all_ext.append(val_aucs)
        val_probs_all_ext.append(val_probs)
        ext_test_probs_all.append(test_probs)
        y_val_all_ext.append(y_vals)
        
        if config['save_models']:
            if not os.path.exists(os.path.join(config['path_save'],'models_ABIDEI','BAnD')):
                os.makedirs(os.path.join(config['path_save'],'models_ABIDEI','BAnD'), exist_ok=True)
            torch.save(min_val_loss_model, os.path.join(config['path_save'],'models_ABIDEI','BAnD','min_val_loss_model_rs' + str(i) + '.pth'))
    
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
        if not os.path.exists(os.path.join(config['path_save'],'results_ABIDEI')):
            os.makedirs(os.path.join(config['path_save'],'results_ABIDEI'),exist_ok=True)
        with open(os.path.join(config['path_save'],'results_ABIDEI','ABIDEI_BAnD_results.pkl'), "wb") as pickle_file:
            pickle.dump(results, pickle_file)
    return results

if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description='Run BAnD on ABIDE I for ASD classification')

    parser.add_argument('--path_data', type=str,
                        help='Path to the dataset')
    parser.add_argument('--path_save', type=str,
                        help='Path to save results')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use for training')
    parser.add_argument('--max_length', type=int, default=50,
                        help='Minimum length for augmentation')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum for SGD')
    parser.add_argument('--milestones', type=int, nargs='+', default=[16,33],
                        help='Epochs at which learning rate is reduced')
    parser.add_argument('--gamma', type=float, default=0.667,
                        help='Gamma value fore reducing learning rate')
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
        'max_length': args.max_length,
        'shuffle': True,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'momentum': args.momentum,
        'milestones': args.milestones,
        'gamma': args.gamma,
        'num_classes': args.num_classes,
        'save_models': args.save_models,
        'save_results': args.save_results,
        'device': args.device,
    }

    results = main(config)

