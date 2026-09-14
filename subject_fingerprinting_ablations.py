from torch.utils.data import DataLoader
import numpy as np
import torch
from utils import InfoNCE
from tqdm import tqdm
from torch.optim import Adam
from utils import DualBranchContrast
from pl_bolts.optimizers import LinearWarmupCosineAnnealingLR
from model_scripts.VarCoNet import VarCoNet_noCNN, VarCoNet_noTransformer, VarCoNet
import os
import pickle
import copy
from utils import test_augment, augment, removeDuplicates, other_augment1, other_augment2
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

def test(encoder_model,test_data1,test_data2,num_winds,batch_size,device):
    encoder_model.eval()
    with torch.no_grad():
        outputs1 = []
        for data in test_data1:
            data = data.to(device)
            outputs = []
            for i in range(0, data.size(0), batch_size):
                batch = data[i:i + batch_size]
                outputs.append(encoder_model(batch))
            outputs1.append(torch.cat(outputs, dim=0))
        outputs1 = torch.stack(outputs1)
        
        outputs2 = []
        for data in test_data2:
            data = data.to(device)
            outputs = []
            for i in range(0, data.size(0), batch_size):
                batch = data[i:i + batch_size]
                outputs.append(encoder_model(batch))
            outputs2.append(torch.cat(outputs, dim=0))
        outputs2 = torch.stack(outputs2)
        
        accuracies_all = []
        mean_accs = []
        std_accs= []
        num_real = int(outputs1.shape[1] / num_winds)
        print('')
        for i in range (num_winds):
            for n in range(num_winds):
                if n >= i:
                    accuracies = []
                    for j in range(num_real):
                        corr_coeffs = torch.corrcoef(torch.cat([outputs1[:, i*num_real+j, :],outputs2[:, n*num_real+j, :]],dim=0))[0:outputs1.shape[0],outputs1.shape[0]:]
                        lower_indices = torch.tril_indices(corr_coeffs.shape[0],corr_coeffs.shape[1], offset=-1)
                        upper_indices = torch.triu_indices(corr_coeffs.shape[0],corr_coeffs.shape[1], offset=1)
                        corr_coeffs1 = corr_coeffs.clone()
                        corr_coeffs2 = corr_coeffs.clone()
                        corr_coeffs1[lower_indices[0],lower_indices[1]] = -2
                        corr_coeffs2[upper_indices[0],upper_indices[1]] = -2
                        counter1 = 0
                        counter2 = 0
                        for j in range(corr_coeffs1.shape[0]):
                            if torch.argmax(corr_coeffs1[j, :]) == j:
                                counter1 += 1
                        for j in range(corr_coeffs2.shape[1]):
                            if torch.argmax(corr_coeffs2[:, j]) == j:
                                counter2 += 1
            
                        total_samples = outputs1.shape[0] + outputs2.shape[0]
                        accuracies.append((counter1 + counter2) / total_samples)
                    mean_acc = np.mean(accuracies)
                    std_acc = np.std(accuracies)
                    print(f'meanAcc{i,n}: {mean_acc:.2f}, stdAcc{i,n}: {std_acc:.3f}')
                    mean_accs.append(mean_acc)
                    std_accs.append(std_acc)
                    accuracies_all.append(accuracies)
        print('')
    return accuracies_all, mean_accs, std_accs


def main(config):
    results = {}
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
        with open(os.path.join(path,'names_train.txt'), 'r') as f:
            for line in f:
                names.append(line.strip())
                
        data = np.load(os.path.join(path,'train_data_HCP_' + atlas + '_resampled.npz'))
        train_data = []
        for key in data:
            train_data.append(data[key])
            
        data = np.load(os.path.join(path,'val_data_HCP_' + atlas + '_1_resampled.npz'))
        val_data1 = []
        for key in data:
            val_data1.append(data[key])
        
        data = np.load(os.path.join(path,'val_data_HCP_' + atlas + '_2_resampled.npz'))
        val_data2 = []
        for key in data:
            val_data2.append(data[key])
        
        data = np.load(os.path.join(path,'test_data_HCP_' + atlas + '_1_resampled.npz'))
        test_data1 = []
        for key in data:
            test_data1.append(data[key])
        
        data = np.load(os.path.join(path,'test_data_HCP_' + atlas + '_2_resampled.npz'))
        test_data2 = []
        for key in data:
            test_data2.append(data[key])      
    
        roi_num = test_data1[0].shape[1]
        
        
        device = torch.device(config['device']) if torch.cuda.is_available() else torch.device("cpu")
        max_length = config['train_length_limits'][-1]
        
        for i,data in enumerate(test_data1):
            data = torch.from_numpy(data.astype(np.float32))
            data = test_augment(data, config['test_winds'], config['num_test_winds'], max_length)
            test_data1[i] = data
            
        for i,data in enumerate(test_data2):
            data = torch.from_numpy(data.astype(np.float32))
            data = test_augment(data, config['test_winds'], config['num_test_winds'], max_length)
            test_data2[i] = data
            
        for i,data in enumerate(val_data1):
            data = torch.from_numpy(data.astype(np.float32))
            data = test_augment(data, config['test_winds'], config['num_test_winds'], max_length)
            val_data1[i] = data
            
        for i,data in enumerate(val_data2):
            data = torch.from_numpy(data.astype(np.float32))
            data = test_augment(data, config['test_winds'], config['num_test_winds'], max_length)
            val_data2[i] = data
            
        train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=config['shuffle'])
    
        
        model_config = config['model_config']
        model_config['max_length'] = max_length
        
        '''-------------------------------------no CNN-------------------------------------'''
      
        encoder_model = VarCoNet_noCNN(model_config, roi_num).to(device)
        contrast_model = DualBranchContrast(loss=InfoNCE(tau=config['tau']),mode='L2L').to(device)   
        optimizer = Adam(encoder_model.parameters(), lr=config['lr'])
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer=optimizer,
            warmup_start_lr = 1e-5,
            warmup_epochs=config['warm_up_epochs'],
            max_epochs=config['epochs'])
                  
        max_val_acc = 0
        losses = []
        res_all = []
        count = 0
        with tqdm(total=config['epochs'], desc='(T)') as pbar:
            for epoch in range(1,config['epochs']+1):
                total_loss = 0.0
                batch_count = 0                              
                for batch_idx, sample_inds in enumerate(train_loader.batch_sampler):
                    sample_inds = removeDuplicates(names,sample_inds)
                    batch_list = [train_data[i] for i in sample_inds]
                    batch_loader = DataLoader(batch_list, batch_size=len(batch_list))
                    batch_data = next(iter(batch_loader))
                    batch_data = augment(batch_data,config['train_length_limits'],device)
                    loss,input_dim = train(batch_data,encoder_model,contrast_model,optimizer)
                    total_loss += loss
                    batch_count += 1
                scheduler.step()
                average_loss = total_loss / batch_count if batch_count > 0 else float('nan')
                losses.append(average_loss)
                pbar.set_postfix({'loss': average_loss})
                pbar.update()        
                
                if epoch in config['eval_epochs']:
                    res = test(encoder_model,val_data1,val_data2,
                               len(config['test_winds']),config['batch_size'],device)
                    res_all.append(res)
                    if np.mean(res[1]) + np.min(res[1]) > max_val_acc:
                        max_val_acc = np.mean(res[1]) + np.min(res[1])
                        max_val_acc_model = copy.deepcopy(encoder_model.state_dict())
                        count = 0
                    else:
                        if epoch > config['eval_epochs'][0]:
                            count += 1
                if count >= config['early_stopping']:
                    print('Early stopping')
                    break
                print('')
    
        max_val_acc_encoder_model = VarCoNet_noCNN(model_config, roi_num).to(device)
        max_val_acc_encoder_model.load_state_dict(max_val_acc_model)
        test_result = test(max_val_acc_encoder_model, test_data1, test_data2,
                                    len(config['test_winds']), config['batch_size'],device)
        results['no_CNN'][atlas] = test_result
        
        '''---------------------------------no Transformer---------------------------------'''
      
        encoder_model = VarCoNet_noTransformer(model_config, roi_num).to(device)
        contrast_model = DualBranchContrast(loss=InfoNCE(tau=config['tau']),mode='L2L').to(device)   
        optimizer = Adam(encoder_model.parameters(), lr=config['lr'])
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer=optimizer,
            warmup_start_lr = 1e-5,
            warmup_epochs=config['warm_up_epochs'],
            max_epochs=config['epochs'])
                  
        max_val_acc = 0
        losses = []
        res_all = []
        count = 0
        with tqdm(total=config['epochs'], desc='(T)') as pbar:
            for epoch in range(1,config['epochs']+1):
                total_loss = 0.0
                batch_count = 0                              
                for batch_idx, sample_inds in enumerate(train_loader.batch_sampler):
                    sample_inds = removeDuplicates(names,sample_inds)
                    batch_list = [train_data[i] for i in sample_inds]
                    batch_loader = DataLoader(batch_list, batch_size=len(batch_list))
                    batch_data = next(iter(batch_loader))
                    batch_data = augment(batch_data,config['train_length_limits'],device)
                    loss,input_dim = train(batch_data,encoder_model,contrast_model,optimizer)
                    total_loss += loss
                    batch_count += 1
                scheduler.step()
                average_loss = total_loss / batch_count if batch_count > 0 else float('nan')
                losses.append(average_loss)
                pbar.set_postfix({'loss': average_loss})
                pbar.update()        
                
                if epoch in config['eval_epochs']:
                    res = test(encoder_model,val_data1,val_data2,
                               len(config['test_winds']),config['batch_size'],device)
                    res_all.append(res)
                    if np.mean(res[1]) + np.min(res[1]) > max_val_acc:
                        max_val_acc = np.mean(res[1]) + np.min(res[1])
                        max_val_acc_model = copy.deepcopy(encoder_model.state_dict())
                        count = 0
                    else:
                        if epoch > config['eval_epochs'][0]:
                            count += 1
                if count >= config['early_stopping']:
                    print('Early stopping')
                    break
                print('')
    
        max_val_acc_encoder_model = VarCoNet_noTransformer(model_config, roi_num).to(device)
        max_val_acc_encoder_model.load_state_dict(max_val_acc_model)
        test_result = test(max_val_acc_encoder_model, test_data1, test_data2,
                                    len(config['test_winds']), config['batch_size'],device)
        results['no_Transformer'][atlas] = test_result
        
        '''------------------------Other augmentations - augmentation 1--------------------'''
      
        encoder_model = VarCoNet(model_config, roi_num).to(device)
        contrast_model = DualBranchContrast(loss=InfoNCE(tau=config['tau']),mode='L2L').to(device)   
        optimizer = Adam(encoder_model.parameters(), lr=config['lr'])
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer=optimizer,
            warmup_start_lr = 1e-5,
            warmup_epochs=config['warm_up_epochs'],
            max_epochs=config['epochs'])
                  
        max_val_acc = 0
        losses = []
        res_all = []
        count = 0
        with tqdm(total=config['epochs'], desc='(T)') as pbar:
            for epoch in range(1,config['epochs']+1):
                total_loss = 0.0
                batch_count = 0                              
                for batch_idx, sample_inds in enumerate(train_loader.batch_sampler):
                    sample_inds = removeDuplicates(names,sample_inds)
                    batch_list = [train_data[i] for i in sample_inds]
                    batch_loader = DataLoader(batch_list, batch_size=len(batch_list))
                    batch_data = next(iter(batch_loader))
                    batch_data = other_augment1(batch_data,config['train_length_limits'],device)
                    loss,input_dim = train(batch_data,encoder_model,contrast_model,optimizer)
                    total_loss += loss
                    batch_count += 1
                scheduler.step()
                average_loss = total_loss / batch_count if batch_count > 0 else float('nan')
                losses.append(average_loss)
                pbar.set_postfix({'loss': average_loss})
                pbar.update()        
                
                if epoch in config['eval_epochs']:
                    res = test(encoder_model,val_data1,val_data2,
                               len(config['test_winds']),config['batch_size'],device)
                    res_all.append(res)
                    if np.mean(res[1]) + np.min(res[1]) > max_val_acc:
                        max_val_acc = np.mean(res[1]) + np.min(res[1])
                        max_val_acc_model = copy.deepcopy(encoder_model.state_dict())
                        count = 0
                    else:
                        if epoch > config['eval_epochs'][0]:
                            count += 1
                if count >= config['early_stopping']:
                    print('Early stopping')
                    break
                print('')
    
        max_val_acc_encoder_model = VarCoNet(model_config, roi_num).to(device)
        max_val_acc_encoder_model.load_state_dict(max_val_acc_model)
        test_result = test(max_val_acc_encoder_model, test_data1, test_data2,
                                    len(config['test_winds']), config['batch_size'],device)
        results['other_augmentations']['augmentation1'][atlas] = test_result
        
        
        '''------------------------Other augmentations - augmentation 2--------------------'''
      
        encoder_model = VarCoNet(model_config, roi_num).to(device)
        contrast_model = DualBranchContrast(loss=InfoNCE(tau=config['tau']),mode='L2L').to(device)   
        optimizer = Adam(encoder_model.parameters(), lr=config['lr'])
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer=optimizer,
            warmup_start_lr = 1e-5,
            warmup_epochs=config['warm_up_epochs'],
            max_epochs=config['epochs'])
                  
        max_val_acc = 0
        losses = []
        res_all = []
        count = 0
        with tqdm(total=config['epochs'], desc='(T)') as pbar:
            for epoch in range(1,config['epochs']+1):
                total_loss = 0.0
                batch_count = 0                              
                for batch_idx, sample_inds in enumerate(train_loader.batch_sampler):
                    sample_inds = removeDuplicates(names,sample_inds)
                    batch_list = [train_data[i] for i in sample_inds]
                    batch_loader = DataLoader(batch_list, batch_size=len(batch_list))
                    batch_data = next(iter(batch_loader))
                    batch_data = other_augment2(batch_data,config['train_length_limits'],device)
                    loss,input_dim = train(batch_data,encoder_model,contrast_model,optimizer)
                    total_loss += loss
                    batch_count += 1
                scheduler.step()
                average_loss = total_loss / batch_count if batch_count > 0 else float('nan')
                losses.append(average_loss)
                pbar.set_postfix({'loss': average_loss})
                pbar.update()        
                
                if epoch in config['eval_epochs']:
                    res = test(encoder_model,val_data1,val_data2,
                               len(config['test_winds']),config['batch_size'],device)
                    res_all.append(res)
                    if np.mean(res[1]) + np.min(res[1]) > max_val_acc:
                        max_val_acc = np.mean(res[1]) + np.min(res[1])
                        max_val_acc_model = copy.deepcopy(encoder_model.state_dict())
                        count = 0
                    else:
                        if epoch > config['eval_epochs'][0]:
                            count += 1
                if count >= config['early_stopping']:
                    print('Early stopping')
                    break
                print('')
    
        max_val_acc_encoder_model = VarCoNet(model_config, roi_num).to(device)
        max_val_acc_encoder_model.load_state_dict(max_val_acc_model)
        test_result = test(max_val_acc_encoder_model, test_data1, test_data2,
                                    len(config['test_winds']), config['batch_size'],device)
        results['other_augmentations']['augmentation2'][atlas] = test_result
        
        
        if config['save_results']:
            if not os.path.exists(os.path.join(config['path_save'],'ablations','results_HCP')):
                os.makedirs(os.path.join(config['path_save'],'ablations','results_HCP'), exist_ok=True)   
            with open(os.path.join(config['path_save'],'ablations','results_HCP','HCP_VarCoNet_ablations.pkl'), 'wb') as f:
                pickle.dump(results,f)
    return results


if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description='Run VarCoNet Ablations on HCP for subject fingerprinting')

    parser.add_argument('--path_data', type=str, 
                        help='Path to the dataset')
    parser.add_argument('--path_save', type=str,
                        help='Path to save results')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use for training')
    parser.add_argument('--train_length_limits', type=int, nargs='+', default=[80,320],
                        help='Minimum and maximum length for augmentation')
    parser.add_argument('--test_winds', type=int, nargs='+', default=[80,200,320],
                        help='Lengths used for testing the model')
    parser.add_argument('--num_test_winds', type=int, default=10,
                        help='Number of tests per length to calculate confidence interval')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--early_stopping', type=int, default=8,
                        help='Maximum number of epochs without improvement')
    parser.add_argument('--warm_up_epochs', type=int, default=10,
                        help='Number of warm up epochs for the lr scheduler')
    parser.add_argument('--save_results', action='store_true',
                        help='Flag to save results')

    args = parser.parse_args()

    config = {
        'path_data': args.path_data,
        'path_save': args.path_save,
        'train_length_limits': args.train_length_limits,
        'test_winds': args.test_winds,
        'num_test_winds': args.num_test_winds,
        'shuffle': True,
        'epochs': args.epochs,
        'early_stopping': args.early_stopping,
        'warm_up_epochs': args.warm_up_epochs,
        'eval_epochs': list(range(1, args.epochs+1)),
        'save_results': args.save_results,
        'device': args.device,
        'model_config': {}
    }

    results = main(config)

