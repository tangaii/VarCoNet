"""
This script performs Bayesian optimization to find optimal paramaters considering 
the architecture of the transformer and the training process. In order to run, this
script recuires the path to the dowmsampled HCP data (config['path_data']). Instructions
on how to bring HCP data into the correct form are provided in the scripts:
"prepare_HCP_data.py" and "subsampling_HCP_data.py". Moreover, one can choose which
atlas to use (AAL3 or AICHA) by setting config['atlas'] = 'AICHA' or 'AAL'.
Important note: Changing the atlas should be followed by changing the 
acceptable values of n_heads when defining the search space for the optimizer.
For AICHA, possible values are 1,2,3,4,6 and for AAL3 1,2. Otherwise, the code
will raise an error. When finishes, the script saves two .pkl files, one containing
the best parameters and one contain information regarding all trials.
"""

from torch.utils.data import DataLoader
import numpy as np
import torch
from utils import InfoNCE
from tqdm import tqdm
from torch.optim import Adam
from utils import DualBranchContrast
from pl_bolts.optimizers import LinearWarmupCosineAnnealingLR
from VarCoNet import VarCoNet
import os
import optuna
import pickle
from utils import test_augment, removeDuplicates, augment_hcp


def train(x, encoder_model, contrast_model, optimizer):
    encoder_model.train()
    optimizer.zero_grad()
    z1 = encoder_model(x[0])
    z2 = encoder_model(x[1])
    loss = contrast_model(z1, z2)
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
                        corr_coeffs = torch.corrcoef(torch.cat([outputs1[:, i*num_real+j, :],outputs2[:, n*num_real+j, :]],dim=0))[0:outputs1.shape[0],outputs1.shape[0]:]#np.corrcoef(outputs1[:, i*num_real+j, :], outputs2[:, n*num_real+j, :])[0:outputs1.shape[0],outputs1.shape[0]:]
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


def build_model(hp,config,model_config,roi_num,device):
    model_config['layers'] = hp['layers']
    model_config['n_heads'] = hp['n_heads']
    model_config['dim_feedforward'] = hp['dim_feedforward']
    encoder_model = VarCoNet(model_config, roi_num).to(device)
    contrast_model = DualBranchContrast(loss=InfoNCE(tau=hp['tau']),mode='L2L').to(device)
        
    optimizer = Adam(encoder_model.parameters(), lr=hp['lr'])
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer=optimizer,
        warmup_epochs=config['warm_up_epochs'],
        warmup_start_lr = 1e-5,
        max_epochs=epochs)
    return encoder_model,contrast_model,optimizer,scheduler


def bayesianOpt(trial):
    cfg = {'lr'             : trial.suggest_float('lr', 5e-5,5e-3),
           'tau'            : trial.suggest_float('tau', 0.05,0.25),
           'layers'         : trial.suggest_categorical('layers',[1,2,3,4]),
           'n_heads'        : trial.suggest_categorical('n_heads',[1,2,3,4,6]), #AICHA:1,2,3,4,6  AAL:1,2
           'batch_size'       : trial.suggest_categorical('batch_size',[32,64,128]),
           'dim_feedforward': trial.suggest_categorical('dim_feedforward',[512,1024,2048])}
    max_accs = []
    for i in range(3):
        encoder_model,contrast_model,optimizer,scheduler = build_model(cfg,config,model_config,roi_num,device)
        train_loader = DataLoader(train_data, batch_size=cfg['batch_size'], shuffle = shuffle)        
        max_acc = 0
        count = 0
        losses = []
        with tqdm(total=epochs, desc='(T)') as pbar:
            for epoch in range(1,epochs+1):
                total_loss = 0.0
                batch_count = 0                              
                for batch_idx, sample_inds in enumerate(train_loader.batch_sampler):
                    sample_inds = removeDuplicates(names,sample_inds)
                    batch_list = [train_data[i] for i in sample_inds]
                    batch_loader = DataLoader(batch_list, batch_size=len(batch_list))
                    batch_data = next(iter(batch_loader))
                    batch_data = augment_hcp(batch_data,train_length_limits,device)
                    loss,input_dim = train(batch_data,encoder_model,contrast_model,optimizer)
                    total_loss += loss
                    batch_count += 1
                scheduler.step()
                average_loss = total_loss / batch_count if batch_count > 0 else float('nan')   
                losses.append(average_loss)
                pbar.set_postfix({'loss': average_loss})
                pbar.update()        
                
                if epoch in eval_epochs:
                    res = test(encoder_model,val_data1,val_data2,
                               len(test_winds),cfg['batch_size'],device)
                    print(np.mean(res[1]) + np.min(res[1]))
                    if np.mean(res[1]) + np.min(res[1]) > max_acc:
                        max_acc = np.mean(res[1]) + np.min(res[1])
                        count = 0
                    else:
                        if epoch > config['eval_epochs'][0]:
                            count += 1
                if count >= 8:
                    print('Early stopping')
                    break
                print('')
        if (losses[0] - losses[-1])/losses[0] < 0.25:
            max_acc = 0
        max_accs.append(max_acc)
    max_accs = np.array(max_accs)
    if (max_accs == 0).any():
        return 0
    else:
        return np.mean(max_accs)


config = {}
config['path_data'] = r'/home/student1/Desktop/Charalampos_Lamprou/SSL_FC_matrix_GNN_data/HCP'
config['atlas'] = 'AICHA' #AICHA, AAL
config['train_length_limits'] = [30,320]
config['test_lengths'] = [30,175,320]
config['test_num_winds'] = 10
config['shuffle'] = True
config['epochs'] = 200
config['warm_up_epochs'] = 10
config['eval_epochs'] = list(range(10,config['epochs']+1))
config['device'] = "cuda:0"

path = config['path_data']

names = []
with open(os.path.join(path,'names_train.txt'), 'r') as f:
    for line in f:
        names.append(line.strip())
        
data = np.load(os.path.join(path,'train_data_HCP_' + config['atlas'] + '_resampled.npz'))
train_data = []
for key in data:
    train_data.append(data[key])

data = np.load(os.path.join(path,'val_data_HCP_' + config['atlas'] + '_1_resampled.npz'))
val_data1 = []
for key in data:
    val_data1.append(data[key])

data = np.load(os.path.join(path,'val_data_HCP_' + config['atlas'] + '_2_resampled.npz'))
val_data2 = []
for key in data:
    val_data2.append(data[key])

roi_num = val_data1[0].shape[1]


device = torch.device(config['device']) if torch.cuda.is_available() else torch.device("cpu")
shuffle = config['shuffle']
epochs = config['epochs']
train_length_limits = config['train_length_limits']
max_length = train_length_limits[-1]
test_winds = config['test_lengths']
eval_epochs = config['eval_epochs']

for i,data in enumerate(val_data1):
    data = torch.from_numpy(data.astype(np.float32))
    data = test_augment(data, config['test_lengths'], config['test_num_winds'], max_length)
    val_data1[i] = data
    
for i,data in enumerate(val_data2):
    data = torch.from_numpy(data.astype(np.float32))
    data = test_augment(data, config['test_lengths'], config['test_num_winds'], max_length)
    val_data2[i] = data

model_config = {}
model_config['max_length'] = max_length

sampler = optuna.samplers.TPESampler()
study = optuna.create_study(sampler=sampler, direction="maximize", pruner=optuna.pruners.SuccessiveHalvingPruner())
study.enqueue_trial({"lr":5e-5, "tau": 0.05, "layers": 1, "n_heads": 2, "dim_feedforward": 2048, "batch_size": 128})
study.optimize(func=bayesianOpt, n_trials=125)
best_params = study.best_params
trials = study.trials

with open('best_params_VarCoNet_' + config['atlas'] + '.pkl','wb') as f:
    pickle.dump(best_params,f)

with open('trials_VarCoNet_' + config['atlas'] + '.pkl','wb') as f:
    pickle.dump(trials,f)


