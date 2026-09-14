parent_path = r'/home/student1/Desktop/Charalampos_Lamprou/VarCoNetV2_extras'
import sys
sys.path.append(parent_path)
from torch.utils.data import DataLoader
import torch
import numpy as np
import os
from utils import ABIDEDataset
from BolT.Models.BolT.model import Model
from BolT.utils import Option
from BolT.Models.BolT.hyperparams import getHyper_bolT
from model_scripts.VarCoNet import VarCoNet
from model_scripts.classifier import MLP
import pickle
from sklearn.metrics import roc_curve
import argparse


def test_varconet(encoder_model, classifier, test_loader, device):
    encoder_model.eval()
    classifier.eval()
    with torch.no_grad():        
        outputs_test= []
        for (x,y) in test_loader:
            x = x.to(device)
            outputs_test.append(encoder_model(x))
        outputs_test = torch.cat(outputs_test, dim=0).clone().detach()
        logits = classifier(outputs_test)
    z = logits[:,-1].cpu().numpy()
    return z


def test_bolt(model, test_data_loader):
    with torch.no_grad():
        zs = []
        for (x,y) in test_data_loader:
            zero_rows = torch.all(x == 0, dim=-1)
            zero_row_indices = torch.where(zero_rows)[1]
            if len(zero_row_indices) > 0:
                x = x[:,:torch.min(zero_row_indices),:]
            x = torch.permute(x,(0,2,1))
            x = (x - torch.mean(x, dim=-1, keepdim=True))/torch.std(x, dim=-1, keepdim=True)
            _, _, test_probs, _ = model.step(x, y, train=False)
            zs.append(test_probs)
        z = torch.cat(zs,dim=0)
        z = z[:,-1].cpu().numpy()
                   
    return z

def cross_entropy(p, q):
    p = np.clip(p, 1e-9, 1)
    q = np.clip(q, 1e-9, 1)
    ce = -np.sum(p * np.log(q))
    return ce


def main(config):
    path = config['path_data']
    
    with open(os.path.join(config['path_save'],'results_ABIDEI',config['atlas'],'ABIDEI_VarCoNet_results.pkl'), 'rb') as f:
        result_varconet = pickle.load(f)
    with open(os.path.join(config['path_save'],'results_ABIDEI',config['atlas'],'ABIDEI_BolT_results.pkl'), 'rb') as f:
        result_bolt = pickle.load(f)
    
    random_states = len(result_varconet['epoch_results_ext'])
    epochs_varconet = len(result_varconet['epoch_results_ext'][0])
    
    if config['atlas'] == 'AAL':
        roi_num = 166
    else:
        roi_num = 384
    test_probs_bolt = []
    test_probs_VarCoNet = []
    change_BolT = 0
    change_VarCoNet = 0
    for j in range(random_states):
        ext_test_aucs_bolt = np.array(result_bolt['ext_test_aucs'][j])
        best_epoch = np.where(ext_test_aucs_bolt == np.max(ext_test_aucs_bolt))[0][0]
        ext_test_probs_bolt = result_bolt['ext_test_probs'][j][best_epoch]
        
        ext_test_aucs_VarCoNet = []
        for i in range(epochs_varconet):
            ext_test_aucs_VarCoNet.append(np.array(result_varconet['epoch_results_ext'][j][i]['best_test_auc']))
        best_epoch = np.where(ext_test_aucs_VarCoNet == np.max(ext_test_aucs_VarCoNet))[0][0]
        ext_test_probs_VarCoNet = result_varconet['epoch_results_ext'][j][best_epoch]['test_probs']
        
        data_list = np.load(os.path.join(path,'ABIDEI_nilearn_' + config['atlas'] + '.npz'))
        Y = np.load(os.path.join(path,'ABIDEI_nilearn_classes.npy'))   
        
        for wind_size in config['test_winds']:
            step_size = (200 - wind_size) // (config['num_winds'] - 1)
            for i in range(0, step_size*config['num_winds'],step_size):
                names = []
                with open(os.path.join(path,'ABIDEI_nilearn_names.txt'), 'r') as f:
                    for line in f:
                        names.append(line.strip())
                names_unique, counts = np.unique(names, return_counts=True)
                ext_test = list(range(51456,51494))
                names_ext_test = []
                for name in ext_test:      
                    if 'sub-00'+str(name) in names_unique:
                        names_ext_test.append('sub-00'+str(name))
                test_data = []
                y_ext_test = []
                for (key,name,y) in zip(data_list,names,Y):
                    if name in names_ext_test:
                        data_temp = data_list[key]
                        max_length = data_temp.shape[0]
                        zero_rows = np.all(data_temp == 0, axis=-1)
                        zero_row_indices = np.where(zero_rows)[0]
                        if len(zero_row_indices) > 0:
                            data_temp = data_temp[:np.min(zero_row_indices),:]
                        data_temp = data_temp[i:i+wind_size,:]
                        data_temp = np.pad(data_temp, ((0, max_length-data_temp.shape[0]), (0, 0)), mode='constant', constant_values=0)
                        test_data.append(data_temp)
                        y_ext_test.append(y)
                y_ext_test = np.array(y_ext_test)
                device = torch.device(config['device']) if torch.cuda.is_available() else torch.device("cpu")
                batch_size = config['batch_size']
                model_config = config['model_config'] 
                model_config['max_length'] = max_length
                
                test_dataset = ABIDEDataset(test_data, y_ext_test)
                test_loader = DataLoader(test_dataset, batch_size=batch_size)  
                
                details = Option({
                    "device" : device,
                    "nOfTrains" : len(test_data),
                    "nOfClasses" : 2,
                    "batchSize" : batch_size,
                    "nOfEpochs" : 1
                })
                hyperParams = getHyper_bolT(rois = roi_num)
                bolt = Model(hyperParams, details)
                state_dict_bolt = torch.load(os.path.join(config['path_save'],'models_ABIDEI',config['atlas'],'BolT','min_val_loss_model_rs' + str(j) + '.pth'))
                bolt.model.load_state_dict(state_dict_bolt)
                test_prob = test_bolt(bolt, test_loader)
                test_probs_bolt.append(test_prob)
                fpr, tpr, thresholds = roc_curve(y_ext_test, test_prob)
                youden_j = tpr - fpr
                best_index = np.argmax(youden_j)
                best_threshold = thresholds[best_index]
                below = (test_prob < best_threshold) & (ext_test_probs_bolt < best_threshold)
                above = (test_prob >= best_threshold) & (ext_test_probs_bolt >= best_threshold)
                change_BolT = change_BolT + np.sum(below | above)
                
                varconet = VarCoNet(model_config, roi_num).to(device)
                classifier = MLP(int(roi_num*(roi_num-1)/2),2).to(device)
                state_dict_varconet = torch.load(os.path.join(config['path_save'],'models_ABIDEI',config['atlas'],'VarCoNet','min_val_loss_model_rs' + str(j) + '.pth'))
                state_dict_cls = torch.load(os.path.join(config['path_save'],'models_ABIDEI',config['atlas'],'VarCoNet','min_val_loss_classifier_rs' + str(j) + '.pth'))
                varconet.load_state_dict(state_dict_varconet)
                classifier.load_state_dict(state_dict_cls)
                test_prob = test_varconet(varconet, classifier, test_loader, device)
                test_probs_VarCoNet.append(test_prob)
                fpr, tpr, thresholds = roc_curve(y_ext_test, test_prob)
                youden_j = tpr - fpr
                best_index = np.argmax(youden_j)
                best_threshold = thresholds[best_index]
                below = (test_prob < best_threshold) & (ext_test_probs_VarCoNet < best_threshold)
                above = (test_prob >= best_threshold) & (ext_test_probs_VarCoNet >= best_threshold)
                change_VarCoNet = change_VarCoNet + np.sum(below | above)
    
    total = y_ext_test.shape[0]*10*config['num_winds']*len(config['test_winds'])
    change_BolT = 100*((total - change_BolT) / total)
    change_VarCoNet = 100*((total - change_VarCoNet) / total)
    
    print('--------------------Prediction Change (%)--------------------')
    print(f'BolT:     {change_BolT:.2f}')
    print(f'VarCoNet: {change_VarCoNet:.2f}')
    print('')
    
    results = {}
    results['bolt'] = {}
    results['bolt']['test_probs'] = test_probs_bolt
    results['bolt']['percent_change'] = change_BolT
    results['bolt']['base_probs'] = ext_test_probs_bolt
    results['VarCoNet'] = {}
    results['VarCoNet']['test_probs'] = test_probs_VarCoNet
    results['VarCoNet']['percent_change'] = change_VarCoNet
    results['VarCoNet']['base_probs'] = ext_test_probs_VarCoNet
    
    if not os.path.exists(os.path.join(config['path_save'],'results_ABIDEI','predictions_stability')):
        os.makedirs(os.path.join(config['path_save'],'results_ABIDEI','predictions_stability'),exist_ok=True)
    with open(os.path.join(config['path_save'],'results_ABIDEI','predictions_stability','ABIDEI_BolT_VarCoNet_' + config['atlas'] + '.pkl'), 'wb') as f:
        pickle.dump(results,f)
    return results
        
if __name__ == '__main__':   
    parser = argparse.ArgumentParser(description='Prediction stability of VarCoNet and BolT')

    parser.add_argument('--path_data', type=str,
                        help='Path to the dataset')
    parser.add_argument('--path_save', type=str,
                        help='Path to save results')
    parser.add_argument('--atlas', type=str, choices=['AICHA', 'AAL'], default='AICHA',
                        help='Atlas type to use')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use for training')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='Number of classes for the classification')
    parser.add_argument('--test_winds', type=int, nargs='+', default=[120,160],
                        help='Length of test windows')
    parser.add_argument('--num_winds', type=int, default=3,
                        help='Number of test windows')

    args = parser.parse_args()

    config = {
        'path_data': args.path_data,
        'path_save': args.path_save,
        'atlas': args.atlas,
        'batch_size': args.batch_size,
        'num_classes': args.num_classes,
        'test_winds': args.test_winds,
        'num_winds': args.num_winds,
        'device': args.device,
        'model_config': {}
    }

    with open(f'best_params_VarCoNet_{config["atlas"]}.pkl', 'rb') as f:
        best_params = pickle.load(f)

    config['model_config']['layers'] = best_params['layers']
    config['model_config']['n_heads'] = best_params['n_heads']
    config['model_config']['dim_feedforward'] = best_params['dim_feedforward']

    results = main(config)