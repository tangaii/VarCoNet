parent_path = r'/home/student1/Desktop/Charalampos_Lamprou/VarCoNetV2_extras'
import sys
sys.path.append(parent_path)
from torch.utils.data import DataLoader
import torch
import numpy as np
import os
import torch.nn.functional as F
from torch import nn
from utils import ABIDEDataset
from model_scripts.VarCoNet import VarCoNet
from model_scripts.classifier import MLP
from BolT.Models.BolT.model import Model
from BolT.utils import Option
from BolT.Models.BolT.hyperparams import getHyper_bolT
from sklearn.metrics import roc_auc_score
import pickle
import argparse


def test_varconet(encoder_model, classifier, test_loader, device):
    encoder_model.eval()
    classifier.eval()
    with torch.no_grad():        
        outputs_test= []
        y_test = []
        for (x,y) in test_loader:
            x = x.to(device)
            y_test.append(y)
            outputs_test.append(encoder_model(x))
        outputs_test = torch.cat(outputs_test, dim=0).clone().detach()
        y_test = torch.cat(y_test,dim=0).to(device)
        logits = classifier(outputs_test)
    loss_func = nn.BCELoss()
    loss = loss_func(logits, F.one_hot(y_test, num_classes=2).float())
    z = logits[:,-1].cpu().numpy()
    y = y_test.cpu().numpy()
    auc_score = roc_auc_score(y, z)
    return loss.item(), auc_score, z, y


def test_bolt(model, test_data_loader):
    with torch.no_grad():
        zs = []
        ys = []
        for (x,y) in test_data_loader:
            zero_rows = torch.all(x == 0, dim=-1)
            zero_row_indices = torch.where(zero_rows)[1]
            if len(zero_row_indices) > 0:
                x = x[:,:torch.min(zero_row_indices),:]
            x = torch.permute(x,(0,2,1))
            x = (x - torch.mean(x, dim=-1, keepdim=True))/torch.std(x, dim=-1, keepdim=True)
            _, _, test_probs, _ = model.step(x, y, train=False)
            zs.append(test_probs)
            ys.append(y)
        z = torch.cat(zs,dim=0)
        y = torch.cat(ys,dim=0)
        loss_func = nn.BCELoss()
        loss = loss_func(z, F.one_hot(y, num_classes=2).float())
        z = z[:,-1].cpu().numpy()
        y = y.numpy()
        auc_score = roc_auc_score(y, z)
                   
    return loss.item(), auc_score, z, y


def main(config):
    path = config['path_data']
    
    data_list = np.load(os.path.join(path,'ABIDEII_nilearn_' + config['atlas'] + '.npz'))
    test_data = []
    for key in data_list:
        test_data.append(data_list[key])
    y = np.load(os.path.join(path,'ABIDEII_nilearn_classes.npy'))  
    
    roi_num = test_data[0].shape[1]
    
    device = torch.device(config['device']) if torch.cuda.is_available() else torch.device("cpu")
    max_length = test_data[0].shape[0]
    batch_size = config['batch_size']
    model_config = config['model_config'] 
    model_config['max_length'] = max_length
    
    test_losses_bolt = []
    test_aucs_bolt = []
    test_probs_bolt = []
    test_losses_varconet = []
    test_aucs_varconet = []
    test_probs_varconet = []
    for i in range(10):
        test_dataset = ABIDEDataset(test_data, y)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)   
        
        details = Option({
            "device" : device,
            "nOfTrains" : len(test_data),
            "nOfClasses" : config['num_classes'],
            "batchSize" : batch_size,
            "nOfEpochs" : 1
        })
        hyperParams = getHyper_bolT(rois = roi_num)
        bolt = Model(hyperParams, details)
        state_dict_bolt = torch.load(os.path.join(config['path_save'],'models_ABIDEI',config['atlas'],'BolT','min_val_loss_model_rs' + str(i) + '.pth'))
        bolt.model.load_state_dict(state_dict_bolt)
        test_loss,test_auc,test_prob,y_ext_test = test_bolt(bolt, test_loader)
        test_losses_bolt.append(test_loss)
        test_aucs_bolt.append(test_auc)
        test_probs_bolt.append(test_prob)
        
        varconet = VarCoNet(model_config, roi_num).to(device)
        classifier = MLP(int(roi_num*(roi_num-1)/2),config['num_classes']).to(device)
        state_dict_varconet = torch.load(os.path.join(config['path_save'],'models_ABIDEI',config['atlas'],'VarCoNet','min_val_loss_model_rs' + str(i) + '.pth'))
        state_dict_cls = torch.load(os.path.join(config['path_save'],'models_ABIDEI',config['atlas'],'VarCoNet','min_val_loss_classifier_rs' + str(i) + '.pth'))
        varconet.load_state_dict(state_dict_varconet)
        classifier.load_state_dict(state_dict_cls)
        test_loss,test_auc,test_prob,y_ext_test = test_varconet(varconet, classifier, test_loader, device)
        test_losses_varconet.append(test_loss)
        test_aucs_varconet.append(test_auc)
        test_probs_varconet.append(test_prob)
        
    results = {}
    results['BolT'] = {}
    results['BolT']['test_losses'] = test_losses_bolt
    results['BolT']['test_aucs'] = test_aucs_bolt
    results['BolT']['test_probs'] = test_probs_bolt
    results['BolT']['y_test'] = y
    results['VarCoNet'] = {}
    results['VarCoNet']['test_losses'] = test_losses_varconet
    results['VarCoNet']['test_aucs'] = test_aucs_varconet
    results['VarCoNet']['test_probs'] = test_probs_varconet
    results['VarCoNet']['y_test'] = y
    
    if not os.path.exists(os.path.join(config['path_save'],'results_ABIDEII',config['atlas'])):
        os.makedirs(os.path.join(config['path_save'],'results_ABIDEII',config['atlas']),exist_ok=True)
    with open(os.path.join(config['path_save'],'results_ABIDEII',config['atlas'],'ABIDEII_results.pkl'), 'wb') as f:
        pickle.dump(results,f)
    return results
        
if __name__ == '__main__':   
    parser = argparse.ArgumentParser(description='Run VarCoNet and BolT on ABIDE II for ASD classification')

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

    args = parser.parse_args()

    config = {
        'path_data': args.path_data,
        'path_save': args.path_save,
        'atlas': args.atlas,
        'batch_size': args.batch_size,
        'num_classes': args.num_classes,
        'device': args.device,
        'model_config': {}
    }

    with open(f'best_params_VarCoNet_{config["atlas"]}.pkl', 'rb') as f:
        best_params = pickle.load(f)

    config['model_config']['layers'] = best_params['layers']
    config['model_config']['n_heads'] = best_params['n_heads']
    config['model_config']['dim_feedforward'] = best_params['dim_feedforward']

    results = main(config)