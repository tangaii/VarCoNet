parent_path = r'/home/student1/Desktop/Charalampos_Lamprou/VarCoNetV2_extras/ksvd-sparse-dictionary'
import sys
sys.path.append(parent_path)
from utils import PCC
import numpy as np
import torch
from model_scripts.VarCoNet import VarCoNet
from model_scripts.competing_models import Model_AE, Model_VAE
import os
import pickle
import pandas as pd
from sparseRep import random_sensor, sense, reconstruct
import argparse    
    
    
def test_varconet(encoder_model,test_data1,test_data2,device):
    encoder_model.eval()
    with torch.no_grad():
        outputs = []
        for data in test_data1:
            data = torch.from_numpy(data).to(device)
            outputs.append(encoder_model(data.unsqueeze(0).float()))
        outputs1 = torch.stack(outputs).cpu().numpy()
        
        outputs = []
        for data in test_data2:
            data = torch.from_numpy(data).to(device)
            outputs.append(encoder_model(data.unsqueeze(0).float()))
        outputs2 = torch.stack(outputs).cpu().numpy()
        
    return outputs1, outputs2
    

def test_vae(model,test_data1,test_data2,D,device):
    model.eval()
    with torch.no_grad():
        outputs = []
        for data in test_data1:
            data = data.to(device)
            outputs.append(model(data.unsqueeze(0).float(),False))
        outputs1 = torch.stack(outputs)
        
        outputs = []
        for data in test_data2:
            data = data.to(device)
            outputs.append(model(data.unsqueeze(0).float(),False))
        outputs2 = torch.stack(outputs)
        
    test_data1 = torch.stack(test_data1)
    test_data2 = torch.stack(test_data2)
    
    outputs1 = outputs1.squeeze()
    outputs2 = outputs2.squeeze()
    outputs1 = outputs1 - test_data1.to(device)
    outputs2 = outputs2 - test_data2.to(device)
    N,K = outputs1.shape
    
    outputs = torch.cat((outputs1, outputs2), dim=0)
    sensor = random_sensor((K, int(0.4*K)), device=device)
    representation = sense(outputs, sensor, device)
    r = reconstruct(representation.T, sensor, D, 25, device).T
    outputs = outputs.cpu().numpy() - r.cpu().numpy()
    del test_data1, test_data2, outputs1, outputs2, sensor, representation, r
    torch.cuda.empty_cache()
    outputs1 = outputs[:int(outputs.shape[0]/2)]
    outputs2 = outputs[int(outputs.shape[0]/2):]
    return outputs1, outputs2


def test_ae(model,test_data1,test_data2,D,device):
    model.eval()
    with torch.no_grad():
        outputs = []
        for data in test_data1:
            data = torch.from_numpy(data).to(device)
            outputs.append(model(data.unsqueeze(0).float()))
        outputs1 = torch.stack(outputs)
        
        outputs = []
        for data in test_data2:
            data = torch.from_numpy(data).to(device)
            outputs.append(model(data.unsqueeze(0).float()))
        outputs2 = torch.stack(outputs)
    
    outputs1 = outputs1.squeeze()
    outputs2 = outputs2.squeeze()
    outputs1 = outputs1 - torch.from_numpy(np.stack(test_data1)[:,:200,:]).to(device)
    outputs2 = outputs2 - torch.from_numpy(np.stack(test_data2)[:,:200,:]).to(device)
    outputs = torch.cat((outputs1, outputs2), dim=0)
    corrs = []
    for output in outputs:
        corr = PCC(output.T)
        triu_indices = torch.triu_indices(corr.shape[0], corr.shape[0], offset=1)
        corrs.append(corr[triu_indices[0], triu_indices[1]])
    corrs = torch.stack(corrs)
    N,K = corrs.shape
    sensor = random_sensor((K, int(0.4*K)), device=device)
    representation = sense(corrs.float(), sensor, device)
    r = reconstruct(representation.T, sensor, D, 25, device).T
    outputs = corrs.cpu().numpy() - r.cpu().numpy()
    del test_data1, test_data2, outputs1, outputs2, sensor, representation, r
    torch.cuda.empty_cache()
    outputs1 = outputs[:int(outputs.shape[0]/2)]
    outputs2 = outputs[int(outputs.shape[0]/2):]
    return outputs1, outputs2

def main(config):
    path = config['path_data']
    
    model_config = config['model_config']
    model_config['max_length'] = config['max_length']
    device = torch.device(config['device']) if torch.cuda.is_available() else torch.device("cpu")
       
    
    data = np.load(os.path.join(path,'test_data_HCP_' + config['atlas'] + '_1_resampled.npz'))
    test_data1 = []
    for key in data:
        test_data1.append(data[key][:config['length'],:])
    
    data = np.load(os.path.join(path,'test_data_HCP_' + config['atlas'] + '_2_resampled.npz'))
    test_data2 = []
    for key in data:
        test_data2.append(data[key][:config['length'],:])
            
    test_data1_varconet = test_data1
    test_data2_varconet = test_data2  
    
    test_data1_ae = test_data1
    test_data2_ae = test_data2  
    
    test_data1_vae = []
    test_data2_vae = []
    for i,data in enumerate(test_data1):
        data = torch.from_numpy(data.astype(np.float32))
        corr = PCC(data.T)
        triu_indices = torch.triu_indices(corr.shape[0], corr.shape[0], offset=1)
        test_data1_vae.append(corr[triu_indices[0], triu_indices[1]])
        
    for i,data in enumerate(test_data2):
        data = torch.from_numpy(data.astype(np.float32))
        corr = PCC(data.T)
        triu_indices = torch.triu_indices(corr.shape[0], corr.shape[0], offset=1)
        test_data2_vae.append(corr[triu_indices[0], triu_indices[1]])
       
    roi_num = test_data1[0].shape[1]
    
    
    
    '''------------------------------------------VarCoNet---------------------------------------'''  
    varconet = VarCoNet(model_config, roi_num).to(device)
    state_dict_varconet = torch.load(os.path.join(config['path_save'],'models_HCP',config['atlas'],'VarCoNet','max_fingerprinting_acc_model.pth'))
    varconet.load_state_dict(state_dict_varconet)
    out1_varconet, out2_varconet = test_varconet(varconet,test_data1_varconet,test_data2_varconet,device)
    out1_varconet = np.squeeze(out1_varconet)
    out2_varconet = np.squeeze(out2_varconet)
    final_varconet = np.zeros((2*out1_varconet.shape[0], out1_varconet.shape[1]))
    subIDs = []
    visits = []
    count = 0
    for i in range(0,out1_varconet.shape[0]):
        final_varconet[count] = out1_varconet[i,:]
        count += 1
        final_varconet[count] = out2_varconet[i,:]
        count += 1
        if i < 10:
            subIDs.append('sub00' + str(i))
            subIDs.append('sub00' + str(i))
        elif i >= 10 and i < 100:
            subIDs.append('sub0' + str(i))
            subIDs.append('sub0' + str(i))
        else:
            subIDs.append('sub' + str(i))
            subIDs.append('sub' + str(i))
        visits.append('time1')
        visits.append('time2')
    
    columns = []
    for i in range(out1_varconet.shape[1]):
        columns.append('ROI.' + str(i))
        
    subIDs = np.expand_dims(np.array(subIDs), axis=1)
    visits = np.expand_dims(np.array(visits), axis=1)
    subIDs = pd.DataFrame(subIDs, columns=['subID'])
    visits = pd.DataFrame(subIDs, columns=['visit'])
    final_varconet = pd.DataFrame(final_varconet, columns=columns)
    final_varconet = pd.concat((subIDs,visits,final_varconet), axis=1)
    if not os.path.exists(os.path.join(config['path_save'],'results_HCP','ReX_files')):
        os.makedirs(os.path.join(config['path_save'],'results_HCP','ReX_files'), exist_ok=True)
    final_varconet.to_csv(os.path.join(config['path_save'],'results_HCP','ReX_files','rex_' + config['atlas'] + '_VarCoNet_' + str(config['length']) + '_samples.csv'))
    
    
    '''------------------------------------------VAE-K-SVD---------------------------------------'''  
    vae = Model_VAE(roi_num).to(device)
    state_dict_vae = torch.load(os.path.join(config['path_save'],'models_HCP',config['atlas'],'VAE_KSVD','min_loss_model.pth'))
    D = torch.load(os.path.join(config['path_save'],'models_HCP',config['atlas'],'VAE_KSVD','dictionary.pt'))
    vae.load_state_dict(state_dict_vae)
    out1_vae, out2_vae = test_vae(vae,test_data1_vae,test_data2_vae,D,device)
    out1_vae = np.squeeze(out1_vae)
    out2_vae = np.squeeze(out2_vae)
    final_vae = np.zeros((2*out1_vae.shape[0], out1_vae.shape[1]))
    count = 0
    for i in range(0,out1_vae.shape[0]):
        final_vae[count] = out1_vae[i,:]
        count += 1
        final_vae[count] = out2_vae[i,:]
        count += 1
        
    final_vae = pd.DataFrame(final_vae, columns=columns)
    final_vae = pd.concat((subIDs,visits,final_vae), axis=1)
    final_vae.to_csv(os.path.join(config['path_save'],'results_HCP','ReX_files','rex_' + config['atlas'] + '_VAE_KSVD_' + str(config['length']) + '_samples.csv'))    
    
    
    '''------------------------------------------AE-K-SVD---------------------------------------'''    
    ae = Model_AE(config['length']).to(device)
    state_dict_ae = torch.load(os.path.join(config['path_save'],'models_HCP',config['atlas'],'AE_KSVD','min_loss_model_length' + str(config['length']) + '.pth'))
    D = torch.load(os.path.join(config['path_save'],'models_HCP',config['atlas'],'AE_KSVD','dictionary.pt'))
    ae.load_state_dict(state_dict_ae)
    out1_ae, out2_ae = test_ae(ae,test_data1_ae,test_data2_ae,D,device)
    out1_ae = np.squeeze(out1_ae)
    out2_ae = np.squeeze(out2_ae)
    final_ae = np.zeros((2*out1_ae.shape[0], out1_ae.shape[1]))
    count = 0
    for i in range(0,out1_ae.shape[0]):
        final_ae[count] = out1_ae[i,:]
        count += 1
        final_ae[count] = out2_ae[i,:]
        count += 1
        
    final_ae = pd.DataFrame(final_ae, columns=columns)
    final_ae = pd.concat((subIDs,visits,final_ae), axis=1)
    final_ae.to_csv(os.path.join(config['path_save'],'results_HCP','ReX_files','rex_' + config['atlas'] + '_AE_KSVD_' + str(config['length']) + '_samples.csv'))  
    
    
    '''------------------------------------------PCC---------------------------------------''' 
    out1_pcc = []
    out2_pcc = []
    for data in test_data1:
        corr = np.corrcoef(data[:config['length'],:].T)
        triu_indices = np.triu_indices(corr.shape[0], k=1)
        upper_triangular_values = corr[triu_indices[0], triu_indices[1]]
        out1_pcc.append(upper_triangular_values)
    for data in test_data2:
        corr = np.corrcoef(data[:config['length'],:].T)
        triu_indices = np.triu_indices(corr.shape[0], k=1)
        upper_triangular_values = corr[triu_indices[0], triu_indices[1]]
        out2_pcc.append(upper_triangular_values)
        
    out1_pcc = np.stack(out1_pcc)
    out2_pcc = np.stack(out2_pcc)
    final_pcc = np.zeros((2*out1_pcc.shape[0], out1_pcc.shape[1]))
    count = 0
    for i in range(0,out1_pcc.shape[0]):
        final_pcc[count] = out1_pcc[i,:]
        count += 1
        final_pcc[count] = out2_pcc[i,:]
        count += 1
        
    final_pcc = pd.DataFrame(final_pcc, columns=columns)
    final_pcc = pd.concat((subIDs,visits,final_pcc), axis=1)
    final_pcc.to_csv(os.path.join(config['path_save'],'results_HCP','ReX_files','rex_' + config['atlas'] + '_PCC_' + str(config['length']) + '_samples.csv'))  



if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description='Run this script to prepare data for ReX toolbox')

    parser.add_argument('--path_data', type=str,
                        help='Path to the dataset')
    parser.add_argument('--path_save', type=str,
                        help='Path to save results')
    parser.add_argument('--atlas', type=str, choices=['AICHA', 'AAL'], default='AICHA',
                        help='Atlas type to use')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use for training')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--length', type=int, default=80,
                        help='Length of input signals')
    parser.add_argument('--max_length', type=int, default=320,
                        help='Maximum length of signals used to train the original models')


    args = parser.parse_args()

    config = {
        'path_data': args.path_data,
        'path_save': args.path_save,
        'device': args.device,
        'atlas': args.atlas,
        'length': args.length,
        'batch_size': args.batch_size,
        'max_length': args.max_length,
        'model_config': {}
    }

    with open(f'best_params_VarCoNet_{config["atlas"]}.pkl', 'rb') as f:
        best_params = pickle.load(f)

    config['model_config']['layers'] = best_params['layers']
    config['model_config']['n_heads'] = best_params['n_heads']
    config['model_config']['dim_feedforward'] = best_params['dim_feedforward']

    main(config)

