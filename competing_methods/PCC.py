import numpy as np
import os
from utils import test_augment_PCC
import pickle
import argparse


def main(config):
    path = config['path_data']
    
    data = np.load(os.path.join(path, 'val_data_HCP_' + config['atlas'] + '_1_resampled.npz'))
    val_data1 = []
    for key in data:
        val_data1.append(data[key])
    
    data = np.load(os.path.join(path, 'val_data_HCP_' + config['atlas'] + '_2_resampled.npz'))
    val_data2 = []
    for key in data:
        val_data2.append(data[key])
        
    data = np.load(os.path.join(path, 'test_data_HCP_' + config['atlas'] + '_1_resampled.npz'))
    test_data1 = []
    for key in data:
        test_data1.append(data[key])
    
    data = np.load(os.path.join(path, 'test_data_HCP_' + config['atlas'] + '_2_resampled.npz'))
    test_data2 = []
    for key in data:
        test_data2.append(data[key])
     
    
    for i,data in enumerate(test_data1):
        data = test_augment_PCC(data, config['test_winds'], config['num_test_winds'])
        test_data1[i] = data
        
    for i,data in enumerate(test_data2):
        data = test_augment_PCC(data, config['test_winds'], config['num_test_winds'])
        test_data2[i] = data
        
    for i,data in enumerate(val_data1):
        data = test_augment_PCC(data, config['test_winds'], config['num_test_winds'])
        val_data1[i] = data
        
    for i,data in enumerate(val_data2):
        data = test_augment_PCC(data, config['test_winds'], config['num_test_winds'])
        val_data2[i] = data
    
    val_data1 = np.stack(val_data1)
    val_data2 = np.stack(val_data2)
    test_data1 = np.stack(test_data1)
    test_data2 = np.stack(test_data2)
    
    accuracies_val = []
    mean_accs_val = []
    std_accs_val = []
    corr_coeffs_val = []
    num_real = int(val_data1.shape[1] / len(config['test_winds']))
    print('PCC Validation')
    for i in range (len(config['test_winds'])):
        for n in range(len(config['test_winds'])):
            if n >= i:
                accuracies = []
                for j in range(num_real):
                    corr_coeffs = np.corrcoef(val_data1[:, i*num_real+j, :], val_data2[:, n*num_real+j, :])[0:val_data1.shape[0],val_data1.shape[0]:]
                    corr_coeffs_val.append(corr_coeffs)
                    lower_indices = np.tril_indices(corr_coeffs.shape[0], k=-1)
                    upper_indices = np.triu_indices(corr_coeffs.shape[0], k=1)
                    corr_coeffs1 = corr_coeffs.copy()
                    corr_coeffs2 = corr_coeffs.copy()
                    corr_coeffs1[lower_indices] = -2
                    corr_coeffs2[upper_indices] = -2
                    counter1 = 0
                    counter2 = 0
                    for j in range(corr_coeffs1.shape[0]):
                        if np.argmax(corr_coeffs1[j, :]) == j:
                            counter1 += 1
                    for j in range(corr_coeffs2.shape[1]):
                        if np.argmax(corr_coeffs2[:, j]) == j:
                            counter2 += 1
        
                    total_samples = val_data1.shape[0] + val_data2.shape[0]
                    accuracies.append((counter1 + counter2) / total_samples)
                mean_acc = np.mean(accuracies)
                std_acc = np.std(accuracies)
                print(f'meanAcc{i,n}: {mean_acc:.2f}, stdAcc{i,n}: {std_acc:.3f}')
                mean_accs_val.append(mean_acc)
                std_accs_val.append(std_acc)
                accuracies_val.append(accuracies)
    print('')
    
    
    
    accuracies_test = []
    mean_accs_test = []
    std_accs_test = []
    corr_coeffs_test = []
    num_real = int(test_data1.shape[1] / len(config['test_winds']))
    print('PCC Test')
    for i in range (len(config['test_winds'])):
        for n in range(len(config['test_winds'])):
            if n >= i:
                accuracies = []
                for j in range(num_real):
                    corr_coeffs = np.corrcoef(test_data1[:, i*num_real+j, :], test_data2[:, n*num_real+j, :])[0:test_data1.shape[0],test_data1.shape[0]:]
                    corr_coeffs_test.append(corr_coeffs)
                    lower_indices = np.tril_indices(corr_coeffs.shape[0], k=-1)
                    upper_indices = np.triu_indices(corr_coeffs.shape[0], k=1)
                    corr_coeffs1 = corr_coeffs.copy()
                    corr_coeffs2 = corr_coeffs.copy()
                    corr_coeffs1[lower_indices] = -2
                    corr_coeffs2[upper_indices] = -2
                    counter1 = 0
                    counter2 = 0
                    for j in range(corr_coeffs1.shape[0]):
                        if np.argmax(corr_coeffs1[j, :]) == j:
                            counter1 += 1
                    for j in range(corr_coeffs2.shape[1]):
                        if np.argmax(corr_coeffs2[:, j]) == j:
                            counter2 += 1
        
                    total_samples = test_data1.shape[0] + test_data2.shape[0]
                    accuracies.append((counter1 + counter2) / total_samples)
                mean_acc = np.mean(accuracies)
                std_acc = np.std(accuracies)
                print(f'meanAcc{i,n}: {mean_acc:.2f}, stdAcc{i,n}: {std_acc:.3f}')
                mean_accs_test.append(mean_acc)
                std_accs_test.append(std_acc)
                accuracies_test.append(accuracies)
    
    results = {}
    results['val_result'] = [accuracies_val, mean_accs_val, std_accs_val, corr_coeffs_val]
    results['test_result'] = [accuracies_test, mean_accs_test, std_accs_test, corr_coeffs_test]
    if config['save_results']:
        if not os.path.exists(config['path_save'],'results_HCP'):
            os.mkdir(config['path_save'],'results_HCP')   
        with open(os.path.join(config['path_save'],'results_HCP',config['atlas'],'HCP_PCC_results.pkl'), 'wb') as f:
            pickle.dump(results,f)

if __name__ == '__main__':   
    parser = argparse.ArgumentParser(description='Run PCC on HCP for subject fingerprinting')

    parser.add_argument('--path_data', type=str,
                        help='Path to the dataset')
    parser.add_argument('--path_save', type=str,
                        help='Path to save results')
    parser.add_argument('--atlas', type=str, choices=['AICHA', 'AAL'], default='AICHA',
                        help='Atlas type to use')
    parser.add_argument('--test_winds', type=int, nargs='+', default=[80,200,320],
                        help='Lengths used for testing the model')
    parser.add_argument('--num_test_winds', type=int, default=10,
                        help='Number of tests per length to calculate confidence interval')
    parser.add_argument('--save_results', action='store_true',
                        help='Flag to save results')

    args = parser.parse_args()

    config = {
        'path_data': args.path_data,
        'path_save': args.path_save,
        'atlas': args.atlas,
        'test_winds': args.test_winds,
        'num_test_winds': args.num_test_winds,
        'save_results': args.save_results,
    }

    results = main(config)