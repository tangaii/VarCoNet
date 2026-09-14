import torch
import numpy as np
import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model_scripts.classifier import MLP
    
path_results = r''  #here, enter the path where results have been saved
atlas = 'AICHA'       #choose atlas (AICHA, AAL)
num_classes = 2
folds = 10
random_states = 10
if atlas == 'AAL':
    roi_num = 166
elif atlas == 'AICHA':
    roi_num = 384

abs_importance = []
for i in range(random_states):
    for j in range(folds):
        model = MLP(int((roi_num*(roi_num-1))/2),num_classes)
        state_dict = torch.load(os.path.join(path_results,'models_ABIDEI',atlas,'VarCoNet','min_val_loss_classifier_rs' + str(i) + '_fold' + str(j) + '.pth'))
        model.load_state_dict(state_dict)
        weights = model.fc[0].weight.detach().cpu().numpy()
        importance = weights[1] - weights[0]
        abs_importance.append(np.abs(importance))

for i in range(random_states):
    model = MLP(int((roi_num*(roi_num-1))/2),num_classes)
    state_dict = torch.load(os.path.join(path_results,'models_ABIDEI',atlas,'VarCoNet','min_val_loss_classifier_rs' + str(i) + '.pth'))
    model.load_state_dict(state_dict)
    weights = model.fc[0].weight.detach().cpu().numpy()
    importance = weights[1] - weights[0]
    abs_importance.append(np.abs(importance))

abs_importance = np.stack(abs_importance)
abs_importance = np.mean(abs_importance, axis=0)
abs_importance = (abs_importance - np.min(abs_importance))/(np.max(abs_importance) - np.min(abs_importance))
abs_importance = abs_importance.astype(np.float32)
abs_importance = pd.DataFrame(abs_importance)
if not os.path.exists(os.path.join(path_results,'results_ABIDEI','feature_importance')):
    os.makedirs(os.path.join(path_results,'results_ABIDEI','feature_importance'), exist_ok=True)
abs_importance.to_csv(os.path.join(path_results,'results_ABIDEI','feature_importance', atlas + '_feature_importance.csv'), index=False, header=False)