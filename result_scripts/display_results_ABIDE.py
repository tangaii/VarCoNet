import pickle
import numpy as np
import os  
from scipy.stats import ttest_ind
from sklearn.metrics import f1_score,roc_curve

atlas = 'AAL'    #choose atlas (AICHA, AAL)
save_path = r'' #here, enter the path where the results are saved
    
with open(os.path.join(save_path,'results_ABIDEI',atlas,'ABIDEI_VarCoNet_results.pkl'), 'rb') as f:
    test_result_VarCoNet = pickle.load(f)
with open(os.path.join(save_path,'results_ABIDEI',atlas,'ABIDEI_BolT_results.pkl'), 'rb') as f:
    test_result_BolT = pickle.load(f)
with open(os.path.join(save_path,'results_ABIDEI',atlas,'ABIDEI_CVFormer_results.pkl'), 'rb') as f:
    test_result_CVFormer = pickle.load(f)
with open(os.path.join(save_path,'results_ABIDEI',atlas,'ABIDEI_FBNETGEN_results.pkl'), 'rb') as f:
    test_result_FBNETGEN = pickle.load(f)
with open(os.path.join(save_path,'results_ABIDEI',atlas,'ABIDEI_DeepFMRI_results.pkl'), 'rb') as f:
    test_result_DeepFMRI = pickle.load(f)
with open(os.path.join(save_path,'results_ABIDEI','ABIDEI_LFB_results.pkl'), 'rb') as f:
    test_result_LFB = pickle.load(f)
with open(os.path.join(save_path,'results_ABIDEI','ABIDEI_BAnD_results.pkl'), 'rb') as f:
    test_result_BAnD = pickle.load(f)
with open(os.path.join(save_path,'results_ABIDEI', atlas,'ABIDEI_BNT_results.pkl'), 'rb') as f:
    test_result_BNT = pickle.load(f)
with open(os.path.join(save_path,'results_ABIDEI', atlas,'ABIDEI_UCGL_results.pkl'), 'rb') as f:
    test_result_UCGL = pickle.load(f)
with open(os.path.join(save_path,'results_ABIDEI', atlas,'ABIDEI_GCDA_results.pkl'), 'rb') as f:
    test_result_GCDA = pickle.load(f)
with open(os.path.join(save_path,'results_ABIDEI', atlas,'ABIDEI_BrainIB_results.pkl'), 'rb') as f:
    test_result_BrainIB = pickle.load(f)
with open(os.path.join(save_path,'results_ABIDEI', atlas,'ABIDEI_A_GCL_results.pkl'), 'rb') as f:
    test_result_A_GCL = pickle.load(f)
    
    
'''-----------------------------------------BolT--------------------------------------------------'''
val_losses = np.array(test_result_BolT['val_losses'])  
val_aucs = np.array(test_result_BolT['val_aucs']) 
val_probs = test_result_BolT['val_probs']
y_vals = test_result_BolT['y_val']
test_losses = np.array(test_result_BolT['test_losses'])    
test_aucs = np.array(test_result_BolT['test_aucs'])
test_probs = test_result_BolT['test_probs']
y_tests = test_result_BolT['y_test']
val_losses_ext = np.array(test_result_BolT['val_losses_ext'])
val_aucs_ext = np.array(test_result_BolT['val_aucs_ext'])
val_probs_ext = test_result_BolT['val_probs_ext'] 
y_vals_ext = test_result_BolT['y_val_ext']
ext_test_losses = np.array(test_result_BolT['ext_test_losses'])
ext_test_aucs = np.array(test_result_BolT['ext_test_aucs'])
ext_test_probs = test_result_BolT['ext_test_probs']
y_ext_test = test_result_BolT['y_ext_test']
min_indices = np.argmin(val_losses, axis=1)
min_indices_ext = np.argmin(val_losses_ext, axis=1)
BolT_val_f1, BolT_test_f1, BolT_val_f1_ext, BolT_ext_test_f1 = np.zeros((len(val_probs),)), np.zeros((len(val_probs),)), np.zeros((len(val_probs_ext),)), np.zeros((len(val_probs_ext),))
for i in range(len(val_probs)):
    val_prob = val_probs[i][min_indices[i]]
    y_val = y_vals[i][min_indices[i]]
    fpr, tpr, thresholds = roc_curve(y_val, val_prob)
    youden_j = tpr - fpr
    best_index = np.argmax(youden_j)
    best_threshold = thresholds[best_index]
    val_prob[val_prob<best_threshold] = 0
    val_prob[val_prob>=best_threshold] = 1
    BolT_val_f1[i] = f1_score(y_val, val_prob)
    test_prob = test_probs[i][min_indices[i]]
    y_test = y_tests[i][min_indices[i]]
    fpr, tpr, thresholds = roc_curve(y_test, test_prob)
    youden_j = tpr - fpr
    best_index = np.argmax(youden_j)
    best_threshold = thresholds[best_index]
    test_prob[test_prob<best_threshold] = 0
    test_prob[test_prob>=best_threshold] = 1
    BolT_test_f1[i] = f1_score(y_test, test_prob)
for i in range(len(val_probs_ext)):
    val_prob_ext = val_probs_ext[i][min_indices_ext[i]]
    y_val_ext = y_vals_ext[i][min_indices_ext[i]]
    fpr, tpr, thresholds = roc_curve(y_val_ext, val_prob_ext)
    youden_j = tpr - fpr
    best_index = np.argmax(youden_j)
    best_threshold = thresholds[best_index]
    val_prob_ext[val_prob_ext<best_threshold] = 0
    val_prob_ext[val_prob_ext>=best_threshold] = 1
    BolT_val_f1_ext[i] = f1_score(y_val_ext, val_prob_ext)
    ext_test_prob = ext_test_probs[i][min_indices_ext[i]]
    fpr, tpr, thresholds = roc_curve(y_ext_test, ext_test_prob)
    youden_j = tpr - fpr
    best_index = np.argmax(youden_j)
    best_threshold = thresholds[best_index]
    ext_test_prob[ext_test_prob<best_threshold] = 0
    ext_test_prob[ext_test_prob>=best_threshold] = 1
    BolT_ext_test_f1[i] = f1_score(y_ext_test, ext_test_prob)

BolT_val_losses = np.take_along_axis(val_losses, min_indices[:, None], axis=1).squeeze()
BolT_val_aucs = np.take_along_axis(val_aucs, min_indices[:, None], axis=1).squeeze()  
BolT_test_losses = np.take_along_axis(test_losses, min_indices[:, None], axis=1).squeeze()
BolT_test_aucs = np.take_along_axis(test_aucs, min_indices[:, None], axis=1).squeeze()
BolT_val_losses_ext = np.take_along_axis(val_losses_ext, min_indices_ext[:, None], axis=1).squeeze()
BolT_val_aucs_ext = np.take_along_axis(val_aucs_ext, min_indices_ext[:, None], axis=1).squeeze()  
BolT_ext_test_losses = np.take_along_axis(ext_test_losses, min_indices_ext[:, None], axis=1).squeeze()
BolT_ext_test_aucs = np.take_along_axis(ext_test_aucs, min_indices_ext[:, None], axis=1).squeeze()


'''-----------------------------------------BNT--------------------------------------------------'''
val_losses = np.array(test_result_BNT['val_losses'])  
val_aucs = np.array(test_result_BNT['val_aucs']) 
val_probs = test_result_BNT['val_probs']
y_vals = test_result_BNT['y_val']
test_losses = np.array(test_result_BNT['test_losses'])    
test_aucs = np.array(test_result_BNT['test_aucs'])
test_probs = test_result_BNT['test_probs']
y_tests = test_result_BNT['y_test']
val_losses_ext = np.array(test_result_BNT['val_losses_ext'])
val_aucs_ext = np.array(test_result_BNT['val_aucs_ext'])
val_probs_ext = test_result_BNT['val_probs_ext'] 
y_vals_ext = test_result_BNT['y_val_ext']
ext_test_losses = np.array(test_result_BNT['ext_test_losses'])
ext_test_aucs = np.array(test_result_BNT['ext_test_aucs'])
ext_test_probs = test_result_BNT['ext_test_probs']
y_ext_tests = test_result_BNT['y_ext_test']
min_indices = np.argmin(val_losses, axis=1)
min_indices_ext = np.argmin(val_losses_ext, axis=1)
BNT_val_f1, BNT_test_f1, BNT_val_f1_ext, BNT_ext_test_f1 = np.zeros((len(val_probs),)), np.zeros((len(val_probs),)), np.zeros((len(val_probs_ext),)), np.zeros((len(val_probs_ext),))
for i in range(len(val_probs)):
    val_prob = val_probs[i][min_indices[i]]
    y_val = y_vals[i][min_indices[i]]
    fpr, tpr, thresholds = roc_curve(y_val, val_prob)
    youden_j = tpr - fpr
    best_index = np.argmax(youden_j)
    best_threshold = thresholds[best_index]
    val_prob[val_prob<best_threshold] = 0
    val_prob[val_prob>=best_threshold] = 1
    BNT_val_f1[i] = f1_score(y_val, val_prob)
    test_prob = test_probs[i][min_indices[i]]
    y_test = y_tests[i][min_indices[i]]
    fpr, tpr, thresholds = roc_curve(y_test, test_prob)
    youden_j = tpr - fpr
    best_index = np.argmax(youden_j)
    best_threshold = thresholds[best_index]
    test_prob[test_prob<best_threshold] = 0
    test_prob[test_prob>=best_threshold] = 1
    BNT_test_f1[i] = f1_score(y_test, test_prob)
for i in range(len(val_probs_ext)):
    val_prob_ext = val_probs_ext[i][min_indices_ext[i]]
    y_val_ext = y_vals_ext[i][min_indices_ext[i]]
    fpr, tpr, thresholds = roc_curve(y_val_ext, val_prob_ext)
    youden_j = tpr - fpr
    best_index = np.argmax(youden_j)
    best_threshold = thresholds[best_index]
    val_prob_ext[val_prob_ext<best_threshold] = 0
    val_prob_ext[val_prob_ext>=best_threshold] = 1
    BNT_val_f1_ext[i] = f1_score(y_val_ext, val_prob_ext)
    ext_test_prob = ext_test_probs[i][min_indices_ext[i]]
    y_ext_test = y_ext_tests[i][min_indices[i]]
    fpr, tpr, thresholds = roc_curve(y_ext_test, ext_test_prob)
    youden_j = tpr - fpr
    best_index = np.argmax(youden_j)
    best_threshold = thresholds[best_index]
    ext_test_prob[ext_test_prob<best_threshold] = 0
    ext_test_prob[ext_test_prob>=best_threshold] = 1
    BNT_ext_test_f1[i] = f1_score(y_ext_test, ext_test_prob)

BNT_val_losses = np.take_along_axis(val_losses, min_indices[:, None], axis=1).squeeze()
BNT_val_aucs = np.take_along_axis(val_aucs, min_indices[:, None], axis=1).squeeze()  
BNT_test_losses = np.take_along_axis(test_losses, min_indices[:, None], axis=1).squeeze()
BNT_test_aucs = np.take_along_axis(test_aucs, min_indices[:, None], axis=1).squeeze()
BNT_val_losses_ext = np.take_along_axis(val_losses_ext, min_indices_ext[:, None], axis=1).squeeze()
BNT_val_aucs_ext = np.take_along_axis(val_aucs_ext, min_indices_ext[:, None], axis=1).squeeze()  
BNT_ext_test_losses = np.take_along_axis(ext_test_losses, min_indices_ext[:, None], axis=1).squeeze()
BNT_ext_test_aucs = np.take_along_axis(ext_test_aucs, min_indices_ext[:, None], axis=1).squeeze()


'''-----------------------------------------CVFormer--------------------------------------------------'''
val_losses = np.array(test_result_CVFormer['val_losses'])  
val_aucs = np.array(test_result_CVFormer['val_aucs']) 
val_probs = test_result_CVFormer['val_probs']
y_vals = test_result_CVFormer['y_val']
test_losses = np.array(test_result_CVFormer['test_losses'])    
test_aucs = np.array(test_result_CVFormer['test_aucs'])
test_probs = test_result_CVFormer['test_probs']
y_tests = test_result_CVFormer['y_test']
val_losses_ext = np.array(test_result_CVFormer['val_losses_ext'])
val_aucs_ext = np.array(test_result_CVFormer['val_aucs_ext'])
val_probs_ext = test_result_CVFormer['val_probs_ext'] 
y_vals_ext = test_result_CVFormer['y_val_ext']
ext_test_losses = np.array(test_result_CVFormer['ext_test_losses'])
ext_test_aucs = np.array(test_result_CVFormer['ext_test_aucs'])
ext_test_probs = test_result_CVFormer['ext_test_probs']
y_ext_test = test_result_CVFormer['y_ext_test']
min_indices = np.argmin(val_losses, axis=1)
min_indices_ext = np.argmin(val_losses_ext, axis=1)

CVFormer_val_f1, CVFormer_test_f1, CVFormer_val_f1_ext, CVFormer_ext_test_f1 = np.zeros((len(val_probs),)), np.zeros((len(val_probs),)), np.zeros((len(val_probs_ext),)), np.zeros((len(val_probs_ext),))
for i in range(len(val_probs)):
    val_prob = val_probs[i][min_indices[i]]
    y_val = y_vals[i][min_indices[i]]
    fpr, tpr, thresholds = roc_curve(y_val, val_prob)
    youden_j = tpr - fpr
    best_index = np.argmax(youden_j)
    best_threshold = thresholds[best_index]
    val_prob[val_prob<best_threshold] = 0
    val_prob[val_prob>=best_threshold] = 1
    CVFormer_val_f1[i] = f1_score(y_val, val_prob)
    test_prob = test_probs[i][min_indices[i]]
    y_test = y_tests[i][min_indices[i]]
    fpr, tpr, thresholds = roc_curve(y_test, test_prob)
    youden_j = tpr - fpr
    best_index = np.argmax(youden_j)
    best_threshold = thresholds[best_index]
    test_prob[test_prob<best_threshold] = 0
    test_prob[test_prob>=best_threshold] = 1
    CVFormer_test_f1[i] = f1_score(y_test, test_prob)
for i in range(len(val_probs_ext)):
    val_prob_ext = val_probs_ext[i][min_indices_ext[i]]
    y_val_ext = y_vals_ext[i][min_indices_ext[i]]
    fpr, tpr, thresholds = roc_curve(y_val_ext, val_prob_ext)
    youden_j = tpr - fpr
    best_index = np.argmax(youden_j)
    best_threshold = thresholds[best_index]
    val_prob_ext[val_prob_ext<best_threshold] = 0
    val_prob_ext[val_prob_ext>=best_threshold] = 1
    CVFormer_val_f1_ext[i] = f1_score(y_val_ext, val_prob_ext)
    ext_test_prob = ext_test_probs[i][min_indices_ext[i]]
    fpr, tpr, thresholds = roc_curve(y_ext_test, ext_test_prob)
    youden_j = tpr - fpr
    best_index = np.argmax(youden_j)
    best_threshold = thresholds[best_index]
    ext_test_prob[ext_test_prob<best_threshold] = 0
    ext_test_prob[ext_test_prob>=best_threshold] = 1
    CVFormer_ext_test_f1[i] = f1_score(y_ext_test, ext_test_prob)

CVFormer_val_losses = np.take_along_axis(val_losses, min_indices[:, None], axis=1).squeeze()
CVFormer_val_aucs = np.take_along_axis(val_aucs, min_indices[:, None], axis=1).squeeze()  
CVFormer_test_losses = np.take_along_axis(test_losses, min_indices[:, None], axis=1).squeeze()
CVFormer_test_aucs = np.take_along_axis(test_aucs, min_indices[:, None], axis=1).squeeze()
CVFormer_val_losses_ext = np.take_along_axis(val_losses_ext, min_indices_ext[:, None], axis=1).squeeze()
CVFormer_val_aucs_ext = np.take_along_axis(val_aucs_ext, min_indices_ext[:, None], axis=1).squeeze()  
CVFormer_ext_test_losses = np.take_along_axis(ext_test_losses, min_indices_ext[:, None], axis=1).squeeze()
CVFormer_ext_test_aucs = np.take_along_axis(ext_test_aucs, min_indices_ext[:, None], axis=1).squeeze()


'''-----------------------------------------FBNETGEN--------------------------------------------------'''
val_losses = np.array(test_result_FBNETGEN['val_losses'])  
val_aucs = np.array(test_result_FBNETGEN['val_aucs']) 
val_probs = test_result_FBNETGEN['val_probs']
y_vals = test_result_FBNETGEN['y_val']
test_losses = np.array(test_result_FBNETGEN['test_losses'])    
test_aucs = np.array(test_result_FBNETGEN['test_aucs'])
test_probs = test_result_FBNETGEN['test_probs']
y_tests = test_result_FBNETGEN['y_test']
val_losses_ext = np.array(test_result_FBNETGEN['val_losses_ext'])
val_aucs_ext = np.array(test_result_FBNETGEN['val_aucs_ext'])
val_probs_ext = test_result_FBNETGEN['val_probs_ext'] 
y_vals_ext = test_result_FBNETGEN['y_val_ext']
ext_test_losses = np.array(test_result_FBNETGEN['ext_test_losses'])
ext_test_aucs = np.array(test_result_FBNETGEN['ext_test_aucs'])
ext_test_probs = test_result_FBNETGEN['ext_test_probs']
y_ext_tests = test_result_FBNETGEN['y_ext_test']
min_indices = np.argmin(val_losses, axis=1)
min_indices_ext = np.argmin(val_losses_ext, axis=1)

FBNETGEN_val_f1, FBNETGEN_test_f1, FBNETGEN_val_f1_ext, FBNETGEN_ext_test_f1 = np.zeros((len(val_probs),)), np.zeros((len(val_probs),)),np.zeros((len(val_probs_ext),)), np.zeros((len(val_probs_ext),))
for i in range(len(val_probs)):
    val_prob = val_probs[i][min_indices[i]]
    y_val = y_vals[i][min_indices[i]]
    fpr, tpr, thresholds = roc_curve(y_val, val_prob)
    youden_j = tpr - fpr
    best_index = np.argmax(youden_j)
    best_threshold = thresholds[best_index]
    val_prob[val_prob<best_threshold] = 0
    val_prob[val_prob>=best_threshold] = 1
    FBNETGEN_val_f1[i] = f1_score(y_val, val_prob)
    test_prob = test_probs[i][min_indices[i]]
    y_test = y_tests[i][min_indices[i]]
    fpr, tpr, thresholds = roc_curve(y_test, test_prob)
    youden_j = tpr - fpr
    best_index = np.argmax(youden_j)
    best_threshold = thresholds[best_index]
    test_prob[test_prob<best_threshold] = 0
    test_prob[test_prob>=best_threshold] = 1
    FBNETGEN_test_f1[i] = f1_score(y_test, test_prob)
for i in range(len(val_probs_ext)):
    val_prob_ext = val_probs_ext[i][min_indices_ext[i]]
    y_val_ext = y_vals_ext[i][min_indices_ext[i]]
    fpr, tpr, thresholds = roc_curve(y_val_ext, val_prob_ext)
    youden_j = tpr - fpr
    best_index = np.argmax(youden_j)
    best_threshold = thresholds[best_index]
    val_prob_ext[val_prob_ext<best_threshold] = 0
    val_prob_ext[val_prob_ext>=best_threshold] = 1
    FBNETGEN_val_f1_ext[i] = f1_score(y_val_ext, val_prob_ext)
    ext_test_prob = ext_test_probs[i][min_indices_ext[i]]
    y_ext_test = y_ext_tests[i][min_indices[i]]
    fpr, tpr, thresholds = roc_curve(y_ext_test, ext_test_prob)
    youden_j = tpr - fpr
    best_index = np.argmax(youden_j)
    best_threshold = thresholds[best_index]
    ext_test_prob[ext_test_prob<best_threshold] = 0
    ext_test_prob[ext_test_prob>=best_threshold] = 1
    FBNETGEN_ext_test_f1[i] = f1_score(y_ext_test, ext_test_prob)

FBNETGEN_val_losses = np.take_along_axis(val_losses, min_indices[:, None], axis=1).squeeze()
FBNETGEN_val_aucs = np.take_along_axis(val_aucs, min_indices[:, None], axis=1).squeeze()  
FBNETGEN_test_losses = np.take_along_axis(test_losses, min_indices[:, None], axis=1).squeeze()
FBNETGEN_test_aucs = np.take_along_axis(test_aucs, min_indices[:, None], axis=1).squeeze() 
FBNETGEN_val_losses_ext = np.take_along_axis(val_losses_ext, min_indices_ext[:, None], axis=1).squeeze()
FBNETGEN_val_aucs_ext = np.take_along_axis(val_aucs_ext, min_indices_ext[:, None], axis=1).squeeze()  
FBNETGEN_ext_test_losses = np.take_along_axis(ext_test_losses, min_indices_ext[:, None], axis=1).squeeze()
FBNETGEN_ext_test_aucs = np.take_along_axis(ext_test_aucs, min_indices_ext[:, None], axis=1).squeeze()


'''-----------------------------------------DeepFMRI--------------------------------------------------'''
val_losses = np.array(test_result_DeepFMRI['val_losses'])  
val_aucs = np.array(test_result_DeepFMRI['val_aucs']) 
val_probs = test_result_DeepFMRI['val_probs']
y_vals = test_result_DeepFMRI['y_val']
test_losses = np.array(test_result_DeepFMRI['test_losses'])    
test_aucs = np.array(test_result_DeepFMRI['test_aucs'])
test_probs = test_result_DeepFMRI['test_probs']
y_tests = test_result_DeepFMRI['y_test']
val_losses_ext = np.array(test_result_DeepFMRI['val_losses_ext'])
val_aucs_ext = np.array(test_result_DeepFMRI['val_aucs_ext'])
val_probs_ext = test_result_DeepFMRI['val_probs_ext'] 
y_vals_ext = test_result_DeepFMRI['y_val_ext']
ext_test_losses = np.array(test_result_DeepFMRI['ext_test_losses'])
ext_test_aucs = np.array(test_result_DeepFMRI['ext_test_aucs'])
ext_test_probs = test_result_DeepFMRI['ext_test_probs']
y_ext_test = test_result_DeepFMRI['y_ext_test']
min_indices = np.argmin(val_losses, axis=1)
min_indices_ext = np.argmin(val_losses_ext, axis=1)

DeepFMRI_val_f1, DeepFMRI_test_f1, DeepFMRI_val_f1_ext, DeepFMRI_ext_test_f1 = np.zeros((len(val_probs),)), np.zeros((len(val_probs),)), np.zeros((len(val_probs_ext),)), np.zeros((len(val_probs_ext),))
for i in range(len(val_probs)):
    val_prob = val_probs[i][min_indices[i]]
    y_val = y_vals[i][min_indices[i]]
    fpr, tpr, thresholds = roc_curve(y_val, val_prob)
    youden_j = tpr - fpr
    best_index = np.argmax(youden_j)
    best_threshold = thresholds[best_index]
    val_prob[val_prob<best_threshold] = 0
    val_prob[val_prob>=best_threshold] = 1
    DeepFMRI_val_f1[i] = f1_score(y_val, val_prob)
    test_prob = test_probs[i][min_indices[i]]
    y_test = y_tests[i][min_indices[i]]
    fpr, tpr, thresholds = roc_curve(y_test, test_prob)
    youden_j = tpr - fpr
    best_index = np.argmax(youden_j)
    best_threshold = thresholds[best_index]
    test_prob[test_prob<best_threshold] = 0
    test_prob[test_prob>=best_threshold] = 1
    DeepFMRI_test_f1[i] = f1_score(y_test, test_prob)
for i in range(len(val_probs_ext)):
    val_prob_ext = val_probs_ext[i][min_indices_ext[i]]
    y_val_ext = y_vals_ext[i][min_indices_ext[i]]
    fpr, tpr, thresholds = roc_curve(y_val_ext, val_prob_ext)
    youden_j = tpr - fpr
    best_index = np.argmax(youden_j)
    best_threshold = thresholds[best_index]
    val_prob_ext[val_prob_ext<best_threshold] = 0
    val_prob_ext[val_prob_ext>=best_threshold] = 1
    DeepFMRI_val_f1_ext[i] = f1_score(y_val_ext, val_prob_ext)
    ext_test_prob = ext_test_probs[i][min_indices_ext[i]]
    fpr, tpr, thresholds = roc_curve(y_ext_test, ext_test_prob)
    youden_j = tpr - fpr
    best_index = np.argmax(youden_j)
    best_threshold = thresholds[best_index]
    ext_test_prob[ext_test_prob<best_threshold] = 0
    ext_test_prob[ext_test_prob>=best_threshold] = 1
    DeepFMRI_ext_test_f1[i] = f1_score(y_ext_test, ext_test_prob)

DeepFMRI_val_losses = np.take_along_axis(val_losses, min_indices[:, None], axis=1).squeeze()
DeepFMRI_val_aucs = np.take_along_axis(val_aucs, min_indices[:, None], axis=1).squeeze()  
DeepFMRI_test_losses = np.take_along_axis(test_losses, min_indices[:, None], axis=1).squeeze()
DeepFMRI_test_aucs = np.take_along_axis(test_aucs, min_indices[:, None], axis=1).squeeze() 
DeepFMRI_val_losses_ext = np.take_along_axis(val_losses_ext, min_indices_ext[:, None], axis=1).squeeze()
DeepFMRI_val_aucs_ext = np.take_along_axis(val_aucs_ext, min_indices_ext[:, None], axis=1).squeeze()  
DeepFMRI_ext_test_losses = np.take_along_axis(ext_test_losses, min_indices_ext[:, None], axis=1).squeeze()
DeepFMRI_ext_test_aucs = np.take_along_axis(ext_test_aucs, min_indices_ext[:, None], axis=1).squeeze()


'''-----------------------------------------BrainIB--------------------------------------------------'''
val_losses = np.array(test_result_BrainIB['val_losses'])  
val_aucs = np.array(test_result_BrainIB['val_aucs']) 
val_probs = test_result_BrainIB['val_probs']
y_vals = test_result_BrainIB['y_val']
test_losses = np.array(test_result_BrainIB['test_losses'])    
test_aucs = np.array(test_result_BrainIB['test_aucs'])
test_probs = test_result_BrainIB['test_probs']
y_tests = test_result_BrainIB['y_test']
val_losses_ext = np.array(test_result_BrainIB['val_losses_ext'])
val_aucs_ext = np.array(test_result_BrainIB['val_aucs_ext'])
val_probs_ext = test_result_BrainIB['val_probs_ext'] 
y_vals_ext = test_result_BrainIB['y_val_ext']
ext_test_losses = np.array(test_result_BrainIB['ext_test_losses'])
ext_test_aucs = np.array(test_result_BrainIB['ext_test_aucs'])
ext_test_probs = test_result_BrainIB['ext_test_probs']
y_ext_test = test_result_BrainIB['y_ext_test']
min_indices = np.argmin(val_losses, axis=1)
min_indices_ext = np.argmin(val_losses_ext, axis=1)

BrainIB_val_f1, BrainIB_test_f1, BrainIB_val_f1_ext, BrainIB_ext_test_f1 = np.zeros((len(val_probs),)), np.zeros((len(val_probs),)), np.zeros((len(val_probs_ext),)), np.zeros((len(val_probs_ext),))
for i in range(len(val_probs)):
    val_prob = val_probs[i][min_indices[i]]
    y_val = y_vals[i][min_indices[i]]
    fpr, tpr, thresholds = roc_curve(y_val, val_prob)
    youden_j = tpr - fpr
    best_index = np.argmax(youden_j)
    best_threshold = thresholds[best_index]
    val_prob[val_prob<best_threshold] = 0
    val_prob[val_prob>=best_threshold] = 1
    BrainIB_val_f1[i] = f1_score(y_val, val_prob)
    test_prob = test_probs[i][min_indices[i]]
    y_test = y_tests[i][min_indices[i]]
    fpr, tpr, thresholds = roc_curve(y_test, test_prob)
    youden_j = tpr - fpr
    best_index = np.argmax(youden_j)
    best_threshold = thresholds[best_index]
    test_prob[test_prob<best_threshold] = 0
    test_prob[test_prob>=best_threshold] = 1
    BrainIB_test_f1[i] = f1_score(y_test, test_prob)
for i in range(len(val_probs_ext)):
    val_prob_ext = val_probs_ext[i][min_indices_ext[i]]
    y_val_ext = y_vals_ext[i][min_indices_ext[i]]
    fpr, tpr, thresholds = roc_curve(y_val_ext, val_prob_ext)
    youden_j = tpr - fpr
    best_index = np.argmax(youden_j)
    best_threshold = thresholds[best_index]
    val_prob_ext[val_prob_ext<best_threshold] = 0
    val_prob_ext[val_prob_ext>=best_threshold] = 1
    BrainIB_val_f1_ext[i] = f1_score(y_val_ext, val_prob_ext)
    ext_test_prob = ext_test_probs[i][min_indices_ext[i]]
    fpr, tpr, thresholds = roc_curve(y_ext_test, ext_test_prob)
    youden_j = tpr - fpr
    best_index = np.argmax(youden_j)
    best_threshold = thresholds[best_index]
    ext_test_prob[ext_test_prob<best_threshold] = 0
    ext_test_prob[ext_test_prob>=best_threshold] = 1
    BrainIB_ext_test_f1[i] = f1_score(y_ext_test, ext_test_prob)

BrainIB_val_losses = np.take_along_axis(val_losses, min_indices[:, None], axis=1).squeeze()
BrainIB_val_aucs = np.take_along_axis(val_aucs, min_indices[:, None], axis=1).squeeze()  
BrainIB_test_losses = np.take_along_axis(test_losses, min_indices[:, None], axis=1).squeeze()
BrainIB_test_aucs = np.take_along_axis(test_aucs, min_indices[:, None], axis=1).squeeze() 
BrainIB_val_losses_ext = np.take_along_axis(val_losses_ext, min_indices_ext[:, None], axis=1).squeeze()
BrainIB_val_aucs_ext = np.take_along_axis(val_aucs_ext, min_indices_ext[:, None], axis=1).squeeze()  
BrainIB_ext_test_losses = np.take_along_axis(ext_test_losses, min_indices_ext[:, None], axis=1).squeeze()
BrainIB_ext_test_aucs = np.take_along_axis(ext_test_aucs, min_indices_ext[:, None], axis=1).squeeze()


'''-----------------------------------------BAnD-------------------------------------------------'''
val_losses = np.array(test_result_BAnD['val_losses'])  
val_aucs = np.array(test_result_BAnD['val_aucs']) 
val_probs = test_result_BAnD['val_probs']
y_vals = test_result_BAnD['y_val']
test_losses = np.array(test_result_BAnD['test_losses'])    
test_aucs = np.array(test_result_BAnD['test_aucs'])
test_probs = test_result_BAnD['test_probs']
y_tests = test_result_BAnD['y_test']
val_losses_ext = np.array(test_result_BAnD['val_losses_ext'])
val_aucs_ext = np.array(test_result_BAnD['val_aucs_ext'])
val_probs_ext = test_result_BAnD['val_probs_ext'] 
y_vals_ext = test_result_BAnD['y_val_ext']
ext_test_losses = np.array(test_result_BAnD['ext_test_losses'])
ext_test_aucs = np.array(test_result_BAnD['ext_test_aucs'])
ext_test_probs = test_result_BAnD['ext_test_probs']
y_ext_test = test_result_BAnD['y_ext_test']
min_indices = np.argmin(val_losses, axis=1)
min_indices_ext = np.argmin(val_losses_ext, axis=1)

BAnD_val_f1, BAnD_test_f1, BAnD_val_f1_ext, BAnD_ext_test_f1 = np.zeros((len(val_probs),)), np.zeros((len(val_probs),)),np.zeros((len(val_probs_ext),)), np.zeros((len(val_probs_ext),))
for i in range(len(val_probs)):
    val_prob = val_probs[i][min_indices[i]]
    y_val = y_vals[i][min_indices[i]]
    fpr, tpr, thresholds = roc_curve(y_val, val_prob)
    youden_j = tpr - fpr
    best_index = np.argmax(youden_j)
    best_threshold = thresholds[best_index]
    val_prob[val_prob<best_threshold] = 0
    val_prob[val_prob>=best_threshold] = 1
    BAnD_val_f1[i] = f1_score(y_val, val_prob)
    test_prob = test_probs[i][min_indices[i]]
    y_test = y_tests[i][min_indices[i]]
    fpr, tpr, thresholds = roc_curve(y_test, test_prob)
    youden_j = tpr - fpr
    best_index = np.argmax(youden_j)
    best_threshold = thresholds[best_index]
    test_prob[test_prob<best_threshold] = 0
    test_prob[test_prob>=best_threshold] = 1
    BAnD_test_f1[i] = f1_score(y_test, test_prob)
for i in range(len(val_probs_ext)):
    val_prob_ext = val_probs_ext[i][min_indices_ext[i]]
    y_val_ext = y_vals_ext[i][min_indices_ext[i]]
    fpr, tpr, thresholds = roc_curve(y_val_ext, val_prob_ext)
    youden_j = tpr - fpr
    best_index = np.argmax(youden_j)
    best_threshold = thresholds[best_index]
    val_prob_ext[val_prob_ext<best_threshold] = 0
    val_prob_ext[val_prob_ext>=best_threshold] = 1
    BAnD_val_f1_ext[i] = f1_score(y_val_ext, val_prob_ext)
    ext_test_prob = ext_test_probs[i][min_indices_ext[i]]
    fpr, tpr, thresholds = roc_curve(y_ext_test, ext_test_prob)
    youden_j = tpr - fpr
    best_index = np.argmax(youden_j)
    best_threshold = thresholds[best_index]
    ext_test_prob[ext_test_prob<best_threshold] = 0
    ext_test_prob[ext_test_prob>=best_threshold] = 1
    BAnD_ext_test_f1[i] = f1_score(y_ext_test, ext_test_prob)

BAnD_val_losses = np.take_along_axis(val_losses, min_indices[:, None], axis=1).squeeze()
BAnD_val_aucs = np.take_along_axis(val_aucs, min_indices[:, None], axis=1).squeeze()  
BAnD_test_losses = np.take_along_axis(test_losses, min_indices[:, None], axis=1).squeeze()
BAnD_test_aucs = np.take_along_axis(test_aucs, min_indices[:, None], axis=1).squeeze() 
BAnD_val_losses_ext = np.take_along_axis(val_losses_ext, min_indices_ext[:, None], axis=1).squeeze()
BAnD_val_aucs_ext = np.take_along_axis(val_aucs_ext, min_indices_ext[:, None], axis=1).squeeze()  
BAnD_ext_test_losses = np.take_along_axis(ext_test_losses, min_indices_ext[:, None], axis=1).squeeze()
BAnD_ext_test_aucs = np.take_along_axis(ext_test_aucs, min_indices_ext[:, None], axis=1).squeeze()

'''-----------------------------------------LFB--------------------------------------------------'''
LFB_val_losses = np.array(test_result_LFB['val_losses'])  
LFB_test_losses = np.array(test_result_LFB['test_losses'])    
LFB_test_aucs = np.array(test_result_LFB['test_aucs'])
test_probs = test_result_LFB['test_probs']
y_tests = test_result_LFB['y_test']
LFB_val_losses_ext = np.array(test_result_LFB['val_losses_ext'])
LFB_ext_test_losses = np.array(test_result_LFB['ext_test_losses'])
LFB_ext_test_aucs = np.array(test_result_LFB['ext_test_aucs'])
ext_test_probs = test_result_LFB['ext_test_probs']
y_ext_test = test_result_LFB['y_ext_test']

LFB_test_f1, LFB_ext_test_f1 = np.zeros((len(val_losses),)), np.zeros((len(val_losses_ext),))
for i in range(len(val_losses)):
    test_prob = test_probs[i]
    y_test = y_tests[i]
    fpr, tpr, thresholds = roc_curve(y_test, test_prob)
    youden_j = tpr - fpr
    best_index = np.argmax(youden_j)
    best_threshold = thresholds[best_index]
    test_prob[test_prob<best_threshold] = 0
    test_prob[test_prob>=best_threshold] = 1
    LFB_test_f1[i] = f1_score(y_test, test_prob)
for i in range(len(val_losses_ext)):
    ext_test_prob = ext_test_probs[i]
    fpr, tpr, thresholds = roc_curve(y_ext_test, ext_test_prob)
    youden_j = tpr - fpr
    best_index = np.argmax(youden_j)
    best_threshold = thresholds[best_index]
    ext_test_prob[ext_test_prob<best_threshold] = 0
    ext_test_prob[ext_test_prob>=best_threshold] = 1
    LFB_ext_test_f1[i] = f1_score(y_ext_test, ext_test_prob)


'''-----------------------------------------UCGL--------------------------------------------------'''
val_losses = np.zeros((100,30))   
val_aucs = np.zeros((100,30))  
val_f1 = np.zeros((100,))  
test_losses = np.zeros((100,30))   
test_aucs = np.zeros((100,30))  
test_f1 = np.zeros((100,))  
val_losses_ext = np.zeros((10,30))   
val_aucs_ext = np.zeros((10,30))  
val_f1_ext = np.zeros((10,)) 
ext_test_losses = np.zeros((10,30))   
ext_test_aucs = np.zeros((10,30))  
ext_test_f1 = np.zeros((10,))
for i,test in enumerate(test_result_UCGL['epoch_results']):
    val_aucs[i,:] = test['val_aucs']
    val_losses[i,:] = test['val_loss']
    val_prob = test['val_probs']
    y_val = test['y_val']
    fpr, tpr, thresholds = roc_curve(y_val, val_prob)
    youden_j = tpr - fpr
    best_index = np.argmax(youden_j)
    best_threshold = thresholds[best_index]
    val_prob[val_prob<best_threshold] = 0
    val_prob[val_prob>=best_threshold] = 1
    val_f1[i] = f1_score(y_val, val_prob)
    test_aucs[i,:] = test['test_auc']
    test_losses[i,:] = test['test_loss']
    test_prob = test['test_probs']
    y_test = test['y_test']
    fpr, tpr, thresholds = roc_curve(y_test, test_prob)
    youden_j = tpr - fpr
    best_index = np.argmax(youden_j)
    best_threshold = thresholds[best_index]
    test_prob[test_prob<best_threshold] = 0
    test_prob[test_prob>=best_threshold] = 1
    test_f1[i] = f1_score(y_test, test_prob)
for i,test in enumerate(test_result_UCGL['epoch_results_ext']):
    val_aucs_ext[i,:] = test['val_aucs']
    val_losses_ext[i,:] = test['val_loss']
    val_prob_ext = test['val_probs']
    y_val_ext = test['y_val']
    fpr, tpr, thresholds = roc_curve(y_val_ext, val_prob_ext)
    youden_j = tpr - fpr
    best_index = np.argmax(youden_j)
    best_threshold = thresholds[best_index]
    val_prob_ext[val_prob_ext<best_threshold] = 0
    val_prob_ext[val_prob_ext>=best_threshold] = 1
    val_f1_ext[i] = f1_score(y_val_ext, val_prob_ext)
    ext_test_aucs[i,:] = test['test_auc']
    ext_test_losses[i,:] = test['test_loss']
    ext_test_prob = test['test_probs']
    y_ext_test = test['y_test']
    fpr, tpr, thresholds = roc_curve(y_ext_test, ext_test_prob)
    youden_j = tpr - fpr
    best_index = np.argmax(youden_j)
    best_threshold = thresholds[best_index]
    ext_test_prob[ext_test_prob<best_threshold] = 0
    ext_test_prob[ext_test_prob>=best_threshold] = 1
    ext_test_f1[i] = f1_score(y_ext_test, ext_test_prob)
        
min_indices = np.argmin(val_losses, axis=1)
min_indices_ext = np.argmin(val_losses_ext, axis=1)
        
UCGL_val_losses = np.take_along_axis(val_losses, min_indices[:, None], axis=1).squeeze()
UCGL_val_aucs = np.take_along_axis(val_aucs, min_indices[:, None], axis=1).squeeze()  
UCGL_val_f1 = val_f1
UCGL_test_losses = np.take_along_axis(test_losses, min_indices[:, None], axis=1).squeeze()
UCGL_test_aucs = np.take_along_axis(test_aucs, min_indices[:, None], axis=1).squeeze() 
UCGL_test_f1 = test_f1
UCGL_val_losses_ext = np.take_along_axis(val_losses_ext, min_indices_ext[:, None], axis=1).squeeze()
UCGL_val_aucs_ext = np.take_along_axis(val_aucs_ext, min_indices_ext[:, None], axis=1).squeeze()  
UCGL_val_f1_ext = val_f1_ext
UCGL_ext_test_losses = np.take_along_axis(ext_test_losses, min_indices_ext[:, None], axis=1).squeeze()
UCGL_ext_test_aucs = np.take_along_axis(ext_test_aucs, min_indices_ext[:, None], axis=1).squeeze()
UCGL_ext_test_f1 = ext_test_f1


'''-----------------------------------------GCDA--------------------------------------------------'''
val_losses = np.zeros((100,30))   
val_aucs = np.zeros((100,30))  
val_f1 = np.zeros((100,))  
test_losses = np.zeros((100,30))   
test_aucs = np.zeros((100,30))  
test_f1 = np.zeros((100,))  
val_losses_ext = np.zeros((10,30))   
val_aucs_ext = np.zeros((10,30))  
val_f1_ext = np.zeros((10,)) 
ext_test_losses = np.zeros((10,30))   
ext_test_aucs = np.zeros((10,30))  
ext_test_f1 = np.zeros((10,))
for i,test in enumerate(test_result_GCDA['epoch_results']):
    val_aucs[i,:] = test['val_aucs']
    val_losses[i,:] = test['val_loss']
    val_prob = test['val_probs'][0]
    y_val = test['y_val']
    fpr, tpr, thresholds = roc_curve(y_val, val_prob)
    youden_j = tpr - fpr
    best_index = np.argmax(youden_j)
    best_threshold = thresholds[best_index]
    val_prob[val_prob<best_threshold] = 0
    val_prob[val_prob>=best_threshold] = 1
    val_f1[i] = f1_score(y_val, val_prob)
    test_aucs[i,:] = test['test_auc']
    test_losses[i,:] = test['test_loss']
    test_prob = test['test_probs'][0]
    y_test = test['y_test']
    fpr, tpr, thresholds = roc_curve(y_test, test_prob)
    youden_j = tpr - fpr
    best_index = np.argmax(youden_j)
    best_threshold = thresholds[best_index]
    test_prob[test_prob<best_threshold] = 0
    test_prob[test_prob>=best_threshold] = 1
    test_f1[i] = f1_score(y_test, test_prob)
for i,test in enumerate(test_result_GCDA['epoch_results_ext']):
    val_aucs_ext[i,:] = test['val_aucs']
    val_losses_ext[i,:] = test['val_loss']
    val_prob_ext = test['val_probs'][0]
    y_val_ext = test['y_val']
    fpr, tpr, thresholds = roc_curve(y_val_ext, val_prob_ext)
    youden_j = tpr - fpr
    best_index = np.argmax(youden_j)
    best_threshold = thresholds[best_index]
    val_prob_ext[val_prob_ext<best_threshold] = 0
    val_prob_ext[val_prob_ext>=best_threshold] = 1
    val_f1_ext[i] = f1_score(y_val_ext, val_prob_ext)
    ext_test_aucs[i,:] = test['test_auc']
    ext_test_losses[i,:] = test['test_loss']
    ext_test_prob = test['test_probs'][0]
    y_ext_test = test['y_test']
    fpr, tpr, thresholds = roc_curve(y_ext_test, ext_test_prob)
    youden_j = tpr - fpr
    best_index = np.argmax(youden_j)
    best_threshold = thresholds[best_index]
    ext_test_prob[ext_test_prob<best_threshold] = 0
    ext_test_prob[ext_test_prob>=best_threshold] = 1
    ext_test_f1[i] = f1_score(y_ext_test, ext_test_prob)
        
min_indices = np.argmin(val_losses, axis=1)
min_indices_ext = np.argmin(val_losses_ext, axis=1)
        
GCDA_val_losses = np.take_along_axis(val_losses, min_indices[:, None], axis=1).squeeze()
GCDA_val_aucs = np.take_along_axis(val_aucs, min_indices[:, None], axis=1).squeeze()  
GCDA_val_f1 = val_f1
GCDA_test_losses = np.take_along_axis(test_losses, min_indices[:, None], axis=1).squeeze()
GCDA_test_aucs = np.take_along_axis(test_aucs, min_indices[:, None], axis=1).squeeze() 
GCDA_test_f1 = test_f1
GCDA_val_losses_ext = np.take_along_axis(val_losses_ext, min_indices_ext[:, None], axis=1).squeeze()
GCDA_val_aucs_ext = np.take_along_axis(val_aucs_ext, min_indices_ext[:, None], axis=1).squeeze()  
GCDA_val_f1_ext = val_f1_ext
GCDA_ext_test_losses = np.take_along_axis(ext_test_losses, min_indices_ext[:, None], axis=1).squeeze()
GCDA_ext_test_aucs = np.take_along_axis(ext_test_aucs, min_indices_ext[:, None], axis=1).squeeze()
GCDA_ext_test_f1 = ext_test_f1


'''-----------------------------------------A-GCL--------------------------------------------------'''
val_losses = np.zeros((100,100))   
val_aucs = np.zeros((100,100))  
val_f1 = np.zeros((100,100))  
test_losses = np.zeros((100,100))   
test_aucs = np.zeros((100,100))  
test_f1 = np.zeros((100,100))  
val_losses_ext = np.zeros((10,100))   
val_aucs_ext = np.zeros((10,100))  
val_f1_ext = np.zeros((10,100)) 
ext_test_losses = np.zeros((10,100))   
ext_test_aucs = np.zeros((10,100))  
ext_test_f1 = np.zeros((10,100))
for i,test in enumerate(test_result_A_GCL['epoch_results']):
    for j,test1 in enumerate(test):
        val_aucs[i,j] = test1['val_auc']
        val_losses[i,j] = test1['val_loss']
        val_prob = test1['val_probs']
        y_val = test1['y_val']
        fpr, tpr, thresholds = roc_curve(y_val, val_prob)
        youden_j = tpr - fpr
        best_index = np.argmax(youden_j)
        best_threshold = thresholds[best_index]
        val_prob[val_prob<best_threshold] = 0
        val_prob[val_prob>=best_threshold] = 1
        val_f1[i,j] = f1_score(y_val, val_prob)
        test_aucs[i,j] = test1['test_auc']
        test_losses[i,j] = test1['test_loss']
        test_prob = test1['test_probs']
        y_test = test1['y_test']
        fpr, tpr, thresholds = roc_curve(y_test, test_prob)
        youden_j = tpr - fpr
        best_index = np.argmax(youden_j)
        best_threshold = thresholds[best_index]
        test_prob[test_prob<best_threshold] = 0
        test_prob[test_prob>=best_threshold] = 1
        test_f1[i,j] = f1_score(y_test, test_prob)
for i,test in enumerate(test_result_A_GCL['epoch_results_ext']):
    for j,test1 in enumerate(test):
        val_aucs_ext[i,j] = test1['val_auc']
        val_losses_ext[i,j] = test1['val_loss']
        val_prob_ext = test1['val_probs']
        y_val_ext = test1['y_val']
        fpr, tpr, thresholds = roc_curve(y_val_ext, val_prob_ext)
        youden_j = tpr - fpr
        best_index = np.argmax(youden_j)
        best_threshold = thresholds[best_index]
        val_prob_ext[val_prob_ext<best_threshold] = 0
        val_prob_ext[val_prob_ext>=best_threshold] = 1
        val_f1_ext[i,j] = f1_score(y_val_ext, val_prob_ext)
        ext_test_aucs[i,j] = test1['test_auc']
        ext_test_losses[i,j] = test1['test_loss']
        ext_test_prob = test1['test_probs']
        y_ext_test = test1['y_test']
        fpr, tpr, thresholds = roc_curve(y_ext_test, ext_test_prob)
        youden_j = tpr - fpr
        best_index = np.argmax(youden_j)
        best_threshold = thresholds[best_index]
        ext_test_prob[ext_test_prob<best_threshold] = 0
        ext_test_prob[ext_test_prob>=best_threshold] = 1
        ext_test_f1[i,j] = f1_score(y_ext_test, ext_test_prob)
        
min_indices = np.argmin(val_losses, axis=1)
min_indices_ext = np.argmin(val_losses_ext, axis=1)
        
A_GCL_val_losses = np.take_along_axis(val_losses, min_indices[:, None], axis=1).squeeze()
A_GCL_val_aucs = np.take_along_axis(val_aucs, min_indices[:, None], axis=1).squeeze()  
A_GCL_val_f1 = np.take_along_axis(val_f1, min_indices[:, None], axis=1).squeeze()  
A_GCL_test_losses = np.take_along_axis(test_losses, min_indices[:, None], axis=1).squeeze()
A_GCL_test_aucs = np.take_along_axis(test_aucs, min_indices[:, None], axis=1).squeeze() 
A_GCL_test_f1 = np.take_along_axis(test_f1, min_indices[:, None], axis=1).squeeze() 
A_GCL_val_losses_ext = np.take_along_axis(val_losses_ext, min_indices_ext[:, None], axis=1).squeeze()
A_GCL_val_aucs_ext = np.take_along_axis(val_aucs_ext, min_indices_ext[:, None], axis=1).squeeze()  
A_GCL_val_f1_ext = np.take_along_axis(val_f1_ext, min_indices_ext[:, None], axis=1).squeeze()  
A_GCL_ext_test_losses = np.take_along_axis(ext_test_losses, min_indices_ext[:, None], axis=1).squeeze()
A_GCL_ext_test_aucs = np.take_along_axis(ext_test_aucs, min_indices_ext[:, None], axis=1).squeeze()
A_GCL_ext_test_f1 = np.take_along_axis(ext_test_f1, min_indices_ext[:, None], axis=1).squeeze()

'''-----------------------------------------VarCoNet--------------------------------------------------'''
val_losses = np.zeros((100,50))   
val_aucs = np.zeros((100,50))  
val_f1 = np.zeros((100,50))  
test_losses = np.zeros((100,50))   
test_aucs = np.zeros((100,50))  
test_f1 = np.zeros((100,50))  
val_losses_ext = np.zeros((10,50))   
val_aucs_ext = np.zeros((10,50))  
val_f1_ext = np.zeros((10,50)) 
ext_test_losses = np.zeros((10,50))   
ext_test_aucs = np.zeros((10,50))  
ext_test_f1 = np.zeros((10,50))
for i,test in enumerate(test_result_VarCoNet['epoch_results']):
    for j,test1 in enumerate(test):
        val_aucs[i,j] = test1['best_val_auc']
        val_losses[i,j] = test1['best_val_loss']
        val_prob = test1['val_probs']
        y_val = test1['y_val']
        fpr, tpr, thresholds = roc_curve(y_val, val_prob)
        youden_j = tpr - fpr
        best_index = np.argmax(youden_j)
        best_threshold = thresholds[best_index]
        val_prob[val_prob<best_threshold] = 0
        val_prob[val_prob>=best_threshold] = 1
        val_f1[i,j] = f1_score(y_val, val_prob)
        test_aucs[i,j] = test1['best_test_auc']
        test_losses[i,j] = test1['best_test_loss']
        test_prob = test1['test_probs']
        y_test = test1['y_test']
        fpr, tpr, thresholds = roc_curve(y_test, test_prob)
        youden_j = tpr - fpr
        best_index = np.argmax(youden_j)
        best_threshold = thresholds[best_index]
        test_prob[test_prob<best_threshold] = 0
        test_prob[test_prob>=best_threshold] = 1
        test_f1[i,j] = f1_score(y_test, test_prob)
for i,test in enumerate(test_result_VarCoNet['epoch_results_ext']):
    for j,test1 in enumerate(test):
        val_aucs_ext[i,j] = test1['best_val_auc']
        val_losses_ext[i,j] = test1['best_val_loss']
        val_prob_ext = test1['val_probs']
        y_val_ext = test1['y_val']
        fpr, tpr, thresholds = roc_curve(y_val_ext, val_prob_ext)
        youden_j = tpr - fpr
        best_index = np.argmax(youden_j)
        best_threshold = thresholds[best_index]
        val_prob_ext[val_prob_ext<best_threshold] = 0
        val_prob_ext[val_prob_ext>=best_threshold] = 1
        val_f1_ext[i,j] = f1_score(y_val_ext, val_prob_ext)
        ext_test_aucs[i,j] = test1['best_test_auc']
        ext_test_losses[i,j] = test1['best_test_loss']
        ext_test_prob = test1['test_probs']
        y_ext_test = test1['y_test']
        fpr, tpr, thresholds = roc_curve(y_ext_test, ext_test_prob)
        youden_j = tpr - fpr
        best_index = np.argmax(youden_j)
        best_threshold = thresholds[best_index]
        ext_test_prob[ext_test_prob<best_threshold] = 0
        ext_test_prob[ext_test_prob>=best_threshold] = 1
        ext_test_f1[i,j] = f1_score(y_ext_test, ext_test_prob)
        
min_indices = np.argmin(val_losses, axis=1)
min_indices_ext = np.argmin(val_losses_ext, axis=1)
        
VarCoNet_val_losses = np.take_along_axis(val_losses, min_indices[:, None], axis=1).squeeze()
VarCoNet_val_aucs = np.take_along_axis(val_aucs, min_indices[:, None], axis=1).squeeze()  
VarCoNet_val_f1 = np.take_along_axis(val_f1, min_indices[:, None], axis=1).squeeze()  
VarCoNet_test_losses = np.take_along_axis(test_losses, min_indices[:, None], axis=1).squeeze()
VarCoNet_test_aucs = np.take_along_axis(test_aucs, min_indices[:, None], axis=1).squeeze() 
VarCoNet_test_f1 = np.take_along_axis(test_f1, min_indices[:, None], axis=1).squeeze() 
VarCoNet_val_losses_ext = np.take_along_axis(val_losses_ext, min_indices_ext[:, None], axis=1).squeeze()
VarCoNet_val_aucs_ext = np.take_along_axis(val_aucs_ext, min_indices_ext[:, None], axis=1).squeeze()  
VarCoNet_val_f1_ext = np.take_along_axis(val_f1_ext, min_indices_ext[:, None], axis=1).squeeze()  
VarCoNet_ext_test_losses = np.take_along_axis(ext_test_losses, min_indices_ext[:, None], axis=1).squeeze()
VarCoNet_ext_test_aucs = np.take_along_axis(ext_test_aucs, min_indices_ext[:, None], axis=1).squeeze()
VarCoNet_ext_test_f1 = np.take_along_axis(ext_test_f1, min_indices_ext[:, None], axis=1).squeeze()


VarCoNet_val_aucs = VarCoNet_val_aucs*100
VarCoNet_val_f1 = VarCoNet_val_f1*100
VarCoNet_test_aucs = VarCoNet_test_aucs*100
VarCoNet_test_f1 = VarCoNet_test_f1*100
VarCoNet_val_aucs_ext = VarCoNet_val_aucs_ext*100
VarCoNet_val_f1_ext = VarCoNet_val_f1_ext*100
VarCoNet_ext_test_aucs = VarCoNet_ext_test_aucs*100
VarCoNet_ext_test_f1 = VarCoNet_ext_test_f1*100

BolT_val_aucs = BolT_val_aucs*100
BolT_val_f1 = BolT_val_f1*100
BolT_test_aucs = BolT_test_aucs*100
BolT_test_f1 = BolT_test_f1*100
BolT_val_aucs_ext = BolT_val_aucs_ext*100
BolT_val_f1_ext = BolT_val_f1_ext*100
BolT_ext_test_aucs =BolT_ext_test_aucs*100
BolT_ext_test_f1 = BolT_ext_test_f1*100

DeepFMRI_val_aucs = DeepFMRI_val_aucs*100
DeepFMRI_val_f1 = DeepFMRI_val_f1*100
DeepFMRI_test_aucs = DeepFMRI_test_aucs*100
DeepFMRI_test_f1 = DeepFMRI_test_f1*100
DeepFMRI_val_aucs_ext = DeepFMRI_val_aucs_ext*100
DeepFMRI_val_f1_ext = DeepFMRI_val_f1_ext*100
DeepFMRI_ext_test_aucs = DeepFMRI_ext_test_aucs*100
DeepFMRI_ext_test_f1 = DeepFMRI_ext_test_f1*100

CVFormer_val_aucs = CVFormer_val_aucs*100
CVFormer_val_f1 = CVFormer_val_f1*100
CVFormer_test_aucs = CVFormer_test_aucs*100
CVFormer_test_f1 = CVFormer_test_f1*100
CVFormer_val_aucs_ext = CVFormer_val_aucs_ext*100
CVFormer_val_f1_ext = CVFormer_val_f1_ext*100
CVFormer_ext_test_aucs = CVFormer_ext_test_aucs*100
CVFormer_ext_test_f1 = CVFormer_ext_test_f1*100

FBNETGEN_val_aucs = FBNETGEN_val_aucs*100
FBNETGEN_val_f1 = FBNETGEN_val_f1*100
FBNETGEN_test_aucs = FBNETGEN_test_aucs*100
FBNETGEN_test_f1 = FBNETGEN_test_f1*100
FBNETGEN_val_aucs_ext = FBNETGEN_val_aucs_ext*100
FBNETGEN_val_f1_ext = FBNETGEN_val_f1_ext*100
FBNETGEN_ext_test_aucs = FBNETGEN_ext_test_aucs*100
FBNETGEN_ext_test_f1 = FBNETGEN_ext_test_f1*100

BAnD_val_aucs = BAnD_val_aucs*100
BAnD_val_f1 = BAnD_val_f1*100
BAnD_test_aucs = BAnD_test_aucs*100
BAnD_test_f1 = BAnD_test_f1*100
BAnD_val_aucs_ext = BAnD_val_aucs_ext*100
BAnD_val_f1_ext = BAnD_val_f1_ext*100
BAnD_ext_test_aucs = BAnD_ext_test_aucs*100
BAnD_ext_test_f1 = BAnD_ext_test_f1*100

UCGL_val_aucs = UCGL_val_aucs*100
UCGL_val_f1 = UCGL_val_f1*100
UCGL_test_aucs = UCGL_test_aucs*100
UCGL_test_f1 = UCGL_test_f1*100
UCGL_val_aucs_ext = UCGL_val_aucs_ext*100
UCGL_val_f1_ext = UCGL_val_f1_ext*100
UCGL_ext_test_aucs = UCGL_ext_test_aucs*100
UCGL_ext_test_f1 = UCGL_ext_test_f1*100

GCDA_val_aucs = GCDA_val_aucs*100
GCDA_val_f1 = GCDA_val_f1*100
GCDA_test_aucs = GCDA_test_aucs*100
GCDA_test_f1 = GCDA_test_f1*100
GCDA_val_aucs_ext = GCDA_val_aucs_ext*100
GCDA_val_f1_ext = GCDA_val_f1_ext*100
GCDA_ext_test_aucs = GCDA_ext_test_aucs*100
GCDA_ext_test_f1 = GCDA_ext_test_f1*100

A_GCL_val_aucs = A_GCL_val_aucs*100
A_GCL_val_f1 = A_GCL_val_f1*100
A_GCL_test_aucs = A_GCL_test_aucs*100
A_GCL_test_f1 = A_GCL_test_f1*100
A_GCL_val_aucs_ext = A_GCL_val_aucs_ext*100
A_GCL_val_f1_ext = A_GCL_val_f1_ext*100
A_GCL_ext_test_aucs = A_GCL_ext_test_aucs*100
A_GCL_ext_test_f1 = A_GCL_ext_test_f1*100

BrainIB_val_aucs = BrainIB_val_aucs*100
BrainIB_val_f1 = BrainIB_val_f1*100
BrainIB_test_aucs = BrainIB_test_aucs*100
BrainIB_test_f1 = BrainIB_test_f1*100
BrainIB_val_aucs_ext = BrainIB_val_aucs_ext*100
BrainIB_val_f1_ext = BrainIB_val_f1_ext*100
BrainIB_ext_test_aucs = BrainIB_ext_test_aucs*100
BrainIB_ext_test_f1 = BrainIB_ext_test_f1*100

BNT_val_aucs = BNT_val_aucs*100
BNT_val_f1 = BNT_val_f1*100
BNT_test_aucs = BNT_test_aucs*100
BNT_test_f1 = BNT_test_f1*100
BNT_val_aucs_ext = BNT_val_aucs_ext*100
BNT_val_f1_ext = BNT_val_f1_ext*100
BNT_ext_test_aucs = BNT_ext_test_aucs*100
BNT_ext_test_f1 = BNT_ext_test_f1*100

LFB_test_aucs = LFB_test_aucs*100
LFB_test_f1 = LFB_test_f1*100
LFB_ext_test_aucs = LFB_ext_test_aucs*100
LFB_ext_test_f1 = LFB_ext_test_f1*100


print('-----------------------' + atlas + '-----------------------' )
print(f"VarCoNet: mean validation      AUC= {np.mean(VarCoNet_val_aucs):.2f}, std={np.std(VarCoNet_val_aucs):.2f}")
print(f"VarCoNet: mean validation      Loss={np.mean(VarCoNet_val_losses):.3f}, std={np.std(VarCoNet_val_losses):.3f}")
print(f"VarCoNet: mean validation      f1={np.mean(VarCoNet_val_f1):.2f}, std={np.std(VarCoNet_val_f1):.2f}")
print(f"VarCoNet: mean test            AUC= {np.mean(VarCoNet_test_aucs):.2f}, std={np.std(VarCoNet_test_aucs):.2f}")
print(f"VarCoNet: mean test            Loss={np.mean(VarCoNet_test_losses):.3f}, std={np.std(VarCoNet_test_losses):.3f}")
print(f"VarCoNet: mean test            f1={np.mean(VarCoNet_test_f1):.2f}, std={np.std(VarCoNet_test_f1):.2f}")
print(f"VarCoNet: mean ext. validation AUC= {np.mean(VarCoNet_val_aucs_ext):.2f}, std={np.std(VarCoNet_val_aucs_ext):.2f}")
print(f"VarCoNet: mean ext. validation Loss={np.mean(VarCoNet_val_losses_ext):.3f}, std={np.std(VarCoNet_val_losses_ext):.3f}")
print(f"VarCoNet: mean ext. validation f1={np.mean(VarCoNet_val_f1_ext):.2f}, std={np.std(VarCoNet_val_f1_ext):.2f}")
print(f"VarCoNet: mean ext. test       AUC= {np.mean(VarCoNet_ext_test_aucs):.2f}, std={np.std(VarCoNet_ext_test_aucs):.2f}")
print(f"VarCoNet: mean ext. test       Loss={np.mean(VarCoNet_ext_test_losses):.3f}, std={np.std(VarCoNet_ext_test_losses):.3f}")
print(f"VarCoNet: mean ext. test       f1={np.mean(VarCoNet_ext_test_f1):.2f}, std={np.std(VarCoNet_ext_test_f1):.2f}")
print('')
print(f"BolT: mean validation          AUC= {np.mean(BolT_val_aucs):.2f}, std={np.std(BolT_val_aucs):.2f}")
print(f"BolT: mean validation          Loss={np.mean(BolT_val_losses):.3f}, std={np.std(BolT_val_losses):.3f}")
print(f"BolT: mean validation          f1={np.mean(BolT_val_f1):.2f}, std={np.std(BolT_val_f1):.2f}")
print(f"BolT: mean test                AUC= {np.mean(BolT_test_aucs):.2f}, std={np.std(BolT_test_aucs):.2f}")
print(f"BolT: mean test                Loss={np.mean(BolT_test_losses):.3f}, std={np.std(BolT_test_losses):.3f}")
print(f"BolT: mean test                f1={np.mean(BolT_test_f1):.2f}, std={np.std(BolT_test_f1):.2f}")
print(f"BolT: mean ext. validation     AUC= {np.mean(BolT_val_aucs_ext):.2f}, std={np.std(BolT_val_aucs_ext):.2f}")
print(f"BolT: mean ext. validation     Loss={np.mean(BolT_val_losses_ext):.3f}, std={np.std(BolT_val_losses_ext):.3f}")
print(f"BolT: mean ext. validation     f1={np.mean(BolT_val_f1_ext):.2f}, std={np.std(BolT_val_f1_ext):.2f}")
print(f"BolT: mean ext. test           AUC= {np.mean(BolT_ext_test_aucs):.2f}, std={np.std(BolT_ext_test_aucs):.2f}")
print(f"BolT: mean ext. test           Loss={np.mean(BolT_ext_test_losses):.3f}, std={np.std(BolT_ext_test_losses):.3f}")
print(f"BolT: mean ext. test           f1={np.mean(BolT_ext_test_f1):.2f}, std={np.std(BolT_ext_test_f1):.2f}")
print('')
print(f"BNT: mean validation           AUC= {np.mean(BNT_val_aucs):.2f}, std={np.std(BNT_val_aucs):.2f}")
print(f"BNT: mean validation           Loss={np.mean(BNT_val_losses):.3f}, std={np.std(BNT_val_losses):.3f}")
print(f"BNT: mean validation           f1={np.mean(BNT_val_f1):.2f}, std={np.std(BNT_val_f1):.2f}")
print(f"BNT: mean test                 AUC= {np.mean(BNT_test_aucs):.2f}, std={np.std(BNT_test_aucs):.2f}")
print(f"BNT: mean test                 Loss={np.mean(BNT_test_losses):.3f}, std={np.std(BNT_test_losses):.3f}")
print(f"BNT: mean test                 f1={np.mean(BNT_test_f1):.2f}, std={np.std(BNT_test_f1):.2f}")
print(f"BNT: mean ext. validation      AUC= {np.mean(BNT_val_aucs_ext):.2f}, std={np.std(BNT_val_aucs_ext):.2f}")
print(f"BNT: mean ext. validation      Loss={np.mean(BNT_val_losses_ext):.3f}, std={np.std(BNT_val_losses_ext):.3f}")
print(f"BNT: mean ext. validation      f1={np.mean(BNT_val_f1_ext):.2f}, std={np.std(BNT_val_f1_ext):.2f}")
print(f"BNT: mean ext. test            AUC= {np.mean(BNT_ext_test_aucs):.2f}, std={np.std(BNT_ext_test_aucs):.2f}")
print(f"BNT: mean ext. test            Loss={np.mean(BNT_ext_test_losses):.3f}, std={np.std(BNT_ext_test_losses):.3f}")
print(f"BNT: mean ext. test            f1={np.mean(BNT_ext_test_f1):.2f}, std={np.std(BNT_ext_test_f1):.2f}")
print('')
print(f"CVFormer: mean validation      AUC= {np.mean(CVFormer_val_aucs):.2f}, std={np.std(CVFormer_val_aucs):.2f}")
print(f"CVFormer: mean validation      Loss={np.mean(CVFormer_val_losses):.3f}, std={np.std(CVFormer_val_losses):.3f}")
print(f"CVFormer: mean validation      f1={np.mean(CVFormer_val_f1):.2f}, std={np.std(CVFormer_val_f1):.2f}")
print(f"CVFormer: mean test            AUC= {np.mean(CVFormer_test_aucs):.2f}, std={np.std(CVFormer_test_aucs):.2f}")
print(f"CVFormer: mean test            Loss={np.mean(CVFormer_test_losses):.3f}, std={np.std(CVFormer_test_losses):.3f}")
print(f"CVFormer: mean test            f1={np.mean(CVFormer_test_f1):.2f}, std={np.std(CVFormer_test_f1):.2f}")
print(f"CVFormer: mean ext. validation AUC= {np.mean(CVFormer_val_aucs_ext):.2f}, std={np.std(CVFormer_val_aucs_ext):.2f}")
print(f"CVFormer: mean ext. validation Loss={np.mean(CVFormer_val_losses_ext):.3f}, std={np.std(CVFormer_val_losses_ext):.3f}")
print(f"CVFormer: mean ext. validation f1={np.mean(CVFormer_val_f1_ext):.2f}, std={np.std(CVFormer_val_f1_ext):.2f}")
print(f"CVFormer: mean ext. test       AUC= {np.mean(CVFormer_ext_test_aucs):.2f}, std={np.std(CVFormer_ext_test_aucs):.2f}")
print(f"CVFormer: mean ext. test       Loss={np.mean(CVFormer_ext_test_losses):.3f}, std={np.std(CVFormer_ext_test_losses):.3f}")
print(f"CVFormer: mean ext. test       f1={np.mean(CVFormer_ext_test_f1):.2f}, std={np.std(CVFormer_ext_test_f1):.2f}")
print('')
print(f"FBNETGEN: mean validation      AUC= {np.mean(FBNETGEN_val_aucs):.2f}, std={np.std(FBNETGEN_val_aucs):.2f}")
print(f"FBNETGEN: mean validation      Loss={np.mean(FBNETGEN_val_losses):.3f}, std={np.std(FBNETGEN_val_losses):.3f}")
print(f"FBNETGEN: mean validation      f1={np.mean(FBNETGEN_val_f1):.2f}, std={np.std(FBNETGEN_val_f1):.2f}")
print(f"FBNETGEN: mean test            AUC= {np.mean(FBNETGEN_test_aucs):.2f}, std={np.std(FBNETGEN_test_aucs):.2f}")
print(f"FBNETGEN: mean test            Loss={np.mean(FBNETGEN_test_losses):.3f}, std={np.std(FBNETGEN_test_losses):.3f}")
print(f"FBNETGEN: mean test            f1={np.mean(FBNETGEN_test_f1):.2f}, std={np.std(FBNETGEN_test_f1):.2f}")
print(f"FBNETGEN: mean ext. validation AUC= {np.mean(FBNETGEN_val_aucs_ext):.2f}, std={np.std(FBNETGEN_val_aucs_ext):.2f}")
print(f"FBNETGEN: mean ext. validation Loss={np.mean(FBNETGEN_val_losses_ext):.3f}, std={np.std(FBNETGEN_val_losses_ext):.3f}")
print(f"FBNETGEN: mean ext. validation f1={np.mean(FBNETGEN_val_f1_ext):.2f}, std={np.std(FBNETGEN_val_f1_ext):.2f}")
print(f"FBNETGEN: mean ext. test       AUC= {np.mean(FBNETGEN_ext_test_aucs):.2f}, std={np.std(FBNETGEN_ext_test_aucs):.2f}")
print(f"FBNETGEN: mean ext. test       Loss={np.mean(FBNETGEN_ext_test_losses):.3f}, std={np.std(FBNETGEN_ext_test_losses):.3f}")
print(f"FBNETGEN: mean ext. test       f1={np.mean(FBNETGEN_ext_test_f1):.2f}, std={np.std(FBNETGEN_ext_test_f1):.2f}")
print('')
print(f"DeepFMRI: mean validation      AUC= {np.mean(DeepFMRI_val_aucs):.2f}, std={np.std(DeepFMRI_val_aucs):.2f}")
print(f"DeepFMRI: mean validation      Loss={np.mean(DeepFMRI_val_losses):.3f}, std={np.std(DeepFMRI_val_losses):.3f}")
print(f"DeepFMRI: mean validation      f1={np.mean(DeepFMRI_val_f1):.2f}, std={np.std(DeepFMRI_val_f1):.2f}")
print(f"DeepFMRI: mean test            AUC= {np.mean(DeepFMRI_test_aucs):.2f}, std={np.std(DeepFMRI_test_aucs):.2f}")
print(f"DeepFMRI: mean test            Loss={np.mean(DeepFMRI_test_losses):.3f}, std={np.std(DeepFMRI_test_losses):.3f}")
print(f"DeepFMRI: mean test            f1={np.mean(DeepFMRI_test_f1):.2f}, std={np.std(DeepFMRI_test_f1):.2f}")
print(f"DeepFMRI: mean ext. validation AUC= {np.mean(DeepFMRI_val_aucs_ext):.2f}, std={np.std(DeepFMRI_val_aucs_ext):.2f}")
print(f"DeepFMRI: mean ext. validation Loss={np.mean(DeepFMRI_val_losses_ext):.3f}, std={np.std(DeepFMRI_val_losses_ext):.3f}")
print(f"DeepFMRI: mean ext. validation f1={np.mean(DeepFMRI_val_f1_ext):.2f}, std={np.std(DeepFMRI_val_f1_ext):.2f}")
print(f"DeepFMRI: mean ext. test       AUC= {np.mean(DeepFMRI_ext_test_aucs):.2f}, std={np.std(DeepFMRI_ext_test_aucs):.2f}")
print(f"DeepFMRI: mean ext. test       Loss={np.mean(DeepFMRI_ext_test_losses):.3f}, std={np.std(DeepFMRI_ext_test_losses):.3f}")
print(f"DeepFMRI: mean ext. test       f1={np.mean(DeepFMRI_ext_test_f1):.2f}, std={np.std(DeepFMRI_ext_test_f1):.2f}")
print('')
print(f"UCGL: mean validation          AUC= {np.mean(UCGL_val_aucs):.2f}, std={np.std(UCGL_val_aucs):.2f}")
print(f"UCGL: mean validation          Loss={np.mean(UCGL_val_losses):.3f}, std={np.std(UCGL_val_losses):.3f}")
print(f"UCGL: mean validation          f1={np.mean(UCGL_val_f1):.2f}, std={np.std(UCGL_val_f1):.2f}")
print(f"UCGL: mean test                AUC= {np.mean(UCGL_test_aucs):.2f}, std={np.std(UCGL_test_aucs):.2f}")
print(f"UCGL: mean test                Loss={np.mean(UCGL_test_losses):.3f}, std={np.std(UCGL_test_losses):.3f}")
print(f"UCGL: mean test                f1={np.mean(UCGL_test_f1):.2f}, std={np.std(UCGL_test_f1):.2f}")
print(f"UCGL: mean ext. validation     AUC= {np.mean(UCGL_val_aucs_ext):.2f}, std={np.std(UCGL_val_aucs_ext):.2f}")
print(f"UCGL: mean ext. validation     Loss={np.mean(UCGL_val_losses_ext):.3f}, std={np.std(UCGL_val_losses_ext):.3f}")
print(f"UCGL: mean ext. validation     f1={np.mean(UCGL_val_f1_ext):.2f}, std={np.std(UCGL_val_f1_ext):.2f}")
print(f"UCGL: mean ext. test           AUC= {np.mean(UCGL_ext_test_aucs):.2f}, std={np.std(UCGL_ext_test_aucs):.2f}")
print(f"UCGL: mean ext. test           Loss={np.mean(UCGL_ext_test_losses):.3f}, std={np.std(UCGL_ext_test_losses):.3f}")
print(f"UCGL: mean ext. test           f1={np.mean(UCGL_ext_test_f1):.2f}, std={np.std(UCGL_ext_test_f1):.2f}")
print('')
print(f"GCDA: mean validation          AUC= {np.mean(GCDA_val_aucs):.2f}, std={np.std(GCDA_val_aucs):.2f}")
print(f"GCDA: mean validation          Loss={np.mean(GCDA_val_losses):.3f}, std={np.std(GCDA_val_losses):.3f}")
print(f"GCDA: mean validation          f1={np.mean(GCDA_val_f1):.2f}, std={np.std(GCDA_val_f1):.2f}")
print(f"GCDA: mean test                AUC= {np.mean(GCDA_test_aucs):.2f}, std={np.std(GCDA_test_aucs):.2f}")
print(f"GCDA: mean test                Loss={np.mean(GCDA_test_losses):.3f}, std={np.std(GCDA_test_losses):.3f}")
print(f"GCDA: mean test                f1={np.mean(GCDA_test_f1):.2f}, std={np.std(GCDA_test_f1):.2f}")
print(f"GCDA: mean ext. validation     AUC= {np.mean(GCDA_val_aucs_ext):.2f}, std={np.std(GCDA_val_aucs_ext):.2f}")
print(f"GCDA: mean ext. validation     Loss={np.mean(GCDA_val_losses_ext):.3f}, std={np.std(GCDA_val_losses_ext):.3f}")
print(f"GCDA: mean ext. validation     f1={np.mean(GCDA_val_f1_ext):.2f}, std={np.std(GCDA_val_f1_ext):.2f}")
print(f"GCDA: mean ext. test           AUC= {np.mean(GCDA_ext_test_aucs):.2f}, std={np.std(GCDA_ext_test_aucs):.2f}")
print(f"GCDA: mean ext. test           Loss={np.mean(GCDA_ext_test_losses):.3f}, std={np.std(GCDA_ext_test_losses):.3f}")
print(f"GCDA: mean ext. test           f1={np.mean(GCDA_ext_test_f1):.2f}, std={np.std(GCDA_ext_test_f1):.2f}")
print('')
print(f"BAnD: mean validation          AUC= {np.mean(BAnD_val_aucs):.2f}, std={np.std(BAnD_val_aucs):.2f}")
print(f"BAnD: mean validation          Loss={np.mean(BAnD_val_losses):.3f}, std={np.std(BAnD_val_losses):.3f}")
print(f"BAnD: mean validation          f1={np.mean(BAnD_val_f1):.2f}, std={np.std(BAnD_val_f1):.2f}")
print(f"BAnD: mean test                AUC= {np.mean(BAnD_test_aucs):.2f}, std={np.std(BAnD_test_aucs):.2f}")
print(f"BAnD: mean test                Loss={np.mean(BAnD_test_losses):.3f}, std={np.std(BAnD_test_losses):.3f}")
print(f"BAnD: mean test                f1={np.mean(BAnD_test_f1):.2f}, std={np.std(BAnD_test_f1):.2f}")
print(f"BAnD: mean ext. validation     AUC= {np.mean(BAnD_val_aucs_ext):.2f}, std={np.std(BAnD_val_aucs_ext):.2f}")
print(f"BAnD: mean ext. validation     Loss={np.mean(BAnD_val_losses_ext):.3f}, std={np.std(BAnD_val_losses_ext):.3f}")
print(f"BAnD: mean ext. validation     f1={np.mean(BAnD_val_f1_ext):.2f}, std={np.std(BAnD_val_f1_ext):.2f}")
print(f"BAnD: mean ext. test           AUC= {np.mean(BAnD_ext_test_aucs):.2f}, std={np.std(BAnD_ext_test_aucs):.2f}")
print(f"BAnD: mean ext. test           Loss={np.mean(BAnD_ext_test_losses):.3f}, std={np.std(BAnD_ext_test_losses):.3f}")
print(f"BAnD: mean ext. test           f1={np.mean(BAnD_ext_test_f1):.2f}, std={np.std(BAnD_ext_test_f1):.2f}")
print('')
print(f"BrainIB: mean validation          AUC= {np.mean(BrainIB_val_aucs):.2f}, std={np.std(BrainIB_val_aucs):.2f}")
print(f"BrainIB: mean validation          Loss={np.mean(BrainIB_val_losses):.3f}, std={np.std(BrainIB_val_losses):.3f}")
print(f"BrainIB: mean validation          f1={np.mean(BrainIB_val_f1):.2f}, std={np.std(BrainIB_val_f1):.2f}")
print(f"BrainIB: mean test                AUC= {np.mean(BrainIB_test_aucs):.2f}, std={np.std(BrainIB_test_aucs):.2f}")
print(f"BrainIB: mean test                Loss={np.mean(BrainIB_test_losses):.3f}, std={np.std(BrainIB_test_losses):.3f}")
print(f"BrainIB: mean test                f1={np.mean(BrainIB_test_f1):.2f}, std={np.std(BrainIB_test_f1):.2f}")
print(f"BrainIB: mean ext. validation     AUC= {np.mean(BrainIB_val_aucs_ext):.2f}, std={np.std(BrainIB_val_aucs_ext):.2f}")
print(f"BrainIB: mean ext. validation     Loss={np.mean(BrainIB_val_losses_ext):.3f}, std={np.std(BrainIB_val_losses_ext):.3f}")
print(f"BrainIB: mean ext. validation     f1={np.mean(BrainIB_val_f1_ext):.2f}, std={np.std(BrainIB_val_f1_ext):.2f}")
print(f"BrainIB: mean ext. test           AUC= {np.mean(BrainIB_ext_test_aucs):.2f}, std={np.std(BrainIB_ext_test_aucs):.2f}")
print(f"BrainIB: mean ext. test           Loss={np.mean(BrainIB_ext_test_losses):.3f}, std={np.std(BrainIB_ext_test_losses):.3f}")
print(f"BrainIB: mean ext. test           f1={np.mean(BrainIB_ext_test_f1):.2f}, std={np.std(BrainIB_ext_test_f1):.2f}")
print('')
print(f"A-GCL: mean validation          AUC= {np.mean(A_GCL_val_aucs):.2f}, std={np.std(A_GCL_val_aucs):.2f}")
print(f"A-GCL: mean validation          Loss={np.mean(A_GCL_val_losses):.3f}, std={np.std(A_GCL_val_losses):.3f}")
print(f"A-GCL: mean validation          f1={np.mean(A_GCL_val_f1):.2f}, std={np.std(A_GCL_val_f1):.2f}")
print(f"A-GCL: mean test                AUC= {np.mean(A_GCL_test_aucs):.2f}, std={np.std(A_GCL_test_aucs):.2f}")
print(f"A-GCL: mean test                Loss={np.mean(A_GCL_test_losses):.3f}, std={np.std(A_GCL_test_losses):.3f}")
print(f"A-GCL: mean test                f1={np.mean(A_GCL_test_f1):.2f}, std={np.std(A_GCL_test_f1):.2f}")
print(f"A-GCL: mean ext. validation     AUC= {np.mean(A_GCL_val_aucs_ext):.2f}, std={np.std(A_GCL_val_aucs_ext):.2f}")
print(f"A-GCL: mean ext. validation     Loss={np.mean(A_GCL_val_losses_ext):.3f}, std={np.std(A_GCL_val_losses_ext):.3f}")
print(f"A-GCL: mean ext. validation     f1={np.mean(A_GCL_val_f1_ext):.2f}, std={np.std(A_GCL_val_f1_ext):.2f}")
print(f"A-GCL: mean ext. test           AUC= {np.mean(A_GCL_ext_test_aucs):.2f}, std={np.std(A_GCL_ext_test_aucs):.2f}")
print(f"A-GCL: mean ext. test           Loss={np.mean(A_GCL_ext_test_losses):.3f}, std={np.std(A_GCL_ext_test_losses):.3f}")
print(f"A-GCL: mean ext. test           f1={np.mean(A_GCL_ext_test_f1):.2f}, std={np.std(A_GCL_ext_test_f1):.2f}")
print('')
print(f"LFB: mean validation           Loss= {np.mean(LFB_val_losses):.3f}, std={np.std(LFB_val_losses):.3f}")
print(f"LFB: mean test                 AUC= {np.mean(LFB_test_aucs):.2f}, std={np.std(LFB_test_aucs):.2f}")
print(f"LFB: mean test                 Loss={np.mean(LFB_test_losses):.3f}, std={np.std(LFB_test_losses):.3f}")
print(f"LFB: mean test                 f1={np.mean(LFB_test_f1):.2f}, std={np.std(LFB_test_f1):.2f}")
print(f"LFB: mean ext. validation      Loss={np.mean(LFB_val_losses_ext):.3f}, std={np.std(LFB_val_losses_ext):.3f}")
print(f"LFB: mean ext. test            AUC= {np.mean(LFB_ext_test_aucs):.2f}, std={np.std(LFB_ext_test_aucs):.2f}")
print(f"LFB: mean ext. test            Loss={np.mean(LFB_ext_test_losses):.3f}, std={np.std(LFB_ext_test_losses):.3f}")
print(f"LFB: mean ext. test            f1={np.mean(LFB_ext_test_f1):.2f}, std={np.std(LFB_ext_test_f1):.2f}")
print('')
print('')

results = {
    # BolT
    ("BolT", "AUC"): ttest_ind(VarCoNet_test_aucs, BolT_test_aucs),
    ("BolT", "Loss"): ttest_ind(VarCoNet_test_losses, BolT_test_losses),
    ("BolT", "F1"): ttest_ind(VarCoNet_test_f1, BolT_test_f1),
    ("BolT", "AUC-ext"): ttest_ind(VarCoNet_ext_test_aucs, BolT_ext_test_aucs),
    ("BolT", "Loss-ext"): ttest_ind(VarCoNet_ext_test_losses, BolT_ext_test_losses),
    ("BolT", "F1-ext"): ttest_ind(VarCoNet_ext_test_f1, BolT_ext_test_f1),

    # BNT
    ("BNT", "AUC"): ttest_ind(VarCoNet_test_aucs, BNT_test_aucs),
    ("BNT", "Loss"): ttest_ind(VarCoNet_test_losses, BNT_test_losses),
    ("BNT", "F1"): ttest_ind(VarCoNet_test_f1, BNT_test_f1),
    ("BNT", "AUC-ext"): ttest_ind(VarCoNet_ext_test_aucs, BNT_ext_test_aucs),
    ("BNT", "Loss-ext"): ttest_ind(VarCoNet_ext_test_losses, BNT_ext_test_losses),
    ("BNT", "F1-ext"): ttest_ind(VarCoNet_ext_test_f1, BNT_ext_test_f1),

    # CVFormer
    ("CVFormer", "AUC"): ttest_ind(VarCoNet_test_aucs, CVFormer_test_aucs),
    ("CVFormer", "Loss"): ttest_ind(VarCoNet_test_losses, CVFormer_test_losses),
    ("CVFormer", "F1"): ttest_ind(VarCoNet_test_f1, CVFormer_test_f1),
    ("CVFormer", "AUC-ext"): ttest_ind(VarCoNet_ext_test_aucs, CVFormer_ext_test_aucs),
    ("CVFormer", "Loss-ext"): ttest_ind(VarCoNet_ext_test_losses, CVFormer_ext_test_losses),
    ("CVFormer", "F1-ext"): ttest_ind(VarCoNet_ext_test_f1, CVFormer_ext_test_f1),

    # FBNETGEN
    ("FBNETGEN", "AUC"): ttest_ind(VarCoNet_test_aucs, FBNETGEN_test_aucs),
    ("FBNETGEN", "Loss"): ttest_ind(VarCoNet_test_losses, FBNETGEN_test_losses),
    ("FBNETGEN", "F1"): ttest_ind(VarCoNet_test_f1, FBNETGEN_test_f1),
    ("FBNETGEN", "AUC-ext"): ttest_ind(VarCoNet_ext_test_aucs, FBNETGEN_ext_test_aucs),
    ("FBNETGEN", "Loss-ext"): ttest_ind(VarCoNet_ext_test_losses, FBNETGEN_ext_test_losses),
    ("FBNETGEN", "F1-ext"): ttest_ind(VarCoNet_ext_test_f1, FBNETGEN_ext_test_f1),

    # DeepFMRI
    ("DeepFMRI", "AUC"): ttest_ind(VarCoNet_test_aucs, DeepFMRI_test_aucs),
    ("DeepFMRI", "Loss"): ttest_ind(VarCoNet_test_losses, DeepFMRI_test_losses),
    ("DeepFMRI", "F1"): ttest_ind(VarCoNet_test_f1, DeepFMRI_test_f1),
    ("DeepFMRI", "AUC-ext"): ttest_ind(VarCoNet_ext_test_aucs, DeepFMRI_ext_test_aucs),
    ("DeepFMRI", "Loss-ext"): ttest_ind(VarCoNet_ext_test_losses, DeepFMRI_ext_test_losses),
    ("DeepFMRI", "F1-ext"): ttest_ind(VarCoNet_ext_test_f1, DeepFMRI_ext_test_f1),
    
    # BrainIB
    ("BrainIB", "AUC"): ttest_ind(VarCoNet_test_aucs, BrainIB_test_aucs),
    ("BrainIB", "Loss"): ttest_ind(VarCoNet_test_losses, BrainIB_test_losses),
    ("BrainIB", "F1"): ttest_ind(VarCoNet_test_f1, BrainIB_test_f1),
    ("BrainIB", "AUC-ext"): ttest_ind(VarCoNet_ext_test_aucs, BrainIB_ext_test_aucs),
    ("BrainIB", "Loss-ext"): ttest_ind(VarCoNet_ext_test_losses, BrainIB_ext_test_losses),
    ("BrainIB", "F1-ext"): ttest_ind(VarCoNet_ext_test_f1, BrainIB_ext_test_f1),

    # UCGL
    ("UCGL", "AUC"): ttest_ind(VarCoNet_test_aucs, UCGL_test_aucs),
    ("UCGL", "Loss"): ttest_ind(VarCoNet_test_losses, UCGL_test_losses),
    ("UCGL", "F1"): ttest_ind(VarCoNet_test_f1, UCGL_test_f1),
    ("UCGL", "AUC-ext"): ttest_ind(VarCoNet_ext_test_aucs, UCGL_ext_test_aucs),
    ("UCGL", "Loss-ext"): ttest_ind(VarCoNet_ext_test_losses, UCGL_ext_test_losses),
    ("UCGL", "F1-ext"): ttest_ind(VarCoNet_ext_test_f1, UCGL_ext_test_f1),

    # GCDA
    ("GCDA", "AUC"): ttest_ind(VarCoNet_test_aucs, GCDA_test_aucs),
    ("GCDA", "Loss"): ttest_ind(VarCoNet_test_losses, GCDA_test_losses),
    ("GCDA", "F1"): ttest_ind(VarCoNet_test_f1, GCDA_test_f1),
    ("GCDA", "AUC-ext"): ttest_ind(VarCoNet_ext_test_aucs, GCDA_ext_test_aucs),
    ("GCDA", "Loss-ext"): ttest_ind(VarCoNet_ext_test_losses, GCDA_ext_test_losses),
    ("GCDA", "F1-ext"): ttest_ind(VarCoNet_ext_test_f1, GCDA_ext_test_f1),

    # BAnD
    ("BAnD", "AUC"): ttest_ind(VarCoNet_test_aucs, BAnD_test_aucs),
    ("BAnD", "Loss"): ttest_ind(VarCoNet_test_losses, BAnD_test_losses),
    ("BAnD", "F1"): ttest_ind(VarCoNet_test_f1, BAnD_test_f1),
    ("BAnD", "AUC-ext"): ttest_ind(VarCoNet_ext_test_aucs, BAnD_ext_test_aucs),
    ("BAnD", "Loss-ext"): ttest_ind(VarCoNet_ext_test_losses, BAnD_ext_test_losses),
    ("BAnD", "F1-ext"): ttest_ind(VarCoNet_ext_test_f1, BAnD_ext_test_f1),
    
    # A-GCL
    ("A-GCL", "AUC"): ttest_ind(VarCoNet_test_aucs, A_GCL_test_aucs),
    ("A-GCL", "Loss"): ttest_ind(VarCoNet_test_losses, A_GCL_test_losses),
    ("A-GCL", "F1"): ttest_ind(VarCoNet_test_f1, A_GCL_test_f1),
    ("A-GCL", "AUC-ext"): ttest_ind(VarCoNet_ext_test_aucs, A_GCL_ext_test_aucs),
    ("A-GCL", "Loss-ext"): ttest_ind(VarCoNet_ext_test_losses, A_GCL_ext_test_losses),
    ("A-GCL", "F1-ext"): ttest_ind(VarCoNet_ext_test_f1, A_GCL_ext_test_f1),

    # LFB
    ("LFB", "AUC"): ttest_ind(VarCoNet_test_aucs, LFB_test_aucs),
    ("LFB", "Loss"): ttest_ind(VarCoNet_test_losses, LFB_test_losses),
    ("LFB", "F1"): ttest_ind(VarCoNet_test_f1, LFB_test_f1),
    ("LFB", "AUC-ext"): ttest_ind(VarCoNet_ext_test_aucs, LFB_ext_test_aucs),
    ("LFB", "Loss-ext"): ttest_ind(VarCoNet_ext_test_losses, LFB_ext_test_losses),
    ("LFB", "F1-ext"): ttest_ind(VarCoNet_ext_test_f1, LFB_ext_test_f1),
}

# Print neatly
count = 0
for (model, metric), res in results.items():
    count += 1
    print(f"VarCoNet vs {model:8s} ({metric:8s}) p-val={res.pvalue:.4e}")
    if count % 6 == 0:
        print('')


'''---------------------------------------Testing on ABIDE II------------------------------------------'''
if atlas == 'AICHA' or atlas == 'AAL':
    with open(os.path.join(save_path,'results_ABIDEII',atlas,'ABIDEII_results.pkl'), 'rb') as f:
        test_result_ABIDEII = pickle.load(f)
    test_result_BolT_ext = test_result_ABIDEII['BolT']
    test_result_VarCoNet_ext = test_result_ABIDEII['VarCoNet']
    
    BolT_test_losses = np.array(test_result_BolT_ext['test_losses'])
    BolT_test_aucs = np.array(test_result_BolT_ext['test_aucs'])
    test_probs = test_result_BolT_ext['test_probs']
    y_test = test_result_BolT_ext['y_test']
    BolT_test_f1 = np.zeros((len(BolT_test_losses),))
    for i in range(len(BolT_test_losses)):
        test_prob = test_probs[i]
        fpr, tpr, thresholds = roc_curve(y_test, test_prob)
        youden_j = tpr - fpr
        best_index = np.argmax(youden_j)
        best_threshold = thresholds[best_index]
        test_prob[test_prob<best_threshold] = 0
        test_prob[test_prob>=best_threshold] = 1
        BolT_test_f1[i] = f1_score(y_test, test_prob)
        
    VarCoNet_test_losses = np.array(test_result_VarCoNet_ext['test_losses'])
    VarCoNet_test_aucs = np.array(test_result_VarCoNet_ext['test_aucs'])
    test_probs = test_result_VarCoNet_ext['test_probs']
    y_test = test_result_VarCoNet_ext['y_test']
    VarCoNet_test_f1 = np.zeros((len(VarCoNet_test_losses),))
    for i in range(len(VarCoNet_test_losses)):
        test_prob = test_probs[i]
        fpr, tpr, thresholds = roc_curve(y_test, test_prob)
        youden_j = tpr - fpr
        best_index = np.argmax(youden_j)
        best_threshold = thresholds[best_index]
        test_prob[test_prob<best_threshold] = 0
        test_prob[test_prob>=best_threshold] = 1
        VarCoNet_test_f1[i] = f1_score(y_test, test_prob)
    
    
    BolT_test_aucs = BolT_test_aucs*100
    BolT_test_f1 = BolT_test_f1*100
    VarCoNet_test_aucs = VarCoNet_test_aucs*100
    VarCoNet_test_f1 = VarCoNet_test_f1*100
    
    
    print('-----------------------Test on ABIDE II-----------------------')
    print('---------------------------' + atlas + '---------------------------' )
    print('')
    print(f"BolT: mean test           AUC= {np.mean(BolT_test_aucs):.2f}, std={np.std(BolT_test_aucs):.2f}")
    print(f"BolT: mean test           Loss={np.mean(BolT_test_losses):.3f}, std={np.std(BolT_test_losses):.3f}")
    print(f"BolT: mean test           f1={np.mean(BolT_test_f1):.2f}, std={np.std(BolT_test_f1):.2f}")
    print('')
    print(f"VarCoNet: mean test       AUC= {np.mean(VarCoNet_test_aucs):.2f}, std={np.std(VarCoNet_test_aucs):.2f}")
    print(f"VarCoNet: mean test       Loss={np.mean(VarCoNet_test_losses):.3f}, std={np.std(VarCoNet_test_losses):.3f}")
    print(f"VarCoNet: mean test       f1={np.mean(VarCoNet_test_f1):.2f}, std={np.std(VarCoNet_test_f1):.2f}")
