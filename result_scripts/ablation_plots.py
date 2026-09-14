import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
from sklearn.metrics import f1_score,roc_curve
import pandas as pd
import seaborn as sns
from statannotations.Annotator import Annotator

path_results = r'' #here, enter the path where results have been saved

if not os.path.exists(os.path.join(path_results,"plots")):
    os.mkdir(os.path.join(path_results,"plots"))
    
with open(os.path.join(path_results,'results_ABIDEI','AICHA','ABIDEI_VarCoNet_results.pkl'), 'rb') as f:
    ASD_result_VarCoNet_aicha = pickle.load(f)
with open(os.path.join(path_results,'results_ABIDEI','AAL','ABIDEI_VarCoNet_results.pkl'), 'rb') as f:
    ASD_result_VarCoNet_aal = pickle.load(f)
with open(os.path.join(path_results,'ablations','results_ABIDEI','ABIDEI_VarCoNet_ablations.pkl'), 'rb') as f:
    ASD_result_VarCoNet_ablations = pickle.load(f)

val_losses = np.array(ASD_result_VarCoNet_ablations['no_SSL']['AICHA']['val_losses'])
test_losses = np.array(ASD_result_VarCoNet_ablations['no_SSL']['AICHA']['test_losses'])
test_aucs = np.array(ASD_result_VarCoNet_ablations['no_SSL']['AICHA']['test_aucs'])
test_probs = ASD_result_VarCoNet_ablations['no_SSL']['AICHA']['test_probs']
y_tests = ASD_result_VarCoNet_ablations['no_SSL']['AICHA']['y_test']
min_indices = np.argmin(val_losses, axis=1)

SL_AICHA_test_f1 = np.zeros((len(val_losses),))
for i in range(len(val_losses)):
    test_prob = test_probs[i][min_indices[i]]
    y_test = y_tests[i][min_indices[i]]
    fpr, tpr, thresholds = roc_curve(y_test, test_prob)
    youden_j = tpr - fpr
    best_index = np.argmax(youden_j)
    best_threshold = thresholds[best_index]
    test_prob[test_prob<best_threshold] = 0
    test_prob[test_prob>=best_threshold] = 1
    SL_AICHA_test_f1[i] = f1_score(y_test, test_prob)

SL_AICHA_test_losses = np.take_along_axis(test_losses, min_indices[:, None], axis=1).squeeze()
SL_AICHA_test_aucs = np.take_along_axis(test_aucs, min_indices[:, None], axis=1).squeeze() 


val_losses = np.array(ASD_result_VarCoNet_ablations['no_SSL']['AAL']['val_losses'])
test_losses = np.array(ASD_result_VarCoNet_ablations['no_SSL']['AAL']['test_losses'])
test_aucs = np.array(ASD_result_VarCoNet_ablations['no_SSL']['AAL']['test_aucs'])
test_probs = ASD_result_VarCoNet_ablations['no_SSL']['AAL']['test_probs']
y_tests = ASD_result_VarCoNet_ablations['no_SSL']['AAL']['y_test']
min_indices = np.argmin(val_losses, axis=1)

repeats, epochs = val_losses.shape

SL_AAL_test_f1 = np.zeros((len(val_losses),))
for i in range(len(val_losses)):
    test_prob = test_probs[i][min_indices[i]]
    y_test = y_tests[i][min_indices[i]]
    fpr, tpr, thresholds = roc_curve(y_test, test_prob)
    youden_j = tpr - fpr
    best_index = np.argmax(youden_j)
    best_threshold = thresholds[best_index]
    test_prob[test_prob<best_threshold] = 0
    test_prob[test_prob>=best_threshold] = 1
    SL_AAL_test_f1[i] = f1_score(y_test, test_prob)

SL_AAL_test_losses = np.take_along_axis(test_losses, min_indices[:, None], axis=1).squeeze()
SL_AAL_test_aucs = np.take_along_axis(test_aucs, min_indices[:, None], axis=1).squeeze() 


test_losses = np.zeros((repeats,epochs))   
test_aucs = np.zeros((repeats,epochs))  
test_f1 = np.zeros((repeats,epochs))  
for i,test in enumerate(ASD_result_VarCoNet_aicha['epoch_results']):
    for j,test1 in enumerate(test):
        val_losses[i,j] = test1['best_val_loss']
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
        
min_indices = np.argmin(val_losses, axis=1)
        
aicha_test_losses = np.take_along_axis(test_losses, min_indices[:, None], axis=1).squeeze()
aicha_test_aucs = np.take_along_axis(test_aucs, min_indices[:, None], axis=1).squeeze() 
aicha_test_f1 = np.take_along_axis(test_f1, min_indices[:, None], axis=1).squeeze() 


test_losses = np.zeros((repeats,epochs))   
test_aucs = np.zeros((repeats,epochs))  
test_f1 = np.zeros((repeats,epochs))  
for i,test in enumerate(ASD_result_VarCoNet_aal['epoch_results']):
    for j,test1 in enumerate(test):
        val_losses[i,j] = test1['best_val_loss']
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
        
min_indices = np.argmin(val_losses, axis=1)
        
aal_test_losses = np.take_along_axis(test_losses, min_indices[:, None], axis=1).squeeze()
aal_test_aucs = np.take_along_axis(test_aucs, min_indices[:, None], axis=1).squeeze() 
aal_test_f1 = np.take_along_axis(test_f1, min_indices[:, None], axis=1).squeeze() 


test_losses = np.zeros((repeats,epochs))   
test_aucs = np.zeros((repeats,epochs))  
test_f1 = np.zeros((repeats,epochs))  
for i,test in enumerate(ASD_result_VarCoNet_ablations['no_CNN']['AICHA']['epoch_results']):
    for j,test1 in enumerate(test):
        val_losses[i,j] = test1['best_val_loss']
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
        
min_indices = np.argmin(val_losses, axis=1)
        
noCNN_AICHA_test_losses = np.take_along_axis(test_losses, min_indices[:, None], axis=1).squeeze()
noCNN_AICHA_test_aucs = np.take_along_axis(test_aucs, min_indices[:, None], axis=1).squeeze() 
noCNN_AICHA_test_f1 = np.take_along_axis(test_f1, min_indices[:, None], axis=1).squeeze() 


test_losses = np.zeros((repeats,epochs))   
test_aucs = np.zeros((repeats,epochs))  
test_f1 = np.zeros((repeats,epochs))  
for i,test in enumerate(ASD_result_VarCoNet_ablations['no_CNN']['AAL']['epoch_results']):
    for j,test1 in enumerate(test):
        val_losses[i,j] = test1['best_val_loss']
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
        
min_indices = np.argmin(val_losses, axis=1)
        
noCNN_AAL_test_losses = np.take_along_axis(test_losses, min_indices[:, None], axis=1).squeeze()
noCNN_AAL_test_aucs = np.take_along_axis(test_aucs, min_indices[:, None], axis=1).squeeze() 
noCNN_AAL_test_f1 = np.take_along_axis(test_f1, min_indices[:, None], axis=1).squeeze() 


test_losses = np.zeros((repeats,epochs))   
test_aucs = np.zeros((repeats,epochs))  
test_f1 = np.zeros((repeats,epochs))  
for i,test in enumerate(ASD_result_VarCoNet_ablations['no_Transformer']['AICHA']['epoch_results']):
    for j,test1 in enumerate(test):
        val_losses[i,j] = test1['best_val_loss']
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
        
min_indices = np.argmin(val_losses, axis=1)
        
noTransformer_AICHA_test_losses = np.take_along_axis(test_losses, min_indices[:, None], axis=1).squeeze()
noTransformer_AICHA_test_aucs = np.take_along_axis(test_aucs, min_indices[:, None], axis=1).squeeze() 
noTransformer_AICHA_test_f1 = np.take_along_axis(test_f1, min_indices[:, None], axis=1).squeeze() 


test_losses = np.zeros((repeats,epochs))   
test_aucs = np.zeros((repeats,epochs))  
test_f1 = np.zeros((repeats,epochs))  
for i,test in enumerate(ASD_result_VarCoNet_ablations['no_Transformer']['AAL']['epoch_results']):
    for j,test1 in enumerate(test):
        val_losses[i,j] = test1['best_val_loss']
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
        
min_indices = np.argmin(val_losses, axis=1)
        
noTransformer_AAL_test_losses = np.take_along_axis(test_losses, min_indices[:, None], axis=1).squeeze()
noTransformer_AAL_test_aucs = np.take_along_axis(test_aucs, min_indices[:, None], axis=1).squeeze() 
noTransformer_AAL_test_f1 = np.take_along_axis(test_f1, min_indices[:, None], axis=1).squeeze()


test_losses = np.zeros((repeats,epochs))   
test_aucs = np.zeros((repeats,epochs))  
test_f1 = np.zeros((repeats,epochs))  
for i,test in enumerate(ASD_result_VarCoNet_ablations['other_augmentations']['augmentation1']['AICHA']['epoch_results']):
    for j,test1 in enumerate(test):
        val_losses[i,j] = test1['best_val_loss']
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
        
min_indices = np.argmin(val_losses, axis=1)
        
augmentation1_AICHA_test_losses = np.take_along_axis(test_losses, min_indices[:, None], axis=1).squeeze()
augmentation1_AICHA_test_aucs = np.take_along_axis(test_aucs, min_indices[:, None], axis=1).squeeze() 
augmentation1_AICHA_test_f1 = np.take_along_axis(test_f1, min_indices[:, None], axis=1).squeeze() 


test_losses = np.zeros((repeats,epochs))   
test_aucs = np.zeros((repeats,epochs))  
test_f1 = np.zeros((repeats,epochs))  
for i,test in enumerate(ASD_result_VarCoNet_ablations['other_augmentations']['augmentation1']['AAL']['epoch_results']):
    for j,test1 in enumerate(test):
        val_losses[i,j] = test1['best_val_loss']
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
        
min_indices = np.argmin(val_losses, axis=1)
        
augmentation1_AAL_test_losses = np.take_along_axis(test_losses, min_indices[:, None], axis=1).squeeze()
augmentation1_AAL_test_aucs = np.take_along_axis(test_aucs, min_indices[:, None], axis=1).squeeze() 
augmentation1_AAL_test_f1 = np.take_along_axis(test_f1, min_indices[:, None], axis=1).squeeze()


test_losses = np.zeros((repeats,epochs))   
test_aucs = np.zeros((repeats,epochs))  
test_f1 = np.zeros((repeats,epochs))  
for i,test in enumerate(ASD_result_VarCoNet_ablations['other_augmentations']['augmentation2']['AICHA']['epoch_results']):
    for j,test1 in enumerate(test):
        val_losses[i,j] = test1['best_val_loss']
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
        
min_indices = np.argmin(val_losses, axis=1)
        
augmentation2_AICHA_test_losses = np.take_along_axis(test_losses, min_indices[:, None], axis=1).squeeze()
augmentation2_AICHA_test_aucs = np.take_along_axis(test_aucs, min_indices[:, None], axis=1).squeeze() 
augmentation2_AICHA_test_f1 = np.take_along_axis(test_f1, min_indices[:, None], axis=1).squeeze() 


test_losses = np.zeros((repeats,epochs))   
test_aucs = np.zeros((repeats,epochs))  
test_f1 = np.zeros((repeats,epochs))  
for i,test in enumerate(ASD_result_VarCoNet_ablations['other_augmentations']['augmentation2']['AAL']['epoch_results']):
    for j,test1 in enumerate(test):
        val_losses[i,j] = test1['best_val_loss']
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
        
min_indices = np.argmin(val_losses, axis=1)
        
augmentation2_AAL_test_losses = np.take_along_axis(test_losses, min_indices[:, None], axis=1).squeeze()
augmentation2_AAL_test_aucs = np.take_along_axis(test_aucs, min_indices[:, None], axis=1).squeeze() 
augmentation2_AAL_test_f1 = np.take_along_axis(test_f1, min_indices[:, None], axis=1).squeeze()

'''
plt.figure()
y = np.concatenate((aal_test_losses, aicha_test_losses,
                    aal_test_aucs, aicha_test_aucs,
                    aal_test_f1, aicha_test_f1), axis=0)
x = np.concatenate((np.full((2*repeats,), 'loss'),np.full((2*repeats,), 'AUC'),np.full((2*repeats,), 'F1')),axis=0)
hue = np.concatenate((np.full((repeats,), 'AAL'),np.full((repeats,), 'AICHA'),
                      np.full((repeats,), 'AAL'),np.full((repeats,), 'AICHA'),
                      np.full((repeats,), 'AAL'),np.full((repeats,), 'AICHA')),axis=0)

d = {'Score': y, 'Atlas': hue, 'Metric': x}
df = pd.DataFrame(data=d)
sns.barplot(df, x="Metric", y="Score", hue="Atlas")
plt.ylabel('Score', fontsize=17)
plt.xlabel('Metric', fontsize=17)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(loc='lower right', fontsize = 16)
plt.tight_layout()
plt.savefig(os.path.join(path_results,"plots","ABIDE_atlas.png"), dpi=600, bbox_inches='tight')
plt.show()
'''

plt.figure()

y = np.concatenate((noCNN_AICHA_test_losses, aicha_test_losses,
                    noCNN_AICHA_test_aucs, aicha_test_aucs,
                    noCNN_AICHA_test_f1, aicha_test_f1), axis=0)
x = np.concatenate((np.full((2*repeats,), 'Loss'),
                    np.full((2*repeats,), 'AUC'),
                    np.full((2*repeats,), 'F1')), axis=0)
hue = np.concatenate((np.full((repeats,), 'No'),
                      np.full((repeats,), 'Yes'),
                      np.full((repeats,), 'No'),
                      np.full((repeats,), 'Yes'),
                      np.full((repeats,), 'No'),
                      np.full((repeats,), 'Yes')), axis=0)

df = pd.DataFrame({'Score': y, 'With CNN': hue, 'Metric': x})

ax = sns.barplot(df, x="Metric", y="Score", hue="With CNN")
plt.ylabel('Score', fontsize=17)
plt.xlabel('Metric', fontsize=17)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(loc='lower right', fontsize=16)
plt.tight_layout()

pairs = [
    (("Loss", "No"), ("Loss", "Yes")),
    (("AUC", "No"), ("AUC", "Yes")),
    (("F1", "No"), ("F1", "Yes"))
]

# Add statistical annotation
annotator = Annotator(ax, pairs, data=df, x="Metric", y="Score", hue="With CNN")
annotator.configure(test='t-test_ind', text_format='star', loc='inside', verbose=2)
annotator.apply_and_annotate()

plt.savefig(os.path.join(path_results, "plots", "ABIDE_noCNN_AICHA.png"),
            dpi=600, bbox_inches='tight')
plt.show()


plt.figure()
y = np.concatenate((noCNN_AAL_test_losses, aal_test_losses,
                    noCNN_AAL_test_aucs, aal_test_aucs,
                    noCNN_AAL_test_f1, aal_test_f1), axis=0)
x = np.concatenate((np.full((2*repeats,), 'Loss'),np.full((2*repeats,), 'AUC'),np.full((2*repeats,), 'F1')),axis=0)
hue = np.concatenate((np.full((repeats,), 'No'),np.full((repeats,), 'Yes'),
                      np.full((repeats,), 'No'),np.full((repeats,), 'Yes'),
                      np.full((repeats,), 'No'),np.full((repeats,), 'Yes')),axis=0)

d = {'Score': y, 'With CNN': hue, 'Metric': x}
df = pd.DataFrame(data=d)
ax = sns.barplot(df, x="Metric", y="Score", hue="With CNN")
plt.ylabel('Score', fontsize=17)
plt.xlabel('Metric', fontsize=17)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(loc='lower right', fontsize = 16)
plt.tight_layout()

pairs = [
    (("Loss", "No"), ("Loss", "Yes")),
    (("AUC", "No"), ("AUC", "Yes")),
    (("F1", "No"), ("F1", "Yes"))
]

# Add statistical annotation
annotator = Annotator(ax, pairs, data=df, x="Metric", y="Score", hue="With CNN")
annotator.configure(test='t-test_ind', text_format='star', loc='inside', verbose=2)
annotator.apply_and_annotate()

plt.savefig(os.path.join(path_results,"plots","ABIDE_noCNN_AAL.png"), dpi=600, bbox_inches='tight')
plt.show()


plt.figure()
y = np.concatenate((SL_AICHA_test_losses, aicha_test_losses,
                    SL_AICHA_test_aucs, aicha_test_aucs,
                    SL_AICHA_test_f1, aicha_test_f1), axis=0)
x = np.concatenate((np.full((2*repeats,), 'Loss'),np.full((2*repeats,), 'AUC'),np.full((2*repeats,), 'F1')),axis=0)
hue = np.concatenate((np.full((repeats,), 'No'),np.full((repeats,), 'Yes'),
                      np.full((repeats,), 'No'),np.full((repeats,), 'Yes'),
                      np.full((repeats,), 'No'),np.full((repeats,), 'Yes')),axis=0)

d = {'Score': y, 'SSL': hue, 'Metric': x}
df = pd.DataFrame(data=d)
ax = sns.barplot(df, x="Metric", y="Score", hue="SSL")
plt.ylabel('Score', fontsize=17)
plt.xlabel('Metric', fontsize=17)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(loc='lower right', fontsize = 16)
plt.tight_layout()

pairs = [
    (("Loss", "No"), ("Loss", "Yes")),
    (("AUC", "No"), ("AUC", "Yes")),
    (("F1", "No"), ("F1", "Yes"))
]

# Add statistical annotation
annotator = Annotator(ax, pairs, data=df, x="Metric", y="Score", hue="SSL")
annotator.configure(test='t-test_ind', text_format='star', loc='inside', verbose=2)
annotator.apply_and_annotate()

plt.savefig(os.path.join(path_results,"plots","ABIDE_SSL_AICHA.png"), dpi=600, bbox_inches='tight')
plt.show()


plt.figure()
y = np.concatenate((SL_AAL_test_losses, aal_test_losses,
                    SL_AAL_test_aucs, aal_test_aucs,
                    SL_AAL_test_f1, aal_test_f1), axis=0)
x = np.concatenate((np.full((2*repeats,), 'Loss'),np.full((2*repeats,), 'AUC'),np.full((2*repeats,), 'F1')),axis=0)
hue = np.concatenate((np.full((repeats,), 'No'),np.full((repeats,), 'Yes'),
                      np.full((repeats,), 'No'),np.full((repeats,), 'Yes'),
                      np.full((repeats,), 'No'),np.full((repeats,), 'Yes')),axis=0)

d = {'Score': y, 'SSL': hue, 'Metric': x}
df = pd.DataFrame(data=d)
ax = sns.barplot(df, x="Metric", y="Score", hue="SSL")
plt.ylabel('Score', fontsize=17)
plt.xlabel('Metric', fontsize=17)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(loc='lower right', fontsize = 16)
plt.tight_layout()

pairs = [
    (("Loss", "No"), ("Loss", "Yes")),
    (("AUC", "No"), ("AUC", "Yes")),
    (("F1", "No"), ("F1", "Yes"))
]

# Add statistical annotation
annotator = Annotator(ax, pairs, data=df, x="Metric", y="Score", hue="SSL")
annotator.configure(test='t-test_ind', text_format='star', loc='inside', verbose=2)
annotator.apply_and_annotate()

plt.savefig(os.path.join(path_results,"plots","ABIDE_SSL_AAL.png"), dpi=600, bbox_inches='tight')
plt.show()

plt.figure()
y = np.concatenate((noTransformer_AICHA_test_losses, aicha_test_losses,
                    noTransformer_AICHA_test_aucs, aicha_test_aucs,
                    noTransformer_AICHA_test_f1, aicha_test_f1), axis=0)
x = np.concatenate((np.full((2*repeats,), 'Loss'),np.full((2*repeats,), 'AUC'),np.full((2*repeats,), 'F1')),axis=0)
hue = np.concatenate((np.full((repeats,), 'No'),np.full((repeats,), 'Yes'),
                      np.full((repeats,), 'No'),np.full((repeats,), 'Yes'),
                      np.full((repeats,), 'No'),np.full((repeats,), 'Yes')),axis=0)

d = {'Score': y, 'With Transformer': hue, 'Metric': x}
df = pd.DataFrame(data=d)
ax = sns.barplot(df, x="Metric", y="Score", hue="With Transformer")
plt.ylabel('Score', fontsize=17)
plt.xlabel('Metric', fontsize=17)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(loc='lower right', fontsize = 16)
plt.tight_layout()

pairs = [
    (("Loss", "No"), ("Loss", "Yes")),
    (("AUC", "No"), ("AUC", "Yes")),
    (("F1", "No"), ("F1", "Yes"))
]

# Add statistical annotation
annotator = Annotator(ax, pairs, data=df, x="Metric", y="Score", hue="With Transformer")
annotator.configure(test='t-test_ind', text_format='star', loc='inside', verbose=2)
annotator.apply_and_annotate()

plt.savefig(os.path.join(path_results,"plots","ABIDE_noTransformer_AICHA.png"), dpi=600, bbox_inches='tight')
plt.show()

plt.figure()
y = np.concatenate((noTransformer_AAL_test_losses, aal_test_losses,
                    noTransformer_AAL_test_aucs, aal_test_aucs,
                    noTransformer_AAL_test_f1, aal_test_f1), axis=0)
x = np.concatenate((np.full((2*repeats,), 'Loss'),np.full((2*repeats,), 'AUC'),np.full((2*repeats,), 'F1')),axis=0)
hue = np.concatenate((np.full((repeats,), 'No'),np.full((repeats,), 'Yes'),
                      np.full((repeats,), 'No'),np.full((repeats,), 'Yes'),
                      np.full((repeats,), 'No'),np.full((repeats,), 'Yes')),axis=0)

d = {'Score': y, 'With Transformer': hue, 'Metric': x}
df = pd.DataFrame(data=d)
ax = sns.barplot(df, x="Metric", y="Score", hue="With Transformer")
plt.ylabel('Score', fontsize=17)
plt.xlabel('Metric', fontsize=17)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(loc='lower right', fontsize = 16)
plt.tight_layout()

pairs = [
    (("Loss", "No"), ("Loss", "Yes")),
    (("AUC", "No"), ("AUC", "Yes")),
    (("F1", "No"), ("F1", "Yes"))
]

# Add statistical annotation
annotator = Annotator(ax, pairs, data=df, x="Metric", y="Score", hue="With Transformer")
annotator.configure(test='t-test_ind', text_format='star', loc='inside', verbose=2)
annotator.apply_and_annotate()

plt.savefig(os.path.join(path_results,"plots","ABIDE_noTransformer_AAL.png"), dpi=600, bbox_inches='tight')
plt.show()

plt.figure()
y = np.concatenate((augmentation2_AICHA_test_losses, aicha_test_losses,
                    augmentation2_AICHA_test_aucs, aicha_test_aucs,
                    augmentation2_AICHA_test_f1, aicha_test_f1), axis=0)
x = np.concatenate((np.full((2*repeats,), 'Loss'),np.full((2*repeats,), 'AUC'),np.full((2*repeats,), 'F1')),axis=0)
hue = np.concatenate((np.full((repeats,), 'UCGL'),np.full((repeats,), 'VarCoNet'),
                      np.full((repeats,), 'UCGL'),np.full((repeats,), 'VarCoNet'),
                      np.full((repeats,), 'UCGL'),np.full((repeats,), 'VarCoNet')),axis=0)

d = {'Score': y, 'Aug. strategy': hue, 'Metric': x}
df = pd.DataFrame(data=d)
ax = sns.barplot(df, x="Metric", y="Score", hue="Aug. strategy")
plt.ylabel('Score', fontsize=17)
plt.xlabel('Metric', fontsize=17)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(loc='lower right', fontsize = 16)
plt.tight_layout()

pairs = [
    (("Loss", "UCGL"), ("Loss", "VarCoNet")),
    (("AUC", "UCGL"), ("AUC", "VarCoNet")),
    (("F1", "UCGL"), ("F1", "VarCoNet"))
]

# Add statistical annotation
annotator = Annotator(ax, pairs, data=df, x="Metric", y="Score", hue="Aug. strategy")
annotator.configure(test='t-test_ind', text_format='star', loc='inside', verbose=2)
annotator.apply_and_annotate()

plt.savefig(os.path.join(path_results,"plots","ABIDE_augment_strategies_AICHA.png"), dpi=600, bbox_inches='tight')
plt.show()

plt.figure()
y = np.concatenate((augmentation2_AAL_test_losses, aal_test_losses,
                    augmentation2_AAL_test_aucs, aal_test_aucs,
                    augmentation2_AAL_test_f1, aal_test_f1), axis=0)
x = np.concatenate((np.full((2*repeats,), 'Loss'),np.full((2*repeats,), 'AUC'),np.full((2*repeats,), 'F1')),axis=0)
hue = np.concatenate((np.full((repeats,), 'UCGL'),np.full((repeats,), 'VarCoNet'),
                      np.full((repeats,), 'UCGL'),np.full((repeats,), 'VarCoNet'),
                      np.full((repeats,), 'UCGL'),np.full((repeats,), 'VarCoNet')),axis=0)

d = {'Score': y, 'Aug. strategy': hue, 'Metric': x}
df = pd.DataFrame(data=d)
ax = sns.barplot(df, x="Metric", y="Score", hue="Aug. strategy")
plt.ylabel('Score', fontsize=17)
plt.xlabel('Metric', fontsize=17)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(loc='lower right', fontsize = 16)
plt.tight_layout()

pairs = [
    (("Loss", "UCGL"), ("Loss", "VarCoNet")),
    (("AUC", "UCGL"), ("AUC", "VarCoNet")),
    (("F1", "UCGL"), ("F1", "VarCoNet"))
]

# Add statistical annotation
annotator = Annotator(ax, pairs, data=df, x="Metric", y="Score", hue="Aug. strategy")
annotator.configure(test='t-test_ind', text_format='star', loc='inside', verbose=2)
annotator.apply_and_annotate()

plt.savefig(os.path.join(path_results,"plots","ABIDE_augment_strategies_AAL.png"), dpi=600, bbox_inches='tight')
plt.show()



with open(os.path.join(path_results,'ablations','results_HCP','HCP_VarCoNet_ablations.pkl'), 'rb') as f:
    fingerprinting_result_VarCoNet_ablations = pickle.load(f)
with open(os.path.join(path_results,'results_HCP','AAL','HCP_VarCoNet_results.pkl'), 'rb') as f:
    fingerprinting_result_VarCoNet_aal = pickle.load(f)
with open(os.path.join(path_results,'results_HCP','AICHA','HCP_VarCoNet_results.pkl'), 'rb') as f:
    fingerprinting_result_VarCoNet_aicha = pickle.load(f)
    
AAL_0_0 = fingerprinting_result_VarCoNet_aal['test_result'][0][0]
AAL_0_1 = fingerprinting_result_VarCoNet_aal['test_result'][0][1]
AAL_0_2 = fingerprinting_result_VarCoNet_aal['test_result'][0][2]
AAL_1_1 = fingerprinting_result_VarCoNet_aal['test_result'][0][3]
AAL_1_2 = fingerprinting_result_VarCoNet_aal['test_result'][0][4]
AAL_2_2 = fingerprinting_result_VarCoNet_aal['test_result'][0][5]

AAL_noCNN_0_0 = fingerprinting_result_VarCoNet_ablations['no_CNN']['AAL'][0][0]
AAL_noCNN_0_1 = fingerprinting_result_VarCoNet_ablations['no_CNN']['AAL'][0][1]
AAL_noCNN_0_2 = fingerprinting_result_VarCoNet_ablations['no_CNN']['AAL'][0][2]
AAL_noCNN_1_1 = fingerprinting_result_VarCoNet_ablations['no_CNN']['AAL'][0][3]
AAL_noCNN_1_2 = fingerprinting_result_VarCoNet_ablations['no_CNN']['AAL'][0][4]
AAL_noCNN_2_2 = fingerprinting_result_VarCoNet_ablations['no_CNN']['AAL'][0][5]

AAL_noTransformer_0_0 = fingerprinting_result_VarCoNet_ablations['no_Transformer']['AAL'][0][0]
AAL_noTransformer_0_1 = fingerprinting_result_VarCoNet_ablations['no_Transformer']['AAL'][0][1]
AAL_noTransformer_0_2 = fingerprinting_result_VarCoNet_ablations['no_Transformer']['AAL'][0][2]
AAL_noTransformer_1_1 = fingerprinting_result_VarCoNet_ablations['no_Transformer']['AAL'][0][3]
AAL_noTransformer_1_2 = fingerprinting_result_VarCoNet_ablations['no_Transformer']['AAL'][0][4]
AAL_noTransformer_2_2 = fingerprinting_result_VarCoNet_ablations['no_Transformer']['AAL'][0][5]

AAL_augmentation2_0_0 = fingerprinting_result_VarCoNet_ablations['other_augmentations']['augmentation2']['AAL'][0][0]
AAL_augmentation2_0_1 = fingerprinting_result_VarCoNet_ablations['other_augmentations']['augmentation2']['AAL'][0][1]
AAL_augmentation2_0_2 = fingerprinting_result_VarCoNet_ablations['other_augmentations']['augmentation2']['AAL'][0][2]
AAL_augmentation2_1_1 = fingerprinting_result_VarCoNet_ablations['other_augmentations']['augmentation2']['AAL'][0][3]
AAL_augmentation2_1_2 = fingerprinting_result_VarCoNet_ablations['other_augmentations']['augmentation2']['AAL'][0][4]
AAL_augmentation2_2_2 = fingerprinting_result_VarCoNet_ablations['other_augmentations']['augmentation2']['AAL'][0][5]


AICHA_0_0 = fingerprinting_result_VarCoNet_aicha['test_result'][0][0]
AICHA_0_1 = fingerprinting_result_VarCoNet_aicha['test_result'][0][1]
AICHA_0_2 = fingerprinting_result_VarCoNet_aicha['test_result'][0][2]
AICHA_1_1 = fingerprinting_result_VarCoNet_aicha['test_result'][0][3]
AICHA_1_2 = fingerprinting_result_VarCoNet_aicha['test_result'][0][4]
AICHA_2_2 = fingerprinting_result_VarCoNet_aicha['test_result'][0][5]

AICHA_noCNN_0_0 = fingerprinting_result_VarCoNet_ablations['no_CNN']['AICHA'][0][0]
AICHA_noCNN_0_1 = fingerprinting_result_VarCoNet_ablations['no_CNN']['AICHA'][0][1]
AICHA_noCNN_0_2 = fingerprinting_result_VarCoNet_ablations['no_CNN']['AICHA'][0][2]
AICHA_noCNN_1_1 = fingerprinting_result_VarCoNet_ablations['no_CNN']['AICHA'][0][3]
AICHA_noCNN_1_2 = fingerprinting_result_VarCoNet_ablations['no_CNN']['AICHA'][0][4]
AICHA_noCNN_2_2 = fingerprinting_result_VarCoNet_ablations['no_CNN']['AICHA'][0][5]

AICHA_noTransformer_0_0 = fingerprinting_result_VarCoNet_ablations['no_Transformer']['AICHA'][0][0]
AICHA_noTransformer_0_1 = fingerprinting_result_VarCoNet_ablations['no_Transformer']['AICHA'][0][1]
AICHA_noTransformer_0_2 = fingerprinting_result_VarCoNet_ablations['no_Transformer']['AICHA'][0][2]
AICHA_noTransformer_1_1 = fingerprinting_result_VarCoNet_ablations['no_Transformer']['AICHA'][0][3]
AICHA_noTransformer_1_2 = fingerprinting_result_VarCoNet_ablations['no_Transformer']['AICHA'][0][4]
AICHA_noTransformer_2_2 = fingerprinting_result_VarCoNet_ablations['no_Transformer']['AICHA'][0][5]

AICHA_augmentation2_0_0 = fingerprinting_result_VarCoNet_ablations['other_augmentations']['augmentation2']['AICHA'][0][0]
AICHA_augmentation2_0_1 = fingerprinting_result_VarCoNet_ablations['other_augmentations']['augmentation2']['AICHA'][0][1]
AICHA_augmentation2_0_2 = fingerprinting_result_VarCoNet_ablations['other_augmentations']['augmentation2']['AICHA'][0][2]
AICHA_augmentation2_1_1 = fingerprinting_result_VarCoNet_ablations['other_augmentations']['augmentation2']['AICHA'][0][3]
AICHA_augmentation2_1_2 = fingerprinting_result_VarCoNet_ablations['other_augmentations']['augmentation2']['AICHA'][0][4]
AICHA_augmentation2_2_2 = fingerprinting_result_VarCoNet_ablations['other_augmentations']['augmentation2']['AICHA'][0][5]


num_test_winds = len(AAL_0_0)

plt.figure()
y = np.concatenate((AAL_noCNN_0_0, AAL_noCNN_0_1, AAL_noCNN_0_2,
                    AAL_noCNN_1_1, AAL_noCNN_1_2, AAL_noCNN_2_2,
                    AAL_0_0, AAL_0_1, AAL_0_2, AAL_1_1, AAL_1_2, AAL_2_2), axis=0)
x = np.concatenate((np.full((num_test_winds,), '2-2'),np.full((num_test_winds,), '2-5'),
                    np.full((num_test_winds,), '2-8'),np.full((num_test_winds,), '5-5'),
                    np.full((num_test_winds,), '5-8'),np.full((num_test_winds,), '8-8'),
                    np.full((num_test_winds,), '2-2'),np.full((num_test_winds,), '2-5'),
                    np.full((num_test_winds,), '2-8'),np.full((num_test_winds,), '5-5'),
                    np.full((num_test_winds,), '5-8'),np.full((num_test_winds,), '8-8'),),axis=0)
hue = np.concatenate((np.full((6*num_test_winds,), 'No'), np.full((6*num_test_winds,), 'Yes')),axis=0)

d = {'Fingerprinting Accuracy': y, 'With CNN': hue, 'Duration Combinations (mins)': x}
df = pd.DataFrame(data=d)
ax = sns.barplot(df, x="Duration Combinations (mins)", y="Fingerprinting Accuracy", hue="With CNN")
plt.ylabel('Fingerprinting Accuracy', fontsize=17)
plt.xlabel('Duration Combinations (mins)', fontsize=17)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(loc='lower right', fontsize = 16)
plt.tight_layout()

pairs = [
    (("2-2", "No"), ("2-2", "Yes")),
    (("2-5", "No"), ("2-5", "Yes")),
    (("2-8", "No"), ("2-8", "Yes")),
    (("5-5", "No"), ("5-5", "Yes")),
    (("5-8", "No"), ("5-8", "Yes")),
    (("8-8", "No"), ("8-8", "Yes"))
]

# Add statistical annotation
annotator = Annotator(ax, pairs, data=df, x="Duration Combinations (mins)", y="Fingerprinting Accuracy", hue="With CNN")
annotator.configure(test='t-test_ind', text_format='star', loc='inside', verbose=2)
annotator.apply_and_annotate()

plt.savefig(os.path.join(path_results,"plots","HCP_noCNN_AAL.png"), dpi=600, bbox_inches='tight')
plt.show()


plt.figure()
y = np.concatenate((AICHA_noCNN_0_0, AICHA_noCNN_0_1, AICHA_noCNN_0_2,
                    AICHA_noCNN_1_1, AICHA_noCNN_1_2, AICHA_noCNN_2_2,
                    AICHA_0_0, AICHA_0_1, AICHA_0_2, AICHA_1_1, AICHA_1_2, AICHA_2_2), axis=0)
x = np.concatenate((np.full((num_test_winds,), '2-2'),np.full((num_test_winds,), '2-5'),
                    np.full((num_test_winds,), '2-8'),np.full((num_test_winds,), '5-5'),
                    np.full((num_test_winds,), '5-8'),np.full((num_test_winds,), '8-8'),
                    np.full((num_test_winds,), '2-2'),np.full((num_test_winds,), '2-5'),
                    np.full((num_test_winds,), '2-8'),np.full((num_test_winds,), '5-5'),
                    np.full((num_test_winds,), '5-8'),np.full((num_test_winds,), '8-8'),),axis=0)
hue = np.concatenate((np.full((6*num_test_winds,), 'No'), np.full((6*num_test_winds,), 'Yes')),axis=0)

d = {'Fingerprinting Accuracy': y, 'With CNN': hue, 'Duration Combinations (mins)': x}
df = pd.DataFrame(data=d)
ax = sns.barplot(df, x="Duration Combinations (mins)", y="Fingerprinting Accuracy", hue="With CNN")
plt.ylabel('Fingerprinting Accuracy', fontsize=17)
plt.xlabel('Duration Combinations (mins)', fontsize=17)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(loc='lower right', fontsize = 16)
plt.tight_layout()

pairs = [
    (("2-2", "No"), ("2-2", "Yes")),
    (("2-5", "No"), ("2-5", "Yes")),
    (("2-8", "No"), ("2-8", "Yes")),
    (("5-5", "No"), ("5-5", "Yes")),
    (("5-8", "No"), ("5-8", "Yes")),
    (("8-8", "No"), ("8-8", "Yes"))
]

# Add statistical annotation
annotator = Annotator(ax, pairs, data=df, x="Duration Combinations (mins)", y="Fingerprinting Accuracy", hue="With CNN")
annotator.configure(test='t-test_ind', text_format='star', loc='inside', verbose=2)
annotator.apply_and_annotate()

plt.savefig(os.path.join(path_results,"plots","HCP_noCNN_AICHA.png"), dpi=600, bbox_inches='tight')
plt.show()

plt.figure()
y = np.concatenate((AAL_noTransformer_0_0, AAL_noTransformer_0_1, AAL_noTransformer_0_2,
                    AAL_noTransformer_1_1, AAL_noTransformer_1_2, AAL_noTransformer_2_2,
                    AAL_0_0, AAL_0_1, AAL_0_2, AAL_1_1, AAL_1_2, AAL_2_2), axis=0)
x = np.concatenate((np.full((num_test_winds,), '2-2'),np.full((num_test_winds,), '2-5'),
                    np.full((num_test_winds,), '2-8'),np.full((num_test_winds,), '5-5'),
                    np.full((num_test_winds,), '5-8'),np.full((num_test_winds,), '8-8'),
                    np.full((num_test_winds,), '2-2'),np.full((num_test_winds,), '2-5'),
                    np.full((num_test_winds,), '2-8'),np.full((num_test_winds,), '5-5'),
                    np.full((num_test_winds,), '5-8'),np.full((num_test_winds,), '8-8'),),axis=0)
hue = np.concatenate((np.full((6*num_test_winds,), 'No'), np.full((6*num_test_winds,), 'Yes')),axis=0)

d = {'Fingerprinting Accuracy': y, 'With Transformer': hue, 'Duration Combinations (mins)': x}
df = pd.DataFrame(data=d)
ax = sns.barplot(df, x="Duration Combinations (mins)", y="Fingerprinting Accuracy", hue="With Transformer")
plt.ylabel('Fingerprinting Accuracy', fontsize=17)
plt.xlabel('Duration Combinations (mins)', fontsize=17)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(loc='lower right', fontsize = 16)
plt.tight_layout()

pairs = [
    (("2-2", "No"), ("2-2", "Yes")),
    (("2-5", "No"), ("2-5", "Yes")),
    (("2-8", "No"), ("2-8", "Yes")),
    (("5-5", "No"), ("5-5", "Yes")),
    (("5-8", "No"), ("5-8", "Yes")),
    (("8-8", "No"), ("8-8", "Yes"))
]

# Add statistical annotation
annotator = Annotator(ax, pairs, data=df, x="Duration Combinations (mins)", y="Fingerprinting Accuracy", hue="With Transformer")
annotator.configure(test='t-test_ind', text_format='star', loc='inside', verbose=2)
annotator.apply_and_annotate()

plt.savefig(os.path.join(path_results,"plots","HCP_noTransformer_AAL.png"), dpi=600, bbox_inches='tight')
plt.show()


plt.figure()
y = np.concatenate((AICHA_noTransformer_0_0, AICHA_noTransformer_0_1, AICHA_noTransformer_0_2,
                    AICHA_noTransformer_1_1, AICHA_noTransformer_1_2, AICHA_noTransformer_2_2,
                    AICHA_0_0, AICHA_0_1, AICHA_0_2, AICHA_1_1, AICHA_1_2, AICHA_2_2), axis=0)
x = np.concatenate((np.full((num_test_winds,), '2-2'),np.full((num_test_winds,), '2-5'),
                    np.full((num_test_winds,), '2-8'),np.full((num_test_winds,), '5-5'),
                    np.full((num_test_winds,), '5-8'),np.full((num_test_winds,), '8-8'),
                    np.full((num_test_winds,), '2-2'),np.full((num_test_winds,), '2-5'),
                    np.full((num_test_winds,), '2-8'),np.full((num_test_winds,), '5-5'),
                    np.full((num_test_winds,), '5-8'),np.full((num_test_winds,), '8-8'),),axis=0)
hue = np.concatenate((np.full((6*num_test_winds,), 'No'), np.full((6*num_test_winds,), 'Yes')),axis=0)

d = {'Fingerprinting Accuracy': y, 'With Transformer': hue, 'Duration Combinations (mins)': x}
df = pd.DataFrame(data=d)
ax = sns.barplot(df, x="Duration Combinations (mins)", y="Fingerprinting Accuracy", hue="With Transformer")
plt.ylabel('Fingerprinting Accuracy', fontsize=17)
plt.xlabel('Duration Combinations (mins)', fontsize=17)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(loc='lower right', fontsize = 16)
plt.tight_layout()

pairs = [
    (("2-2", "No"), ("2-2", "Yes")),
    (("2-5", "No"), ("2-5", "Yes")),
    (("2-8", "No"), ("2-8", "Yes")),
    (("5-5", "No"), ("5-5", "Yes")),
    (("5-8", "No"), ("5-8", "Yes")),
    (("8-8", "No"), ("8-8", "Yes"))
]

# Add statistical annotation
annotator = Annotator(ax, pairs, data=df, x="Duration Combinations (mins)", y="Fingerprinting Accuracy", hue="With Transformer")
annotator.configure(test='t-test_ind', text_format='star', loc='inside', verbose=2)
annotator.apply_and_annotate()

plt.savefig(os.path.join(path_results,"plots","HCP_noTransformer_AICHA.png"), dpi=600, bbox_inches='tight')
plt.show()


plt.figure()
y = np.concatenate((AAL_augmentation2_0_0, AAL_augmentation2_0_1, AAL_augmentation2_0_2, 
                    AAL_augmentation2_1_1, AAL_augmentation2_1_2, AAL_augmentation2_2_2,
                    AAL_0_0, AAL_0_1, AAL_0_2, AAL_1_1, AAL_1_2, AAL_2_2), axis=0)
x = np.concatenate((np.full((num_test_winds,), '2-2'),np.full((num_test_winds,), '2-5'),
                    np.full((num_test_winds,), '2-8'),np.full((num_test_winds,), '5-5'),
                    np.full((num_test_winds,), '5-8'),np.full((num_test_winds,), '8-8'),
                    np.full((num_test_winds,), '2-2'),np.full((num_test_winds,), '2-5'),
                    np.full((num_test_winds,), '2-8'),np.full((num_test_winds,), '5-5'),
                    np.full((num_test_winds,), '5-8'),np.full((num_test_winds,), '8-8'),),axis=0)
hue = np.concatenate((np.full((6*num_test_winds,), 'UCGL'), np.full((6*num_test_winds,), 'VarCoNet')),axis=0)

d = {'Fingerprinting Accuracy': y, 'Aug. strategy': hue, 'Duration Combinations (mins)': x}
df = pd.DataFrame(data=d)
ax = sns.barplot(df, x="Duration Combinations (mins)", y="Fingerprinting Accuracy", hue="Aug. strategy")
plt.ylabel('Fingerprinting Accuracy', fontsize=17)
plt.xlabel('Duration Combinations (mins)', fontsize=17)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(loc='lower right', fontsize = 16)
plt.tight_layout()

pairs = [
    (("2-2", "UCGL"), ("2-2", "VarCoNet")),
    (("2-5", "UCGL"), ("2-5", "VarCoNet")),
    (("2-8", "UCGL"), ("2-8", "VarCoNet")),
    (("5-5", "UCGL"), ("5-5", "VarCoNet")),
    (("5-8", "UCGL"), ("5-8", "VarCoNet")),
    (("8-8", "UCGL"), ("8-8", "VarCoNet"))
]

# Add statistical annotation
annotator = Annotator(ax, pairs, data=df, x="Duration Combinations (mins)", y="Fingerprinting Accuracy", hue="Aug. strategy")
annotator.configure(test='t-test_ind', text_format='star', loc='inside', verbose=2)
annotator.apply_and_annotate()

plt.savefig(os.path.join(path_results,"plots","HCP_augment_strategies_AAL.png"), dpi=600, bbox_inches='tight')
plt.show()


plt.figure()
y = np.concatenate((AICHA_augmentation2_0_0, AICHA_augmentation2_0_1, AICHA_augmentation2_0_2, 
                    AICHA_augmentation2_1_1, AICHA_augmentation2_1_2, AICHA_augmentation2_2_2,
                    AICHA_0_0, AICHA_0_1, AICHA_0_2, AICHA_1_1, AICHA_1_2, AICHA_2_2), axis=0)
x = np.concatenate((np.full((num_test_winds,), '2-2'),np.full((num_test_winds,), '2-5'),
                    np.full((num_test_winds,), '2-8'),np.full((num_test_winds,), '5-5'),
                    np.full((num_test_winds,), '5-8'),np.full((num_test_winds,), '8-8'),
                    np.full((num_test_winds,), '2-2'),np.full((num_test_winds,), '2-5'),
                    np.full((num_test_winds,), '2-8'),np.full((num_test_winds,), '5-5'),
                    np.full((num_test_winds,), '5-8'),np.full((num_test_winds,), '8-8'),),axis=0)
hue = np.concatenate((np.full((6*num_test_winds,), 'UCGL'), np.full((6*num_test_winds,), 'VarCoNet')),axis=0)

d = {'Fingerprinting Accuracy': y, 'Aug. strategy': hue, 'Duration Combinations (mins)': x}
df = pd.DataFrame(data=d)
ax = sns.barplot(df, x="Duration Combinations (mins)", y="Fingerprinting Accuracy", hue="Aug. strategy")
plt.ylabel('Fingerprinting Accuracy', fontsize=17)
plt.xlabel('Duration Combinations (mins)', fontsize=17)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(loc='lower right', fontsize = 16)
plt.tight_layout()

pairs = [
    (("2-2", "UCGL"), ("2-2", "VarCoNet")),
    (("2-5", "UCGL"), ("2-5", "VarCoNet")),
    (("2-8", "UCGL"), ("2-8", "VarCoNet")),
    (("5-5", "UCGL"), ("5-5", "VarCoNet")),
    (("5-8", "UCGL"), ("5-8", "VarCoNet")),
    (("8-8", "UCGL"), ("8-8", "VarCoNet"))
]

# Add statistical annotation
annotator = Annotator(ax, pairs, data=df, x="Duration Combinations (mins)", y="Fingerprinting Accuracy", hue="Aug. strategy")
annotator.configure(test='t-test_ind', text_format='star', loc='inside', verbose=2)
annotator.apply_and_annotate()

plt.savefig(os.path.join(path_results,"plots","HCP_augment_strategies_AICHA.png"), dpi=600, bbox_inches='tight')
plt.show()


'''
plt.figure()
y = np.concatenate((AAL_0_0, AAL_0_1, AAL_0_2, AAL_1_1, AAL_1_2, AAL_2_2,
                    AICHA_0_0, AICHA_0_1, AICHA_0_2, AICHA_1_1, AICHA_1_2, AICHA_2_2), axis=0)
x = np.concatenate((np.full((num_test_winds,), '2-2'),np.full((num_test_winds,), '2-5'),
                    np.full((num_test_winds,), '2-8'),np.full((num_test_winds,), '5-5'),
                    np.full((num_test_winds,), '5-8'),np.full((num_test_winds,), '8-8'),
                    np.full((num_test_winds,), '2-2'),np.full((num_test_winds,), '2-5'),
                    np.full((num_test_winds,), '2-8'),np.full((num_test_winds,), '5-5'),
                    np.full((num_test_winds,), '5-8'),np.full((num_test_winds,), '8-8'),),axis=0)
hue = np.concatenate((np.full((6*num_test_winds,), 'AAL'), np.full((6*num_test_winds,), 'AICHA')),axis=0)

d = {'Fingerprinting Accuracy': y, 'Atlas': hue, 'Duration Combinations (mins)': x}
df = pd.DataFrame(data=d)
sns.barplot(df, x="Duration Combinations (mins)", y="Fingerprinting Accuracy", hue="Atlas")
plt.ylabel('Fingerprinting Accuracy', fontsize=17)
plt.xlabel('Duration Combinations (mins)', fontsize=17)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(loc='lower right', fontsize = 16)
plt.tight_layout()
plt.savefig(os.path.join(path_results,"plots","HCP_atlas.png"), dpi=600, bbox_inches='tight')
plt.show()
'''
