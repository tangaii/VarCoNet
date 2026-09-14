import numpy as np
import os
import shutil
import pickle

atlas = 'AAL'
folds = 10
seeds = 10
path = os.path.join('...', 'FBNETGEN', atlas) #add the path where results are saved from the fbnetgen script
path_save = os.path.join('...','results_ABIDEI', atlas) #add the main path where all results are saved (same as in other scripts)
path_save_model = os.path.join('...','models_ABIDEI', atlas, 'FBNETGEN') #add the main path where all results are saved (same as in other scripts)
val_losses_all = []
val_aucs_all = []
test_losses_all = []
test_aucs_all = []
losses_all = []
val_probs_all = []
y_val_all = []
test_probs_all = []
y_test_all = []
for rs in range(seeds):
    for fold in range(folds):
        path_res = os.path.join(path,'rs' + str(rs) + '_fold' + str(fold), 'training_process.npy')
        path_model = os.path.join(path,'rs' + str(rs) + '_fold' + str(fold), 'model.pt')
        new_folder = path_save_model
        new_name = 'model_rs' + str(rs) + '_fold' + str(fold) + '.pt'
        target_path = os.path.join(new_folder, new_name)
        os.makedirs(new_folder, exist_ok=True)
        shutil.move(path_model, target_path)
        res = np.load(path_res, allow_pickle=True)
        val_losses = []
        val_aucs = []
        test_losses = []
        test_aucs = []
        losses = []
        val_probs = []
        y_val = []
        test_probs = []
        y_test = []
        for i in range(len(res)):
            val_losses.append(res[i]['bce_loss_val'])
            val_aucs.append(res[i]['val_result'][0])
            test_losses.append(res[i]['bce_loss_test'])
            test_aucs.append(res[i]['test_result'][0])
            losses.append(res[i]['train_loss'])
            val_probs.append(res[i]['val_probs'])
            y_val.append(np.array(res[i]['y_val']))
            test_probs.append(res[i]['test_probs'])
            y_test.append(np.array(res[i]['y_test']))
        losses_all.append(losses)
        val_losses_all.append(val_losses)
        val_aucs_all.append(val_aucs)
        test_losses_all.append(test_losses)
        test_aucs_all.append(test_aucs)
        val_probs_all.append(val_probs)
        y_val_all.append(y_val)
        test_probs_all.append(test_probs)
        y_test_all.append(y_test)

results = {}
results['losses'] = losses_all
results['val_losses'] = val_losses_all
results['val_aucs'] = val_aucs_all
results['test_losses'] = test_losses_all
results['test_aucs'] = test_aucs_all
results['y_val'] = y_val_all
results['y_test'] = y_test_all
results['val_probs'] = val_probs_all
results['test_probs'] = test_probs_all        
       

val_losses_all = []
val_aucs_all = []
test_losses_all = []
test_aucs_all = []
losses_all = []
val_probs_all = []
y_val_all = []
test_probs_all = []
y_test_all = []
for rs in range(seeds):
    path_res = os.path.join(path,'rs' + str(rs), 'training_process.npy')
    path_model = os.path.join(path,'rs' + str(rs), 'model.pt')
    new_folder = path_save_model
    new_name = 'model_rs' + str(rs) + '.pt'
    target_path = os.path.join(new_folder, new_name)
    os.makedirs(new_folder, exist_ok=True)
    shutil.move(path_model, target_path)
    res = np.load(path_res, allow_pickle=True)
    val_losses = []
    val_aucs = []
    test_losses = []
    test_aucs = []
    losses = []
    val_probs = []
    y_val = []
    test_probs = []
    y_test = []
    for i in range(len(res)):
        val_losses.append(res[i]['bce_loss_val'])
        val_aucs.append(res[i]['val_result'][0])
        test_losses.append(res[i]['bce_loss_test'])
        test_aucs.append(res[i]['test_result'][0])
        losses.append(res[i]['train_loss'])
        val_probs.append(res[i]['val_probs'])
        y_val.append(np.array(res[i]['y_val']))
        test_probs.append(res[i]['test_probs'])
        y_test.append(np.array(res[i]['y_test']))
    losses_all.append(losses)
    val_losses_all.append(val_losses)
    val_aucs_all.append(val_aucs)
    test_losses_all.append(test_losses)
    test_aucs_all.append(test_aucs)
    val_probs_all.append(val_probs)
    y_val_all.append(y_val)
    test_probs_all.append(test_probs)
    y_test_all.append(y_test)
    
    
results['losses_ext'] = losses_all
results['val_losses_ext'] = val_losses_all
results['val_aucs_ext'] = val_aucs_all
results['ext_test_losses'] = test_losses_all
results['ext_test_aucs'] = test_aucs_all
results['y_val_ext'] = y_val_all
results['y_ext_test'] = y_test_all
results['val_probs_ext'] = val_probs_all
results['ext_test_probs'] = test_probs_all


with open(os.path.join(path_save,'ABIDEI_FBNETGEN_results.pkl'), "wb") as pickle_file:
    pickle.dump(results, pickle_file)