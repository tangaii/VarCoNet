import numpy as np
import os
import pickle

atlas = 'AICHA'
folds = 10
seeds = 10
path = r'/.../baselines/BrainNetworkTransformerSpyder/source/result' #In the three dots, add the path to the "baselines" folder
path_save = r'' #the main path where all results are saved (same as in other scripts)
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
        path_res = os.path.join(path,'rs' + str(rs) + '_' + atlas + '_kfold','training_process_rs' + str(rs) + '_fold' + str(fold) + '.npy')
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
            val_losses.append(res[i]['Val LossBCE'])
            val_aucs.append(res[i]['Val AUC'])
            test_losses.append(res[i]['Test LossBCE'])
            test_aucs.append(res[i]['Test AUC'])
            losses.append(res[i]['Train Loss'])
            val_probs.append(res[i]['val_probs'][0])
            y_val.append(res[i]['y_val'][0])
            test_probs.append(res[i]['test_probs'][0])
            y_test.append(res[i]['y_test'][0])
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
    path_res = os.path.join(path,'rs' + str(rs) + '_' + atlas,'training_process_rs' + str(rs) + '_fold' + '.npy')
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
        val_losses.append(res[i]['Val LossBCE'])
        val_aucs.append(res[i]['Val AUC'])
        test_losses.append(res[i]['Test LossBCE'])
        test_aucs.append(res[i]['Test AUC'])
        losses.append(res[i]['Train Loss'])
        val_probs.append(res[i]['val_probs'][0])
        y_val.append(res[i]['y_val'][0])
        test_probs.append(res[i]['test_probs'][0])
        y_test.append(res[i]['y_test'][0])
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

with open(os.path.join(path_save, 'results_ABIDEI',atlas,'ABIDEI_BNT_results.pkl'), "wb") as pickle_file:
    pickle.dump(results, pickle_file)