import numpy as np
import pickle
import os
import seaborn as sns
import matplotlib.pyplot as plt

path_results = r''  #here, enter the path where results have been saved
atlas = 'AICHA'         #choose atlas (AICHA, AAL)

if not os.path.exists(os.path.join(path_results,"plots")):
    os.mkdir(os.path.join(path_results,"plots"))

if atlas == 'AICHA':
    vmax = 0.7
elif atlas == 'AAL':
    vmax = 0.77
    
with open(os.path.join(path_results, 'results_HCP', atlas, 'HCP_VarCoNet_results.pkl'), 'rb') as f:
    results_VarCoNet = pickle.load(f)
similarities = np.stack(results_VarCoNet['test_result'][-1])
similarities = np.mean(similarities, axis=0)
similarities = similarities[150:250,150:250]
plt.figure(figsize=(10, 8))
ax=sns.heatmap(similarities, cmap='coolwarm', center=0, square=True, vmin=0, vmax=vmax)
plt.xticks([])
plt.yticks([])
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=20)
plt.savefig(os.path.join(path_results,"plots","VarCoNet_" + atlas + "_100_subjs.png"), dpi=600, bbox_inches='tight')
plt.show()


with open(os.path.join(path_results, 'results_HCP', atlas, 'HCP_VAE_KSVD_results.pkl'), 'rb') as f:
    results_VAE = pickle.load(f)
similarities = np.stack(results_VAE['test_result'][-1])
similarities = np.mean(similarities, axis=0)
similarities = similarities[150:250,150:250]
plt.figure(figsize=(10, 8))
ax=sns.heatmap(similarities, cmap='coolwarm', center=0, square=True, vmin=0, vmax=vmax)
plt.xticks([])
plt.yticks([])
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=20)
plt.savefig(os.path.join(path_results,"plots","VAE_" + atlas + "_100_subjs.png"), dpi=600, bbox_inches='tight')
plt.show()


with open(os.path.join(path_results, 'results_HCP', atlas, 'HCP_AE_KSVD_results.pkl'), 'rb') as f:
    results_AE = pickle.load(f)
similarities = np.stack(results_AE['test_result'][-1])
similarities = np.mean(similarities, axis=0)
similarities = similarities[150:250,150:250]
plt.figure(figsize=(10, 8))
ax=sns.heatmap(similarities, cmap='coolwarm', center=0, square=True, vmin=0, vmax=vmax)
plt.xticks([])
plt.yticks([])
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=20)
plt.savefig(os.path.join(path_results,"plots","AE_" + atlas + "_100_subjs.png"), dpi=600, bbox_inches='tight')
plt.show()



with open(os.path.join(path_results, 'results_HCP', atlas, 'HCP_PCC_results.pkl'), 'rb') as f:
    results_PCC = pickle.load(f)
similarities = np.stack(results_PCC['test_result'][-1])
similarities = np.mean(similarities, axis=0)
similarities = similarities[150:250,150:250]
plt.figure(figsize=(10, 8))
ax=sns.heatmap(similarities, cmap='coolwarm', center=0, square=True, vmin=0, vmax=vmax)
plt.xticks([])
plt.yticks([])
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=20)
plt.savefig(os.path.join(path_results,"plots","PCC_" + atlas + "_100_subjs.png"), dpi=600, bbox_inches='tight')
plt.show()