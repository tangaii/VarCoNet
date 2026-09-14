import pickle
import numpy as np
import os
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


path_results = r'' #here, enter the path where results have been saved
atlas = 'AICHA'         #choose atlas (AICHA, AAL)

if not os.path.exists(os.path.join(path_results,"plots")):
    os.mkdir(os.path.join(path_results,"plots"))

with open(os.path.join(path_results, 'results_HCP', atlas, 'HCP_VarCoNet_results.pkl'), 'rb') as f:
    test_result_VarCoNet = pickle.load(f)
with open(os.path.join(path_results, 'results_HCP', atlas, 'HCP_PCC_results.pkl'), 'rb') as f:
    test_result_PCC = pickle.load(f)
with open(os.path.join(path_results, 'results_HCP', atlas, 'HCP_VAE_KSVD_results.pkl'), 'rb') as f:
    test_result_VAE = pickle.load(f)
with open(os.path.join(path_results, 'results_HCP', atlas, 'HCP_AE_KSVD_results.pkl'), 'rb') as f:
    test_result_AE = pickle.load(f)

losses = test_result_VarCoNet['losses']

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=np.arange(1, len(losses) + 1),
    y=losses,
    mode='lines',
    name='Training Loss',
    line=dict(color='red', width=3),
))
fig.update_layout(
    xaxis_title='Epoch',
    yaxis_title='Loss',
    font=dict(size=20),
    xaxis=dict(
        tickmode='linear',
        tick0=0,
        dtick=5,
        tickfont=dict(size=18),
    ),
    yaxis=dict(
        showgrid=True,
        tickfont=dict(size=18),
    ),
)

fig.write_image(os.path.join(path_results,'plots','HCP_training_loss_' + atlas + '.png'), scale=8, engine = "orca")


VarCoNet_0_0 = test_result_VarCoNet['test_result'][0][0]
VarCoNet_0_1 = test_result_VarCoNet['test_result'][0][1]
VarCoNet_0_2 = test_result_VarCoNet['test_result'][0][2]
VarCoNet_1_1 = test_result_VarCoNet['test_result'][0][3]
VarCoNet_1_2 = test_result_VarCoNet['test_result'][0][4]
VarCoNet_2_2 = test_result_VarCoNet['test_result'][0][5]
PCC_0_0 = test_result_PCC['test_result'][0][0]
PCC_0_1 = test_result_PCC['test_result'][0][1]
PCC_0_2 = test_result_PCC['test_result'][0][2]
PCC_1_1 = test_result_PCC['test_result'][0][3]
PCC_1_2 = test_result_PCC['test_result'][0][4]
PCC_2_2 = test_result_PCC['test_result'][0][5]
VAE_0_0 = test_result_VAE['test_result'][0][0]
VAE_0_1 = test_result_VAE['test_result'][0][1]
VAE_0_2 = test_result_VAE['test_result'][0][2]
VAE_1_1 = test_result_VAE['test_result'][0][3]
VAE_1_2 = test_result_VAE['test_result'][0][4]
VAE_2_2 = test_result_VAE['test_result'][0][5]
AE_0_0 = test_result_AE['test_result'][0][0]
AE_0_1 = test_result_AE['test_result'][0][1]
AE_0_2 = test_result_AE['test_result'][0][2]
AE_1_1 = test_result_AE['test_result'][0][3]
AE_1_2 = test_result_AE['test_result'][0][4]
AE_2_2 = test_result_AE['test_result'][0][5]

data = np.concatenate((VarCoNet_0_0, VarCoNet_0_1, VarCoNet_0_2,
                    VarCoNet_1_1, VarCoNet_1_2, VarCoNet_2_2,
                    PCC_0_0, PCC_0_1, PCC_0_2, PCC_1_1, PCC_1_2, PCC_2_2,
                    VAE_0_0, VAE_0_1, VAE_0_2, VAE_1_1, VAE_1_2, VAE_2_2,
                    AE_0_0, AE_0_1, AE_0_2, AE_1_1, AE_1_2, AE_2_2), axis=0)

groups = np.concatenate((np.full((10,), '2-2'),np.full((10,), '2-5'),
                    np.full((10,), '2-8'),np.full((10,), '5-5'),
                    np.full((10,), '5-8'),np.full((10,), '8-8'),
                    np.full((10,), '2-2'),np.full((10,), '2-5'),
                    np.full((10,), '2-8'),np.full((10,), '5-5'),
                    np.full((10,), '5-8'),np.full((10,), '8-8'),
                    np.full((10,), '2-2'),np.full((10,), '2-5'),
                    np.full((10,), '2-8'),np.full((10,), '5-5'),
                    np.full((10,), '5-8'),np.full((10,), '8-8'),
                    np.full((10,), '2-2'),np.full((10,), '2-5'),
                    np.full((10,), '2-8'),np.full((10,), '5-5'),
                    np.full((10,), '5-8'),np.full((10,), '8-8')),axis=0)
labels = np.concatenate((np.full((60,), 'VarCoNet'), np.full((60,), 'PCC'),
                         np.full((60,), '(Lu et al., 2024)'), np.full((60,), '(Cai et al., 2021)')),axis=0)    

df = pd.DataFrame({'Value': data, 'Method': labels, 'Group': groups})

plt.figure(figsize=(10, 6))
sns.set(style="whitegrid")

ax = sns.boxplot(x='Group', y='Value', hue='Method', data=df, palette='Set2')

plt.ylabel('Accuracy', fontsize=20)
plt.xlabel('Duration Combinations (mins)', fontsize=20)

plt.legend(loc='lower right', fontsize = 16)
plt.xticks(rotation=0, fontsize=20)
plt.yticks(fontsize=20)
plt.ylim(0, 1)
plt.tight_layout()

plt.savefig(os.path.join(path_results,'plots','fingerprinting_comparison_boxplot_' + atlas + '.png'), dpi=600, bbox_inches='tight')
plt.show()