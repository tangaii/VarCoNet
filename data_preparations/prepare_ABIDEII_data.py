import numpy as np
import os
from scipy.signal import resample
import pandas as pd


def resample_signal(signal,site):
    if 'EMC' in site or 'ETH' in site or 'GU' in site or 'NYU' in site or 'SDSU' in site or 'SU' in site or 'TCD' in site or 'UCD' in site or 'USM' in site or 'MIA' in site:
        TR = 2.0
    elif 'BNI' in site or 'UCLA' in site:
        TR = 3.0
    elif 'OHSU' in site or 'KUL' in site or 'KKI' in site:
        TR = 2.5
    elif 'IU':
        TR = 0.813
    elif 'OILH' in site:
        TR = 0.475
    elif 'IP' in site:
        TR = 2.7
    else:
        raise Exception('Did not find correspondance')
    if TR > 1.5:
        ratio = TR/1.5
        new_len = int(np.round(signal.shape[0]*ratio))
        signal = resample(signal,new_len)
    return signal

max_size = 320
path_data_AICHA = r'E:\ABIDEII_nilearn\fmriprep\ROISignals_AICHA'
path_data_AAL = r'E:\ABIDEII_nilearn\fmriprep\ROISignals_AAL'
save_path = r'C:\Users\100063082\Desktop\SSL_FC_matrix_GNN_data\ABIDEII\fmriprep'
if not os.path.exists(save_path):
    os.makedirs(save_path, exist_ok=True)
path_phenotypic = r'C:\Users\100063082\Desktop\prepare_ADHD200_ABIDE_phenotypics\ABIDEII_Composite_Phenotypic.csv'


data_dir = os.listdir(path_data_AICHA)


phenotypic = pd.read_csv(path_phenotypic,encoding="latin-1")
sub_id = phenotypic["SUB_ID"].values
sites = phenotypic["SITE_ID"].values
group = phenotypic['DX_GROUP'].values
group[group == 2] = 0

names = []
Times_AICHA = []
Times_AAL = []
classes = []
for data in data_dir:
    if len(os.listdir(os.path.join(path_data_AICHA,data))) > 0:
        pos_ses = data.find('_ses')
        ID = data[4:pos_ses]
        pos_id = np.where(sub_id == int(ID))[0]
        if len(pos_id) != 1:
            raise TypeError("ID false identification")
        feature_dir_aicha = os.path.join(path_data_AICHA, data,'ROISignals.npy')
        feature_dir_aal = os.path.join(path_data_AAL, data,'ROISignals.npy')
        feature_aicha = np.load(feature_dir_aicha)#np.array(sio.loadmat(feature_dir_aicha)['ROISignals'])
        feature_aal = np.load(feature_dir_aal)#np.array(sio.loadmat(feature_dir_aal)['ROISignals'])
        if not np.any(np.all(feature_aicha == 0, axis=0)) and not np.any(np.all(feature_aal == 0, axis=0)):
            site = sites[pos_id[0]]
            classes.append(group[pos_id[0]])
            feature_aicha = resample_signal(feature_aicha, site)          
            feature_aal = resample_signal(feature_aal, site)   
            if feature_aicha.shape[0] > max_size:
                feature_aicha = feature_aicha[:max_size,:]
                feature_aal = feature_aal[:max_size,:]
            elif feature_aicha.shape[0] < max_size:
                feature_aicha = np.concatenate((feature_aicha,np.zeros((max_size-feature_aicha.shape[0],feature_aicha.shape[1]))),axis=0)
                feature_aal = np.concatenate((feature_aal,np.zeros((max_size-feature_aal.shape[0],feature_aal.shape[1]))),axis=0)
            Times_AICHA.append(feature_aicha)
            Times_AAL.append(feature_aal)
            names.append(data)
        else:
            print(data)



classes = np.array(classes) 
np.save(os.path.join(save_path,'ABIDEII_nilearn_classes.npy'),classes) 
np.savez(os.path.join(save_path,'ABIDEII_nilearn_AICHA'),*Times_AICHA)
np.savez(os.path.join(save_path,'ABIDEII_nilearn_AAL'),*Times_AAL) 
for i,name in enumerate(names):
    names[i] = name[:9]
with open(os.path.join(save_path,'ABIDEII_nilearn_names.txt'), 'w') as f:
    for item in names:
        f.write("%s\n" % item)
