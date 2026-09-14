import numpy as np
import os
from scipy.signal import resample
import pandas as pd


def resample_signal(signal,file,site):
    if 'Caltech' in file or 'CMU_a' in file or 'SDSU' in file or 'Stanford' in file or 'UM' in file or 'Yale' in file or 'Trinity' in file or 'USM' in file or 'NYU' in file:
        TR = 2.0
    elif 'KKI' in file or 'OHSU' in file:
        TR = 2.5
    elif 'Leuven' in file:
        TR = 1.6669999999999998
    elif 'MaxMun' in file or 'UCLA' in file:
        TR = 3
    elif 'Olin' in file or 'Pitt' in file or 'CMU_b' in file:
        TR = 1.5
    elif 'SBL' in file:
        TR = 2.2
    elif 'no_filename' in file:
        if 'CALTECH' in site or 'SDSU' in site or 'STANFORD' in site or 'UM' in site or 'YALE' in site or 'TRINITY' in site or 'USM' in site or 'NYU' in site:
            TR = 2.0
        elif 'KKI' in site or 'OHSU' in site:
            TR = 2.5
        elif 'LEUVEN' in site:
            TR = 1.6669999999999998
        elif 'MAX_MUN' in site or 'UCLA' in site:
            TR = 3
        elif 'OLIN' in site or 'PITT' in site:
            TR = 1.5
        elif 'SBL' in site:
            TR = 2.2
    else:
        raise Exception('Did not find correspondance')
    samples = signal.shape[0]
    duration = samples*TR/60
    if TR > 1.5:
        ratio = TR/1.5
        new_len = int(np.round(signal.shape[0]*ratio))
        signal = resample(signal,new_len)
    return signal,duration

use_ICA_AOMA = True
max_size = 320
if use_ICA_AOMA:
    path_data_AICHA = r'E:\ABIDEI_nilearn_0.11.1\fmriprep\ROISignals_AICHA'
    path_data_AAL = r'E:\ABIDEI_nilearn_0.11.1\fmriprep\ROISignals_AAL'
    save_path = r'C:\Users\100063082\Desktop\SSL_FC_matrix_GNN_data_nilearn_0.11.1\ABIDEI\fmriprep'

path_phenotypic = r'C:\Users\100063082\Desktop\prepare_ADHD200_ABIDE_phenotypics\Phenotypic_V1_0b_preprocessed1.csv'


data_dir = os.listdir(path_data_AICHA)


phenotypic = pd.read_csv(path_phenotypic)
sub_id = phenotypic['SUB_ID'].values
group = phenotypic['DX_GROUP'].values
file_ids = phenotypic['FILE_ID'].values
sites = phenotypic['SITE_ID'].values
age = phenotypic['AGE_AT_SCAN'].values
gender = phenotypic['SEX'].values
group[group == 2] = 0

names = []
Times_AICHA = []
Times_AAL = []
classes = []
ages = []
genders = []
durations = []
for data in data_dir:
    pos00 = data.find('00')
    pos_task = data.find('_task')
    ID = data[pos00+2:pos_task]
    pos_id = np.where(sub_id == int(ID))[0]
    if len(pos_id) != 1:
        raise TypeError("ID false identification")
    feature_dir_aicha = os.path.join(path_data_AICHA, data,'ROISignals.npy')
    feature_dir_aal = os.path.join(path_data_AAL, data,'ROISignals.npy')
    feature_aicha = np.load(feature_dir_aicha)
    feature_aal = np.load(feature_dir_aal)
    if not np.any(np.all(feature_aicha == 0, axis=0)) and not np.any(np.all(feature_aal == 0, axis=0)):
        file_id = file_ids[pos_id[0]]
        site = sites[pos_id[0]]
        classes.append(group[pos_id[0]])
        ages.append(age[pos_id[0]])
        genders.append(gender[pos_id[0]])
        feature_aicha,duration = resample_signal(feature_aicha, file_id, site)          
        feature_aal,_ = resample_signal(feature_aal, file_id, site)   
        durations.append(duration)
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
np.save(os.path.join(save_path,'ABIDEI_nilearn_classes.npy'),classes) 
np.savez(os.path.join(save_path,'ABIDEI_nilearn_AICHA'),*Times_AICHA)
np.savez(os.path.join(save_path,'ABIDEI_nilearn_AAL'),*Times_AAL) 
for i,name in enumerate(names):
    names[i] = name[:11]
with open(os.path.join(save_path,'ABIDEI_nilearn_names.txt'), 'w') as f:
    for item in names:
        f.write("%s\n" % item)
