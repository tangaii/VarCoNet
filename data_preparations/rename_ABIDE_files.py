import os
import numpy as np
import pandas as pd

path = r'E:\ABIDE-fmriprep_BIDS\ABIDE\data\derivatives\fmriprep'
phenotypics = pd.read_csv(r'C:\Users\100063082\Desktop\prepare_ADHD200_ABIDE_phenotypics\Phenotypic_V1_0b_preprocessed1.csv')
sub_ids = phenotypics['SUB_ID'].values
group = phenotypics['DX_GROUP'].values
sub_ids_new = []
for sub_id in sub_ids:
    sub_ids_new.append('sub-00' + str(sub_id))
path_dir = os.listdir(path)
for file in path_dir:
    if 'sub' in file and os.path.isdir(os.path.join(path,file)):
        subject_path = os.path.join(path,file,'func')
        subject_dir = os.listdir(subject_path)
        for item in subject_dir:
            source_file = os.path.join(subject_path,item)
            if 'smoothAROMAnonaggr' in item:
                pos = [i for i, char in enumerate(item) if char == "_"]
                pos_id = np.where(np.array(sub_ids_new) == item[:pos[0]]) 
                label = group[pos_id]
                if label == 1:
                    new_file = item[:pos[0]] + '_task-restASD_' + item[pos[1]+1:]
                else:
                    new_file = item[:pos[0]] + '_task-restControl_' + item[pos[1]+1:]
                #new_file = item[:pos[0]] + '_task-rest_' + item[pos[1]+1:]
                os.rename(source_file, os.path.join(subject_path,new_file))
            if 'confounds' in item:
                pos = [i for i, char in enumerate(item) if char == "_"]
                pos_id = np.where(np.array(sub_ids_new) == item[:pos[0]]) 
                label = group[pos_id]
                if label == 1:
                    new_file = item[:pos[0]] + '_task-restASD_' + item[pos[1]+1:]
                else:
                    new_file = item[:pos[0]] + '_task-restControl_' + item[pos[1]+1:]
                #new_file = item[:pos[0]] + '_task-rest_' + item[pos[1]+1:]
                new_file = new_file[:-14] + 'regressors.tsv'
                os.rename(source_file, os.path.join(subject_path,new_file))
