import os
from nilearn.input_data import  NiftiLabelsMasker
import pandas as pd
import numpy as np

#FIX IF CONFOUNDS IS MISSING
atlas = 'AICHA'

if atlas == "AAL":
    path_atlas = r'C:\Users\100063082\Desktop\atlases\AAL3v1.nii'
else:
    path_atlas = r'C:\Users\100063082\Desktop\atlases\AICHA.nii'


path = r'E:\ABIDE-fmriprep_BIDS\ABIDE\data\derivatives\fmriprep'
path_phenotypic = r'C:\Users\100063082\Desktop\prepare_ADHD200_ABIDE_phenotypics\Phenotypic_V1_0b_preprocessed1.csv'
phenotypic = pd.read_csv(path_phenotypic)
ids = phenotypic["SUB_ID"].values
sites = phenotypic["SITE_ID"].values
files = phenotypic["FILE_ID"].values

path_save = r'E:\ABIDEI_nilearn_0.11.1\fmriprep\ROIsignals_' + atlas
if not os.path.exists(path_save):
    os.makedirs(path_save,exist_ok=True)

dire = os.listdir(path);
for i in range(len(dire)):
    if 'sub' in dire[i] and os.path.isdir(os.path.join(path,dire[i])):
        path_subject = os.path.join(path,dire[i],'func')
        if os.path.exists(path_subject):
            dir_sucject = os.listdir(path_subject)
            for j in range(len(dir_sucject)):
                file = dir_sucject[j]
                if 'smoothAROMAnonaggr' in file and 'gz' in file:
                    pos_space = file.find("_space")   
                    pos_underscore = file.find("_")
                    path_data = os.path.join(path_subject,file)
                    path_confounds = os.path.join(path_subject,file[:pos_space] + '_desc-confounds_regressors.tsv')
                    try:
                        confounds = pd.read_table(path_confounds).iloc[:,:2]
                    except:
                        continue
                    if pos_underscore == -1:
                        raise ValueError("Invalid filename format")
                    
                    try:
                        id_val = int(file[4:pos_underscore])
                    except ValueError:
                        raise ValueError("Invalid ID extraction")
                    
                    pos_id = np.where(ids == id_val)[0]
                    
                    if len(pos_id) == 0:
                        raise ValueError("Subject not identified")
                    
                    pos_id = pos_id[0]
                    file_id = files[pos_id]
                    site = sites[pos_id]
                    
                    tr_values = {
                        "Caltech": 2.0, "CMU_a": 2.0, "SDSU": 2.0, "Stanford": 2.0, "UM": 2.0, 
                        "Yale": 2.0, "Trinity": 2.0, "USM": 2.0, "NYU": 2.0, "KKI": 2.5, "OHSU": 2.5, 
                        "Leuven": 1.667, "MaxMun": 3, "UCLA": 3, "Olin": 1.5, "Pitt": 1.5, "CMU_b": 1.5, "SBL": 2.2
                    }
                    
                    for key, tr in tr_values.items():
                        if key in file_id:
                            TR = tr
                    
                    if file_id == "no_filename":
                        site_tr_values = {
                            "CALTECH": 2.0, "SDSU": 2.0, "STANFORD": 2.0, "UM": 2.0, "YALE": 2.0, "TRINITY": 2.0,
                            "USM": 2.0, "NYU": 2.0, "KKI": 2.5, "OHSU": 2.5, "LEUVEN": 1.667, "MAX_MUN": 3, 
                            "UCLA": 3, "OLIN": 1.5, "PITT": 1.5, "SBL": 2.2
                        }
                        
                        for key, tr in site_tr_values.items():
                            if key in site:
                                TR = tr
                    
                    path_save_new = os.path.join(path_save,file[:pos_space])
                
                    if os.path.isfile(path_data) and not os.path.exists(os.path.join(path_save_new,'ROISignals.npy')):
                        try:
                            masker = NiftiLabelsMasker(
                                    labels_img=path_atlas,
                                    memory="nilearn_cache",
                                    low_pass=0.1,
                                    high_pass=0.01,
                                    t_r=TR,
                                    verbose=0,
                                )
                            time_series = masker.fit_transform(path_data, confounds=confounds)
                            if not os.path.exists(path_save_new):
                                os.mkdir(path_save_new)
                            np.save(os.path.join(path_save_new,'ROISignals.npy'), time_series)
                            print(file)
                        except:
                            continue
