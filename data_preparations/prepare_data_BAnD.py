import os
from nilearn.image import clean_img
import pandas as pd
import numpy as np
import nibabel as nib

path = r'E:\ABIDE-fmriprep_BIDS\ABIDE\data\derivatives\fmriprep'
path_phenotypic = r'C:\Users\100063082\Desktop\prepare_ADHD200_ABIDE_phenotypics\Phenotypic_V1_0b_preprocessed1.csv'
phenotypic = pd.read_csv(path_phenotypic)
ids = phenotypic["SUB_ID"].values
sites = phenotypic["SITE_ID"].values
files = phenotypic["FILE_ID"].values

path_save = r'E:\ABIDEI_BAnD_affine'
if not os.path.exists(path_save):
    os.makedirs(path_save,exist_ok=True)

dire = os.listdir(path)
for i in range(len(dire)):
    if i != 1824:
        if 'sub' in dire[i] and os.path.isdir(os.path.join(path,dire[i])):
            path_subject = os.path.join(path,dire[i],'func')
            dir_sucject = os.listdir(path_subject)
            for j in range(len(dir_sucject)):
                file = dir_sucject[j]
                if 'smoothAROMAnonaggr' in file:
                    pos_space = file.find("_space")   
                    pos_run = file.find("_run")
                    pos_underscore = file.find("_")
                    path_data = os.path.join(path_subject,file)
                    path_confounds = os.path.join(path_subject,file[:pos_space] + '_desc-confounds_regressors.tsv')
                    confounds = pd.read_table(path_confounds).iloc[:,:2]
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
                    parts = file.split("_")
                    path_save_new = os.path.join(path_save,parts[0] + '_' + parts[2] + '.npy')
                
                    if os.path.isfile(path_data):
                        try:
                            imgs = nib.load(path_data)
                            affine = imgs.affine
                            #clean_data = clean_img(imgs,
                            #        low_pass=0.1,
                            #        high_pass=0.01,
                            #        t_r=TR,
                            #        confounds=confounds
                            #    )
                            #clean_data = np.array(clean_data.dataobj)
                            np.save(os.path.join(path_save_new), affine)
                            print(file)
                        except:
                            continue
