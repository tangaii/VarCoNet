import os
import nibabel as nib
from nilearn.input_data import NiftiLabelsMasker
import numpy as np

atlas = 'AAL' #AICHA, AAL
if atlas == 'AAL':
    atlas_path = r'path_to\AAL3v1.nii'
else:
    atlas_path = r'path_to\AICHA.nii'
tr = 0.72
bandpass = (0.01, 0.1)

def extract_roi_signals(fmri_path, atlas_path, save_path, tr, bandpass):
    masker = NiftiLabelsMasker(labels_img=atlas_path,
                               standardize=True,
                               t_r=tr,
                               low_pass=bandpass[1],
                               high_pass=bandpass[0],
                               detrend=True,
                               memory='nilearn_cache')

    time_series = masker.fit_transform(fmri_path)

    np.save(save_path + '.npy', time_series)  # Save as .npy

def process_task(drive, task, atlas_path, atlas):
    path_task = os.path.join(drive, task)
    path_save = os.path.join(drive, f"{task}_ROIsignals"+atlas)
    os.makedirs(path_save, exist_ok=True)

    subjects = [f for f in os.listdir(path_task) if os.path.isdir(os.path.join(path_task, f))]

    for subject in subjects:
        path_data_LR = os.path.join(path_task, subject, 'MNINonLinear', 'Results',
                                    f'rfMRI_{task}_LR', f'rfMRI_{task}_LR.nii.gz')
        path_data_RL = os.path.join(path_task, subject, 'MNINonLinear', 'Results',
                                    f'rfMRI_{task}_RL', f'rfMRI_{task}_RL.nii.gz')

        path_save_subject = os.path.join(path_save, subject)
        os.makedirs(path_save_subject, exist_ok=True)

        if len(os.listdir(path_save_subject)) <= 2:
            if os.path.isfile(path_data_LR):
                save_path_LR = os.path.join(path_save_subject, f"{subject}_LR")
                extract_roi_signals(path_data_LR, atlas_path, save_path_LR, tr, bandpass)

            if os.path.isfile(path_data_RL):
                save_path_RL = os.path.join(path_save_subject, f"{subject}_RL")
                extract_roi_signals(path_data_RL, atlas_path, save_path_RL, tr, bandpass)

# === Run for both tasks ===
process_task('E:', 'REST1', atlas_path, atlas)
process_task('D:', 'REST2', atlas_path, atlas)
