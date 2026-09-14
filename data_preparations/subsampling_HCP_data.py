import os
from scipy.signal import resample
import numpy as np

path = r'/home/student1/Desktop/Charalampos_Lamprou/SSL_FC_matrix_GNN_data'

length = 1200
TR = 0.72
ratio = TR/1.5
new_length = int(np.round(length*ratio))

data = np.load(os.path.join(path,'train_data_HCP_AICHA.npz'))
train_data = []
for key in data:
    train_data.append(resample(data[key], new_length))

data = np.load(os.path.join(path,'val_data_HCP_AICHA_1.npz'))
val_data1 = []
for key in data:
    val_data1.append(resample(data[key], new_length))

data = np.load(os.path.join(path,'val_data_HCP_AICHA_2.npz'))
val_data2 = []
for key in data:
    val_data2.append(resample(data[key], new_length))

data = np.load(os.path.join(path,'test_data_HCP_AICHA_1.npz'))
test_data1 = []
for key in data:
    test_data1.append(resample(data[key], new_length))

data = np.load(os.path.join(path,'test_data_HCP_AICHA_2.npz'))
test_data2 = []
for key in data:
    test_data2.append(resample(data[key], new_length))
    
    
np.savez(os.path.join(path, 'train_data_HCP_AICHA_resampled'), *train_data)
np.savez(os.path.join(path, 'test_data_HCP_AICHA_1_resampled'), *test_data1)
np.savez(os.path.join(path, 'test_data_HCP_AICHA_2_resampled'), *test_data2)
np.savez(os.path.join(path, 'val_data_HCP_AICHA_1_resampled'), *val_data1)
np.savez(os.path.join(path, 'val_data_HCP_AICHA_2_resampled'), *val_data2)


data = np.load(os.path.join(path,'train_data_HCP_AAL.npz'))
train_data = []
for key in data:
    train_data.append(resample(data[key], new_length))

data = np.load(os.path.join(path,'val_data_HCP_AAL_1.npz'))
val_data1 = []
for key in data:
    val_data1.append(resample(data[key], new_length))

data = np.load(os.path.join(path,'val_data_HCP_AAL_2.npz'))
val_data2 = []
for key in data:
    val_data2.append(resample(data[key], new_length))

data = np.load(os.path.join(path,'test_data_HCP_AAL_1.npz'))
test_data1 = []
for key in data:
    test_data1.append(resample(data[key], new_length))

data = np.load(os.path.join(path,'test_data_HCP_AAL_2.npz'))
test_data2 = []
for key in data:
    test_data2.append(resample(data[key], new_length))
    
    
np.savez(os.path.join(path, 'train_data_HCP_AAL_resampled'), *train_data)
np.savez(os.path.join(path, 'test_data_HCP_AAL_1_resampled'), *test_data1)
np.savez(os.path.join(path, 'test_data_HCP_AAL_2_resampled'), *test_data2)
np.savez(os.path.join(path, 'val_data_HCP_AAL_1_resampled'), *val_data1)
np.savez(os.path.join(path, 'val_data_HCP_AAL_2_resampled'), *val_data2)
