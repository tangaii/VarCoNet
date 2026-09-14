import numpy as np
import scipy.io as sio
from os.path import join
import os
from random import sample, randint


train_subjects = 500

path_data_1_AICHA = r'E:\REST1_ROIsignals_AICHA'
path_data_2_AICHA = r'E:\REST2_ROIsignals_AICHA'
path_data_1_AAL = r'E:\REST1_ROIsignals_AAL'
path_data_2_AAL = r'E:\REST2_ROIsignals_AAL'
path_save = r'C:\Users\100063082\Desktop\SSL_FC_matrix_GNN_data'

dir1 = os.listdir(path_data_1_AICHA)
dir2 = os.listdir(path_data_2_AICHA)

one_not_two = list(set(dir1).difference(dir2))
two_not_one = list(set(dir2).difference(dir1))

pos_remove = []
for ind in one_not_two:
    pos_remove.append(dir1.index(ind))
pos_remove.sort(reverse=True)
for pos in pos_remove:
    dir1.pop(pos)
    
pos_remove = []
for ind in two_not_one:
    pos_remove.append(dir2.index(ind))
pos_remove.sort(reverse=True)
for pos in pos_remove:
    dir2.pop(pos)

one_not_two.extend(two_not_one)

remaining_subjects = train_subjects - len(one_not_two)

rand_inds = sample(range(1, len(dir1)), remaining_subjects)

rand_subjects = [dir1[i] for i in rand_inds]

rand_subjects.extend(one_not_two)

signals_AICHA = []
signals_AAL = []
names_train = []
for name in rand_subjects:
    name_path1_aicha = join(path_data_1_AICHA, name)
    name_path2_aicha = join(path_data_2_AICHA, name)
    name_path1_aal = join(path_data_1_AAL, name)
    name_path2_aal = join(path_data_2_AAL, name)
    feature = []
    if os.path.exists(name_path1_aicha):
        pe = randint(0,1)
        if pe == 0:
            pe = 'LR'
        else:
            pe = 'RL'
        flag = 0
        for time_dir in os.listdir(name_path1_aicha):
            if 'ROISignals' in time_dir and '.mat' in time_dir and pe in time_dir:
                feature_dir_aicha = os.path.join(name_path1_aicha, time_dir)
                feature_dir_aal = os.path.join(name_path1_aal, time_dir)  
                temp = np.array(sio.loadmat(feature_dir_aicha)['ROISignals'])
                if temp.shape[0] == 1200:
                    signals_AICHA.append(temp)
                    names_train.append(name)
                    flag = 1
                temp = np.array(sio.loadmat(feature_dir_aal)['ROISignals'])
                if temp.shape[0] == 1200:
                    signals_AAL.append(temp)
        if flag == 0:
            if pe == 'RL':
                pe = 'LR'
            elif pe == 'LR':
                pe = 'RL'
            for time_dir in os.listdir(name_path1_aicha):
                if 'ROISignals' in time_dir and '.mat' in time_dir and pe in time_dir:
                    feature_dir_aicha = os.path.join(name_path1_aicha, time_dir)   
                    feature_dir_aal = os.path.join(name_path1_aal, time_dir)   
                    temp = np.array(sio.loadmat(feature_dir_aicha)['ROISignals'])
                    if temp.shape[0] == 1200:
                        signals_AICHA.append(temp)
                        names_train.append(name)
                    temp = np.array(sio.loadmat(feature_dir_aal)['ROISignals'])
                    if temp.shape[0] == 1200:
                        signals_AAL.append(temp)
    if os.path.exists(name_path2_aicha):
        pe = randint(0,1)
        if pe == 0:
            pe = 'LR'
        else:
            pe = 'RL'
        flag = 0
        for time_dir in os.listdir(name_path2_aicha):
            if 'ROISignals' in time_dir and '.mat' in time_dir and pe in time_dir:
                feature_dir_aicha = os.path.join(name_path2_aicha, time_dir)  
                feature_dir_aal = os.path.join(name_path2_aal, time_dir)  
                temp = np.array(sio.loadmat(feature_dir_aicha)['ROISignals'])
                if temp.shape[0] == 1200:
                    signals_AICHA.append(temp)
                    names_train.append(name)
                    flag = 1
                temp = np.array(sio.loadmat(feature_dir_aal)['ROISignals'])
                if temp.shape[0] == 1200:
                    signals_AAL.append(temp)
        if flag == 0:
            if pe == 'RL':
                pe = 'LR'
            elif pe == 'LR':
                pe = 'RL'
            for time_dir in os.listdir(name_path2_aicha):
                if 'ROISignals' in time_dir and '.mat' in time_dir and pe in time_dir:
                    feature_dir_aicha = os.path.join(name_path2_aicha, time_dir)   
                    feature_dir_aal = os.path.join(name_path2_aal, time_dir)   
                    temp = np.array(sio.loadmat(feature_dir_aicha)['ROISignals'])
                    if temp.shape[0] == 1200:
                        signals_AICHA.append(temp)
                        names_train.append(name)   
                    temp = np.array(sio.loadmat(feature_dir_aal)['ROISignals'])
                    if temp.shape[0] == 1200:
                        signals_AAL.append(temp) 


dir1 = os.listdir(path_data_1_AICHA)
dir2 = os.listdir(path_data_2_AICHA)
        
test_dir1 =[x for x in dir1 if x not in names_train]
test_dir2 =[x for x in dir2 if x not in names_train]


test_names = list(set(test_dir1).intersection(test_dir2))
signals1_AICHA = []
signals2_AICHA = []
signals1_AAL = []
signals2_AAL = []
for name in test_names:
    name_path1_aicha = join(path_data_1_AICHA, name)
    name_path2_aicha = join(path_data_2_AICHA, name)
    name_path1_aal = join(path_data_1_AAL, name)
    name_path2_aal = join(path_data_2_AAL, name)
    feature = []
    if os.path.exists(name_path1_aicha):
        pe = randint(0,1)
        if pe == 0:
            pe = 'LR'
        else:
            pe = 'RL'
        flag = 0
        for time_dir in os.listdir(name_path1_aicha):
            if 'ROISignals' in time_dir and '.mat' in time_dir and pe in time_dir:
                feature_dir_aicha = os.path.join(name_path1_aicha, time_dir)    
                feature_dir_aal = os.path.join(name_path1_aal, time_dir)
                temp = np.array(sio.loadmat(feature_dir_aicha)['ROISignals'])
                if temp.shape[0] == 1200:
                    signals1_AICHA.append(temp)
                    flag = 1
                temp = np.array(sio.loadmat(feature_dir_aal)['ROISignals'])
                if temp.shape[0] == 1200:
                    signals1_AAL.append(temp)
        if flag == 0:
            if pe == 'RL':
                pe = 'LR'
            elif pe == 'LR':
                pe = 'RL'
            for time_dir in os.listdir(name_path1_aicha):
                if 'ROISignals' in time_dir and '.mat' in time_dir and pe in time_dir:
                    feature_dir_aicha = os.path.join(name_path1_aicha, time_dir)   
                    feature_dir_aal = os.path.join(name_path1_aal, time_dir) 
                    temp = np.array(sio.loadmat(feature_dir_aicha)['ROISignals'])
                    if temp.shape[0] == 1200:
                        signals1_AICHA.append(temp)
                        flag = 1
                    temp = np.array(sio.loadmat(feature_dir_aal)['ROISignals'])
                    if temp.shape[0] == 1200:
                        signals1_AAL.append(temp)

    if os.path.exists(name_path2_aicha):
        pe = randint(0,1)
        if pe == 0:
            pe = 'LR'
        else:
            pe = 'RL'
        flag = 0
        for time_dir in os.listdir(name_path2_aicha):
            if 'ROISignals' in time_dir and '.mat' in time_dir and pe in time_dir:
                feature_dir_aicha = os.path.join(name_path2_aicha, time_dir)  
                feature_dir_aal = os.path.join(name_path2_aal, time_dir)  
                temp = np.array(sio.loadmat(feature_dir_aicha)['ROISignals'])
                if temp.shape[0] == 1200:
                    signals2_AICHA.append(temp)
                    flag = 1
                temp = np.array(sio.loadmat(feature_dir_aal)['ROISignals'])
                if temp.shape[0] == 1200:
                    signals2_AAL.append(temp)
        if flag == 0:
            if pe == 'RL':
                pe = 'LR'
            elif pe == 'LR':
                pe = 'RL'
            for time_dir in os.listdir(name_path2_aicha):
                if 'ROISignals' in time_dir and '.mat' in time_dir and pe in time_dir:
                    feature_dir_aicha = os.path.join(name_path2_aicha, time_dir)   
                    feature_dir_aal = os.path.join(name_path2_aal, time_dir)  
                    temp = np.array(sio.loadmat(feature_dir_aicha)['ROISignals'])
                    if temp.shape[0] == 1200:
                        signals2_AICHA.append(temp)
                        flag = 1
                    temp = np.array(sio.loadmat(feature_dir_aal)['ROISignals'])
                    if temp.shape[0] == 1200:
                        signals2_AAL.append(temp)
    if len(signals1_AICHA) > len(signals2_AICHA):
        signals1_AICHA = signals1_AICHA[:-1]
        signals1_AAL = signals1_AAL[:-1]
    if len(signals2_AICHA) > len(signals1_AAL):
        signals2_AICHA = signals2_AICHA[:-1]
        Tsignals2_AAL = signals2_AAL[:-1]
                        
                        
np.savez(os.path.join(path_save,'train_data_HCP_AICHA'),*signals_AICHA)
np.savez(os.path.join(path_save,'train_data_HCP_AAL'),*signals_AAL)
np.savez(os.path.join(path_save,'test_data_HCP_AICHA_1'),*signals1_AICHA)
np.savez(os.path.join(path_save,'test_data_HCP_AICHA_2'),*signals2_AICHA)
np.savez(os.path.join(path_save,'test_data_HCP_AAL_1'),*signals1_AAL)
np.savez(os.path.join(path_save,'test_data_HCP_AAL_2'),*signals2_AAL)

with open(os.path.join(path_save,'names_train.txt'), 'w') as f:
    for item in names_train:
        f.write("%s\n" % item)

with open(os.path.join(path_save,'names_test.txt'), 'w') as f:
    for item in test_names:
        f.write("%s\n" % item)