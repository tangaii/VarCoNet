import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from torch.utils.data import Dataset
# The VarCoNet/ABIDE workflow below only needs ``torch.utils.data.Dataset``.
# These aliases retain import compatibility for optional baseline datasets when
# PyTorch Geometric is not installed in the execution environment.
Dataset_geo = Dataset
InMemoryDataset = Dataset
Data_geo = None
Data = None
from collections import defaultdict
import numpy as np
import random
import os
from random import randint
from scipy.signal import detrend
import scipy.sparse as sp
import torch.utils.data as utils


class Loss(ABC):
    @abstractmethod
    def compute(self, anchor, sample, pos_mask, neg_mask, *args, **kwargs) -> torch.FloatTensor:
        pass

    def __call__(self, anchor, sample, pos_mask=None, neg_mask=None, *args, **kwargs) -> torch.FloatTensor:
        loss = self.compute(anchor, sample, pos_mask, neg_mask, *args, **kwargs)
        return loss

def _similarity(h1: torch.Tensor, h2: torch.Tensor):
    h1 = F.normalize(h1)
    h2 = F.normalize(h2)
    return h1 @ h2.t()

class InfoNCE(Loss):
    def __init__(self, tau):
        super(InfoNCE, self).__init__()
        self.tau = tau

    def compute(self, anchor, sample, pos_mask, neg_mask, *args, **kwargs):
        sim = _similarity(anchor, sample) / self.tau
        exp_sim = torch.exp(sim) * (pos_mask + neg_mask)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))
        loss = log_prob * pos_mask
        loss = loss.sum(dim=1) / pos_mask.sum(dim=1)
        return -loss.mean()

class Sampler(ABC):
    def __init__(self, intraview_negs=False):
        self.intraview_negs = intraview_negs

    def __call__(self, anchor, sample, *args, **kwargs):
        ret = self.sample(anchor, sample, *args, **kwargs)
        if self.intraview_negs:
            ret = self.add_intraview_negs(*ret)
        return ret

def get_sampler(mode: str, intraview_negs: bool) -> Sampler:
    if mode in {'L2L', 'G2G'}:
        return SameScaleSampler(intraview_negs=intraview_negs)
    else:
        raise RuntimeError(f'unsupported mode: {mode}')

class SameScaleSampler(Sampler):
    def __init__(self, *args, **kwargs):
        super(SameScaleSampler, self).__init__(*args, **kwargs)

    def sample(self, anchor, sample, *args, **kwargs):
        assert anchor.size(0) == sample.size(0)
        num_nodes = anchor.size(0)
        device = anchor.device
        pos_mask = torch.eye(num_nodes, dtype=torch.float32, device=device)
        neg_mask = 1. - pos_mask
        return anchor, sample, pos_mask, neg_mask

class BatchSampler(Sampler):
    def __init__(self, names, batch_size, shuffle=True, drop_last=True):
        for i,name in enumerate(names):
            pos = name.find('_')
            if pos != -1:
                name = name[:pos]
                names[i] = name
        self.names = names
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.name_to_indices = defaultdict(list)
        for idx, name in enumerate(names):
            self.name_to_indices[name].append(idx)

        self.unique_names = list(np.unique(names))

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.unique_names)

        # Flatten name-grouped indices into batches
        batch = []
        name_in_batch = set()

        for name in self.unique_names:
            indices = self.name_to_indices[name]
            random.shuffle(indices)

            if name in name_in_batch:
                continue

            batch.append(indices[0])
            name_in_batch.add(name)

            if len(batch) == self.batch_size:
                yield batch
                batch = []
                name_in_batch = set()
                
        if not self.drop_last and len(batch) > 0:
            yield batch
            batch = []
            name_in_batch = set()

    def __len__(self):
        return (len(self.unique_names) + self.batch_size - 1) // self.batch_size

def add_extra_mask(pos_mask, neg_mask=None, extra_pos_mask=None, extra_neg_mask=None):
    if extra_pos_mask is not None:
        pos_mask = torch.bitwise_or(pos_mask.bool(), extra_pos_mask.bool()).float()
    if extra_neg_mask is not None:
        neg_mask = torch.bitwise_and(neg_mask.bool(), extra_neg_mask.bool()).float()
    else:
        neg_mask = 1. - pos_mask
    return pos_mask, neg_mask

    
class DualBranchContrast(torch.nn.Module):
    def __init__(self, loss: Loss, mode: str, intraview_negs: bool = False, **kwargs):
        super(DualBranchContrast, self).__init__()
        self.loss = loss
        self.mode = mode
        self.sampler = get_sampler(mode, intraview_negs=intraview_negs)
        self.kwargs = kwargs

    def forward(self, h1=None, h2=None, g1=None, g2=None, batch=None, h3=None, h4=None,
                extra_pos_mask=None, extra_neg_mask=None):
        if self.mode == 'L2L':
            assert h1 is not None and h2 is not None
            anchor1, sample1, pos_mask1, neg_mask1 = self.sampler(anchor=h1, sample=h2)
            anchor2, sample2, pos_mask2, neg_mask2 = self.sampler(anchor=h2, sample=h1)
        elif self.mode == 'G2G':
            assert g1 is not None and g2 is not None
            anchor1, sample1, pos_mask1, neg_mask1 = self.sampler(anchor=g1, sample=g2)
            anchor2, sample2, pos_mask2, neg_mask2 = self.sampler(anchor=g2, sample=g1)

        pos_mask1, neg_mask1 = add_extra_mask(pos_mask1, neg_mask1, extra_pos_mask, extra_neg_mask)
        pos_mask2, neg_mask2 = add_extra_mask(pos_mask2, neg_mask2, extra_pos_mask, extra_neg_mask)
        l1 = self.loss(anchor=anchor1, sample=sample1, pos_mask=pos_mask1, neg_mask=neg_mask1, **self.kwargs)
        l2 = self.loss(anchor=anchor2, sample=sample2, pos_mask=pos_mask2, neg_mask=neg_mask2, **self.kwargs)

        return (l1 + l2) * 0.5
    
class ABIDEDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y
    
    
class ABIDEDataset_BAnD(Dataset):
    def __init__(self, data, labels, path, max_length, mode):
        self.data = data
        self.labels = labels
        self.path = path
        self.max_length = max_length
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = np.load(os.path.join(self.path,self.data[idx]))
        if self.mode == 'train':
            if x.shape[-1] < self.max_length:
                repeats = (self.max_length // x.shape[-1]) + 1
                x = np.tile(x, repeats)
                x = x[:,:,:,:self.max_length]
            else:
                start_idx = np.random.randint(0, x.shape[-1] - self.max_length + 1)
                x = x[:, :, :, start_idx:start_idx + self.max_length]
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y
    
def upper_triangular_cosine_similarity(x):
    N, M, D = x.shape
    x_norm = F.normalize(x, p=2, dim=-1)
    cosine_similarity = torch.matmul(x_norm, x_norm.transpose(1, 2))
    triu_indices = torch.triu_indices(M, M, offset=1)
    upper_triangular_values = cosine_similarity[:, triu_indices[0], triu_indices[1]]
    return upper_triangular_values    

def removeDuplicates(names,inds):
    names_batch = []
    for ind in inds:
        names_batch.append(names[ind])
    names_unique,counts = np.unique(names_batch,return_counts=True)
    if len(names_unique) == len(names_batch):
        return inds
    else:
        non_common = list(set(names).symmetric_difference(set(names_batch)))
        positions = np.where(counts>1)[0]
        for pos in positions:
            name_dupl = names_unique[pos]
            pos_name = np.where(np.array(names_batch) == name_dupl)[0][1]
            names_batch[pos_name] = non_common[random.randint(0, len(non_common)-1)]
            possible_pos = np.where(np.array(names) == names_batch[pos_name])[0]
            inds[pos_name] = possible_pos[random.randint(0,len(possible_pos)-1)]
            non_common = list(set(non_common).symmetric_difference(set(names_batch)))
        return inds

def test_augment(data,wind_sizes,num_winds,max_length):
    windows_all = []
    for wind_size in wind_sizes:
        windows = []
        step_size = (data.shape[0] - wind_size) // (num_winds - 1)
        for i in range(0, step_size*num_winds,step_size):
            temp = torch.zeros(max_length,data.shape[1])
            temp[:wind_size] = data[i:i+wind_size]
            windows.append(temp)
        windows_all.append(torch.stack(windows))
    return torch.cat(windows_all,dim=0)



def test_augment_AE(data,wind_sizes,num_winds,max_length):
    for wind_size in wind_sizes:
        windows = []
        step_size = (data.shape[0] - wind_size) // (num_winds - 1)
        for i in range(0, step_size*num_winds,step_size):
            temp = torch.zeros(max_length,data.shape[1])
            temp[:wind_size] = data[i:i+wind_size]
            windows.append(temp)
    return torch.stack(windows)

"""
def augment_hcp(data, train_length_limits, device):
    max_length = train_length_limits[-1]   
    data1 = []
    data2 = []
    for dat in data:
        length1 = random.randint(train_length_limits[0],train_length_limits[-1]-1)
        length2 = random.randint(train_length_limits[0],train_length_limits[-1]-1)

        a11 = int(length1/2+1)
        a12 = dat.shape[0]-int(length1/2+1)
        a21 = int(length2/2+1)
        a22 = dat.shape[0]-int(length2/2+1)
        if a12 > a11:
            c1 = random.randint(a11,a12)
        else:
            c1 = a11
        if a22 > a21:
            c2 = random.randint(a21,a22)   
        else:
            c2 = a21  
        counter = 0            
        c1s = [c1]
        c2s = [c2]
        dists = [abs(c1 - c2)]
        while counter < 100:
            if a12 > a11:
                c1 = random.randint(a11,a12)
            else:
                c1 = a11
            if a22 > a21:
                c2 = random.randint(a21,a22)   
            else:
                c2 = a21
            c1s.append(c1)
            c2s.append(c2)
            dists.append(abs(c1 - c2))
            counter = counter + 1
        max_dist = max(dists)
        max_pos = dists.index(max_dist)
        c1 = c1s[max_pos]
        c2 = c2s[max_pos]
        temp = torch.zeros(max_length,dat.shape[1])
        if length1 % 2 == 0:
            temp[:length1,:] = dat[c1-int(length1/2):c1+int(length1/2),:]
        else:
            temp[:length1,:] = dat[c1-int(length1/2):c1+int(length1/2)+1,:]
        data1.append(temp) 
        temp = torch.zeros(max_length,dat.shape[1])
        if length2 % 2 == 0:
            temp[:length2,:] = dat[c2-int(length2/2):c2+int(length2/2),:]
        else:
            temp[:length2,:] = dat[c2-int(length2/2):c2+int(length2/2)+1,:]
        data2.append(temp)
    data1 = torch.stack(data1).to(device)
    data2 = torch.stack(data2).to(device)
    return [data1,data2]
"""

def augment(data, train_length_limits, device):
    data = data.to(device)  
    max_length = train_length_limits[-1]
    batch_size, _, feat_dim = data.shape
    data1 = torch.zeros((batch_size, max_length, feat_dim), device=device)
    data2 = torch.zeros((batch_size, max_length, feat_dim), device=device)

    for i, dat in enumerate(data):
        zero_rows = torch.all(dat == 0, dim=1)
        zero_row_indices = torch.where(zero_rows)[0]
        if len(zero_row_indices) > 0:
            dat = dat[:torch.min(zero_row_indices), :]

        low_limit = train_length_limits[0]
        up_limit = min(dat.shape[0], train_length_limits[1])

        length1 = torch.randint(low_limit, up_limit, (1,), device=device).item()
        length2 = torch.randint(low_limit, up_limit, (1,), device=device).item()

        a11, a12 = length1 // 2 + 1, dat.shape[0] - length1 // 2 - 2
        a21, a22 = length2 // 2 + 1, dat.shape[0] - length2 // 2 - 2

        # Compute 100 alternative distances on GPU
        c1_candidates = torch.randint(a11, max(a12, a11 + 1), (100,), device=device)
        c2_candidates = torch.randint(a21, max(a22, a21 + 1), (100,), device=device)
        dists = torch.abs(c1_candidates - c2_candidates)

        # Select the maximum distance
        max_pos = torch.argmax(dists).item()
        c1, c2 = c1_candidates[max_pos].item(), c2_candidates[max_pos].item()

        # Slice & store in preallocated tensors
        if length1 % 2 == 0:
            data1[i, :length1, :] = dat[c1 - length1 // 2 : c1 + length1 // 2, :]
        else:
            data1[i, :length1, :] = dat[c1 - length1 // 2 : c1 + length1 // 2 + 1, :]

        if length2 % 2 == 0:
            data2[i, :length2, :] = dat[c2 - length2 // 2 : c2 + length2 // 2, :]
        else:
            data2[i, :length2, :] = dat[c2 - length2 // 2 : c2 + length2 // 2 + 1, :]

    return [data1, data2]


def other_augment1(data, train_length_limits, device):
    data = data.to(device)  
    max_length = train_length_limits[-1]
    batch_size, _, feat_dim = data.shape
    data1 = torch.zeros((batch_size, max_length, feat_dim), device=device)
    data2 = torch.zeros((batch_size, max_length, feat_dim), device=device)
    data3 = torch.zeros((batch_size, max_length, feat_dim), device=device)
    data4 = torch.zeros((batch_size, max_length, feat_dim), device=device)
    for i, dat in enumerate(data):
        zero_rows = torch.all(dat == 0, dim=1)
        zero_row_indices = torch.where(zero_rows)[0]
        if len(zero_row_indices) > 0:
            dat = dat[:torch.min(zero_row_indices), :]
        windowsize = int(dat.shape[0]/3)
        strite = int(windowsize/2)
        t = int((dat.shape[0]-windowsize)/strite)+1
        k1 = randint(0, t-1)*strite
        k2 = randint(0, t-1)*strite
        while k1 == k2:
            k2 = randint(0, t-1)*strite
        data1[i, :windowsize, :] = dat[k1 : k1+windowsize, :]
        data2[i, :windowsize, :] = dat[k2 : k2+windowsize, :]
        
        windowsizes = [int(dat.shape[0]/5),int(dat.shape[0]/4),int(dat.shape[0]/3),int(dat.shape[0]/2),dat.shape[0]]
        windowsizes = [x for x in windowsizes if x <= max_length]
        windowsize1 = windowsizes[randint(0,len(windowsizes)-1)]
        windowsize2 = windowsizes[randint(0,len(windowsizes)-1)]
        while windowsize2 == windowsize1:
            windowsize2 = windowsizes[randint(0,len(windowsizes)-1)]
        if windowsize1 > windowsize2:
            strite = int(windowsize1/2)
            t = int((dat.shape[0]-windowsize1)/strite)+1
        else:
            strite = int(windowsize2/2)
            t = int((dat.shape[0]-windowsize2)/strite)+1
        if t > 1:
            k = randint(1, t-1)*strite
        else:
            k=strite
        if windowsize1 % 2 == 0:
            data3[i, :windowsize1, :] = dat[k - windowsize1 // 2 : k + windowsize1 // 2, :]
        else:
            data3[i, :windowsize1, :] = dat[k - windowsize1 // 2 : k + windowsize1 // 2 + 1, :]

        if windowsize2 % 2 == 0:
            data4[i, :windowsize2, :] = dat[k - windowsize2 // 2 : k + windowsize2 // 2, :]
        else:
            data4[i, :windowsize2, :] = dat[k - windowsize2 // 2 : k + windowsize2 // 2 + 1, :]
        
    return [data1, data2, data3, data4]

def other_augment2(data, train_length_limits, device):
    data = data.to(device)  
    max_length = train_length_limits[-1]
    batch_size, _, feat_dim = data.shape
    data1 = torch.zeros((batch_size, max_length, feat_dim), device=device)
    data2 = torch.zeros((batch_size, max_length, feat_dim), device=device)
    for i, dat in enumerate(data):
        zero_rows = torch.all(dat == 0, dim=1)
        zero_row_indices = torch.where(zero_rows)[0]
        if len(zero_row_indices) > 0:
            dat = dat[:torch.min(zero_row_indices), :]
        if dat.shape[0] <= max_length:
            windowsize = int(0.9*dat.shape[0])
        else:
            dat = dat[:max_length,:]
            windowsize = int(0.9*dat.shape[0])
        data1[i, :windowsize, :] = dat[:windowsize, :]
        data2[i, :windowsize, :] = dat[-windowsize:, :] 
        
    return [data1, data2]


def test_augment_overlap(data, wind_sizes, overlap, min_length, max_length, device):
    data = data.to(device)  

    zero_rows = torch.all(data == 0, dim=1)
    zero_row_indices = torch.where(zero_rows)[0]
    if len(zero_row_indices) > 0:
        data = data[:torch.min(zero_row_indices), :]

    windows_all = []
    data_len = data.shape[0]

    for i in range(len(wind_sizes) - 1):
        if data_len // wind_sizes[i+1] >= min_length:
            wind_size_min = max(min_length, wind_sizes[i])
            wind_size_max = min(data_len // wind_sizes[i+1], data_len)
            wind_size = torch.randint(wind_size_min, wind_size_max + 1, (1,), device=device).item()

            step_size = int(wind_size * (1 - overlap))
            num_windows = max(1, (data_len - wind_size) // step_size + 1)

            indices = torch.arange(0, num_windows * step_size, step_size, device=device).unsqueeze(1)
            range_indices = torch.arange(wind_size, device=device).unsqueeze(0)
            window_indices = indices + range_indices  # Shape: (num_windows, wind_size)

            windows = data[window_indices]  # Gather windows using tensor indexing

            # Pad windows to max_length (if needed)
            pad_size = max_length - wind_size
            if pad_size > 0:
                pad_tensor = torch.zeros((windows.shape[0], pad_size, data.shape[1]), device=device)
                windows = torch.cat((windows, pad_tensor), dim=1)

            windows_all.append(windows)

    return torch.cat(windows_all, dim=0) if windows_all else torch.empty(0, max_length, data.shape[1], device=device)


def repeat_to_fill(x):
    batch_size, seq_len, num_channels = x.shape
    new_x = torch.zeros_like(x)
    for i in range(batch_size):
        nonzero_rows = (x[i] != 0).any(dim=1)
        N = nonzero_rows.sum().item()
        if N == 0:
            continue
        signal = x[i, :N,:]
        if seq_len // N > 1:
            for i in range(seq_len // N):
                if i == 0:
                    new_signal = torch.concat((signal,signal),dim=0)
                elif i < seq_len // N - 1:
                    new_signal = torch.concat((new_signal,signal),dim=0)
                else:
                    new_signal = torch.concat((new_signal,signal[:seq_len-new_signal.shape[0],:]),dim=0)
        elif seq_len // N == 1: 
            new_signal = torch.concat((signal,signal[:seq_len-signal.shape[0],:]),dim=0)
        else:
            new_signal = signal
        new_x[i] = new_signal
    return new_x


def PCC(X):
    M, N = X.shape
    mean = X.mean(dim=1, keepdim=True)
    X_centered = X - mean
    std = X_centered.std(dim=1, unbiased=True, keepdim=True)
    std[std == 0] = 1
    Z = X_centered / std
    corr_matrix = torch.matmul(Z, Z.T) / (N - 1)
    return corr_matrix

def augment_VAE(data,a,b,device):
    x = []
    for dat in data:
        length = random.randint(a, b)
        max_start = dat.shape[0] - length
        start = random.randint(0, max_start)
        dat = dat[start:start + length,:]
        corr = PCC(dat.T)
        triu_indices = torch.triu_indices(corr.shape[0], corr.shape[0], offset=1)
        upper_triangular_values = corr[triu_indices[0], triu_indices[1]]
        x.append(upper_triangular_values)
    return torch.stack(x).float().to(device)


def test_augment_PCC(data,wind_sizes,num_winds):
    new_data = []
    for wind_size in wind_sizes:
        step_size = (data.shape[0] - wind_size) // (num_winds - 1)
        for i in range(0, step_size*num_winds,step_size):
            segment = data[i:i+wind_size]
            corr = np.corrcoef(segment.T)
            triu_indices = np.triu_indices(corr.shape[0], k=1)
            upper_triangular_values = corr[triu_indices[0], triu_indices[1]]
            new_data.append(upper_triangular_values)
    return np.stack(new_data)


def compute_alff_falff(AllVolume, ASamplePeriod, HighCutoff, LowCutoff):

    sampleFreq = 1.0 / ASamplePeriod
    sampleLength = AllVolume.shape[0]
    paddedLength = 2**int(np.ceil(np.log2(sampleLength)))

    if LowCutoff >= sampleFreq / 2:
        idx_LowCutoff = paddedLength // 2 + 1
    else:
        idx_LowCutoff = int(np.ceil(LowCutoff * paddedLength * ASamplePeriod + 1))

    if (HighCutoff >= sampleFreq / 2) or (HighCutoff == 0):
        idx_HighCutoff = paddedLength // 2 + 1
    else:
        idx_HighCutoff = int(np.fix(HighCutoff * paddedLength * ASamplePeriod + 1))
        
    AllVolume = detrend(AllVolume, axis=0)


    # Zero padding
    padding = paddedLength - sampleLength
    if padding > 0:
        AllVolume = np.vstack([AllVolume, np.zeros((padding, AllVolume.shape[1]))])

    AllVolume = 2 * np.abs(np.fft.fft(AllVolume, axis=0)) / sampleLength

    ALFF_2D = AllVolume[idx_LowCutoff:idx_HighCutoff, :].mean(axis=0)

    return ALFF_2D

class AGCLDataset(InMemoryDataset):
    def __init__(self, root, data, y, atlas, transform=None, pre_transform=None):
        self.root = root
        self.data = data
        self.atlas = atlas
        self.y = y

        super(AGCLDataset, self).__init__(root,transform, pre_transform)
        path = os.path.join(self.processed_dir, 'data_' + atlas + '.pt')
        self.data, self.slices = torch.load(path)

    @property
    def processed_dir(self):
        return os.path.join(self.root, 'processed')
    
    @property
    def processed_file_names(self):
        return  'data_' + self.atlas + '.pt'


    def process(self):
        atlas = self.atlas
        data_list = []
        for signal,y in zip(self.data, self.y):  
            label = torch.squeeze(torch.tensor(y, dtype=torch.long))
            if np.any(np.all(signal == 0, axis=1)):
                min_index = np.min(np.where(np.all(signal == 0, axis=1)))
                signal = signal[:min_index,:]
                signal = (signal - np.mean(signal, axis=0, keepdims=True))/np.std(signal, axis=0, keepdims=True)
            pcc = np.corrcoef(signal.T)
            alff_s5 = compute_alff_falff(signal, 1.5, 0.027, 0.01)
            alff_s4 = compute_alff_falff(signal, 1.5, 0.073, 0.027)
            alff_classic = compute_alff_falff(signal, 1.5, 0.08, 0.01)
            alffs = np.stack([alff_s5, alff_s4, alff_classic])
            alffs = (alffs - np.min(alffs)/np.max(alffs) - np.min(alffs))
            alffs = alffs.T
            pcc -= np.eye(pcc.shape[0])
            pcc /= np.max(np.abs(pcc))
            pcc += np.eye(pcc.shape[0])  
            x = np.nan_to_num(alffs)
            x = torch.Tensor(x)

            edge_index = pcc

            edge_index = np.nan_to_num(edge_index)
            edge_index_temp = sp.coo_matrix(edge_index)
            edge_weight = torch.Tensor(edge_index_temp.data)

            edge_index = torch.Tensor(edge_index)  
            edge_index = edge_index.nonzero(as_tuple=False).t().contiguous()
            num_nodes = int(edge_index.max()) + 1

            data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight,y=label)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)
        
        torch.save(self.collate(data_list),
                os.path.join(self.processed_dir, 'data_' + atlas + '.pt'))

    def __repr__(self):
        return '{}({})'.format(self.name, len(self))

class AGCLDataset2(Dataset_geo):
    def __init__(self, signals, labels):
        self.signals = signals
        self.labels = labels

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = self.signals[idx]
        label = self.labels[idx]
        label = torch.squeeze(torch.tensor(label, dtype=torch.long))
        if np.any(np.all(signal == 0, axis=1)):
            min_index = np.min(np.where(np.all(signal == 0, axis=1)))
            signal = signal[:min_index,:]
            signal = (signal - np.mean(signal, axis=0, keepdims=True))/np.std(signal, axis=0, keepdims=True)
        pcc = np.corrcoef(signal.T)
        alff_s5 = compute_alff_falff(signal, 1.5, 0.027, 0.01)
        alff_s4 = compute_alff_falff(signal, 1.5, 0.073, 0.027)
        alff_classic = compute_alff_falff(signal, 1.5, 0.08, 0.01)
        alffs = np.stack([alff_s5, alff_s4, alff_classic])
        alffs = (alffs - np.min(alffs)/np.max(alffs) - np.min(alffs))
        alffs = alffs.T
        pcc -= np.eye(pcc.shape[0])
        pcc /= np.max(np.abs(pcc))
        pcc += np.eye(pcc.shape[0])  
        x = np.nan_to_num(alffs)
        x = torch.Tensor(x)
        edge_index = pcc
        edge_index = np.nan_to_num(edge_index)
        edge_index_temp = sp.coo_matrix(edge_index)
        edge_weight = torch.Tensor(edge_index_temp.data)
        edge_index = torch.Tensor(edge_index)  
        edge_index = edge_index.nonzero(as_tuple=False).t().contiguous()
        data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight,y=label)
        return data
    
    
class UCGLDataset(Dataset):
    def __init__(self, signals, labels):
        self.signals = signals
        self.labels = labels

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = self.signals[idx]
        label = self.labels[idx]
        T = signal.shape[0]
        S = int(0.9*T)
        data1 = signal[:S,:]
        data2 = signal[-S:,:]
        data1 = torch.from_numpy(data1).unsqueeze(0).unsqueeze(-1)
        data2 = torch.from_numpy(data2).unsqueeze(0).unsqueeze(-1)
        y = torch.squeeze(torch.tensor(label, dtype=torch.long))
        return data1, data2, y
    
class UCGLDataset2(Dataset):
    def __init__(self, signals, labels):
        self.signals = signals
        self.labels = labels

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = self.signals[idx]
        label = self.labels[idx]
        data = torch.from_numpy(signal).unsqueeze(0).unsqueeze(-1)
        y = torch.squeeze(torch.tensor(label, dtype=torch.long))
        return data, y
    
def LoadAllUCGL(signals):
    data_alla1 = None
    data_alla2 = None
    
    for i in range(len(signals)): #subjects
        signal = signals[i]
        if np.any(np.all(signal == 0, axis=1)):
            min_index = np.min(np.where(np.all(signal == 0, axis=1)))
            signal = signal[:min_index,:] 
        T = signal.shape[0]
        S = int(0.9*T)
        X1=signal[:S,:]
        X2=signal[-S:,:]
        
        if data_alla1 is None:              
           data_alla1 = X1
        else:
           data_alla1 = np.concatenate((data_alla1, X1), axis=0)
           
        if data_alla2 is None:              
           data_alla2 = X2
        else:
           data_alla2 = np.concatenate((data_alla2, X2), axis=0) 
    return data_alla1, data_alla2


class GCDADataset(Dataset):
    def __init__(self, signals, labels):
        self.signals = signals
        self.labels = labels

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = self.signals[idx]
        label = self.labels[idx]
        if np.any(np.all(signal == 0, axis=1)):
            min_index = np.min(np.where(np.all(signal == 0, axis=1)))
            signal = signal[:min_index,:] 
        pc = np.corrcoef(signal.T)
        pc = np.nan_to_num(pc)
        pc = abs(pc)              
        E = torch.from_numpy(pc)
        E = torch.where(E >= 0.3, torch.tensor(1.0), torch.tensor(0.0)).long()
        E = F.one_hot(E, num_classes=2)  
        X = torch.from_numpy(pc)
        y_0 = torch.zeros([0]).float()
        y = torch.from_numpy(y_0.numpy())
        label = torch.squeeze(torch.tensor(label, dtype=torch.long))
        return X, E, y, label
    
class GCDADataset2(Dataset):
    def __init__(self, signals, labels):
        self.signals = signals
        self.labels = labels

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = self.signals[idx]
        label = self.labels[idx]
        if np.any(np.all(signal == 0, axis=1)):
            min_index = np.min(np.where(np.all(signal == 0, axis=1)))
            signal = signal[:min_index,:] 
        pc = np.corrcoef(signal.T)
        pc = np.nan_to_num(pc)
        pc = abs(pc)              
        E = torch.from_numpy(pc)
        E = torch.where(E >= 0.3, torch.tensor(1.0), torch.tensor(0.0)).long()
        X = torch.from_numpy(pc)
        E = E.unsqueeze(0)
        X = X.unsqueeze(0)
        label = torch.squeeze(torch.tensor(label, dtype=torch.long))
        return X, E, label
    
def GCDA_dims(atlas):
    if atlas == 'AAL':
        roi_num = 166
    elif atlas == 'AICHA':
        roi_num = 384
    A20 = np.random.rand(10, 320, roi_num)
    pc_l = []
    y_list = []      
    for i in range(len(A20)): 
        pc = np.corrcoef(A20[i].T)
        pc = np.nan_to_num(pc)
        pc = abs(pc)         
        pc_l.append(pc)
        y_0 = torch.zeros([0]).float()
        y_np = y_0.numpy()
        y_list.append(y_np)
    X = np.array(pc_l) 
    E = torch.from_numpy(X)
    E = torch.where(E >= 0.3, torch.tensor(1.0), torch.tensor(0.0)).long()
    E = F.one_hot(E, num_classes=2)
    y = np.array(y_list) 
    X = torch.from_numpy(X)
    y = torch.from_numpy(y)
    return  X,E,y


def BrainIBDataset(signals, labels):
    graphs = []
    for signal, label in zip(signals, labels):
        if np.any(np.all(signal == 0, axis=1)):
            min_index = np.min(np.where(np.all(signal == 0, axis=1)))
            signal = signal[:min_index,:] 
        corr = np.corrcoef(signal.T)
        corr = np.nan_to_num(corr)
        abs_corr = np.abs(corr)
        threshold = np.percentile(abs_corr, 80)
        adj = (abs_corr > threshold).astype(int)
        np.fill_diagonal(adj, 0)
        corr_triu = np.triu(corr, k=1)
        r, c = np.where(abs_corr > threshold)
        r = r.astype(int)
        c = c.astype(int)
        rows, columns, vals = [], [], []
        neighbors = {}
        for j in range(corr.shape[0]):
            pos = np.where(r == j)[0]
            rows.extend(r[pos])
            columns.extend(c[pos])
            linear_idx = np.ravel_multi_index((r[pos], c[pos]), corr_triu.shape)
            vals.extend(linear_idx)
            neighbors[j] = list(c[pos])
        
        edge = np.vstack((rows, columns, vals)).T
        corr_no_diag = corr.copy()
        np.fill_diagonal(corr_no_diag, 0)
        node_tags = np.arange(corr.shape[0])
    
        edge = torch.Tensor(edge)
        ROI = torch.Tensor(corr_no_diag)
        node_tags = torch.Tensor(node_tags)
        adj = torch.Tensor(adj)
        y = torch.squeeze(torch.tensor(label, dtype=torch.long))
        A = torch.sparse_coo_tensor(
            indices = edge[:, :2].t().long(),
            values = edge[:, -1].reshape(-1,).float(),
            size = (corr.shape[0], corr.shape[0])
            )
        G = (A.t() + A).coalesce()
        graph = Data_geo(x=ROI.reshape(-1,corr.shape[0]).float(),
                     edge_index=G.indices().reshape(2,-1).long(),
                     edge_attr=G.values().reshape(-1,1).float(),
                     y=y.long())
        graphs.append(graph)
    return graphs




class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def FBNETGENLoader(final_fc, labels, batch_size):
    final_pearson = []
    for signal in final_fc:
        corr = np.corrcoef(signal)
        final_pearson.append(corr)
    final_pearson = np.stack(final_pearson)

    timeseries = signal.shape[1]

    _, node_size, node_feature_size = final_pearson.shape

    scaler = StandardScaler(mean=np.mean(
        final_fc), std=np.std(final_fc))
    
    final_fc = scaler.transform(final_fc)

    final_fc, final_pearson, labels = [torch.from_numpy(
        data).float() for data in (final_fc, final_pearson, labels)]


    dataset = utils.TensorDataset(
        final_fc,
        final_pearson,
        labels
    )

    dataloader = utils.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    return dataloader, node_size, node_feature_size, timeseries
