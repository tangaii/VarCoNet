import torch.nn as nn
from torch.nn.functional import avg_pool1d
import torch.nn.functional as F
from torch.nn import Conv1d, MaxPool1d, Linear
import torch

def upper_triangular_cosine_similarity(x):
    N, M, D = x.shape
    x_norm = F.normalize(x, p=2, dim=-1)
    cosine_similarity = torch.matmul(x_norm, x_norm.transpose(1, 2))
    triu_indices = torch.triu_indices(M, M, offset=1)
    upper_triangular_values = cosine_similarity[:, triu_indices[0], triu_indices[1]]
    return upper_triangular_values 


class PositionalEncodingTrainable(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncodingTrainable, self).__init__()
        pe = torch.zeros(1, max_seq_length, d_model)
        self.pe = nn.Parameter(pe)#

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class CNN_Transformer(nn.Module):
    def __init__(self, d_model, n_layers, n_heads, dim_feedforward, max_len):
        super(CNN_Transformer, self).__init__()
        self.d_model = d_model
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward, batch_first = True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        self.conv1 = Conv1d(in_channels=1, out_channels=16, kernel_size=4, stride=2)
        self.IN = MaskedInstanceNorm1d(d_model)
        new_len = (max_len - 4) // 2 + 1
        self.pos_enc = PositionalEncodingTrainable(d_model,new_len)
        
        
    def forward(self, x, x_mask=None):   
        b, k, d = x.shape
        mask = (x[:,:,0] != 0)
        x = self.IN(x,mask)
        x = torch.transpose(x, 1, 2)
        x = x.contiguous()
        x = x.view((b*d, 1, k))
        x = self.conv1(x)
        x = avg_pool1d(x.permute(0, 2, 1), kernel_size=16).permute(0, 2, 1)
        x = x.view((b, d, -1))
        x = torch.transpose(x, 1, 2)
        last_vals = x[:, -1:, :]
        matches = (x == last_vals)
        counts = matches.sum(dim=1)
        mask = (counts >= 2).unsqueeze(1).expand_as(matches)
        x = x.masked_fill(matches & mask, 0)
        x_mask = (x[:, :, 0] == 0).bool()    
        x = self.pos_enc(x)
        x[x_mask,:] = 0
        x = self.transformer_encoder(x, src_key_padding_mask=x_mask)  
        x[x_mask,:] = 0
        x = torch.transpose(x, 1, 2)
        return x

class MaskedInstanceNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("gamma", None)
            self.register_parameter("beta", None)

    def forward(self, x, mask):
        """
        x: [B, L, C]  (batch, length, channels)
        mask: [B, L]  (1 for valid positions, 0 for padding)
        """
        B, L, C = x.shape
        mask = mask.unsqueeze(-1).to(x.dtype)  # [B, L, 1]
        x_masked = x * mask
        valid_counts = mask.sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1, 1]
        mean = x_masked.sum(dim=1, keepdim=True) / valid_counts
        var = ((x_masked - mean) * mask).pow(2).sum(dim=1, keepdim=True) / valid_counts
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        x_norm = x_norm * mask
        if self.affine:
            x_norm = x_norm * self.gamma + self.beta

        return x_norm


class Transformer(nn.Module):
    def __init__(self, d_model, n_layers, n_heads, dim_feedforward, max_len):
        super(Transformer, self).__init__()
        self.d_model = d_model
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward, batch_first = True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        self.pos_enc = PositionalEncodingTrainable(d_model,max_len)
        self.IN = MaskedInstanceNorm1d(d_model)
        
        
    def forward(self, x, x_mask=None):   
        b, k, d = x.shape
        mask = (x[:,:,0] != 0)
        x = self.IN(x,mask)
        x_mask = (x[:, :, 0] == 0).bool()    
        x = self.pos_enc(x)
        x[x_mask,:] = 0
        x = self.transformer_encoder(x, src_key_padding_mask=x_mask)     
        x[x_mask,:] = 0
        x = torch.transpose(x, 1, 2)
        return x
    
class ConvKRegion(nn.Module):

    def __init__(self, k=1, out_size=8, kernel_size=8, pool_size=8, time_series=180, channels=166):
        super().__init__()
        self.conv1 = Conv1d(in_channels=k, out_channels=32,
                            kernel_size=kernel_size, stride=2)
        output_dim_1 = (time_series-kernel_size)//2+1

        self.conv2 = Conv1d(in_channels=32, out_channels=32,
                            kernel_size=16)
        output_dim_2 = output_dim_1 - 16 + 1
        self.conv3 = Conv1d(in_channels=32, out_channels=16,
                            kernel_size=8)
        output_dim_3 = output_dim_2 - 8 + 1
        self.max_pool1 = MaxPool1d(pool_size)
        output_dim_4 = output_dim_3 // pool_size * 16
        self.in0 = MaskedInstanceNorm1d(channels)
        self.in1 = nn.BatchNorm1d(32)
        self.in2 = nn.BatchNorm1d(32)
        self.in3 = nn.BatchNorm1d(16)

        self.linear = nn.Sequential(
            Linear(output_dim_4, 32),
            nn.LeakyReLU(negative_slope=0.2),
            Linear(32, out_size)
        )

    def forward(self, x):

        b, k, d = x.shape
        mask = (x[:,:,0] != 0)
        x = self.in0(x,mask)
        x = torch.transpose(x, 1, 2)
        x = x.contiguous()
        x = x.view((b*d, 1, k))
        x = self.conv1(x)
        x = self.in1(x)
        x = self.conv2(x)
        x = self.in2(x)
        x = self.conv3(x)
        x = self.in3(x)
        x = self.max_pool1(x)
        x = x.view((b, d, -1))
        last_vals = x[:, :, -1:]
        matches = (x == last_vals)
        counts = matches.sum(dim=2)
        mask = (counts >= 2).unsqueeze(2).expand_as(matches)
        x = x.masked_fill(matches & mask, 0)
        x = self.linear(x)
        return x

class VarCoNet(nn.Module):

    def __init__(self, model_config, roi_num):
        super().__init__()

        self.extract = CNN_Transformer(
                d_model=roi_num,
                n_layers=model_config['layers'],n_heads=model_config['n_heads'],
                dim_feedforward=model_config['dim_feedforward'],
                max_len=model_config['max_length'])
    def forward(self, x):
        x = self.extract(x)
        x = upper_triangular_cosine_similarity(x)
        return x 
    
class VarCoNet_noCNN(nn.Module):

    def __init__(self, model_config, roi_num):
        super().__init__()

        self.extract = Transformer(
                d_model=roi_num,
                n_layers=model_config['layers'],n_heads=model_config['n_heads'],
                dim_feedforward=model_config['dim_feedforward'],
                max_len=model_config['max_length'])
    def forward(self, x):
        x = self.extract(x)
        x = upper_triangular_cosine_similarity(x)
        return x 
    
class VarCoNet_noTransformer(nn.Module):

    def __init__(self, model_config, roi_num):
        super().__init__()

        self.extract = ConvKRegion(
            out_size=128, 
            kernel_size=16,
            pool_size=4,
            time_series=model_config['max_length'],
            channels=roi_num
            )
    def forward(self, x):
        x = self.extract(x)
        x = upper_triangular_cosine_similarity(x)
        return x 
    
class VarCoNet_noSSL(nn.Module):

    def __init__(self, model_config, roi_num, num_classes):
        super().__init__()

        self.extract = Transformer(
                d_model=roi_num,
                n_layers=model_config['layers'],n_heads=model_config['n_heads'],
                dim_feedforward=model_config['dim_feedforward'],
                max_len=model_config['max_length'])
        self.linear = nn.Sequential(nn.Linear(int(roi_num*(roi_num-1)/2),
                                              num_classes),nn.Softmax(dim=-1))
    def forward(self, x):
        x = self.extract(x)
        x = upper_triangular_cosine_similarity(x)
        x = self.linear(x)
        return x 