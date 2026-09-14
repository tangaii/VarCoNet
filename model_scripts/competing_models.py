import torch
import torch.nn as nn
from torch.nn import Linear, Conv1d, MaxPool1d, GRU
import torch.nn.functional as F
import numpy as np
import math

class VAE(nn.Module):
    def __init__(self, input_dim=64620, hidden_dims=[3000, 2000, 1000, 500, 100]):
        super(VAE, self).__init__()
        
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.Tanh(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.Tanh(),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.BatchNorm1d(hidden_dims[2]),
            nn.Tanh(),
            nn.Linear(hidden_dims[2], hidden_dims[3]),
            nn.BatchNorm1d(hidden_dims[3]),
            nn.Tanh(),
        )

        # Latent space
        self.fc_mu = nn.Linear(hidden_dims[3], hidden_dims[4])
        self.fc_logvar = nn.Linear(hidden_dims[3], hidden_dims[4])

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dims[4], hidden_dims[3]),
            nn.BatchNorm1d(hidden_dims[3]),
            nn.Tanh(),
            nn.Linear(hidden_dims[3], hidden_dims[2]),
            nn.BatchNorm1d(hidden_dims[2]),
            nn.Tanh(),
            nn.Linear(hidden_dims[2], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.Tanh(),
            nn.Linear(hidden_dims[1], hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.Tanh(),
            nn.Linear(hidden_dims[0], input_dim),
        )

        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)  # Scaled variance initialization
                torch.nn.init.zeros_(m.bias)

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar    


class Model_VAE(nn.Module):

    def __init__(self, roi_num):
        super().__init__()

        self.vae = VAE(input_dim=int((roi_num*(roi_num-1))/2))
        
    def forward(self, x, train):
        y,mu,logvar = self.vae(x)
        if train:
            return y,mu,logvar 
        else:
            return y
        
        

class AE(nn.Module):
    def __init__(self, input_dim=80, hidden_dims=[500, 500, 2000], bottleneck=10):
        super(AE, self).__init__()
        
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU(),
            nn.Linear(hidden_dims[2], bottleneck),
            nn.ReLU(),
        )

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, hidden_dims[2]),
            nn.ReLU(),
            nn.Linear(hidden_dims[2], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], input_dim)
        )  
        
    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)
        
    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x


class Model_AE(nn.Module):

    def __init__(self, length):
        super().__init__()
        self.ae = AE(input_dim=length)
        self.length = length
        
    def forward(self, x):
        y = self.ae(x.permute(0,2,1).contiguous().view(-1,self.length))
        y = y.view(x.shape[0],x.shape[2],x.shape[1]).permute(0,2,1)
        return y
    
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length = 320):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_layers, max_len):
        super(TransformerBlock, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=4,
            activation='gelu',
            norm_first = True,
            batch_first = True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer,num_layers)
        self.pos_enc = PositionalEncoding(d_model,max_len)

    def forward(self, x):
        x_mask = (x[:, :, 0] == 0)      
        x = self.pos_enc(x)
        return self.transformer(x, src_key_padding_mask = x_mask)


class CrossViewModel(nn.Module):
    def __init__(self, model_config):
        super(CrossViewModel, self).__init__()
        self.transformer1 = TransformerBlock(model_config['d_model1'], 
                                             model_config['num_layers1'],
                                             model_config['length'])
        self.transformer2 = TransformerBlock(model_config['d_model2'], 
                                             model_config['num_layers2'],
                                             model_config['length'])
        self.transformer3 = TransformerBlock(model_config['d_model1'], 
                                             model_config['num_layers'],
                                             model_config['length'])
        self.transformer4 = TransformerBlock(model_config['d_model2'], 
                                             model_config['num_layers'],
                                             model_config['length'])
        self.linear_layer = nn.Linear(model_config['window_size']**2, model_config['patch_proj'])
        self.classifier = nn.Sequential(nn.Linear(2*model_config['length'], 
                                                  model_config['num_classes']),nn.Softmax(dim=-1))
        self.cls_token1 = torch.nn.Parameter(torch.randn(1, model_config['length'], 1))
        self.cls_token2 = torch.nn.Parameter(torch.randn(1, model_config['length'], 1))
        self.linear1 = nn.Sequential(nn.Linear(model_config['d_model1']-1, 128), nn.Linear(128, 64))
        self.linear2 = nn.Sequential(nn.Linear(model_config['d_model2']-1, 128), nn.Linear(128, 64))

    def forward(self, input1, input2, window_size):
        input2 = torch.permute(process_windows(input2, self.linear_layer, window_size),(0,2,1))
        input1 = input1[:,:,:-1]
        input2 = input2[:,:,:-1]
        
        cls_token1 = self.cls_token1.expand(input1.shape[0], -1, -1)
        cls_token2 = self.cls_token2.expand(input2.shape[0], -1, -1)
        
        input1 = torch.cat((cls_token1, input1), dim=-1)
        input2 = torch.cat((cls_token2, input2), dim=-1)
        
        # First-stage transformers
        output1 = self.transformer1(input1)
        output2 = self.transformer2(input2)

        # Cross-view encoder
        cls1, cls2 = output1[:,:,0], output2[:,:,0]
        swapped_input1 = torch.cat([cls2.unsqueeze(-1), output1[:,:,1:]], dim=-1)
        swapped_input2 = torch.cat([cls1.unsqueeze(-1), output2[:,:,1:]], dim=-1)

        # Second-stage transformers
        final_output1 = self.transformer3(swapped_input1)
        final_output2 = self.transformer4(swapped_input2)

        # Extract CLS tokens for classification
        cls_final1, cls_final2 = final_output1[:,:,0], final_output2[:,:,0]
        emb1 = final_output1[:,:,1:]
        emb2 = final_output2[:,:,1:]
        emb1 = self.linear1(emb1)
        emb2 = self.linear2(emb2)
        emb1 = emb1.contiguous().view(emb1.shape[0],-1)
        emb2 = emb2.contiguous().view(emb2.shape[0],-1)
        combined_cls = torch.cat([cls_final1, cls_final2], dim=-1)

        return self.classifier(combined_cls),emb1,emb2
     

def process_windows(input_tensor, linear_layer, window_size=12):
    batch_size, height, width = input_tensor.shape
    windows = input_tensor.unfold(1, window_size, window_size).unfold(2, window_size, window_size)
    windows = windows.contiguous().view(batch_size, -1, window_size * window_size)
    output = linear_layer(windows)  
    return output



class CNN_1D(nn.Module):
    
    def __init__(self, time_series=120):
        super().__init__()
        self.conv1 = Conv1d(in_channels=1, out_channels=32,
                            kernel_size=3, stride=1)
        self.max_pool1 = MaxPool1d(kernel_size=2,stride=1)
        output1 = time_series - 3 + 1
        output1 = np.floor((output1 - 2)/1 + 1)

        self.conv2 = Conv1d(in_channels=32, out_channels=64,
                            kernel_size=3)
        self.max_pool2 = MaxPool1d(kernel_size=2,stride=1)
        output2 = output1 - 3 + 1
        output2 = np.floor((output2 -2)/1 + 1)
        
        self.conv3 = Conv1d(in_channels=64, out_channels=96,
                            kernel_size=3)
        output3 = output2 -3 +1
        
        self.conv4 = Conv1d(in_channels=96, out_channels=64,
                            kernel_size=3)
        output4 = output3 - 3 + 1
        
        self.conv5 = Conv1d(in_channels=64, out_channels=64,
                            kernel_size=3)
        self.max_pool3 = MaxPool1d(kernel_size=2,stride=2)
        output5 = output4 - 3 + 1
        output5 = np.floor((output5 - 2)/2 + 1)
        
        self.in1 = nn.BatchNorm1d(32)
        self.in2 = nn.BatchNorm1d(64)
        self.in3 = nn.BatchNorm1d(96)
        
        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()
        self.prelu3 = nn.PReLU()
        
        self.linear = Linear(int(output5*64), 32)
        

    def forward(self, x):

        b, k, d = x.shape

        x = torch.transpose(x, 1, 2)
        x = x.contiguous()

        x = x.view((b*d, 1, k))

        x = self.conv1(x)
        x = self.in1(x)
        x = self.prelu1(x)
        x = self.max_pool1(x)
        
        x = self.conv2(x)
        x = self.in2(x)
        x = self.prelu2(x)
        x = self.max_pool2(x)
        
        x = self.conv3(x)
        x = self.in3(x)
        x = self.prelu3(x)

        x = self.conv4(x)
        x = self.conv5(x)
        x = self.max_pool3(x)

        x = x.view((b, d, -1))        
        x = self.linear(x)
        return x


class FCN(nn.Module):
    
    def __init__(self):
        super().__init__()
    
        self.linear1 = Linear(2*32, 32)
        self.linear2 = Linear(32, 32)
        self.linear3 = Linear(32, 2)
        self.softmax = nn.Softmax(dim=-1)
        

    def forward(self, x):
        
        batch_size, num_regions, feature_dim = x.shape
        idx = torch.triu_indices(num_regions, num_regions, offset=1)
        i_idx, j_idx = idx[0], idx[1]
        region_i_features = x[:, i_idx, :]
        region_j_features = x[:, j_idx, :]
        pairs = torch.cat([region_i_features, region_j_features], dim=2)
        x1 = self.linear1(pairs)
        x1 = self.linear2(x1)
        x1 = self.linear3(x1)
        x1 = self.softmax(x1)
        fc = x1[:, :, 0]
        return fc
    
    
class CLSN(nn.Module):
    
    def __init__(self, num_roi):
        super().__init__()
    
        self.linear1 = Linear(int((num_roi*(num_roi-1))/2), 100)
        self.linear2 = Linear(100, 50)
        self.linear3 = Linear(50, 50)
        self.linear4 = Linear(50, 2)
        self.in1 = nn.BatchNorm1d(100)
        self.in2 = nn.BatchNorm1d(50)
        self.in3 = nn.BatchNorm1d(50)
        self.softmax = nn.Softmax(dim=-1)
        

    def forward(self, x):
        x = self.linear1(x)
        x = self.in1(x)
        x = self.linear2(x)
        x = self.in2(x)
        x = self.linear3(x)
        x = self.in3(x)
        x = self.linear4(x)
        x = self.softmax(x)
        return x


class DeepFMRI(nn.Module):

    def __init__(self, roi_num):
        super().__init__()
        self.extract = CNN_1D()
        self.fcn = FCN()
        self.clsn = CLSN(roi_num)
    def forward(self, x):
        x = self.extract(x)
        x = self.fcn(x)
        x = self.clsn(x)
        return x
    
    