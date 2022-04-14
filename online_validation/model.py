from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as torch_functional



class NeuralLinearRegressor(nn.Module):
    def __init__(self, in_dim, n_hidden_units, activation):
        super(NeuralLinearRegressor, self).__init__()
        self.in_dim=in_dim
        self.n_hidden_units=list(n_hidden_units)
        self.layers=self.cascade_layers(l=[self.in_dim]+self.n_hidden_units, activation=activation)

    def cascade_layers(self, l, activation):
        nlayers = len(l)
        if (nlayers <= 0):
            print('Error: Null input. No model constructed!')
            return None
        else:
            nunits_list = [nunits for nunits in l] + [1]
            modules = OrderedDict()
            ilayer=1
            for idx in range(0, len(nunits_list) - 1):
                out_dim,in_dim = nunits_list[idx],nunits_list[idx + 1]
                layer=nn.Linear(out_dim, in_dim)
                modules[f'Layer{ilayer}']=layer
                torch.nn.init.xavier_uniform_(layer.weight)
                ilayer+=1
                modules[f'Layer{ilayer}']=activation()
                ilayer+=1

            return nn.Sequential(modules)

    def forward(self, t):
        t = t.view(-1, self.in_dim)
        return self.layers(t)



class NerualConvRegressor(nn.Module):
    def __init__(self, in_dim, seq_len):
        super(NerualConvRegressor, self).__init__()
        self.in_dim=in_dim
        self.seq_len=seq_len

        self.conv_module=nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(4,4)),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(8,4)),
            nn.LeakyReLU(),

            nn.Dropout(),
            nn.Flatten(),)

        self.conv_out_dim=None
        self.probe_conv_out_dim()
        assert(self.conv_out_dim is not None)

        self.layers=nn.Sequential(
            self.conv_module,
            nn.Linear(self.conv_out_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128,1)
        )

    def probe_conv_out_dim(self):
        probe_tensor=torch.randn(1,1,self.seq_len, self.in_dim)
        probe_tensor_conv_out=self.conv_module(probe_tensor)
        assert(probe_tensor_conv_out.dim()==2 and probe_tensor_conv_out.shape[0]==1)
        self.conv_out_dim=probe_tensor_conv_out.shape[1]

    def forward(self, t):
        return self.layers(t)

class TransformerRegressor(nn.Module):
    def __init__(self, in_size, seq_len, embed_size, encoder_nlayers, encoder_nheads, dim_feedforward):
        super(TransformerRegressor, self).__init__()
        self.in_size=in_size
        self.embed_size=embed_size
        self.seq_len=seq_len
        self.embedding=nn.Linear(self.in_size, self.embed_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.embed_size, nhead=encoder_nheads, batch_first=True,
                                                        dim_feedforward=dim_feedforward, activation='relu')
        self.encoder_module = nn.TransformerEncoder(self.encoder_layer, num_layers=encoder_nlayers)
        self.decoder_flatten=nn.Flatten()
        self.decoder_module = nn.Linear(self.seq_len*self.embed_size, 1)

    def forward(self, t):
        t=self.embedding(t)
        t=self.encoder_module(t)
        t=self.decoder_flatten(t)
        t=self.decoder_module(t)
        return t