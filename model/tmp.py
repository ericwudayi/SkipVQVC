import torch
import torch.nn as nn
import torch.nn.functional as F
class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal
    
class Discriminator(nn.Module):
    """discriminator model"""
    def __init__(self, dim_deck=80, dim_pre=128):
        super(Discriminator, self).__init__()
        convolutions = []
        for i in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(dim_deck if i==0 else dim_pre,
                         dim_pre,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.InstanceNorm1d(dim_deck) if i==0 else nn.InstanceNorm1d(dim_pre))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)
        self.dense = nn.Linear(dim_pre,1)
    def forward(self, x):
        for conv in self.convolutions:
            x = F.relu(conv(x))
        patch_size = 32
        x_sample = torch.randint(low = 0, high = x.size(2) - patch_size, size = (1,))
        
        x = x[:,:,x_sample:x_sample+patch_size]
        x = torch.mean(x,dim=2)
        mean_val = self.dense(x)
        #mean_val = torch.clamp(mean_val, 0, 1, out=None) 
        return mean_val

    

    
class LatentClassifier(nn.Module):
    """discriminator model"""
    def __init__(self, nc = 376, ns=0.2, dim_pre=128, dim_deck=64):
        super(LatentClassifier, self).__init__()
        self.ns = ns
        self.nc = nc
        
        self.lstm1 = nn.LSTM(dim_deck, dim_pre, 2, batch_first=True)
        convolutions = []
        for i in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(dim_deck if i==0 else dim_pre,
                         dim_pre,
                         kernel_size=3, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)
        #self.lstm2 = nn.LSTM(dim_pre, 32, 2, batch_first=True)
        self.softmax = nn.LogSoftmax(dim=1)
        self.dense = nn.Linear(dim_pre,nc)
    def forward(self, x):
        #x,_ = self.lstm1(x)
        #x = x.transpose(1,2)
        for conv in self.convolutions:
            x = F.relu(conv(x))
        patch_size = 32
        x_sample = torch.randint(low = 0, high = x.size(2) - patch_size, size = (1,))
        
        x = x[:,:,x_sample:x_sample+patch_size]
        
        x = torch.mean(x,dim=2)
        x = self.dense(x)
        x = self.softmax(x)
        
        return x
class Speakernet(nn.Module):
    def __init__(self, dim_pre=128, dim_deck=64):
        super(Speakernet, self).__init__()
        
        self.lstm1 = nn.LSTM(dim_deck, dim_pre, 2, batch_first=True)
        convolutions = []
        for i in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(dim_pre,
                         dim_pre,
                         kernel_size=3, stride=1,
                         padding=1,
                         dilation=1, w_init_gain='relu'))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)
        
    def forward(self, x):
        x = x.transpose(1,2)
        x,_ = self.lstm1(x)
        x = x.transpose(1,2)
        for conv in self.convolutions:
            x = F.relu(conv(x))
        
        return x

class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(80, 512,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(512))
        )

        for i in range(1, 5 - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(512,
                             512,
                             kernel_size=5, stride=1,
                             padding=2,
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(512))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(512, 80,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(80))
            )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = torch.tanh(self.convolutions[i](x))

        x = self.convolutions[-1](x)

        return x    