import sys
import os
import torch
import numpy as np
from torch import nn, optim

sys.path.append('../logger')
from logger import Logger
from logger_utils import prepare_directories_and_logger
from plotting_utils import  plot_spectrogram_to_numpy

sys.path.append('../utils')
from optim_step import *
sys.path.append('../loss')
from began_loss import BEGANRecorder, BEGANLoss
from save_and_load import save_checkpoint, load_checkpoint
import random
from torch.nn import functional as F
from torch.nn.utils import spectral_norm
vocoder = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')

###########################################
#  BEGAN Parameters.....                  #
###########################################


class BNSNConv1dDBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, dilation):
        super().__init__()
        ks = kernel_size
        ksm1d2 = (ks - 1) // 2

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.dilation = dilation

        block = [
            spectral_norm(nn.Conv1d(input_dim, output_dim, ks,
                                    1, dilation*ksm1d2, dilation=dilation)),
            nn.BatchNorm1d(output_dim),
            nn.LeakyReLU(),
        ]
        self.block = nn.Sequential(*block)

    def forward(self, x):

        x = self.block(x)

        return x

class BNSNConv2dDBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, frequency_stride, time_dilation):
        super().__init__()
        ks = kernel_size
        ksm1d2 = (ks - 1) // 2

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.time_dilation = time_dilation
        self.frequency_stride = frequency_stride

        block = [
            spectral_norm(nn.Conv2d(
                input_dim, output_dim, ks,
                (frequency_stride, 1),
                (1, time_dilation*ksm1d2),
                dilation=(1, time_dilation))),
            nn.BatchNorm2d(output_dim),
            nn.LeakyReLU(),
        ]
        self.block = nn.Sequential(*block)

    def forward(self, x):

        x = self.block(x)
        
        return x


class NetD(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        ks = 3  # filter size
        mfd = 512

        self.mfd = mfd
        self.input_size = input_size

        # ### Main body ###
        blocks2d = [
            BNSNConv2dDBlock(1, 4, ks, 2, 2),
            BNSNConv2dDBlock(4, 16, ks, 2, 4),
            BNSNConv2dDBlock(16, 64, ks, 2, 8),
            #BNSNConv2dDBlock(64, 128, ks, 2, 16)
        ]

        blocks1d = [
            BNSNConv1dDBlock(64*10 * input_size//80, mfd, 3, 1),
            BNSNConv1dDBlock(mfd, mfd, ks, 16),
            BNSNConv1dDBlock(mfd, mfd, ks, 32),
            BNSNConv1dDBlock(mfd, mfd, ks, 64),
            BNSNConv1dDBlock(mfd, mfd, ks, 128),
            #BNSNConv1dDBlock(mfd, mfd, ks, 256),
        ]

        self.body2d = nn.Sequential(*blocks2d)
        self.body1d = nn.Sequential(*blocks1d)

        self.head = spectral_norm(nn.Conv1d(mfd, input_size, 3, 1, 1))

    def forward(self, x):
        '''
        x.shape=(batch_size, feat_dim, num_frames)
        cond.shape=(batch_size, cond_dim, num_frames)
        '''
        bs, fd, nf = x.size()

        # ### Process generated ###
        # shape=(bs, 1, fd, nf)
        x = x.unsqueeze(1)

        # shape=(bs, 64, 10, nf_)
        x = self.body2d(x)
        # shape=(bs, 64*10, nf_)
        x = x.view(bs, -1, x.size(3))

        # ### Merging ###
        x = self.body1d(x)

        # ### Head ###
        # shape=(bs, input_size, nf)
        # out = torch.sigmoid(self.head(x))
        out = self.head(x)

        return out

#####################################################


def train_(args, model, opt, latent_loss_weight, criterion, loader, epochs, inf_iterator_test, logger, iteration):
    dis = NetD(80).cuda()
    opt_dis = optim.Adam(dis.parameters())
    gamma = 1.0
    lambda_k = 0.01
    init_k = 0.0
    recorder = BEGANRecorder(lambda_k, init_k, gamma)
    k = recorder.k.item()
    opt_dec = optim.Adam(model.dec.parameters())
    for epoch in range(epochs):
        mse_sum = 0
        mse_n = 0
        
        for i, audio in enumerate(loader):

            audio = audio.cuda()
            audio = (audio*25 + 50) / 50
            factor = 32

            time_step = audio.size(2)
            
            audio_shuffle = [[] for i in range (time_step//factor)]
            nums = [x for x in range(time_step//factor)]
            random.shuffle(nums)
            
            for i_n, n in enumerate(nums):
                sf = random.uniform(0.5, 2)
                audio_shuffle[n] = F.interpolate(audio[...,factor*n : factor*(n+1)], scale_factor=sf, mode='nearest')
            
            audio_shuffle = torch.cat(audio_shuffle,dim=2)   

            audio = audio_shuffle#F.interpolate(audio, scale_factor= audio_shuffle.size(2)/time_step)
            audio = audio[...,:audio.size(2)//16*16]
            
            audio_middile =  F.interpolate(audio, scale_factor= 1/2)
            audio_middile = audio_middile[:, :audio_middile.size(1)//2, :]

            audio_low = F.interpolate(audio_middile, scale_factor= 1/2)
            audio_low = audio_low[:, :audio_low.size(1)//2, :]
            
            audio_list = [audio_low, audio_middile, audio]
            
            out, latent_loss, index_list = model(audio)

            recon_loss = 0
            for num in range(3):
                recon_loss += criterion(out[num], audio_list[num])
            
            latent_loss = latent_loss.mean()  

            #OptimStep([(model, opt,  recon_loss + latent_loss_weight*latent_loss , True)], 3)# True),

            
            #################################
            # BEGAN TRAINING PHASE          #
            #################################
            model.zero_grad()
            index_list_ = []
            for l in index_list:
                idx = torch.randperm(l.size(0))
                index_list_ += [l[idx]]
            out_code = model.index_to_decode(index_list_)
            loss_gan, loss_dis, real_dloss, fake_dloss = BEGANLoss(dis, audio, out_code[-1], k)
            OptimStep([(model, opt,  recon_loss + latent_loss_weight*latent_loss , True), (model.dec, opt_dec,  0.2*(loss_gan), True), 
                (dis, opt_dis, loss_dis, False)], 3)
            
            k, convergence = recorder(real_dloss, fake_dloss, update_k=True)
            iteration += 1
            print (iteration)
            model.zero_grad()

            if i% 5 == 0 :
                logger.log_training(iteration = iteration, loss_gan = loss_gan, 
            loss_dis = loss_dis, loss_recon = recon_loss, latent_loss = latent_loss, k = k, convergence = convergence)

            if i % 50 == 0 :
                model.eval()
                a = torch.stack([audio[0], out[-1][0], out_code[-1][0]], dim = 0)
                a = (a*50 - 50)/25
                a = vocoder.inverse(a)
                a = a.detach().cpu().numpy()
                logger.log_validation(iteration = iteration,
                    mel_ori = ("image", plot_spectrogram_to_numpy(), audio[0]),
                    mel_recon = ("image", plot_spectrogram_to_numpy(), out[-1][0]),
                    mel_code = ("image", plot_spectrogram_to_numpy(), out_code[-1][0]),

                    audio_ori = ("audio", 22050, a[0]),
                    audio_recon = ("audio", 22050, a[1]),
                    audio_code = ("audio", 22050, a[2]),
                )
                
                save_checkpoint(model, opt, iteration, f'checkpoint/{args.model}_n{args.n_embed}_ch{args.channel}_{args.trainer}/gen')
                save_checkpoint(dis, opt_dis, iteration, f'checkpoint/{args.model}_n{args.n_embed}_ch{args.channel}_{args.trainer}/dis')

                model.train()
                logger.close()