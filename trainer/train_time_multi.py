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
from wgangp_loss import WGANGP
from gan_loss import GANLOSS
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
            (nn.Conv1d(input_dim, output_dim, ks,
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
            (nn.Conv2d(
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
            #BNSNConv1dDBlock(mfd, mfd, ks, 256),
        ]

        self.body2d = nn.Sequential(*blocks2d)
        self.body1d = nn.Sequential(*blocks1d)

        self.head = (nn.Conv1d(mfd, 1, 3, 1, 1))
        self.dense = nn.Linear(128,1)
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
        out = self.head(x).squeeze()
        out = F.sigmoid(self.dense(out))
        #out = self.head(x)

        return out.squeeze()

#####################################################


def train_(args, model, opt, latent_loss_weight, criterion, loader, epochs, inf_iterator_test, logger, iteration, inf_iterator_enc):
    dis = NetD(80).cuda()
    opt_dis = optim.Adam(dis.parameters())
    '''
    gamma = 1.0
    lambda_k = 0.01
    init_k = 0.0
    recorder = BEGANRecorder(lambda_k, init_k, gamma)
    k = recorder.k.item()
    '''
    opt_dec = optim.Adam(model.dec.parameters())
    lj_mean = torch.tensor(np.load("/home/ericwudayi/nas189/homes/ericwudayi/LJSpeech/mean.npy")).unsqueeze(0).unsqueeze(2).cuda()
    lj_std = torch.tensor(np.load("/home/ericwudayi/nas189/homes/ericwudayi/LJSpeech/std.npy")).unsqueeze(0).unsqueeze(2).cuda()
    vctk_mean = torch.tensor(np.load("/home/ericwudayi/nas189/homes/ericwudayi/VCTK-Corpus/mel3/mean.npy")).unsqueeze(0).unsqueeze(2).cuda()
    vctk_std = torch.tensor(np.load("/home/ericwudayi/nas189/homes/ericwudayi/VCTK-Corpus/mel3/std.npy")).unsqueeze(0).unsqueeze(2).cuda()
    lj_mean = vctk_mean
    lj_std = vctk_std
    if args.load_checkpoint==True:
        dis, opt_dis, iteration = load_checkpoint(f'checkpoint/{args.model}_n{args.n_embed}_ch{args.channel}_{args.trainer}/dis', dis, opt_dis)       

    for epoch in range(80000):
        
        for i, audio in enumerate(loader):

            audio = audio.cuda()
            audio = (audio-lj_mean)/lj_std
            #audio = (audio*25 + 50) / 50
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
            
            audio = audio[...,:audio.size(2)//32*32]
            
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

            if iteration% 1 == 0:
                model.zero_grad()
                
                audio_enc = next(inf_iterator_enc)
                audio_enc = audio_enc.cuda()
                audio_enc = (audio_enc - vctk_mean)/vctk_std
                if audio_enc.size(0) > audio.size(0):
                    audio_enc = audio_enc[:audio.size(0)]
                else:
                    audio = audio[:audio_enc.size(0)]
                audio_enc =  F.interpolate(audio_enc, scale_factor= audio.size(2)/audio_enc.size(2))
                out_code, latent_loss_enc, index_list = model(audio_enc)
                #latent_loss += latent_loss_enc.mean()
                #latent_loss *= 0 
                #loss_dis, loss_gan = GANLOSS(dis, audio, out_code[-1])
                #if iteration%4==0:
                #    OptimStep([(dis, opt_dis, loss_dis, False)],3)
                #else:
                #    OptimStep([(model, opt, recon_loss + latent_loss_weight*latent_loss + 0.1*loss_gan , False)],3)
                OptimStep([(model, opt, recon_loss + latent_loss_weight*latent_loss, False)],3)

            #else:
            #latent_loss *= 0
            #OptimStep([(model, opt,  recon_loss + latent_loss_weight*latent_loss , True)], 3)# True),

            
            #################################
            # BEGAN TRAINING PHASE          #
            #################################
            model.zero_grad()

            if iteration% 5 == 0 :
                logger.log_training(iteration = iteration, loss_recon = recon_loss, latent_loss = latent_loss)

            if iteration % 200 == 0 :
                model.eval()
                a = torch.stack([audio[0], out[-1][0], out_code[-1][0], audio_enc[0]], dim = 0)
                a = a*lj_std + lj_mean
                a[3] = (a[3] - lj_mean)/lj_std*vctk_std+vctk_mean
                image = a
                a = vocoder.inverse(a)
                a = a.detach().cpu().numpy()
                logger.log_validation(iteration = iteration,
                    mel_ori = ("image", plot_spectrogram_to_numpy(), image[0]),
                    mel_recon = ("image", plot_spectrogram_to_numpy(), image[1]),
                    mel_code = ("image", plot_spectrogram_to_numpy(), image[2]),
                    mel_target = ("image", plot_spectrogram_to_numpy(), image[3]),

                    audio_ori = ("audio", 22050, a[0]),
                    audio_recon = ("audio", 22050, a[1]),
                    audio_code = ("audio", 22050, a[2]),
                    audio_enc = ("audio", 22050, a[3]),
                )
                
                save_checkpoint(model, opt, iteration, f'checkpoint/{args.model}_n{args.n_embed}_ch{args.channel}_{args.trainer}/gen')
                save_checkpoint(dis, opt_dis, iteration, f'checkpoint/{args.model}_n{args.n_embed}_ch{args.channel}_{args.trainer}/dis')

                model.train()
                logger.close()
            iteration += 1
            