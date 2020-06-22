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
#


def train_(args, model, opt, latent_loss_weight, criterion, loader, epochs, inf_iterator_test, logger, iteration):
    vctk_mean = torch.tensor(np.load("/home/ericwudayi/nas189/homes/ericwudayi/VCTK-Corpus/mel3/mean.npy")).unsqueeze(0).unsqueeze(2).cuda()
    vctk_std = torch.tensor(np.load("/home/ericwudayi/nas189/homes/ericwudayi/VCTK-Corpus/mel3/std.npy")).unsqueeze(0).unsqueeze(2).cuda()    
    for epoch in range(epochs):
        mse_sum = 0
        mse_n = 0
        
        for i, audio in enumerate(loader):
            cluster_size = audio.size(1)
            audio = audio.cuda()
            audio = (audio - vctk_mean)/vctk_std
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
            
            out, out_conversion, enc_content, spk, latent_loss, idx = model(audio)
            
            recon_loss = 0

            for num in range(3):
                recon_loss += criterion(out[num], audio_list[num])
            
            latent_loss = latent_loss.mean()  
            #print ("recon_loss:", recon_loss)
            OptimStep([(model, opt,  recon_loss + latent_loss_weight*latent_loss , False)], 3)# True),
            


            if i % 200 == 0 :
                
                logger.log_training(iteration = iteration,  loss_recon = recon_loss, latent_loss = latent_loss)

                model.eval()

                audio = next(inf_iterator_test)
                audio = audio.cuda()
                audio = (audio - vctk_mean)/vctk_std
                
                out, out_conversion, enc_content, spk, latent_loss, idx = model(audio)
                
                
                audio = audio*vctk_std + vctk_mean
                out[-1] = out[-1]*vctk_std + vctk_mean
                out_conversion[-1] = out_conversion[-1]*vctk_std+vctk_mean
                a = torch.stack([audio[0], audio[idx[0]], out[-1][0], out_conversion[-1][0]], dim = 0)

                a = vocoder.inverse(a)
                a = a.detach().cpu().numpy()
                logger.log_validation(iteration = iteration,
                    mel_ori = ("image", plot_spectrogram_to_numpy(), audio[0]),
                    mel_target = ("image", plot_spectrogram_to_numpy(), audio[idx[0]]),
                    mel_recon = ("image", plot_spectrogram_to_numpy(), out[-1][0]),
                    mel_conversion = ("image", plot_spectrogram_to_numpy(), out_conversion[-1][0]),
                    
                    mel_recon_middle = ("image", plot_spectrogram_to_numpy(), out[-2][0]),
                    mel_conversion_middle = ("image", plot_spectrogram_to_numpy(), out_conversion[-2][0]),
                    
                    mel_recon_low = ("image", plot_spectrogram_to_numpy(), out[-3][0]),
                    mel_conversion_low = ("image", plot_spectrogram_to_numpy(), out_conversion[-3][0]),

                    audio_ori = ("audio", 22050, a[0]),
                    audio_target = ("audio", 22050, a[1]),
                    audio_recon = ("audio", 22050, a[2]),
                    audio_conversion = ("audio", 22050, a[3]),

                )
                logger.close()
                save_checkpoint(model, opt, iteration, f'checkpoint/{args.model}_n{args.n_embed}_ch{args.channel}_{args.trainer}/gen')
                
                model.train()
            iteration += 1