import sys
import os
import torch
import numpy as np
from torch import nn, optim
import random
sys.path.append('../logger')
from logger import Logger
from logger_utils import prepare_directories_and_logger
from plotting_utils import  plot_spectrogram_to_numpy

sys.path.append('../utils')
from optim_step import *
from save_and_load import save_checkpoint, load_checkpoint
from torch.nn import functional as F

vocoder = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')

base_dir = "/home/ericwudayi/nas189/homes/ericwudayi/VCTK-Corpus/mel3/"
mean_fp = os.path.join(base_dir, f'mean.mel.melgan.npy')
std_fp = os.path.join(base_dir, f'std.mel.melgan.npy')
mean = torch.from_numpy(np.load(mean_fp)).float().cuda().view(1, 80, 1)
std = torch.from_numpy(np.load(std_fp)).float().cuda().view(1, 80, 1)

def train_(args, model, opt, latent_loss_weight, criterion, loader, epochs, inf_iterator_test, logger):
    iteration = 0
    for epoch in range(epochs):
        factor = 32
        for i, audio in enumerate(loader):
            time_step = audio.size(2)
            audio = audio.cuda()
            audio_shuffle = [[] for i in range (time_step//factor)]
            nums = [x for x in range(time_step//factor)]
            random.shuffle(nums)
            
            for i_n, n in enumerate(nums):
                sf = random.uniform(0.5, 2)
                audio_shuffle[n] = F.interpolate(audio[...,factor*n : factor*(n+1)], scale_factor=sf, mode='nearest')
            
            audio_shuffle = torch.cat(audio_shuffle,dim=2)   
            audio = F.interpolate(audio, scale_factor= audio_shuffle.size(2)/time_step)
            audio = audio[...,:audio.size(2)//16*16]
            audio_shuffle = audio_shuffle[...,:audio_shuffle.size(2)//16*16]
            out, out_conversion, enc_content, spk, latent_loss, idx = model(audio, audio_shuffle)
            
            recon_loss = criterion(out, audio) #+ criterion(out_conversion, audio_shuffle)
            latent_loss = latent_loss.mean()  

            OptimStep([(model, opt,  recon_loss + latent_loss_weight*latent_loss , False)], 3)# True),

            if i% 50 == 0 :
                
                logger.log_training(iteration = iteration,  loss_recon = recon_loss, latent_loss = latent_loss)


            if i % 200 == 0 :
                model.eval()

                audio = next(inf_iterator_test)
                audio = audio.cuda()
                audio_shuffle = [[] for i in range (time_step//factor)]
                
                for i_n, n in enumerate(nums):
                    sf = random.uniform(0.5, 1.5)
                    audio_shuffle[n] = F.interpolate(audio[...,factor*n : factor*(n+1)], scale_factor=sf, mode='nearest')

                audio_shuffle = torch.cat(audio_shuffle,dim=2)
                audio = F.interpolate(audio, scale_factor= audio_shuffle.size(2)/time_step)
                audio = audio[...,:audio.size(2)//16*16]
                audio_shuffle = audio_shuffle[...,:audio_shuffle.size(2)//16*16]
                out, out_conversion, enc_content, spk, latent_loss, idx = model(audio, audio_shuffle)
                a = torch.stack([audio[0], audio_shuffle[idx[0]], out[0], out_conversion[0]], dim = 0)
                
                a = vocoder.inverse(a)
                a = a.detach().cpu().numpy()
                logger.close()
                logger.log_validation(iteration = iteration,
                    mel_ori = ("image", plot_spectrogram_to_numpy(), audio[0]),
                    mel_target = ("image", plot_spectrogram_to_numpy(), audio_shuffle[idx[0]]),
                    mel_recon = ("image", plot_spectrogram_to_numpy(), out[0]),
                    mel_conversion = ("image", plot_spectrogram_to_numpy(), out_conversion[0]),
                    
                    audio_ori = ("audio", 22050, a[0]),
                    audio_target = ("audio", 22050, a[1]),
                    audio_recon = ("audio", 22050, a[2]),
                    audio_conversion = ("audio", 22050, a[3]),

                )
                
                save_checkpoint(model, opt, iteration, f'checkpoint/{args.model}_n{args.n_embed}_ch{args.channel}_{args.trainer}/gen')
                model.train()
            iteration += 1