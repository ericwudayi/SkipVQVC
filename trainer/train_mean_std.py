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
from save_and_load import save_checkpoint, load_checkpoint

vocoder = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')

base_dir = "/home/ericwudayi/nas189/homes/ericwudayi/VCTK-Corpus/mel3/"
mean_fp = os.path.join(base_dir, f'mean.mel.melgan.npy')
std_fp = os.path.join(base_dir, f'std.mel.melgan.npy')
mean = torch.from_numpy(np.load(mean_fp)).float().cuda().view(1, 80, 1)
std = torch.from_numpy(np.load(std_fp)).float().cuda().view(1, 80, 1)

def train_(args, model, opt, latent_loss_weight, criterion, loader, epochs, inf_iterator_test, logger):
    iteration = 0
    for epoch in range(epochs):
        mse_sum = 0
        mse_n = 0
        
        for i, audio in enumerate(loader):
            cluster_size = audio.size(1)
            audio = audio.cuda()
            audio = (audio - mean)/std/3
            out, out_conversion, enc_content, spk, latent_loss, idx = model(audio)
            recon_loss = criterion(out, audio)
            latent_loss = latent_loss.mean()  

            OptimStep([(model, opt,  recon_loss + latent_loss_weight*latent_loss , False)], 3)# True),

            mse_sum += recon_loss.item() * audio.shape[0]
            mse_n += audio.shape[0]
            if i% 5 == 0 :
                logger.log_training(iteration = iteration,  loss_recon = recon_loss, latent_loss = latent_loss)


            if i % 100 == 0 :
                model.eval()

                audio = next(inf_iterator_test)
                audio = audio.cuda()
                
                audio = (audio - mean)/std/3
                
                
                out, out_conversion, enc_content, spk, latent_loss, idx = model(audio)
                a = torch.stack([audio[0], audio[idx[0]], out[0], out_conversion[0]], dim = 0)
                
                a = a*std*3 + mean
                a = vocoder.inverse(a)
                a = a.detach().cpu().numpy()
                logger.log_validation(iteration = iteration,
                    mel_ori = ("image", plot_spectrogram_to_numpy(), audio[0]),
                    mel_target = ("image", plot_spectrogram_to_numpy(), audio[idx[0]]),
                    mel_recon = ("image", plot_spectrogram_to_numpy(), out[0]),
                    mel_conversion = ("image", plot_spectrogram_to_numpy(), out_conversion[0]),

                    audio_ori = ("audio", 22050, a[0]),
                    audio_target = ("audio", 22050, a[1]),
                    audio_recon = ("audio", 22050, a[2]),
                    audio_conversion = ("audio", 22050, a[3]),

                )
                logger.close()
                save_checkpoint(model, opt, iteration, f'checkpoint/{args.model}_n{args.n_embed}_ch{args.channel}_{args.trainer}/gen')
                model.train()
            iteration += 1