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
from torch.nn import functional as F
sys.path.append('/home/ericwudayi/AiVocal/SkipVQVC/vocoder/melgan-neurips')
from mel2wav.interface import MelVocoder

#vocoder = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')
vocoder = MelVocoder(path = "/home/ericwudayi/AiVocal/ai_singing/vocoder/melgan-neurips/scripts/logs/NUS")

#vocoder = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')

def train_(args, model, opt, latent_loss_weight, criterion, loader, epochs, inf_iterator_test, logger, iteration):
    
    for epoch in range(epochs):
        mse_sum = 0
        mse_n = 0
        
        for i, (audio,pitch) in enumerate(loader):
            
            audio = audio.cuda().float()
            pitch = pitch.cuda().float()

            audio = (audio*25 + 50) / 50
            
            #Normalize pitch
            #print (pitch.size())
            pitch_non_sil = (pitch>20)
            pitch_sil = pitch<20
            pitch_mean_non_sil = torch.sum(pitch*pitch_non_sil)/torch.sum(pitch_non_sil)
            pitch -= pitch_mean_non_sil#torch.mean(pitch,dim = 1, keepdim = True)
            pitch = (pitch+20) / 50
            pitch[pitch_sil] = 0.0
            #print (pitch[0,:50])

            pitch = pitch.unsqueeze(1)
            audio_middle =  F.interpolate(audio, scale_factor= 1/2)
            audio_middle = audio_middle[:, :audio_middle.size(1)//2, :]
            pitch_middle = F.interpolate(pitch, scale_factor= 1/2)
            

            audio_low = F.interpolate(audio_middle, scale_factor= 1/2)
            audio_low = audio_low[:, :audio_low.size(1)//2, :]
            pitch_low = F.interpolate(pitch_middle, scale_factor= 1/2)
            

            audio_list = [audio_low, audio_middle, audio]
            pitch_list = [pitch, pitch_middle, pitch_low]
            out, out_conversion, enc_content, spk, latent_loss, idx = model(audio, pitch_list)
            
            recon_loss = 0
            #print (i)
            for num in range(3):
                recon_loss += criterion(out[num], audio_list[num])
            
            latent_loss = latent_loss.mean()  
            #print ("recon_loss:", recon_loss)
            OptimStep([(model, opt,  recon_loss + latent_loss_weight*latent_loss , False)], 3)# True),
            


            if i % 100 == 0 :
                
                logger.log_training(iteration = iteration,  loss_recon = recon_loss, latent_loss = latent_loss)

                model.eval()

                audio, pitch = next(inf_iterator_test)
                
                audio = audio.cuda().float()
                pitch = pitch.cuda().float()
                
                audio = (audio*25 + 50) / 50
                pitch_non_sil = (pitch>20)
                pitch_sil = pitch<20
                pitch_mean_non_sil = torch.sum(pitch*pitch_non_sil)/torch.sum(pitch_non_sil)
                pitch -= pitch_mean_non_sil#torch.mean(pitch,dim = 1, keepdim = True)
                pitch = (pitch+20) / 50
                pitch[pitch_sil] = 0.0
                pitch = pitch.unsqueeze(1)
                pitch_middle = F.interpolate(pitch, scale_factor= 1/2)

                pitch_low = F.interpolate(pitch_middle, scale_factor= 1/2)

                
                pitch_list = [pitch, pitch_middle, pitch_low]
                out, out_conversion, enc_content, spk, latent_loss, idx = model(audio, pitch_list)
            
                
                a = torch.stack([audio[0], audio[idx[0]], out[-1][0], out_conversion[-1][0]], dim = 0)
                
                a = (a*50 - 50)/25
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