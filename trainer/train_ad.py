import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import os
from os import listdir
import sys
import argparse
from librosa.filters import mel as librosa_mel_fn
import librosa.core as core
import numpy as np

from vqvae import VQVAE, NetD
from utils.dataloader import AudioNpyLoader, VCTK_collate
import utils.fileio as fio
from trainer import train

sys.path.append('logger')
from logger import Logger
from logger_utils import prepare_directories_and_logger
from plotting_utils import  plot_spectrogram_to_numpy

sys.path.append('loss')
sys.path.append('utils')
from began_loss import BEGANRecorder, BEGANLoss
from optim_step import *
from save_and_load import save_checkpoint, load_checkpoint
from tqdm import tqdm

vocoder = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output_directory', type=str,
                    help='directory to save checkpoints')
parser.add_argument('-c', '--checkpoint_path', type=str, default=None,
                    required=False, help='checkpoint path')
parser.add_argument('--rank', type=str, default="0",
                    required=False, help='rank of current gpu')
parser.add_argument('--load_checkpoint', type=bool, default=False,
                    required=False)

args = parser.parse_args()
logger = prepare_directories_and_logger(Logger, output_directory = f'output/{args.output_directory}')
os.environ["CUDA_VISIBLE_DEVICES"] = args.rank
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
print (args.rank)
'''
Dataset and loader
'''
def make_inf_iterator(data_iterator):
    while True:
        for data in data_iterator:
            yield data

audio_dir = "/home/ericwudayi/nas189/homes/ericwudayi/VCTK-Corpus/mel_clip_sil3/mel.melgan"
dataset = AudioNpyLoader(audio_dir)

loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4,collate_fn=VCTK_collate)

audio_dir_test = "/home/ericwudayi/nas189/homes/ericwudayi/VCTK-Corpus/mel_clip_sil3/mel.test"
dataset_test = AudioNpyLoader(audio_dir_test)
test_loader = DataLoader(dataset_test, batch_size=8, shuffle=True, num_workers=4,collate_fn=VCTK_collate)
inf_iterator_test = make_inf_iterator(test_loader)

'''
Model Initilization
'''
model = (VQVAE(in_channel=80,channel=512,n_embed=128, n_res_block=2,n_res_channel=64,embed_dim=80//8)).cuda()
scheduler = None
opt = optim.Adam(model.parameters())

dis = NetD(80).cuda()
opt_dis = optim.Adam(dis.parameters())

'''
Training
'''
criterion = nn.L1Loss()
latent_loss_weight = 0.1

iteration = 0

gamma = 1.0
lambda_k = 0.01
init_k = 0.0
recorder = BEGANRecorder(lambda_k, init_k, gamma)
k = recorder.k.item()

if args.load_checkpoint==True:
    model, opt, iteration = load_checkpoint(f'checkpoint/{args.checkpoint_path}/gen', model, opt)       
    dis, opt_dis, iteration = load_checkpoint(f'checkpoint/{args.checkpoint_path}/dis', dis, opt_dis)

for epoch in range(800):
    mse_sum = 0
    mse_n = 0

    for i, audio in enumerate(loader):
        cluster_size = audio.size(1)
        audio = audio.cuda()
        
        out, out_conversion, enc_content, spk, latent_loss, idx = model(audio)
        recon_loss = criterion(out, audio)
        latent_loss = latent_loss.mean()  

        loss_gan, loss_dis, real_dloss, fake_dloss = BEGANLoss(dis, out, out_conversion, k)#song_len_padded[...,:fake_song_padded.size(2)]
       
        OptimStep([(model, opt,  0.5*(loss_gan) + recon_loss + latent_loss_weight*latent_loss , True),
            (dis, opt_dis, loss_dis, False)], 3)
        k, convergence = recorder(real_dloss, fake_dloss, update_k=True)


        mse_sum += recon_loss.item() * audio.shape[0]
        mse_n += audio.shape[0]


        if i% 5 == 0 :
            logger.log_training(iteration = iteration, loss_gan = loss_gan, 
            loss_dis = loss_dis, loss_recon = recon_loss, latent_loss = latent_loss, k = k, convergence = convergence)


        if i % 100 == 0 :
            model.eval()

            audio = next(inf_iterator_test)
            audio = audio.cuda()
            out, out_conversion, enc_content, spk, latent_loss, idx = model(audio)
            a = torch.stack([audio[0], audio[idx[0]], out[0], out_conversion[0]], dim = 0)
            
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
            save_checkpoint(model, opt, iteration, f'checkpoint/{args.checkpoint_path}/gen')
            save_checkpoint(dis, opt_dis, iteration, f'checkpoint/{args.checkpoint_path}/dis')

            model.train()
        iteration += 1