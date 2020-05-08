import sys
import os
import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from utils.dataloader import AudioNpyLoader, VCTK_collate

sys.path.append('logger')
from logger import Logger
from logger_utils import prepare_directories_and_logger

sys.path.append('utils')
from save_and_load import load_checkpoint

parser = argparse.ArgumentParser()
parser.add_argument('-train_dir', '--train_dir', type=str, required = True,
                    help = 'preprocessed npy files in train dir')
parser.add_argument('-test_dir','--test_dir', type=str, required = False, default=None,
                    help = 'preprocessed npy files of test dir')
parser.add_argument('-m', '--model', type=str, required= True,
                    help='model type in model dir')
parser.add_argument('-n', '--n_embed', type=str,required= True,
                    help='number of vectors in codebook')
parser.add_argument('-ch', '--channel', type=str, required= True,
                     help='channel number in VQVC+')
parser.add_argument('-t', '--trainer', type=str, required= True,
                    help = 'which trainer do you want? (rhythm, mean_std, normal)')
parser.add_argument('--load_checkpoint', type=bool, default=False,
                    required=False)


args = parser.parse_args()
logger = prepare_directories_and_logger(Logger, output_directory = f'output/{args.model}_n{args.n_embed}_ch{args.channel}_{args.trainer}')

import importlib

trainer = importlib.import_module(f'trainer.{args.trainer}')
train_ = getattr(trainer, 'train_')

model = importlib.import_module(f'model.{args.model}.vq_model')
model = getattr(model, 'VC_MODEL')
'''
Dataset and loader
'''
def make_inf_iterator(data_iterator):
    while True:
        for data in data_iterator:
            yield data

audio_dir = args.train_dir#"/home/ericwudayi/nas189/homes/ericwudayi/VCTK-Corpus/mel3/mel.melgan"

dataset = AudioNpyLoader(audio_dir)
loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4,collate_fn=VCTK_collate)

if args.test_dir != None:
    audio_dir_test = args.test_dir#"/home/ericwudayi/nas189/homes/ericwudayi/VCTK-Corpus/mel3/mel.test"
else:
    audio_dir_test = audio_dir
    print ("None test dir, use train dir instead")
dataset_test = AudioNpyLoader(audio_dir_test)
test_loader = DataLoader(dataset_test, batch_size=8, shuffle=True, num_workers=4,collate_fn=VCTK_collate)
inf_iterator_test = make_inf_iterator(test_loader)
'''
Model Initilization
'''
model = model(in_channel=80,channel=int(args.channel),n_embed=int(args.n_embed)).cuda()
opt = optim.Adam(model.parameters())
'''
Training
'''
criterion = nn.L1Loss()
latent_loss_weight = 0.1

if args.load_checkpoint==True:
    model, opt, iteration = load_checkpoint(f'checkpoint/{args.model}_n{args.n_embed}_ch{args.channel}/gen', model, opt)       

train_(args, model, opt, latent_loss_weight, criterion, loader, 800, inf_iterator_test, logger)