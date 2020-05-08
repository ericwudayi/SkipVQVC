import torch
from torch.utils.data import Dataset
import numpy as np
import os
import random

def VCTK_collate(batch):
    maxn = 256#864//2
    audio = []
    #name = []
    for item in batch:
    
        item_len = int(item.shape[1])
        if item_len>maxn:
            rand = np.random.randint(item_len-maxn)
            item_128 = item[:,rand:rand+maxn]
        else:
            item_128 = item
        audio += [item_128]

    for i in range(len(audio)):
        a = audio[i]
        a = np.pad(a,((0,0),(0,maxn-a.shape[1])),'reflect')
        audio[i] = a
        
    return torch.tensor((np.array(audio)))#, torch.tensor(np.array(name))
    
class AudioNpyLoader(torch.utils.data.Dataset):
    """
        1) loads audio
    """
    def __init__(self, audio_path):
        self.audio_path = audio_path
        self.audios = os.listdir(self.audio_path)
        
        random.seed(1234)
        random.shuffle(self.audios)

    def __getitem__(self, index):
        item = f'{self.audio_path}/{self.audios[index]}'
        item = np.load(item)
        
        return item

    def __len__(self):
        return len(self.audios)

class AudioNpyNameLoader(torch.utils.data.Dataset):
    """
        1) loads audio
    """
    def __init__(self, audio_path):
        self.audio_path = audio_path
        self.audios = os.listdir(self.audio_path)
        
        random.seed(1234)
        random.shuffle(self.audios)

    def __getitem__(self, index):
        item = f'{self.audio_path}/{self.audios[index]}'
        item = np.load(item)
        
        return item, item.split('/')

    def __len__(self):
        return len(self.audios)