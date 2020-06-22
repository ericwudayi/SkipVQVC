import torch
from torch.utils.data import Dataset
import numpy as np
import os
import random
import pandas
from torch.nn import functional as F
import librosa
def VCTK_collate(batch):
    maxn = 128
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

def VCTK_NAME_collate(batch):
    maxn = 256
    audio = []
    name = []
    for item in batch:
        n = item[1]
        item = item[0]
        item_len = int(item.shape[1])
        if item_len>maxn:
            rand = np.random.randint(item_len-maxn)
            item_128 = item[:,rand:rand+maxn]
        else:
            item_128 = item
        audio += [item_128]
        name+= [n]
    for i in range(len(audio)):
        a = audio[i]
        a = np.pad(a,((0,0),(0,maxn-a.shape[1])),'reflect')
        audio[i] = a
        
    return torch.tensor((np.array(audio))), name

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
        name = self.audios[index].split('_')[0]
        item = np.load(item)
        
        return item, name

    def __len__(self):
        return len(self.audios)



class AudioNpyPitchLoader(torch.utils.data.Dataset):
    """
        1) loads audio
    """
    def __init__(self, audio_path):
        self.audio_path = audio_path
        self.audios = os.listdir(self.audio_path)
        
        self.pitch_path = '/'.join(audio_path.split('/')[:-2])
        random.seed(1234)
        random.shuffle(self.audios)

    def __getitem__(self, index):
        item = f'{self.audio_path}/{self.audios[index]}'
        item = np.load(item)
        
        item_pitch = "_".join(self.audios[index].split('_')[-3:])[:-4]
        pitch_csv = f'{self.pitch_path}/10seconds_pitch/{item_pitch}.f0.csv'

        pitch_csv = pandas.read_csv(pitch_csv)
        
        pitch = pitch_csv['frequency'].values.tolist()
        confident= pitch_csv['confidence'].values.tolist()
        
        for i in range(len(confident)):
            if confident[i]<0.3 or pitch[i]<60 or pitch[i]>1100:
                pitch[i] = 1
        pitch = np.array(pitch)
        pitch = torch.from_numpy(pitch).unsqueeze(0).unsqueeze(0)

        pitch = F.interpolate(pitch, scale_factor=(item.shape[1]/pitch.size(2)), mode='nearest')
        pitch = torch.squeeze(pitch)
        
        pitch = librosa.core.hz_to_midi(pitch)

        return item, pitch

    def __len__(self):
        return len(self.audios)

def Pitch_collate(batch):
    maxn = 640
    audio = []
    pitch_list = []
    for item in batch:
        pitch = item[1]
        item = item[0]
        item_len = int(item.shape[1])
        if item_len>maxn:
            rand = np.random.randint(item_len-maxn)
            item_128 = item[:,rand:rand+maxn]
            pitch = pitch[rand:rand+maxn]
        else:
            item_128 = item
        audio += [item_128]
        pitch_list += [pitch]
    for i in range(len(audio)):
        a = audio[i]
        a = np.pad(a,((0,0),(0,maxn-a.shape[1])),'reflect')
        audio[i] = a
        
        p = pitch_list[i]
        #rint (p.shape)
        p = np.pad(p,((0,maxn-p.shape[0])),'reflect')
        pitch_list[i] = p
    return torch.tensor((np.array(audio))), torch.tensor((np.array(pitch_list)))