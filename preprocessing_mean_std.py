import sys
import os
import librosa
import numpy as np

from multiprocessing import Pool
import pickle
import torch
from torch import nn
from torch.nn import functional as F
import warnings
import pyworld as pw
warnings.filterwarnings('ignore')



def process_audios(path):
    
    try:
        file = np.load(path)

    except Exception:
        return "none", "none"
    return np.mean(file,axis=1), np.std(file,axis=1)


if __name__ == "__main__":
    base_out_dir = sys.argv[1]
    

    audio_dir = sys.argv[2]
    

    feat_type = 'mel.melgan'
    extension = '.npy'
    peak_norm = False

    n_fft = 1024
    hop_length = 256
    win_length = 1024
    sampling_rate = 22050
    n_mel_channels = 80
    sr = sampling_rate

    audio_files = []
    for dirPath, dirNames, fileNames in os.walk(f"{audio_dir}"):
        #print (dirPath)
        for f in fileNames:
            if f.endswith(extension):
                audio_files += [os.path.join(dirPath, f)]

    print (audio_files[:5])
    
    if len(audio_files) == 0:

        print('Please point wav_path in hparams.py to your dataset,')
        print('or use the --path option.\n')

    else:

        pool = Pool(processes=20)
        # pool = Pool(processes=cpu_count())
        mean_arr = []
        std_arr = []
        length_arr = []
        for i, (mean, std) in enumerate(pool.imap_unordered(process_audios, audio_files), 1):
            if mean=="none":
                print ("error")
            print(mean.shape)
            mean_arr += [mean]
            std_arr += [std]

        mean_arr = np.mean(np.array(mean_arr),axis=0)
        std_arr = np.mean(np.array(std_arr),axis=0)

        np.save(f"{base_out_dir}/mean.npy", mean_arr)
        np.save(f"{base_out_dir}/std.npy", std_arr)
        
        print('\n\nCompleted. ')
