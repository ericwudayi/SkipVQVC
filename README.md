# vq-vc-pytorch
Implementation of VQVC

## Requisite

* Python >= 3.6
* numpy >= 1.17.2
* PyTorch >= 1.1
* librosa
> pip3 -r requirements.txt
## Usage

1. One-Shot convert the voice to target (training)

> python3 main.py source_wav target_wav 

2. the output melspectrogram is in mel_out/, and the wav is in wav_out/
## Sample

> python3 main.py sample/sample01/input.wav sample/sample01/target.wav


## Training

1. Download VCTK dataset into data/vctk/

2. python3 preprocessing.py "path_to_vctk" (example: ./data/vctk/VCTK-Corpus/wav48)

3. python3 train.py
