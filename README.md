# SkipVQVC
Implementation of SkipVQVC with variant settings. Skip connection is an powerful technique in deep learning. However, in auto-encoder based voice conversion(VC) domain, skip connection is often no-used. Skip-connection cause model learning too fast, and overfitting on reconstruction, and such a model cannot fullfill VC anymore. In this paper, we discuss how quantization can form a strong bottleneck that skip-connection VC can fullfilled.

## Usage

# preprocessing
> python preprocessing.py [input_dir (VCTK/wav48)] [output_dir npy dir]

```bash
# File 
- SkipVQVC
  |- logger (some utlis used in tensorboard)
  |  |.
  |
  |- trainer (differnt trainer have different properties)
  |  |- train_normal.py
  |  |- train_rhythm.py (split speech to rhythm fator, shoud use vqvc+_rhythm model)
  |  |- train_mean_std.py (train with input normalized by mean and std)
  |
  |- model (different models like normal, speaker vae, rhythm, )
  | |- .
  | |- .
  |
  |- utils
#  Training config

- **train\_dir** is your training dir


