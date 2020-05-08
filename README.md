# SkipVQVC
Implementation of SkipVQVC with variant settings. Skip connection is an powerful technique in deep learning. However, in auto-encoder based voice conversion(VC) domain, skip connection is often no-used. Skip-connection cause model learning too fast, and overfitting on reconstruction, and such a model cannot fullfill VC anymore. In this paper, we discuss how quantization can form a strong bottleneck that skip-connection VC can fullfilled.

## Usage

# preprocessing
> python preprocessing.py [input_dir (VCTK/wav48)] [output_dir npy dir]

# File 
- SkipVQVC
  |- logger (some utlis used in tensorboard)
  |  |.
  |
  |- trainer
#  Training config

- **train\_dir** is your training dir


