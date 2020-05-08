import torch
from tensorboardX import SummaryWriter
from plotting_utils import  plot_spectrogram_to_numpy, plot_onset_to_numpy
class Logger(SummaryWriter):
    def __init__(self, logdir):
        super(Logger, self).__init__(logdir)

    def log_training(self, iteration, **kwarg):
        self.add_scalars("training.loss", kwarg , iteration)
    
    def log_validation(self, iteration, **kwarg):
        for key in (kwarg.keys()):
            (type_, method_, data) = kwarg[key]
            
            if type_=="audio":
                self.add_audio(
                f'{key}',
                data, iteration, sample_rate=method_)
            elif type_ == "scalars":
                self.add_scalars("validation.loss", data, iteration)
            elif type_ == "image":
                data = data.detach().cpu().numpy()
                self.add_image(
                f'{key}',
                method_(data),
                iteration, dataformats='HWC')