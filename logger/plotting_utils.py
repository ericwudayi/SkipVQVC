import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt
import numpy as np


def save_figure_to_numpy(fig):
    # save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


class plot_spectrogram_to_numpy():

    def __call__(self, spectrogram):
        fig, ax = plt.subplots(figsize=(12, 3))
        im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                       interpolation='none')
        plt.colorbar(im, ax=ax)
        plt.xlabel("Frames")
        plt.ylabel("Channels")
        plt.tight_layout()

        fig.canvas.draw()
        data = save_figure_to_numpy(fig)
        plt.close()
        return data
class plot_alignment():

    def __call__(self, alignment):
        fig, ax = plt.subplots(figsize=(12, 3))
        alignment = alignment.transpose(1,0)
        im = ax.imshow(alignment, aspect="auto", origin="lower",
                       interpolation='none')
        plt.colorbar(im, ax=ax)
        plt.xlabel("Frames")
        plt.ylabel("Input")
        plt.tight_layout()

        fig.canvas.draw()
        data = save_figure_to_numpy(fig)
        plt.close()
        return data

class plot_onset_to_numpy():
    def __call__(onset_target, onset_predict):
        fig, ax = plt.subplots(figsize=(12, 3))
        plt.plot(range(onset_target.shape[0]), onset_target)
        plt.plot(range(onset_predict.shape[0]), onset_predict)
        plt.xlabel("Frames")
        plt.ylabel("Strength")
        plt.legend(['onset_target', 'onset_predict'])
        plt.tight_layout()
        fig.canvas.draw()
        data = save_figure_to_numpy(fig)
        plt.close()
        return data