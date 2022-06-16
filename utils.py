import glob
import os
import matplotlib
import torch
from torch.nn.utils import weight_norm

matplotlib.use("Agg")
import matplotlib.pylab as plt


def set_init_dict(model_dict, checkpoint_state, skip_layers=None):
    # Partial initialization: if there is a mismatch with new and old layer, it is skipped.
    for k, v in checkpoint_state.items():
        if k not in model_dict:
            print(" | > Layer missing in the model definition: {}".format(k))
    # 1. filter out unnecessary keys
    filtered_dict = {k: v for k, v in checkpoint_state.items() if k in model_dict}

    if skip_layers:
        filtered_dict = {
            k: v
            for k, v in checkpoint_state.items()
            if not any([layer in k for layer in skip_layers])
        }

    # 2. filter out different size layers
    pretrained_dict = {
        k: v for k, v in filtered_dict.items() if v.numel() == model_dict[k].numel()
    }

    for k, v in pretrained_dict.items():
        if k not in model_dict.keys():
            print("Missing:", k)

    # 3. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)

    print(
        " | > {} / {} layers are restored.".format(
            len(pretrained_dict), len(model_dict)
        )
    )
    return model_dict


def plot_spectrogram(spectrogram):
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)

    fig.canvas.draw()
    plt.close()

    return fig


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def apply_weight_norm(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        weight_norm(m)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def save_checkpoint(filepath, obj):
    print("Saving checkpoint to {}".format(filepath))
    torch.save(obj, filepath)
    print("Complete.")


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + "????????")
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]
