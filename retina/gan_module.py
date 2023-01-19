import matplotlib.pyplot as plt  # plotting library
import numpy as np  # this module is useful to work with numerical arrays
import pandas as pd
import scipy.io as sio  # Tai siirrä ApricotData luokkaan

# import random
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm

import pdb


class GAN(nn.Module):
    """Generative Adversarial Network class"""

    def __init__(self, apricot_data_folder, gc_type, response_type):

        # Get the data. From apricot_to_pillow.ipynb
        datafile = "mosaicGLM_apricot_ONParasol-1-mat.mat"
        data = sio.loadmat(datafile)
        data = data["mosaicGLM"][0]
        cellnum = 0
        i = 0
        stimulus_filter = data[cellnum][0][0][i][0][0][3][0][0][0]

        pass

    def __call__(self, *args, **kwargs):
        return self

    def __getattr__(self, *args, **kwargs):
        return self
