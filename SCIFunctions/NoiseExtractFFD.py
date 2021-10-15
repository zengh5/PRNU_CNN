import numpy as np
import torch
from torch.autograd import Variable
from denoise_FFD import denoise_FFD
# 以下为自己编写的函数
from SCIFunctions.ZeroMeanTotal import ZeroMeanTotal
from SCIFunctions.Zeromean import Zeromean
from SCIFunctions.WienerInDFT import WienerInDFT


def NoiseExtractFFD(imorig, noise_sigma, cuda, model, postprocess):
    imorig = np.expand_dims(imorig, 0)
    Noisex = denoise_FFD(imorig, noise_sigma, cuda, model)

    Noisexnp = Noisex
    stdValue = np.std(Noisex)
    if postprocess:
        Noisex = ZeroMeanTotal(Noisex)
        std = np.std(Noisex)
        Noisex = WienerInDFT(Noisex, std)

    return Noisex, Noisexnp, stdValue