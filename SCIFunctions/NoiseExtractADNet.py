import numpy as np
import torch
from torch.autograd import Variable
# 以下为自己编写的函数
from SCIFunctions.Zeromean import Zeromean
from SCIFunctions.WienerInDFT import WienerInDFT


def NoiseExtractADNet(imorig, model, postprocess):

    Img = np.float32(imorig)/255.
    Img = np.expand_dims(Img, 0)
    Img = np.expand_dims(Img, 1)

    # the probe image is used as noisy image
    INoisy = torch.Tensor(Img)
    INoisy = Variable(INoisy)
    INoisy = INoisy.cuda()
    with torch.no_grad(): # this can save much memory
        Out = model(INoisy)
        # since we changed the output from denoised image into residual
    Noisex = Out.cpu().numpy().squeeze()
    # 对SPN进行后处理，去除那些同型号相机公共的成分
    if postprocess:
        Noisex = Zeromean(Noisex)
        std = np.std(Noisex)
        Noisex = WienerInDFT(Noisex, std)

    return Noisex