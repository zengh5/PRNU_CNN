import numpy as np
import torch
import time
# userdefined
from SCIFunctions.ZeroMeanTotal import ZeroMeanTotal
from SCIFunctions.WienerInDFT import WienerInDFT
from skimage import img_as_float32


def NoiseExtractDA(imorig, model, postprocess):
    inputs = img_as_float32(imorig).transpose([2, 0, 1])
    INoisy = torch.from_numpy(inputs).unsqueeze(0).cuda()
    # timestart = time.time()
    with torch.no_grad():  # this can save much memory
        Out = model(INoisy)
        # since we changed the output from denoised image into residual
    # timeend = time.time()
    # print("Took", timeend - timestart, "seconds to run")
    temp = Out.cpu().numpy()[0,]
    im_noisex_rgb = temp.transpose([1, 2, 0])
    Noisex = 0.2989 * im_noisex_rgb[:, :, 0] + \
             0.5870 * im_noisex_rgb[:, :, 1] + \
             0.1140 * im_noisex_rgb[:, :, 2]

    # remove NUA, we find it is only needed in the RP side
    Noisexnp = Noisex
    stdValue = np.std(Noisex)
    if postprocess:
        Noisex = ZeroMeanTotal(Noisex)
        std = np.std(Noisex)
        Noisex = WienerInDFT(Noisex, std)

    return Noisex, Noisexnp, stdValue