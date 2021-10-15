# 202110 Compare CNN denoisers for PRNU extraction purpose
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from models import ADNetres, DnCNN, DnCNN2, FFDNet
import torch
import torch.nn as nn
# 以下为自己编写的函数
from SCIFunctions.crosscorr import crosscorr
from SCIFunctions.PCE1 import PCE1
from SCIFunctions.ReadDir40New import ReadDirnew
from SCIFunctions.NoiseExtractDL import NoiseExtractDL
from SCIFunctions.NoiseExtractFFD import NoiseExtractFFD
from SCIFunctions.NoiseExtractDA import NoiseExtractDA
from DAnetworks import UNetD

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device_ids = [0]

# postprocess = True
postprocess = False
########
## 1 Load saved models
print('Loading model ...\n')

# FFDNet
net_FFDNet = FFDNet(num_input_channels=1)
model_FFDNet = nn.DataParallel(net_FFDNet, device_ids=device_ids).cuda()
model_FFDNet.load_state_dict(torch.load('Mymodels/FFDNet.pth'))
model_FFDNet.eval()

# ADNet
net_ADNet = ADNetres(channels=1, num_of_layers=17)
model_ADNet = nn.DataParallel(net_ADNet, device_ids=device_ids).cuda()
model_ADNet.load_state_dict(torch.load('Mymodels/ADNet.pth'))
model_ADNet.eval()

 # DANet, the size is too large to upload. Contact the authors for this model
# model_DANet = UNetD(3, wf=32, depth=5).cuda()
# model_DANet.load_state_dict(torch.load('Mymodels/GDANet.pt', map_location='cpu')['D'])
# model_DANet.eval()

# DnCNN
net_DnCNN = DnCNN(channels=1, num_of_layers=17)
model_DnCNN = nn.DataParallel(net_DnCNN, device_ids=device_ids).cuda()
dict_DnCNN = torch.load('Mymodels/DnCNN.pth')
model_DnCNN.load_state_dict(dict_DnCNN)
model_DnCNN.eval()

## 2 read the RP image
RPname = 'samples/FP01_OlympusC0.png'
RP = cv2.imread(RPname, cv2.IMREAD_GRAYSCALE)
RP = (np.float32(RP) - 127.5)/32.5

## 3 read the probe image
imxname = 'samples/Olympus_mju_1050SW_0_23679.JPG'
Probeimage = cv2.imread(imxname, cv2.IMREAD_GRAYSCALE)
ProbeimageC = cv2.imread(imxname)
ProbeimageC = ProbeimageC[:, :, ::-1]

HalfB = 256
# If the image is too large, crop it for speed
if RP.shape[0] > 2*HalfB:
    # center crop
    centerY = np.round(RP.shape[0] / 2).astype(int)
    centerX = np.round(RP.shape[1] / 2).astype(int)
    RP = RP[centerY-HalfB: centerY+HalfB,  centerX-HalfB: centerX+HalfB]
    Probeimage = Probeimage[centerY - HalfB: centerY + HalfB, centerX - HalfB: centerX + HalfB]
    ProbeimageC = ProbeimageC[centerY - HalfB: centerY + HalfB, centerX - HalfB: centerX + HalfB, :]
Probeimagefloat = np.float32(Probeimage)

## 4 Extract noisex form probe
Noisex, NoisexnpDnCNN, stdValue = NoiseExtractDL(Probeimage, model_DnCNN, postprocess)
# Since FFDNet can denoise images with a wide range of sigmas, we need specify the sigma in denoising
Noisex, NoisexnpFFDNet, stdValue = NoiseExtractFFD(Probeimage, 3/255, True, model_FFDNet, postprocess)
Noisex, NoisexnpADNet, stdValue = NoiseExtractDL(Probeimage, model_ADNet, postprocess)
# DANet only accept rgb images
# Noisex, NoisexnpDANet, stdValue = NoiseExtractDA(ProbeimageC, model_DANet, postprocess)

## 5 Compute the correlation between probe and RP
KI = Probeimagefloat * RP

C = crosscorr(NoisexnpDnCNN, KI)
PCE_value_DnCNN = PCE1(C)
print("PCE of model_DnCNN")
print(PCE_value_DnCNN)

C = crosscorr(NoisexnpFFDNet, KI)
PCE_value_FFDNet = PCE1(C)
print("PCE of model_FFDNet")
print(PCE_value_FFDNet)

C = crosscorr(NoisexnpADNet, KI)
PCE_value_ADNet = PCE1(C)
print("PCE of model_ADNet")
print(PCE_value_ADNet)

# C = crosscorr(NoisexnpDANet, KI)
# PCE_value_DANet = PCE1(C)
# print("PCE of model_DANet")
# print(PCE_value_DANet)

## 6 Visualize
plt.figure(1)
plt.subplot(2, 3, 1)
plt.title("Probe image")
plt.imshow(Probeimage, cmap="gray")

plt.subplot(2, 3, 2)
plt.title("DnCNN")
plt.imshow(NoisexnpDnCNN, cmap="gray")

plt.subplot(2, 3, 3)
plt.title("FFDNet")
plt.imshow(NoisexnpFFDNet, cmap="gray")

plt.subplot(2, 3, 4)
plt.title("ADNet")
plt.imshow(NoisexnpADNet, cmap="gray")

# plt.subplot(2, 3, 5)
# plt.title("DANet")
# plt.imshow(NoisexnpDANet, cmap="gray")

plt.show()
Finish = 1

