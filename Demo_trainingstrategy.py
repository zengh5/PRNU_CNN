# 202110 Compare CNN denoisers for PRNU extraction purpose
import os
import cv2
import numpy as np
import time
import glob
import matplotlib.pyplot as plt
from models import ADNetres, DnCNN, DnCNN2, FFDNet
import torch
import torch.nn as nn
# user defined
from SCIFunctions.crosscorr import crosscorr
from SCIFunctions.PCE1 import PCE1
from SCIFunctions.NoiseExtractDL import NoiseExtractDL


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device_ids = [0]

# postprocess = True
postprocess = False
########
## 1 Load saved models
print('Loading model ...\n')

# SPNCNN-MSE
# DnCNN() is used for reproduce DnCNN, MSE
net_DnCNN = DnCNN(channels=1, num_of_layers=17)
net_MSE = DnCNN(channels=1, num_of_layers=17)

model_DnCNN = nn.DataParallel(net_DnCNN, device_ids=device_ids).cuda()
model_MSE = nn.DataParallel(net_MSE, device_ids=device_ids).cuda()

dict_DnCNN = torch.load('Mymodels/DnCNN.pth')
dict_MSE = torch.load('Mymodels/DnCNN_SPN_MSE.pth')

model_DnCNN.load_state_dict(dict_DnCNN)
model_DnCNN.eval()
model_MSE.load_state_dict(dict_MSE)
model_MSE.eval()

# SPNCNN-Rho
model_Rho = DnCNN2()
model_Rho.cuda()
dict_Rho = torch.load('Mymodels/DnCNN_SPN_Rho.pth')
model_Rho.load_state_dict(dict_Rho)
model_Rho.eval()

## 2 read the RP image
RPname = 'samples/FP01_OlympusC0.png'
# RPname = 'samples/Canon_Ixus55_0_2662_s1_112.png'
RP = cv2.imread(RPname, cv2.IMREAD_GRAYSCALE)
RP = (np.float32(RP) - 127.5)/32.5

## 3 read the probe image
imxname = 'samples/Olympus_mju_1050SW_0_23679.JPG'
# imxname = 'samples/Canon_Ixus55_0_2662_s1_112_1_.png'
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

## 4 Extract noisex
Noisex, NoisexnpDnCNN, stdValueDnCNN = NoiseExtractDL(Probeimage, model_DnCNN, postprocess)
Noisex, NoisexnpMSE, stdValueMSE = NoiseExtractDL(Probeimage, model_MSE, postprocess)
Noisex, NoisexnpRho, stdValueRho = NoiseExtractDL(Probeimage, model_Rho, postprocess)

## 5 Compute the correlation between probe and RP
KI = Probeimagefloat * RP

C = crosscorr(NoisexnpDnCNN, KI)
PCE_value_DnCNN = PCE1(C)
print("PCE of model_DnCNN")
print(PCE_value_DnCNN)

C = crosscorr(NoisexnpMSE, KI)
PCE_value_MSE = PCE1(C)
print("PCE of model_MSE")
print(PCE_value_MSE)

C = crosscorr(NoisexnpRho, KI)
PCE_value_Rho = PCE1(C)
print("PCE of model_Rho")
print(PCE_value_Rho)

## 6  Visualize
plt.figure(1)
plt.subplot(2, 2, 1)
plt.title("Probe image")
plt.imshow(Probeimage, cmap="gray")
plt.axis('off')

plt.subplot(2, 2, 2)
plt.title("DnCNN, PCE= %.1f" % PCE_value_DnCNN)
plt.imshow(NoisexnpDnCNN, cmap="gray", vmin=-stdValueDnCNN, vmax=stdValueDnCNN)
plt.axis('off')

plt.subplot(2, 2, 3)
plt.title("DnCNN-SPN-MSE, PCE= %.1f" % PCE_value_MSE)
plt.imshow(NoisexnpMSE, cmap="gray", vmin=-stdValueMSE, vmax=stdValueMSE)
plt.axis('off')

plt.subplot(2, 2, 4)
plt.title("DnCNN-SPN-Rho, PCE= %.1f" % PCE_value_Rho)
plt.imshow(NoisexnpRho, cmap="gray", vmin=-stdValueRho, vmax=stdValueRho)
plt.axis('off')

# plt.savefig("3strategies.png")
plt.show()
Finish = 1

