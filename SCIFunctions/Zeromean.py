import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

# 同样来去除SPN中同型号相机的共同部分
def Zeromean(X):
    M,N = X.shape[0],X.shape[1]
    Xzm = X - np.mean(X)
    MeanR = [np.mean(Xzm, 0)]
    MeanC = [np.mean(Xzm, 1)]

    OneCol = np.ones_like(np.transpose(MeanC))
    LPR = OneCol @ MeanR
    # plt.imshow(LPR,cmap="gray",vmin=-1, vmax=1)
    # plt.colorbar()
    # plt.show()

    OneRow = np.ones_like(MeanR)
    LPC = np.transpose(MeanC) @ OneRow
    # plt.imshow(LPC,cmap="gray",vmin=-1, vmax=1)
    # plt.colorbar()
    # plt.show()

    ZMX = Xzm-LPR-LPC

    return ZMX
