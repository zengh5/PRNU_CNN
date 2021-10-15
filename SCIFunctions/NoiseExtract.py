# 需安装pywavelets
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import pywt

def NoiseExtract(img, qmf, sigma, L):
    L = 4
    height, width = img.shape[0], img.shape[1]
    img = img.astype(np.float32)
# 小波变换要求输入图像的宽，高都要被 2^L 整除，不能整除的话要进行填充
    m = np.power(2,L)
    nr = math.ceil(height / m) * m
    nc = math.ceil(width / m) * m
    padr = ((nr-height)/2).astype(np.int)
    padc = ((nc-width)/2).astype(np.int)
    img_pad = np.pad(img, ((padr, padr), (padc, padc)), 'symmetric')
    # plt.figure(figsize=(3, 3))
    # plt.imshow(np.array(img_pad,dtype='uint8'), cmap="gray", vmin=0, vmax=255)
    # plt.show()
    NoiseVar =  np.power(sigma,2)
## 小波变换结果和Matlab wavedec2相同，但和mdwt有区别，是因为filter的方向和shift的关系
## 效果上无差别

    #小波分解
    wave_trans = pywt.wavedec2(img_pad, 'db4', level=L, mode='periodization')

    LL = wave_trans[0]
    # 将低频分量用0代替
    wave_trans_dn=[np.zeros_like(LL, dtype=np.float32)]
    for i in range(1, L + 1):
        HH = wave_trans[i][2]
        LH = wave_trans[i][1]
        HL = wave_trans[i][0]
        # 此处要理解wave_trans的结构，对小波系数进行去噪，这是整个算法的灵魂所在
        temp3 = WaveNoise(HH, NoiseVar)
        temp2 = WaveNoise(LH, NoiseVar)
        temp1 = WaveNoise(HL, NoiseVar)
        t = (temp1, temp2, temp3)
        wave_trans_dn.append(t)

    rec_im = pywt.waverec2(wave_trans_dn, 'db4', mode='periodization')
    image_noise = rec_im[padr:(padr + height), padc: (padc + width)]
    return image_noise
    # return wave_im
    # wave_trans = mdwt(img_pad, qmf, L);

# 算法思想：相邻小波系数之间具有相关性，具体来说就是小波系数的局部方差是平稳的：一块变化剧烈的小波系数周围
# 区域的小波系数变化往往也会剧烈，一块平坦的小波系数周围区域的小波系数往往也会平坦
def WaveNoise(coef,NoiseVar):
    # 因为小波系数每个区域的均值约为0，D(X) = E(X^2)-(E(X)^2) =E(X^2)
    # 所以EstVar1就是局部方差
    tc = coef*coef
    # cv2.blur() 是求邻域均值
    EstVar1 = cv2.blur(tc, (3, 3))
    # 把方差小于NoiseVar的值置为0，这就是去噪
    temp = EstVar1 - NoiseVar
    coefVar = np.maximum(temp,0)
    # 为了算法更可靠，我们用不同大小的窗口来计算方差，取其中的最小值为最终的方差
    for w in range(5, 10, 2):
        EstVar1 = cv2.blur(tc, (w, w))
        temp = EstVar1- NoiseVar
        EstVar = np.maximum(temp,0)
        coefVar = np.minimum(coefVar, EstVar)

    # 理解这个公式，coefVar=0 时，说明这个位置小波系数全部是噪声，所以tc =coef
    #               coefVar>>NoiseVar时，说明这个位置小波系数主要是图像纹理，所以tc = 0
    tc = (coef* NoiseVar)/ (coefVar + NoiseVar)
    return tc
    # return wave_im
    # wave_trans = mdwt(img_pad, qmf, L);