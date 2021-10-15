import numpy as np
# 暂且只考虑对准的情况
def PCE1(C):
    r = 5
    M,N= C.shape[0],C.shape[1]
    Cinrange = C[M-1:M, N-1:N]
    # 循环位移使得尖峰邻域位于右下角
    temp = np.roll(C,-r,axis=0)
    Rollc = np.roll(temp, -r, axis=1)
    # 将右下角置为0
    Rollc[M-2*r-1:M,N-2*r-1:N] = 0
    # 实现与原文有差异，但结果相同
    PCE_energy = np.sum(Rollc * Rollc)/ (M * N - np.power(2 * r + 1, 2))
    peakheight = C[M-1,N-1]
    # Square of the peakvalue with SIGN
    Height2 = np.sign(peakheight)* np.power(peakheight, 2)
    PCE = Height2/ PCE_energy
    Out = {'peakheight': peakheight,'PCE':PCE}
    return Out['PCE']
