# By HuiZeng 20210329
import numpy as np
from SCIFunctions.Zeromean import Zeromean

# remove NUAs sharing in cameras of the camera model
def ZeroMeanTotal(X):
    Y = np.zeros_like(X)
    X11 = X[::2, ::2]
    X12 = X[::2, 1::2]
    X21 = X[1::2, ::2]
    X22 = X[1::2, 1::2]

    Y[::2, ::2] = Zeromean(X11)
    Y[::2, 1::2] = Zeromean(X12)
    Y[1::2, ::2] = Zeromean(X21)
    Y[1::2, 1::2] = Zeromean(X22)

    return Y
