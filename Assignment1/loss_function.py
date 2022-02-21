import numpy as np

class Categorical_CE:

    def __init__(self):
        pass

    def loss(self, y_hat, y_true):
        y_hat = np.clip(y_hat, 1e-15, 1 - 1e-15)
        y_hat_wrt_true = np.multiply(y_hat, y_true)
        return -1*np.sum(y_hat_wrt_true)

    def gradient(self, y_hat, y_true):
        y_hat = np.clip(y_hat, 1e-15, 1 - 1e-15)
        y_hat_wrt_true = np.multiply(y_hat, y_true)
        y_hat_wrt_true_vec = -1/np.sum(y_hat_wrt_true, axis=1, keepdims=True)
        #return np.multiply(y_hat_wrt_true_vec, y_true)
        return y_hat_wrt_true_vec