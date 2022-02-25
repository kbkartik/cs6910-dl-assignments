import numpy as np

class Categorical_CE:

    def __init__(self):
        pass

    def loss(self, y_hat, y_true):
        y_hat = np.clip(y_hat, 1e-15, 1 - 1e-15)
        y_hat_wrt_true = np.multiply(y_hat, y_true)
        y_hat_wrt_true_vec = np.sum(y_hat_wrt_true, axis=1, keepdims=True)
        return -1*np.sum(np.log(y_hat_wrt_true_vec))

    def gradient(self, y_hat, y_true):
        y_hat = np.clip(y_hat, 1e-15, 1 - 1e-15)
        y_hat_wrt_true = np.multiply(y_hat, y_true)
        y_hat_wrt_true_vec = -1/np.sum(y_hat_wrt_true, axis=1, keepdims=True)
        return y_hat_wrt_true_vec

class MSE:
    
    def __init__(self):
        pass
        
    def loss(self, y_hat, y_true):
        y_hat = np.clip(y_hat, 1e-15, 1 - 1e-15)
        return 0.5*np.sum(np.power(y_hat - y_true, 2))
    
    def gradient(self, y_hat, y_true):
        y_hat = np.clip(y_hat, 1e-15, 1 - 1e-15)
        return -1 * (y_true - y_hat)