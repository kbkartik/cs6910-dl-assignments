import numpy as np
import math

class Sigmoid:

    # Numerically stable sigmoid
    # https://stackoverflow.com/a/64717799/10886420

    def _positive_sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _negative_sigmoid(self, x):
        exp = np.exp(x)
        return exp / (exp + 1)

    def __call__(self, x):

        positive = x >= 0
        # Boolean array inversion is faster than another comparison
        negative = ~positive

        # empty contains junk hence will be faster to allocate
        # Zeros has to zero-out the array after allocation, no need for that
        result = np.empty_like(x)
        result[positive] = 1 / (1 + np.exp(-x[positive])) #positive sigmoid
        exp = np.exp(x[negative])
        result[negative] = exp / (exp + 1) # Negative sigmoid

        return result

    def gradient(self, hidden_activation):
        #p = self.__call__(x)
        return hidden_activation * (1-hidden_activation)

class Tanh:

    # Numerically stable tanh

    def _positive_tanh(self, x):
        t = np.exp(-2*x)
        return (1-t)/(1+t)

    def _negative_tanh(self, x):
        t = np.exp(2*x)
        return (t - 1) / (t + 1)

    def __call__(self, x):

        positive = x >= 0
        # Boolean array inversion is faster than another comparison
        negative = ~positive

        # empty contains junk hence will be faster to allocate
        # Zeros has to zero-out the array after allocation, no need for that
        result = np.empty_like(x)
        t = np.exp(-2*x[positive])
        result[positive] = (1 - t)/(1 + t) # positive tanh
        t = np.exp(2*x[negative])
        result[negative] = (t - 1) / (t + 1) # Negative tanh

        return result

    def gradient(self, hidden_activation):
        # p = self.__call__(x)
        return 1 - np.power(hidden_activation, 2)

class ReLU:

    def __call__(self, x):
        return np.where(x >= 0, x, 0)
        
    def gradient(self, x):
        return np.where(x >= 0, 1, 0)

class Softmax:

    def __call__(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x/np.sum(exp_x, axis=1, keepdims=True)

    def gradient(self, y_hat, y_true):
        y_hat = np.clip(y_hat, 1e-15, 1 - 1e-15)
        y_hat_wrt_true = np.multiply(y_hat, y_true)
        y_hat_wrt_true_vec = np.sum(y_hat_wrt_true, axis=1, keepdims=True)
        #return np.multiply(y_hat_wrt_true,  (y_true - y_hat))
        return y_hat_wrt_true - np.multiply(y_hat_wrt_true_vec, y_hat)

activation_dict = {'sigmoid': (Sigmoid(), 1), 'tanh': (Tanh(), float(5/3)), 'relu': (ReLU(), np.sqrt(2))}