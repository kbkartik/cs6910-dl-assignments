import numpy as np

class SGD:

    def __init__(self, lr, lamda, *args):
        self.lr = lr
        self.lamda = lamda
        self.type_ = 'sgd'

    def step(self, weight, grad_w, *args):
        weight -= (self.lr * grad_w + self.lamda * grad_w)
        
        return weight

class SGDMomentum:

    def __init__(self, lr, lamda, n_hidden_layers, *args):
        self.lr = lr
        self.lamda = lamda
        self.mu = 0.9
        self.type_ = 'sgdm'
        self.n_layer_weights = n_hidden_layers + 1
        self.update = []

    def step(self, weight, grad_w, i, *args):
        u = self.lr * grad_w
        i = self.n_layer_weights -1-i
        if len(self.update) != self.n_layer_weights:
            # Same as initializing update to 0
            self.update.append(u)
        else:
            self.update[i] = self.mu * self.update[i] + u

        # Updating layer weights
        weight -= (self.update[i] + self.lamda * grad_w)
        
        return weight

class NAG:

    def __init__(self, lr, lamda, n_hidden_layers, *args):
        self.lr = lr
        self.lamda = lamda
        self.mu = 0.9
        self.type_ = 'NAG'
        self.n_layer_weights = n_hidden_layers + 1
        self.update = []

    def get_curr_update(self, i):
        if len(self.update) != self.n_layer_weights:
            return 0
        else:
            i = self.n_layer_weights -1-i
            return self.mu*self.update[i]

    def step(self, weight, grad_w_lookahead, i):
        u = self.lr * grad_w_lookahead
        i = self.n_layer_weights -1-i
        if len(self.update) != self.n_layer_weights:
            self.update.append(u)
        else:
            self.update[i] = self.mu * self.update[i] +  u

        weight -= (self.update[i] + self.lamda * grad_w_lookahead)
        return weight

class RMSProp:

    def __init__(self, lr, lamda, n_hidden_layers, *args):
        self.lr = lr
        self.lamda = lamda
        self.beta = 0.999
        self.epsilon = 1e-5
        self.type_ = 'rmsprop'
        self.history = []
        self.n_layer_weights = n_hidden_layers + 1
        
    def step(self, weight, grad_w, i, *args):

        h = (1 - self.beta) * np.power(grad_w, 2)
        i = self.n_layer_weights -1-i
        if len(self.history) != self.n_layer_weights:
            # Same as initializing history to 0
            self.history.append(h)
        else:
            self.history[i] = self.beta * self.history[i] + h

        per_weight_hist = np.sqrt(self.history[i] + self.epsilon)
        # Updating layer weights
        weight -= (self.lr * np.divide(grad_w, per_weight_hist) + self.lamda * grad_w)
    
        return weight

class Adam:

    def __init__(self, lr, lamda, n_hidden_layers, *args):
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-5
        self.lr = lr
        self.lamda = lamda
        self.t = 0
        self.type_ = 'adam'
        self.update = []
        self.history = []
        self.n_layer_weights = n_hidden_layers + 1

    def step(self, weight, grad_w, i, *args):

        u = self.lr * grad_w
        h = (1 - self.beta2) * np.power(grad_w, 2)
        i = self.n_layer_weights -1-i

        if len(self.history) != self.n_layer_weights or len(self.update) != self.n_layer_weights:
            # Same as initializing update and history to 0
            self.update.append(u)
            self.history.append(h)
        else:
            self.update[i] = self.beta1 * self.update[i] + u
            self.history[i] = self.beta2 * self.history[i] + h

        update_hat = self.update[i]/(1 - np.power(self.beta1, self.t))
        history_hat = self.history[i]/(1 - np.power(self.beta2, self.t))

        per_weight_hist = np.sqrt(history_hat + self.epsilon)
        # Updating layer weights
        weight -= (self.lr * np.divide(update_hat, per_weight_hist) + self.lamda * grad_w)

        return weight

optimizer_dict = {'sgd': SGD, 'sgdm': SGDMomentum, 'rmsprop': RMSProp, 'adam': Adam, 'NAG': NAG}#, 'nadam': NAdam()}