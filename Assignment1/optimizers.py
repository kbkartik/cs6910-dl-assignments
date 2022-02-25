class SGD:

    def __init__(self, lr):
        self.lr = lr
        self.type_ = 'SGD'

    def step(self, network, grads):
        
        n_layer_weights = len(network)
        for i in range(n_layer_weights):
            network[i].weights -= self.lr * grads[i]
        
        return network

class SGDMomentum:

    def __init__(self, lr):
        self.lr = lr
        self.mu = 0.9
        self.update = []
        self.type_ = 'SGDM'

    def step(self, network, grads):

        n_layer_weights = len(network)
        for i in range(n_layer_weights):
            u = self.lr * grads[i]
            if len(self.update) != n_layer_weights:
                # Same as initializing update to 0
                self.update.append(u)
            else:
                self.update[i] = self.mu * self.update[i] + u

            # Updating layer weights
            network[i].weights -= self.update[i]
        
        return network

class RMSProp:

    def __init__(self, lr):
        self.lr = lr
        self.beta = 0.999
        self.epsilon = 1e-8
        self.history = []
        self.type_ = 'RMSProp'
        
    def step(self, network, grads):

        n_layer_weights = len(network)
        for i in range(n_layer_weights):
            h = (1 - self.beta) * np.power(grads[i], 2)
            if len(self.history) != n_layer_weights:
                # Same as initializing history to 0
                self.history.append(h)
            else:
                self.history[i] = self.beta * self.history[i] + h

            per_weight_hist = np.sqrt(self.history[i] + self.epsilon)
            # Updating layer weights
            network[i].weights -= self.lr * np.divide(grads[i], per_weight_hist)
    
        return network

class Adam:

    def __init__(self, lr):
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.lr = lr
        self.t = 0
        self.type_ = 'Adam'
        self.update = []
        self.history = []
        
    def step(self, network, grads):

        self.t += 1

        n_layer_weights = len(network)
        for i in range(n_layer_weights):
            u = self.lr * grads[i]
            h = (1 - self.beta2) * np.power(grads[i], 2)
            if len(self.history) != n_layer_weights:
                # Same as initializing update and history to 0
                self.update.append(u)
                self.history.append(h)
            else:
                self.update[i] = self.beta1 * self.update[i] + u
                self.history[i] = self.beta2 * self.history[i] + h

            update_hat = self.update[i]/(1 - np.power(self.beta1, self.t))
            history_hat = self.history[i](1 - np.power(self.beta2, self.t))

            per_weight_hist = np.sqrt(history_hat + self.epsilon)
            # Updating layer weights
            network[i].weights -= self.lr * np.divide(update_hat, per_weight_hist)

        return network

class NAG:

    def __init__(self, lr, n_hidden_layers):
        self.lr = lr
        self.mu = 0.9
        self.update = []
        self.type_ = 'NAG'
        self.n_layer_weights = n_hidden_layers + 1

    def get_curr_update(self, i):

        if len(self.update) != self.n_layer_weights:
            return 0
        else:
            return self.update[i]

    def update(self, weight, grad_w_lookahead, i):
        u = self.lr * grad_w_lookahead
        if len(self.update) != self.n_layer_weights:
            self.update.append(u)
        else:
            self.update[i] = self.mu * self.update[i] +  u
        weight -= self.update[i]
        return weight

activation_dict = {'sgd': SGD(), 'sgdm': SGDMomentum(), 'rmsprop': RMSProp(), 'adam': Adam(), 'nesterov': NAG()}#, 'nadam': NAdam()}