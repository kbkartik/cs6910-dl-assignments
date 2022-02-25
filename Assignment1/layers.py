import numpy as np

class Linear:
    def __init__(self, in_features, out_features, weight_init_type, activation_type, gain):
        self.in_features = in_features
        self.out_features = out_features
        self.reset_params(activation_type, gain, weight_init_type)

    def reset_params(self, activation_type, gain, weight_init_type):
        if weight_init_type == 'xavier':
            #limit = gain * np.sqrt(6/(self.in_features+self.out_features))
            limit = gain * np.sqrt(1/self.in_features)
            #self.weights = np.random.uniform(-limit, limit, (self.in_features, self.out_features))
            self.weights = np.random.normal(0, limit, (self.in_features, self.out_features))
        elif weight_init_type == 'random':
            self.weights = np.random.uniform(-1, 1, (self.in_features, self.out_features))
            #self.weights = np.random.normal(0, 0.1, (self.in_features, self.out_features))
            
        self.weights[:, -1] = np.zeros(self.weights.shape[0])

    def __call__(self, x):
        return np.dot(x, self.weights)