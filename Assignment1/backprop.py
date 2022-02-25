import numpy as np

class Backprop:

    def __init__(self, loss_fn, optimizer):
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def backward(self, layer_activation, layer_wise_output, network, y_true):
        
        n_layer_weights = len(network)
        
        accum_grad = self.loss_fn.gradient(layer_wise_output[-1], y_true)
        if self.optimizer.type_ == 'adam':
            self.optimizer.t += 1
        
        for i in range(n_layer_weights)[::-1]:
            if i == n_layer_weights - 1:
                # last layer
                accum_grad = np.multiply(accum_grad, layer_activation[i].gradient(layer_wise_output[i+1], y_true))
            else:
                accum_grad = np.multiply(accum_grad, layer_activation[i].gradient(layer_wise_output[i+1]))
            
            grad_w = np.dot(layer_wise_output[i].T, accum_grad)
            
            curr_w = np.copy(network[i].weights)
            if self.optimizer.type_ == 'NAG':
                accum_grad = np.dot(accum_grad, (curr_w - self.optimizer.get_curr_update(i)).T)
            else:
                accum_grad = np.dot(accum_grad, network[i].weights.T)
            
            network[i].weights = self.optimizer.step(network[i].weights, grad_w, i)
        
        return network