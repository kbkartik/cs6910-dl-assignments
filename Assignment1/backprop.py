import numpy as np

class Backprop:

    def backward(self, loss_fn, layer_activation, layer_wise_output, network, y_true):
        
        n_layer_weights = len(network)
        layer_gradients = []
        accum_grad = loss_fn.gradient(layer_wise_output[-1], y_true)
        for i in range(n_layer_weights)[::-1]:
            if i == n_layer_weights - 1:
                # last layer
                accum_grad = np.multiply(accum_grad, layer_activation[i].gradient(layer_wise_output[i+1], y_true))
            else:
                accum_grad = np.multiply(accum_grad, layer_activation[i].gradient(layer_wise_output[i+1]))
            grad_w = np.dot(layer_wise_output[i].T, accum_grad)
            accum_grad = np.dot(accum_grad, network[i].weights.T)
            
            # Append gradients
            layer_gradients.append(grad_w)
        
        layer_gradients.reverse()
        
        return layer_gradients