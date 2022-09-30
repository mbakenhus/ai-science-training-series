# this code is based on pieces of the first assignment from Stanford's CSE231n course, 
# hosted at https://github.com/cs231n/cs231n.github.io with the MIT license

from builtins import object
import numpy as np

from layers import *
from layer_utils import *

class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss (categorical cross-entropy) that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure is affine - relu - affine - softmax.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        self.params['W1'] = np.random.normal(scale=weight_scale, size=(input_dim, hidden_dim))
        self.params['W2'] = np.random.normal(scale=weight_scale, size=(hidden_dim, num_classes))
        self.params['b1'] = np.zeros((hidden_dim,))
        self.params['b2'] = np.zeros((num_classes,))

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, D)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """

        # forward pass, keeping intermediate values to avoid recomputing
        out_layer1, cache_layer1 = affine_relu_forward(X, self.params["W1"], self.params["b1"])
        scores, cache_layer2 = affine_forward(out_layer1, self.params["W2"], self.params["b2"])

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}

        # L2 regularization is L2 norm of the weights (sum of squares)
        regularization = np.sum(np.square(self.params["W1"])) + np.sum(np.square(self.params["W2"])) 
        loss_softmax, dx_softmax = softmax_loss(scores, y) 

        # total loss is softmax loss plus regularization term. Can change balance between the two by
        # changing self.reg
        # the 0.5 term is sometimes included in order for convenience to cancel out a 2 in the gradient
        loss = loss_softmax + self.reg * 0.5 * regularization

        # backward pass (computing gradients)
        dx_affine_backward, grads['W2'], grads['b2'] = affine_backward(dx_softmax, cache_layer2)
        dx, grads['W1'], grads['b1'] = affine_relu_backward(dx_affine_backward, cache_layer1) 

        # adjusting W1 and W2 gradients to incorporate the L2 regularization
        grads['W2'] += self.reg * self.params['W2']
        grads['W1'] += self.reg * self.params['W1']

        return loss, grads

    
    
class MultiLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss (categorical cross-entropy) that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure is affine - relu - affine - softmax.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=[100,100],
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg
        
        self.layers = len(hidden_dim)+1
        
        hidden_dim.insert(0,input_dim)
        hidden_dim.append(num_classes)
        
        self.params['W'] = {}
        self.params['b'] = {}
        
        for l in range(0,self.layers):
            self.params['W'][l] = np.random.normal(scale=weight_scale, size=(hidden_dim[l], hidden_dim[l+1]))
            self.params['b'][l] = np.zeros((hidden_dim[l+1],))
        

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, D)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """

        # forward pass, keeping intermediate values to avoid recomputing
        #out_layer1, cache_layer1 = affine_relu_forward(X, self.params["W1"], self.params["b1"])
        #scores, cache_layer2 = affine_forward(out_layer1, self.params["W2"], self.params["b2"])
        
        out_layer, cache_layer = {},{}
        
        
        out_layer[0], cache_layer[0] = affine_relu_forward(X, self.params['W'][0], self.params['b'][0])
        for l in range(1,self.layers-1):
            out_layer[l], cache_layer[l] = affine_relu_forward(out_layer[l-1], self.params['W'][l], self.params['b'][l])
        scores, cache_layer[self.layers-1] = affine_forward(out_layer[self.layers-2], self.params['W'][self.layers-1], self.params['b'][self.layers-1])
        
        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores
        
        loss, dx_affine_backward, grads = 0, {}, {'W' : {}, 'b' : {}}
        
        # L2 regularization is L2 norm of the weights (sum of squares)
        regularization = np.sum([np.sum(np.square(self.params['W'][l])) for l in range(0,self.layers)])
        loss_softmax, dx_softmax = softmax_loss(scores, y) 

        # total loss is softmax loss plus regularization term. Can change balance between the two by
        # changing self.reg
        # the 0.5 term is sometimes included in order for convenience to cancel out a 2 in the gradient
        loss = loss_softmax + self.reg * 0.5 * regularization / (self.layers-1)

        # backward pass (computing gradients)
        
        dx_affine_backward[self.layers-1], grads['W'][self.layers-1], grads['b'][self.layers-1] = affine_backward(dx_softmax, cache_layer[self.layers-1])
        grads['W'][self.layers-1] += self.reg * self.params['W'][self.layers-1]
        
        for l in range(1,self.layers):
            dx_affine_backward[self.layers-1-l], grads['W'][self.layers-1-l], grads['b'][self.layers-1-l] = affine_relu_backward(dx_affine_backward[self.layers-l], 
                                                                       cache_layer[self.layers-1-l]) 
            grads['W'][self.layers-1-l] += self.reg * self.params['W'][self.layers-1-l]

        return loss, grads
