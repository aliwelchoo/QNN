import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    
def sigmoid_derivative(x):
    return sigmoid(x)*(1-sigmoid(x))

class QuantumNeuralNetwork:
    def __init__(self, x, y):
        self.set_instance(x, y)
        self.probs   = np.random.rand(self.input.shape[0])
        self.weights   = np.eye(2 * self.input.shape[0],y.shape[1]) 
        self.output     = np.zeros(self.y.shape)

    def feedforward(self):
        self.layer1 = np.zeros(2 * self.input.shape[0])
        for i in range(2 * self.input.shape[0]) :
            self.layer1[i] = sigmoid(self.input[int(np.floor(i/2))]) * (i%2 + (2 * (i%2) - 1) * self.probs[int(np.floor(i/2))])
        self.output = sigmoid(np.dot(self.layer1, self.weights))
        self.layer1 = np.matrix(self.layer1)
        return(sum(np.square(self.output - self.y)))
        
    def backprop(self, lr):
        # application of the chain rule to find derivative of the loss function with respect to probs and weights
        d_layer = np.matrix(np.multiply(2*(self.y - self.output), (sigmoid_derivative(self.output))))
        d_weights = np.dot(self.layer1.T, d_layer)
        d_probs = np.zeros(self.probs.shape)
        for i in range(2 * self.input.shape[0]) :
            d_probs[int(np.floor(i/2))] += (2 * (i%2) - 1) * sigmoid(self.input[int(np.floor(i/2))]) * np.multiply(np.multiply(d_layer, self.weights[i,:].T), self.layer1[:,i]).sum()

        # update the weights with the derivative (slope) of the loss function
        self.weights += lr * d_weights
        self.probs += lr * d_probs
        
    def set_instance(self, x, y):
        inputx = np.ones((x.shape[1] + 1,1))
        inputx = x.T
        self.input      = inputx
        self.y          = y
        
    def step_train(self, x, y):
        self.set_instance(x, y)
        cost = self.feedforward()
        self.backprop(0.3)
        return cost