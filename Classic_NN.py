import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    
def sigmoid_derivative(x):
    return sigmoid(x)*(1-sigmoid(x))

class NeuralNetwork:
    def __init__(self, x, y, hn):
        self.set_instance(x, y)
        self.weights1   = np.eye(x.shape[1] + 1,hn) 
        self.weights2   = np.eye(hn + 1,y.shape[1]) 
        self.output     = np.zeros(self.y.shape)
        self.layer1     = np.ones((1, hn + 1))

    def feedforward(self):
        self.layer1[:,1:] = sigmoid(np.dot(self.input.T, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))
        return(sum(np.square(self.output - self.y)))
        
    def backprop(self, lr):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, np.multiply(2*(self.y - self.output), (sigmoid_derivative(self.output))))
        d_weights1 = np.dot(self.input,  np.multiply(np.dot(np.multiply(2*(self.y - self.output), (sigmoid_derivative(self.output))), self.weights2[1:,:].T), sigmoid_derivative(self.layer1[:,1:])))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += lr * d_weights1
        self.weights2 += lr * d_weights2
        
    def set_instance(self, x, y):
        inputx = np.ones((x.shape[1] + 1,1))
        inputx[1:] = x.T
        self.input      = inputx
        self.y          = y
        
    def step_train(self, x, y):
        self.set_instance(x, y)
        cost = self.feedforward()
        self.backprop(1)
        return cost