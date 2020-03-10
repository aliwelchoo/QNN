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
        
from sklearn import datasets
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import pandas as pd
iris = datasets.load_iris()

X = iris.data
y = np.array(pd.get_dummies(pd.DataFrame(iris.target)[0]))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

y_train = np.matrix(y_train)
X_train = np.matrix(X_train)
X_scale = X_train.max(axis = 0)
X_train = X_train/X_scale

TC = []
NN = NeuralNetwork(X_train[0], y_train[0], 5)
fig = plt.figure(num="LIVE", figsize=(14, 6), dpi=80, facecolor='w', edgecolor='k')
for _ in range(100):
    C = []
    Result = []
    for it in range(len(X_train)) :
        C.append(NN.step_train(X_train[it], y_train[it]))
        Result.append(NN.output - NN.y)
    TC.append(np.mean(C))
    plt.clf()
    plt.plot(TC, color = 'black')
    fig.canvas.draw()
    fig.canvas.flush_events()
    
plt.close()
fig = plt.figure(num="Complete", figsize=(14, 6), dpi=80, facecolor='w', edgecolor='k')
plt.plot(TC, color = 'black')

print("Training:" + str(int(len(X_train) - np.rint(Result).sum(axis = 0).sum())) + "/" + str(len(X_train)))

y_test = np.matrix(y_test)
X_test = np.matrix(X_test)
X_test = X_test/X_scale
Correct = []
for it in range(len(X_test)) :
    NN.step_train(X_test[it], y_test[it])
    Correct.append((np.rint(NN.output) - NN.y).sum() == 0)
print("Testing:" + str(np.array(Correct).sum()) + "/" + str(len(X_test)))
