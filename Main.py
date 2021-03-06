import numpy as np
import QNN
#import Classic_NN as nn
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
QNN = QNN.QuantumNeuralNetwork(X_train[0], y_train[0])
plt.close()
fig = plt.figure(num="LIVE", figsize=(14, 6), dpi=80, facecolor='w', edgecolor='k')
for _ in range(500):
    C = []
    Result = []
    for it in range(len(X_train)) :
        C.append(QNN.step_train(X_train[it], y_train[it]))
        Result.append(QNN.output - QNN.y)
    TC.append(np.mean(C))
    plt.clf()
    plt.subplot(111) 
    plt.plot(TC, color = 'black')
    plt.ylim(0,0.3)
    fig.canvas.draw()
    fig.canvas.flush_events()
    
plt.close()
fig = plt.figure(num="Complete", figsize=(14, 6), dpi=80, facecolor='w', edgecolor='k')
plt.plot(TC, color = 'black')
plt.ylim(0,0.3)

print("Training:" + str((np.matrix(np.rint(Result)).sum(axis = 1) == 0).sum()) + "/" + str(len(X_train)))

y_test = np.matrix(y_test)
X_test = np.matrix(X_test)
X_test = X_test/X_scale
Correct = []
for it in range(len(X_test)) :
    QNN.step_train(X_test[it], y_test[it])
    Correct.append((np.rint(QNN.output) - QNN.y).sum() == 0)
print("Testing:" + str(np.array(Correct).sum()) + "/" + str(len(X_test)))