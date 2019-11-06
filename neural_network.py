
from matplotlib import pyplot as plt
import numpy as np
import math

def calculate_loss(model, X, y):
    return

def predict(model, x):
    return

def build_model(X,y,nn_hdim, num_passes=20000,printloss=False):
    sampleSize = np.size(X, 1)
    print(sampleSize)
    step = 0.01
    b1 = np.zeros((1,nn_hdim))
    b2 = np.zeros((1,nn_hdim))
    nn = None
    W1 = np.full((sampleSize, nn_hdim), 100)
    W2 = np.full((nn_hdim, sampleSize), 100)


    for i in range(0, num_passes):
        a = X.dot(W1) + b1
        h = np.tanh(a)
        z = h.dot(W2) + b2
        y1 = softmax(z)
        #if(math.isnan(y1[0][0])):
            #print(i)

        back2 = y1
        a1 = 1 - ((np.tanh(a))**2)
        for k in range(sampleSize):
            for j in range(nn_hdim):
                back2[k][j] = back2[k][j] - y[j]

        gW2 = (h.T).dot(back2)
        gb2 = np.sum(back2, axis=0, keepdims=True)

        back1 = a1 * back2.dot(W2.T)
        gW1 = np.dot(X.T, back1)
        gb1 = np.sum(back1, axis=0)
        if(i % 5000 == 0):
            print(W1, W2, gW1, gW2)

        #gW2 += 0.01 * W2
        #gW1 += 0.01 * W1

        W1 = W1 - (step * gW1)
        b1 = b1 - (step * gb1)
        W2 = W2 - (step * gW2)
        b2 = b2 - (step * gb2)


        nn = {'W1': W1, 'b1': b1, 'W2':W2, 'b2': b2}


    print(nn)
    return nn


def softmax(z):
    i = np.exp(z)
    y = i / np.sum(i, axis=1, keepdims=True)
    #print("Sum of i is this", np.sum(i, axis=1, keepdims=True))
    return y

def plot_decision_boundary(pred_func, X, y):
    x_min, x_max = X[:,0].min() - .5, X[:,0].max()+.5
    y_min, y_max = X[:,1].min() - .5, X[:,1].max()+.5
    h=0.01
    xx,yy=np.meshgrid(np.arange(x_min,x_max,h), np.arange(y_min,y_max,h))
    Z=pred_function(np.c_[xx.ravel(),yy.ravel()])
    Z=Z.reshape(xx.shape)
    plt.contour(xx,yy,Z,cmap=plt.cm.Spectral)
    plt.scatter(X[:,0],X[:,1], c=y, cmap=plt.cm.Spectral)
