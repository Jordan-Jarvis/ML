# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 12:47:07 2019

@author: Jordan
This is an implementation and 2 tests of a neural network, made with numpy
and standard libraries. SKlearn datasets used for testing.
"""
import matplotlib.pylab as plt
import numpy.random as r
from sklearn import datasets
import numpy as np
from statistics import mode
from collections import Counter
iris = datasets.load_iris()
diabetes = datasets.load_wine()
X1 = diabetes.data
Y1 = diabetes.target
X = iris.data
Y = iris.target
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = .8)
nn_structure = [4, 12, 10]


X1_train, X1_test, y1_train, y1_test = train_test_split(X1, Y1, test_size = .8)
nn_structure1 = [13, 20, 10]
class NN:
    def __init__(self, nn_structure):
        self.nn_structure = nn_structure

    def convert_y_to_vect(self, y):
        y_vect = np.zeros((len(y), 10))
        for i in range(len(y)):
            y_vect[i, y[i]] = 1
        return y_vect

    def f(self, x):
        return 1 / (1 + np.exp(-x))
    def f_deriv(self, x):
        return self.f(x) * (1 - self.f(x))


    def setup_and_init_weights(self, nn_structure):
        W = {}
        b = {}
        for l in range(1, len(nn_structure)):
            W[l] = r.random_sample((nn_structure[l], nn_structure[l-1]))
            b[l] = r.random_sample((nn_structure[l],))
        return W, b

    def init_tri_values(self, nn_structure):
        tri_W = {}
        tri_b = {}
        for l in range(1, len(nn_structure)):
            tri_W[l] = np.zeros((nn_structure[l], nn_structure[l-1]))
            tri_b[l] = np.zeros((nn_structure[l],))
        return tri_W, tri_b
    def feed_forward(self, x, W, b):
        h = {1: x}
        z = {}
        for l in range(1, len(W) + 1):
            # if it is the first layer, then the input into the weights is x, otherwise, 
            # it is the output from the last layer
            if l == 1:
                node_in = x
            else:
                node_in = h[l]
            z[l+1] = W[l].dot(node_in) + b[l] # z^(l+1) = W^(l)*h^(l) + b^(l)  
            h[l+1] = self.f(z[l+1]) # h^(l) = f(z^(l)) 
        return h, z
    def calculate_out_layer_delta(self, y, h_out, z_out):
        # delta^(nl) = -(y_i - h_i^(nl)) * f'(z_i^(nl))
        return -(y-h_out) * self.f_deriv(z_out)

    def calculate_hidden_delta(self, delta_plus_1, w_l, z_l):
        # delta^(l) = (transpose(W^(l)) * delta^(l+1)) * f'(z^(l))
        return np.dot(np.transpose(w_l), delta_plus_1) * self.f_deriv(z_l)

    def train_nn(self, nn_structure, X, y, iter_num=30000, alpha=0.05):
        W, b = self.setup_and_init_weights(nn_structure)
        cnt = 0
        m = len(y)
        avg_cost_func = []
        while cnt < iter_num:
            tri_W, tri_b = self.init_tri_values(nn_structure)
            avg_cost = 0
            for i in range(len(y)):
                delta = {}
                # perform the feed forward pass and return the stored h and z values, to be used in the
                # gradient descent step
                h, z = self.feed_forward(X[i, :], W, b)
                # loop from nl-1 to 1 backpropagating the errors
                for l in range(len(nn_structure), 0, -1):
                    if l == len(nn_structure):
                        delta[l] = self.calculate_out_layer_delta(y[i,:], h[l], z[l])
                        avg_cost += np.linalg.norm((y[i,:]-h[l]))
                    else:
                        if l > 1:
                            delta[l] = self.calculate_hidden_delta(delta[l+1], W[l], z[l])
                        # triW^(l) = triW^(l) + delta^(l+1) * transpose(h^(l))
                        tri_W[l] += np.dot(delta[l+1][:,np.newaxis], np.transpose(h[l][:,np.newaxis])) 
                        # trib^(l) = trib^(l) + delta^(l+1)
                        tri_b[l] += delta[l+1]
            # perform the gradient descent step for the weights in each layer
            for l in range(len(nn_structure) - 1, 0, -1):
                W[l] += -alpha * (1.0/m * tri_W[l])
                b[l] += -alpha * (1.0/m * tri_b[l])
            # complete the average cost calculation
            avg_cost = 1.0/m * avg_cost
            avg_cost_func.append(avg_cost)
            cnt += 1
        return W, b, avg_cost_func

    def predict_y(self, W, b, X, n_layers):
        m = X.shape[0]
        y = np.zeros((m,))
        for i in range(m):
            h, z = self.feed_forward(X[i, :], W, b)
            y[i] = np.argmax(h[n_layers])
        return y
nn = NN(nn_structure)
y_v_train = nn.convert_y_to_vect(y_train)
y_v_test = nn.convert_y_to_vect(y_test)
y_train[0], y_v_train[0]

W, b, avg_cost_func = nn.train_nn(nn_structure, X_train, y_v_train)
plt.plot(avg_cost_func)
plt.ylabel('Average J')
plt.xlabel('Iteration number')
plt.show()
from sklearn.metrics import accuracy_score
y_pred = nn.predict_y(W, b, X_test, 3)

print("Using neural network the algorithm is {} accurate".format(accuracy_score(y_test, y_pred)*100))


print(y1_train)
y_v_train1 = nn.convert_y_to_vect(y1_train)
y_v_test1 = nn.convert_y_to_vect(y1_test)
y1_train[0], y_v_train1[0]


W, b, avg_cost_func = nn.train_nn(nn_structure1, X1_train, y_v_train1)
plt.plot(avg_cost_func)
plt.ylabel('Average J')
plt.xlabel('Iteration number')
plt.show()


y1_pred = nn.predict_y(W, b, X1_test, 3)

print("Using neural network the algorithm is {} accurate".format(accuracy_score(y_test, y_pred)*100))