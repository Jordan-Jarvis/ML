# -*- coding: utf-8 -*-
"""
@author: Jordan
This is an implementation and a test of a KNN algorithm, made with numpy
and standard libraries. SKlearn datasets used for testing.
"""
from sklearn import datasets
import numpy as np
from statistics import mode
from collections import Counter
iris = datasets.load_iris()
X = iris.data
Y = iris.target

class KNN:
    
    def __init__(self):
        self.x = []
        self.y = []
        self.k = 1
    
    def fit(self, X, Y, k):
        self.x = X
        self.y = Y
        self.k = k
        return 0
    def predict(self, X_Test):
        d = []
        PredictionsX = []
        predictions = (self.distance(X_Test, self.x, self.y))
        #print(predictions)
        guesses = []
        for p in predictions:
            PredictionsY = []
            tempX = self.x
            tempY = self.y

            for i in range(0,self.k):
                smallestIDX = p.index(min(p))
                #print(p[smallestIDX])
                PredictionsX.append(tempX[smallestIDX])
                PredictionsY.append(tempY[smallestIDX])

                tempX = np.delete(tempX,smallestIDX)
                tempY = np.delete(tempY,tempY[smallestIDX])
                p.pop(smallestIDX)
            #print(predictions[smallestIDX])
            guesses.append(mode(PredictionsY))
            #print(self.x)
        return guesses
        #for i in range(0,k):
#        print(d)
    def distance(self, a,b,y):

        DistanceMatrix = []

        
        for g in a:
            i = 0
            z = []
            for e in b:
                z.append(((e[0]-g[0])**2 + (e[1]-g[1])**2 + (e[2]-g[2])**2 + (e[3]-g[3])**2))
                i = i+1
            DistanceMatrix.append(z)

            i = i+1
            
        return DistanceMatrix
    
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = .5)
testt = KNN()
testt.fit(X_train, y_train, 4)
predictions = testt.predict(X_test)

from sklearn.metrics import accuracy_score
print("Using KNN the algorithm is {} accurate".format(100*accuracy_score(y_test, predictions)))




