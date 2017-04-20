#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 16:11:55 2017

@author: administorzz
"""

import numpy as np
import pandas as pd


def accuracy_score(y_true, y_pred):
   
    score = y_true == y_pred
    return np.average(score)


def confusion_matrix(y_true, y_pred):
 
    return pd.crosstab(y_true, y_pred,
                       rownames=['Actual'], colnames=['Predicted'])


def label_binarize(y):
   
    n_instances = len(y)
    classes_ = np.unique(y)

    y_bin = np.zeros((n_instances, len (classes_)))
    for y_i in classes_:
        i = classes_.searchsorted(y_i)
        idx = np.where(y == y_i)
        y_bin[idx, i] = 1

    return y_bin


def softmax(X):

    tmp = X - X.max(axis=1)[:, np.newaxis]
    np.exp(tmp, out=X)
    X /= X.sum(axis=1)[:, np.newaxis]

    return X

def compute_output(x,w,b):
    return np.dot(w.transpose(), x)+b


class SLNNClassifier(object):
   

    def __init__(self, learning_rate=0.01, max_iter=500, random_state=None):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = random_state

    def _init_parameters(self, fan_in, fan_out):

        W = np.random.uniform(0., 1., (fan_in, fan_out))
        b = np.random.uniform(0., 1., (1, fan_out))
        self.weight_ = W
        self.bias_ = b

    def _forward_pass(self, X):
       
        s=0
        for n in range(self.n_instances):
            x_n = X[n,:]
            for k in range(self.classes_):
                weight = self.weight_[:,k]
                bias = self.bias_[k]
                s = compute_output(x_n,weight,bias)
                self.x_out[n][k] = s                     
  
    def _compute_gradient(self, X, y):
        grad_1 = np.zeros(self.n_features,self.classes_)
        grad_2 = np.zeros(self.classes_)
        for n in range(self.n_instances):
            y_n = y[n]
            x_n = X[n,:]
            p_output = self.x_out[n][y_n]
            sum_output = np.sum(self.x_out[n,:])
            
            p_yn_xn = np.exp(p_output) / np.exp(sum_output)
            grad_1[:,y_n] += x_n * (1- p_yn_xn)
            grad_2[y_n] += (1- p_yn_xn)
            
        self.w_grad = -1/self.n_instances * grad_1
        self.b_grad = -1/self.n_instances * grad_2
    
    
    def _update_params(self):
        
        for k in range(self.classes_):
            self.weight_[:,k] = self.weight_[:,k] - self.learning_rate * self.w_grad[:,k]
            self.bias_[k] = self.bias_[k] - self.learning_rate * self.b_grad[k]
                
        
    def fit(self, X, y):
        self.acc = np.zeros(10)
        np.random.seed(seed=self.random_state)
        self.n_instances, self.n_features = X.shape  
        self.classes_ = np.unique(y)
        self.n_outputs_ = len(self.classes_)
        
        self.y_true = y

        
        for i in range(self.max_iter):
            self._forward_pass(X)
            self._compute_gradient(X,y)
            self._update_params()
            self.predict(self.x_out)
            
            if (i == 49):
                self.acc[0] = accuracy_score(self.y_true , self.y_pred)
            elif (i == 99):
                self.acc[1] = accuracy_score(self.y_true , self.y_pred)
            elif (i == 149):
                self.acc[2] = accuracy_score(self.y_true , self.y_pred)
            elif (i == 199):
                self.acc[3] = accuracy_score(self.y_true , self.y_pred)
            elif (i == 249):
                self.acc[4] = accuracy_score(self.y_true , self.y_pred)
            elif (i == 299):
                self.acc[5] = accuracy_score(self.y_true , self.y_pred)
            elif (i == 349):
                self.acc[6] = accuracy_score(self.y_true , self.y_pred)
            elif (i == 399):
                self.acc[7] = accuracy_score(self.y_true , self.y_pred)
            elif (i == 449):
                self.acc[8] = accuracy_score(self.y_true , self.y_pred)
            elif (i == 499):
                self.acc[9] = accuracy_score(self.y_true , self.y_pred)
                
    def predict(self, x_out):
        
        for n in range(self.n_instances):
            x_n = x_out[n,:]
            h = np.argmax(x_n)
            self.y_pred = h



if __name__ == "__main__":
    with open('output.txt', 'w') as f_out:
        df = pd.read_csv("/Users/administorzz/Downloads/digits.csv")

        X = df.ix[:, :-1]
        y = df.ix[:, -1]

        clf = SLNNClassifier(random_state=0)
        clf.fit(X, y)
       
        f_out.write("{0:.3f}".format(clf.acc) + '\n')
        f_out.write('\n')
        f_out.write(str(clf.confusion_matrix(clf.y_ture, clf.y_pred)))
        