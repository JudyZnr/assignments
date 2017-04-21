#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 14:23:51 2017

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

    y_bin = np.zeros((n_instances, len(classes_)))
    for y_i in classes_:
        i = classes_.searchsorted(y_i)
        idx = np.where(y == y_i)
        y_bin[idx, i] = 1

    return y_bin


def tanh(X):
    

    return np.tanh(X, out=X)


def tanh_derivative(Z):
   
    return (1 - Z ** 2)


def softmax(X):
    
    tmp = X - X.max(axis=1)[:, np.newaxis]
    np.exp(tmp, out=X)
    X /= X.sum(axis=1)[:, np.newaxis]

    return X


class MLNNClassifier(object):
    

    def __init__(self, hidden_dim=(100,), batch_size=100, learning_rate=0.01,
                 max_iter=500, reg_lambda=1.0, random_state=None):
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.reg_lambda = reg_lambda
        self.random_state = random_state

    def _initialize(self, layer_dim):
       
        self.n_layers_ = len(layer_dim)

        # Initialize weights and biases.
        self.weight_ = []
        self.bias_ = []
        for i in range(self.n_layers_ - 1):
            W, b = self._init_normalized(layer_dim[i], layer_dim[i + 1])
            self.weight_.append(W)
            self.bias_.append(b)

    def _init_normalized(self, fan_in, fan_out):
        init_bound = np.sqrt(6. / (fan_in + fan_out))
        W = np.random.uniform(-init_bound, init_bound,
                              (fan_in, fan_out))
        b = np.random.uniform(-init_bound, init_bound,
                              (1, fan_out))
        return W, b

    def _forward_pass(self, x_in):
        weight_0 = self.weight_[0]
        bias_0 = self.bias_[0]
        weight_1 = self.weight_[1]
        bias_1 = self.bias_[1]
        for n in range(100):
            x_n = x_in[n,:]
            for h in range(self.hidden_dim):
                self.s_h = np.dot(weight_0[h,:].transpose,x_n)
                self.actv_hidden[n][h] = tanh(self.s_h + bias_0[h])
                for k in range(self.classes_):
                    self.s_o = np.dot(weight_1[k,:].transpose,self.actv_hidden[h,:])
                    self.actv_output[n][k] = softmax(self.s_o + bias_1[k])
                    
      

    def _backprop(self, y, x_in):
  
        weight_1 = 0
        x_1 = 0
        x_2 = 0
        weight_1 = self.weight_[1]
        for n in range(self.batch_size):
            x_n = x_in[n,:]
            for i in range(self.hidden_dim):
                max_v = np.max(self.actv_output[i,:])
                for k in range(self.classes_):
                    if(self.actv_output[i][k]==max_v):
                        self.delta_output[i][k] = 1 - max_v                  
                    else:
                        self.delta_output[i][k] = self.actv_output[i][k]
                x_2 = np.dot(weight_1[i,:],self.delta_output[i,:])
                x_1 = tanh_derivative(x_n)
                self.delta_hidden[n][i] = np.multiply(x_1,x_2)

    def _compute_gradient(self,x_in, batch_size):
       
        self.w_grad[0] = self.w_grad_hidden
        self.w_grad[1] = self.w_grad_output          
        for n in range(batch_size):
            x_n = x_in[n,:]
            self.w_grad_output = np.dot(self.actv_hidden[n,:], self.delta_output[n,:].transpose)
            self.b_grad_output = self.delta_output[n,:]
            
            self.w_grad_hidden = np.dot(x_n, self.delta_hidden.transpose)
            self.b_grad_hidden = np.dot.delta_hidden
                            
        self.w_grad_output = self.w_grad_output + (1/batch_size) * self.w_grad_output
        self.b_grad_output = self.b_grad_output + (1/batch_size) * self.b_grad_output                                                  
        self.w_grad_hidden = self.w_grad_hidden + (1/batch_size) * self.w_grad_hidden
        self.b_grad_hidden = self.b_grad_hidden + (1/batch_size) * self.b_grad_hidden    


    def _update_params(self,  batch_size):
       
        for i in range(self.n_layers_ -1):
            self.w_grad[i] = self.w_grad[i] + ((2 * self.reg_lambda)/batch_size) * self.weight_[i]
            self.b_grad[i] = self.b_grad[i] + ((2 * self.reg_lambda)/batch_size) * self.bias_[i]

            self.weight_[i] = self.weight_[i] - self.learning_rate * self.w_grad[i]
            self.bias_[i] = self.bias_[i] - self.learning_rate * self.b_grad[i]


    def fit(self, X, y):
       
        np.random.seed(seed=self.random_state)

        self.n_instances, self.n_features = X.shape
        hidden_dim = list(self.hidden_dim)

        self.classes_ = np.unique(y)
        self.n_outputs_ = len(self.classes_)

        self.y_bin = label_binarize(y)
        self.batch_numbers = 55


        layer_dim = ([self.n_features] + hidden_dim + [self.n_outputs_]) 
        self._initialize(layer_dim)
# assign all instances into 55 batches randomly
#for each batch, it has to go through "forward pass, backwardpropagation,compute gradient descent, update parameters"
        for m in range(self.max_iter):
            self.data_preprocessing(X)
            for b in range(self.batch_numbers):
                x_in = self.array_batch[b]
# b is the bth batch in all the batches, each batch is denoted as x_in                
                self._forward_pass(x_in)
                self._backprop(x_in)
                self._compute_gradient(x_in, 100)
                self._update_params(100)
                if m in {0,50,100,150,200,250,300,350,400,450}:
                    for n in range(self.batch_size):
                        max_index = np.argmax(self.actv_output[n,:])
                        self.y_pred[b][n] = max_index
                        self.y_true[b][n] = x_in[n:,(self.n_features-1)]
                        self.acc[b] = accuracy_score(self.y_true,self.y_pred)
                if(m == 499):
                    for n in range(self.batch_size):
                        max_index = np.argmax(self.actv_output[n,:])
                        self.y_pred[b][n] = max_index
                        self.y_true[b][n] = x_in[n:,(self.n_features-1)]
                        self.final_table = confusion_matrix(self.y_true,self.y_pred)

                       
                    
    def data_preprocessing(self,X):
        y = np.zeros(self.n_instances)
        x_temp = np.zeros(self.n_instances,self.n_features)
        for n in range(self.n_instances):
            x_n = X[n,:]
            x_temp[n,:] = x_n
        self.x_batches = self.shuffle_rows(x_temp)
        for n in range(self.n_instances):
            y[n] = self.x_batches[n,(self.n_features-1)]
        self.array_batch = np.array_split(self.x_batches,100)
                  
    def shuffle_rows(self,arr):
        rows = []
        return np.random.shuffle(arr[rows[0]:rows[self.n_instances]])
                
            
if __name__ == "__main__":
    with open('output.txt', 'w') as f_out:
        df = pd.read_csv('digits.csv')

        X = df.ix[:, :-1]
        y = df.ix[:, -1]

        clf = MLNNClassifier(random_state=0)
        clf.fit(X, y)
        
        f_out.write("{0:.3f}".format(clf.acc) + '\n')
        f_out.write('\n')
        f_out.write(str(clf.final_table))
        