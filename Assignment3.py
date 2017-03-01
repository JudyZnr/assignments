#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 15:46:10 2017

@author: administorzz
"""

import numpy as np
import pandas as pd


class GaussianNB():
    
    
    
    def __init__(self):
        self.offset = 10e-5
        self.p_x_yi = 0
               
    def accurateScore(self,y1,y2):
        score = y1==y2
        return np.average(score)
    
    def confusion_matrix(self,y1,y2):
        return pd.crosstab(y1,y2,
                           rownames=['Actual'],colnames=['Predicted'])
    
    def computeMeans(self, X, y):
# mean value of each feature per class(conditional)
        num_classes = len(np.unique(y))
        self.sum_mean = np.zeros((num_classes, self.n_features))
        self.sum_var = np.zeros(self.n_features)
        
        for y_n in range(num_classes):
            X_i = X[y == y_n]
            self.sum_mean[y_n] = np.mean(X_i)
     
    def computeVariance(self, X, y):
        self.sum_var = np.var(X)
        self.sum_var += self.offset
                       
    
    def prior_class(self,y):
        num_classes = len(np.unique(y))
        self.p_y = np.zeros(num_classes)
        
        for y_n in range(num_classes):
            self.p_y[y_n] = len(y[y == y_n]) / float(len(y))

    def joint_likelihood(self,x):
        temp =0
        var1=0
        var2=0
        # var3 = log(P(y=class|X=instance))
        var3=0
        var4=0
        var5=[]
        self.p_yi_x = np.zeros(len(self.p_y))
        for i in range(len(self.p_y)):
            var2 = 0
            for j in range(self.n_features):
                x_nf = x[j]
                temp = np.square(x_nf - self.sum_mean[i,j]) / (2 * self.sum_var[j])
                self.p_xj_yi= (1/ np.sqrt(2* np.pi * np.sqrt(self.sum_var[j]))) * np.exp(-temp)
                # calculate the P(X=instance|y=class)
                var1 = np.log(self.p_xj_yi)
                #np.log(self.p_x_yi) = np.log(self.p_x_yi) + np.log(self.p_xj_yi)
            # calculate the P(y=class|X=instance) we can get the predict value
                var2 += var1
            var3 = np.log(self.p_y[i])
            var4 = var2 + var3
            var5.append(var4)
            
        #return np.argmax(self.p_yi_x)
        return np.argmax(var5)
    
  
    def fit(self, X, y):
        
        self.n_instances,self.n_features = np.shape(X)
        self.computeMeans(X,y)
        
        self.computeVariance(X,y)
        self.prior_class(y)
       
        return self
    
    def predict(self,X,y):
        y_pred = np.zeros(len(y))
        for i in range(self.n_instances):
            x_n = X.ix[i,:]
            y_pred[i] = self.joint_likelihood(x_n)
        
        return y_pred

if __name__=="__main__": 
    nb = GaussianNB()
    with open("output.txt","w") as f_out:
        df = pd.read_csv('/Users/administorzz/Downloads/digits.csv')
        X = df.ix[:,:-1]
        y = df.ix[:,-1]
        y_true = y
        
        nb.fit(X,y)
        y_pred = nb.predict(X,y)
        
        f_out.write("{0:.3f}".format(nb.accurateScore(y_true, y_pred)))
        f_out.write('\n')
        f_out.write(str(nb.confusion_matrix(y_true, y_pred)))
        
    
            
        
    
        
    
    