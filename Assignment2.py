#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 20:38:35 2017

@author: administorzz
"""

import pandas as pd
import numpy as np
np.random.seed(seed=0)

class Logic_Regression(object):
    
    def __init__(self):
        self.weight = None
        self.max_iteration = 500
        self.learning_rate = 0.01
        
    def _accScore(self, y1, y2):
        score= y1==y2
        self.acc = np.average(score)
        
        
    def _gradDescent(self , weight ,X , y):#parameter should be X instead of x_n
        
        self.grad = np.zeros(self.n_features)
    
        sum_1=0


        for r in range(self.n_instances):
            x_n=X[r,:]
            y_n=y[r]
            sum_1=sum_1+((y_n * x_n)/(1+np.exp(y_n * self.weight.T * x_n)))
            #sum_2=sum_2+np.log(1+np.exp(y_n*self.weight.T*x_n))
            
            self.grad=-1/self.n_instances * sum_1
        return self.grad             
        
    
    def _updateWeight(self , grad):
        
            self.weight =  self.weight-self.learning_rate * grad
        
    def _decision(self , weight , X):
        self.y_predVector = np.zeros(self.n_instances)
        for i in range(self.n_instances):
            x_n = X[i,:]
            self.h = np.sign(self.sigmoid(np.dot(weight , x_n))-0.5)
            self.y_predVector[i]=self.h
        return self.y_predVector
        
    def sigmoid(self,x):
        sig_value = 1. /( 1. + np.exp(-x))
        return sig_value
        
    def _fit(self, X , y):
        self.n_instances , self.n_features = np.shape(X)
        self.accVector = np.zeros(500)
        self.y_predVector = np.zeros(self.n_instances)
        self.y_trueVector = y
        
        bias = np.ones(shape=(self.n_instances ,1))
        X = np.hstack((X,bias))
        self.weight = np.random.rand(self.n_features + 1) 
        
        for j in range(self.max_iteration):
                          
                self._updateWeight(self._gradDescent(self.weight , X , y))
                self._decision(self.weight,X)
                self._accScore(self.y_trueVector,self.y_predVector)
                self.accVector[j]=self.acc
                    
            
if __name__ == "__main__":
    lr =Logic_Regression()
    index=0
    
    with open("output.txt","w") as f_out:
        df=pd.read_csv("/Users/administorzz/Downloads/digits_binary.csv")
        X = df.ix[:, :-1]
        y = df.ix[:, -1]
        lr._fit(X,y)
        m=0
        while m < len(lr.accVector):
                
#modify the format of output data  
                    f_out.write(str(index)+' '+"{0:.3f}".format(lr.accVector[m])+'\n')
                    m+=1
#output of misclassification with index                    
                    index = index +1
                
        
        
            
        
        
