#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 15:46:10 2017

@author: administorzz
"""

import numpy as np
import pandas as pd
import math

class GaussianNB():
    
    
    
    def __init__(self):
        self.offset = 10e-5
               
    def accurateScore(self,y1,y2):
        score = y1==y2
        return np.average(score)
    
    def computeMeans(self, N, X, y):
# mean value of each feature per class(conditional)
        y_n = y
        self.sum_mean_0 = np.zeros(self.n_features)
        self.sum_mean_1 = np.zeros(self.n_features)
        self.sum_mean_2 = np.zeros(self.n_features)
        self.sum_mean_3 = np.zeros(self.n_features)
        self.sum_mean_4 = np.zeros(self.n_features)
        self.sum_mean_5 = np.zeros(self.n_features)
        self.sum_mean_6 = np.zeros(self.n_features)
        self.sum_mean_7 = np.zeros(self.n_features)
        self.sum_mean_8 = np.zeros(self.n_features)
        self.sum_mean_9 = np.zeros(self.n_features)
        
        for i in range(self.n_instances):
            if y_n[i] == 0:
                x_n = X[i,:]
                self.function_0(self,x_n)
            elif y_n[i] == 1:
                x_n = X[i,:]
                self.function_1(self,x_n)
            elif y_n[i] == 2:
                x_n = X[i,:]
                self.function_2(self,x_n)
            elif y_n[i] == 3:
                x_n = X[i,:]
                self.function_3(self,x_n)
            elif y_n[i] == 4:
                x_n = X[i,:]
                self.function_4(self,x_n)
            elif y_n[i] == 5:
                x_n = X[i,:]
                self.function_5(self,x_n)
            elif y_n[i] == 6:
                x_n = X[i,:]
                self.function_6(self,x_n)
            elif y_n[i] == 7:
                x_n = X[i,:]
                self.function_7(self,x_n)
            elif y_n[i] == 8:
                x_n = X[i,:]
                self.function_8(self,x_n)
            elif y_n[i] == 9:
                x_n = X[i,:]
                self.function_9(self,x_n)
                
    def function_0(self, x):
        for j in range(self.n_features):
            self.sum_mean_0[j] = self.sum_mean_0[j] + x[j]
        self.meansArray_0 = np.zeros(self.n_features)
        self.meansArray_0 = 1/self.n_instances * sum_mean_0
    
    def function_1(self, x):
        for j in range(self.n_features):
            self.sum_mean_1[j] = self.sum_mean_1[j] + x[j]
        self.meansArray_1 = np.zeros(self.n_features)
        self.meansArray_1 = 1/self.n_instances * sum_mean_1
    
    def function_2(self, x):
        for j in range(self.n_features):
            self.sum_mean_2[j] = self.sum_mean_2[j] + x[j]
        self.meansArray_2 = np.zeros(self.n_features)
        self.meansArray_2 = 1/self.n_instances * sum_mean_2
    
    def function_3(self, x):
        for j in range(self.n_features):
            self.sum_mean_3[j] = self.sum_mean_3[j] + x[j]
        self.meansArray_3 = np.zeros(self.n_features)
        self.meansArray_3 = 1/self.n_instances * sum_mean_3
    
    def function_4(self, x):
        for j in range(self.n_features):
            self.sum_mean_4[j] = self.sum_mean_4[j] + x[j]
        self.meansArray_4 = np.zeros(self.n_features)
        self.meansArray_4 = 1/self.n_instances * sum_mean_4
    
    def function_5(self, x):
        for j in range(self.n_features):
            self.sum_mean-6[j] = self.sum_mean_6[j] + x[j]
        self.meansArray_5 = np.zeros(self.n_features)
        self.meansArray_5 = 1/self.n_instances * sum_mean_5
    
    def function_6(self, x):
        for j in range(self.n_features):
            self.sum_mean_6[j] = self.sum_mean_6[j] + x[j]
        self.meansArray_6 = np.zeros(self.n_features)
        self.meansArray_6 = 1/self.n_instances * sum_mean_6
    
    def function_7(self, x):
        for j in range(self.n_features):
            self.sum_mean_7[j] = self.sum_mean_7[j] + x[j]
        self.meansArray_7 = np.zeros(self.n_features)
        self.meansArray_7 = 1/self.n_instances * sum_mean_7
    
    def function_8(self, x):
        for j in range(self.n_features):
            self.sum_mean_8[j] = self.sum_mean_8[j] + x[j]
        self.meansArray_8 = np.zeros(self.n_features)
        self.meansArray_8 = 1/self.n_instances * sum_mean_8
    
    def function_9(self, x):
        for j in range(self.n_features):
            self.sum_mean_9[j] = self.sum_mean_9[j] + x[j]
        self.meansArray_9 = np.zeros(self.n_features)
        self.meansArray_9 = 1/self.n_instances * sum_mean_9
        
        
        
    def computeVariance(self, N, X, y):        
    # variance value of each class plus offset
        y_n = y
        for i in range(self.n_instances):
            if y_n[i] == 0:
                x_n = X[i,:]
                for j in range(self.n_features):
                    sum_0[j] = sum_0[j] + math.pow((x_n[j] - self.meansArray_0[j]),2)
                
            elif y_n[i] == 1:
                x_n = X[i,:]
                for j in range(self.n_features):
                    sum_1[j] = sum_1[j] + math.pow((x_n[j] - self.meansArray_1[j]),2)
                    
            elif y_n[i] == 2:
                x_n = X[i,:]
                for j in range(self.n_features):
                    sum_2[j] = sum_2[j] + math.pow((x_n[j] - self.meansArray_2[j]),2)
                    
            elif y_n[i] == 3:
                x_n = X[i,:]
                for j in range(self.n_features):
                    sum_3[j] = sum_3[j] + math.pow((x_n[j] - self.meansArray_0[j]),2)
                    
            elif y_n[i] == 4:
                x_n = X[i,:]
                for j in range(self.n_features):
                    sum_4[j] = sum_4[j] + math.pow((x_n[j] - self.meansArray_4[j]),2)
                    
            elif y_n[i] == 5:
                x_n = X[i,:]
                for j in range(self.n_features):
                    sum_5[j] = sum_5[j] + math.pow((x_n[j] - self.meansArray_5[j]),2)
                    
            elif y_n[i] == 6:
                x_n = X[i,:]
                for j in range(self.n_features):
                    sum_6[j] = sum_6[j] + math.pow((x_n[j] - self.meansArray_6[j]),2)
                    
            elif y_n[i] == 7:
                x_n = X[i,:]
                for j in range(self.n_features):
                    sum_7[j] = sum_7[j] + math.pow((x_n[j] - self.meansArray_7[j]),2)
                    
            elif y_n[i] == 8:
                x_n = X[i,:]
                for j in range(self.n_features):
                    sum_8[j] = sum_8[j] + math.pow((x_n[j] - self.meansArray_8[j]),2)
                    
            elif y_n[i] == 9:
                x_n = X[i,:]
                for j in range(self.n_features):
                    sum_9[j] = sum_9[j] + math.pow((x_n[j] - self.meansArray_9[j]),2)
        
        self.Var_0 = 1/(self.n_instances - 1) * sum_0
        self.Var_1 = 1/(self.n_instances - 1) * sum_1
        self.Var_2 = 1/(self.n_instances - 1) * sum_2
        self.Var_3 = 1/(self.n_instances - 1) * sum_3
        self.Var_4 = 1/(self.n_instances - 1) * sum_4
        self.Var_5 = 1/(self.n_instances - 1) * sum_5
        self.Var_6 = 1/(self.n_instances - 1) * sum_6
        self.Var_7 = 1/(self.n_instances - 1) * sum_7
        self.Var_8 = 1/(self.n_instances - 1) * sum_8
        self.Var_9 = 1/(self.n_instances - 1) * sum_9
        
        self.varc = np.array([self.Var_0,self.Var_1,self.Var_2,self.Var_3,self.Var_4,
                                  self.Var_5,self.Var_6,self.Var_7,self.Var_8,self.Var_9])
                       
                       
    
    def prior_class(self,y):
        y_n = y
        count_0 = None
        count_1 = None
        count_2 = None
        count_3 = None
        count_4 = None
        count_5 = None
        count_6 = None
        count_7 = None
        count_8 = None
        count_9 = None
        for t in range(self.n_instances):
            if y_n[t]==0:
                count_0 = count_0 +1
            elif y_n[t]==1:
                count_1 = count_1 +1
            elif y_n[t]==2:
                count_1 = count_2 +1
            elif y_n[t]==3:
                count_1 = count_3 +1
            elif y_n[t]==4:
                count_1 = count_4 +1
            elif y_n[t]==5:
                count_1 = count_5 +1
            elif y_n[t]==6:
                count_1 = count_6 +1
            elif y_n[t]==7:
                count_1 = count_7 +1
            elif y_n[t]==8:
                count_1 = count_8 +1
            elif y_n[t]==8:
                count_1 = count_9 +1
                
        self.p_y0 = count_0 / self.n_instances
        self.p_y1 = count_1 / self.n_instances
        self.p_y2 = count_2 / self.n_instances
        self.p_y3 = count_3 / self.n_instances
        self.p_y4 = count_4 / self.n_instances
        self.p_y5 = count_5 / self.n_instances
        self.p_y6 = count_6 / self.n_instances
        self.p_y7 = count_7 / self.n_instances
        self.p_y8 = count_8 / self.n_instances
        self.p_y9 = count_9 / self.n_instances
        self.p_y = np.array([self.p_y0,self.p_y1,self.p_y2,self.p_y3,self.p_y4,
                             self.p_y5,self.p_y6,self.p_y7,self.p_y8,self.p_y9])
            
    def joint_likelihood(self,X,y):
        temp =None
        for i in range(self.n_instances):
            x_n = X[i,:]
            y_n = y[i]
            for j in range(self.n_features):
                x_nf = x_n[j]
                temp = math.pow((x_nf - meansArray[y_n][j]),2)/2*math.pow((self.varc[y_n]),2)
                p_xj_yn= (1/ np.sqrt(2*pi*self.varc[y_n]))* math.exp(-temp)
                
            p_x_yn =  np.log(p_xj_yn) + np.log(self.p_y[y_n])
                
            
                   
            
            
        
                 
                
        
    
    def fit(self, X, y):
        
        self.n_instances,self.n_features = np.shape(X)
        self.computeMeans(self.n_instances, X,y)
        self.meansArray = np.array([self.meansArray_0,self.meansArray_1,self.meansArray_2,self.meansArray_3,self.meansArray_4,
                                    self.meansArray_5,self.meansArray_6,self.meansArray_7,self.meansArray_8,self.meansArray_9])
        self.computeVariance(self.n_instances, X,y)
        self.prior_class(self, y)
        return self
    
    