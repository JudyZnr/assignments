#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 22:21:40 2017

@author: administorzz
"""

import pandas as pd
import numpy as np


np.random.seed(seed = 0)

class Perceptron:

    def _init_(self):
        self.w_vector=None

    def _updateWeight( self, x , y ):

        self.w_vector = self.w_vector + y*x
      

    def _decision(self, x):

        return np.sign(np.dot(self.w_vector , x))

    
    def _accurateScore(self,y1,y2):
         score= y1==y2
         self.acc = np.average(score)

    def _fit (self,X,y):
        
        n_instances, n_features = np.shape(X)
        bias = np.negative(np.ones(shape=(n_instances ,1)))
        X = np.hstack((X,bias))
        self.w_vector = np.random.rand(n_features + 1) 
  
        i=0
#define a list to save the accuracy of each test
        self.accVector = []
        while True:
     
             for i in range(n_instances):
                 x=X[i,:]
                 self.y_true = y[i]
                 self.y_predVector = np.zeros(n_instances)
                 for j in range(n_instances):
                     self.y_predVector[j] = self._decision(X[j,:])
# build predVector to store predict values, using the preditc value calculated in the function decision
                 if self._decision(x) != self.y_true:
                     self._updateWeight(x,self.y_true)
# using true vector and predic vector to calculate accuracy 
                 self.y_trueVector = y
                 self._accurateScore(self.y_trueVector,self.y_predVector)
                 #print(self.acc)
                 self.accVector.append(self.acc)
                 #if i didn't use self(pointer), then I have to define the acc and return the acc value
                 if self.acc==1.0:
                     return self        
        return self  



#main function in python
if __name__ == "__main__":
   
    p=Perceptron()
    indexVector =[]
    index = 0
    with open('output.txt','w') as f_out:
 #f_out is a varibale to represent the output file, for later reference   
        df=pd.read_csv('/Users/administorzz/Dropbox/digits_binary.csv')
        X=df.ix[0:,:64]
        y=df.ix[0:,64]
        p._fit(X,y)
#initialize a list to save the indexer         
        
        m=0
        while m < len(p.accVector):
                if p.accVector[m]!=p.accVector[m-1] :
#modify the format of output data  
                    f_out.write(str(index)+' '+"{0:.3f}".format(p.accVector[m])+'\n')
                    m+=1
#output of misclassification with index                    
                    index = index +1
                else:
                    m+=1
#f_out.close()
            
        
    
