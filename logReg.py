#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 23:17:01 2017

@author: dell1
"""

from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
import numpy as np

import time
from sklearn.cross_validation import cross_val_score
import os

from pandas import Series, DataFrame
from pylab import rcParams
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics 
from sklearn.metrics import classification_report

from sklearn.datasets import load_iris
from sklearn.datasets import make_classification
from sklearn.datasets import make_moons
from sklearn.datasets import make_circles
from sklearn.datasets import make_swiss_roll

#import system

def pause():
    programPause = input("Press the <ENTER> key to continue...")




def rbf_kernel(X, gamma, n_components):
    
    # Calculating the squared Euclidean distances for every pair of points
    # in the MxN dimensional dataset.
    sq_dists = pdist(X, 'sqeuclidean')

    # Converting the pairwise distances into a symmetric MxM matrix.
    mat_sq_dists = squareform(sq_dists)

    # Computing the MxM kernel matrix.
    K = exp(-gamma * mat_sq_dists)

    return(K)


def linear_kernel(X,n_components):
    K = X.dot(X.T)
   

    return(K)


def poly_kernel(X,n_components,P):
  
    A=1
    B=1
    K=((X.dot(X.T))+1)**P
    

    return(K)
    
    
def laplacian_kernel(X,gamma,n_components):
    
    #sq_dists = pdist(X, 'sqeuclidean')
    ma_dists=pdist(X,'cityblock')
    # Converting the pairwise distances into a symmetric MxM matrix.
    mat_ma_dists = squareform(ma_dists)

    # Computing the MxM kernel matrix.
    K = exp(-gamma * mat_ma_dists)    
    
    return(K)    
    
    
def logreg(X,y):

                        
           ############normal logistic regression##############
    
  print("basic logistic regression without kernal")
  start=time.time()
  score=cross_val_score(LogisticRegression(),X,y,scoring='accuracy',cv=5)
  score=score.mean()
  print('score is ',score)
  end=time.time()
  print("time spent on the current task is ",end - start)
  pause()
  
            ##################linear#####################
            
  print("linear kernal")
  start=time.time()
  X_linear=linear_kernel(X,n_components=3)
  score=cross_val_score(LogisticRegression(),X_linear,y,scoring='accuracy',cv=5)
  score=score.mean()
  print('score is ',score)
  print()
  end=time.time()
  print("time spent on the current task is ",end - start)
  pause()

           ####################poly#######################""

  for i in (2,3,4,5,10):
    print('polynomial kernal with degree ',i)  
    start=time.time()
    X_poly=poly_kernel(X,n_components=3,P=i)
    score=cross_val_score(LogisticRegression(),X_poly,y,scoring='accuracy',cv=5)
    score=score.mean()
    print('score is ',score)
    end=time.time()
    print("time spent on the current  task is", end - start)
    pause()

               #############RBF##########################

  for i in (1,3,5,10,15,20):
    start=time.time()  
    print("rbf kernal with gamma = ",i)
    X_rbf=rbf_kernel(X,gamma=i, n_components=3)
    score=cross_val_score(LogisticRegression(),X_rbf,y,scoring='accuracy',cv=5)
    score=score.mean()
    print('score is ',score)
    end=time.time()
    print("the time spent on the current task is ",end - start)  
    pause()  
    
         ###################laplacian######################
         
  for i in (1,3,5,10,15,20):
    start=time.time()  
    print("laplacian kernal with gamma = ",i)
    X_laplacian=laplacian_kernel(X,gamma=i, n_components=3)
    score=cross_val_score(LogisticRegression(),X_laplacian,y,scoring='accuracy',cv=5)
    score=score.mean()
    print('score is ',score)
    end=time.time()
    print("the time spent on the current task is ",end - start)  
    pause()    
    
#####################################################################"    
data=[make_moons,make_circles,make_classification]
for i in data:
 print("data set is ",i)   
 X,y=i(n_samples=2000,random_state=123)
 logreg(X,y)
     