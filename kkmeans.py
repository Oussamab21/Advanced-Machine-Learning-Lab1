#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 15:17:14 2017

@author: Oussama 
"""
from scipy.spatial.distance import pdist, squareform
from scipy import exp
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons
from sklearn.datasets import make_circles
from sklearn.datasets import make_classification
from sklearn.datasets.samples_generator import make_swiss_roll
from sklearn import cluster
#import sklearn.cross_validation 
from sklearn import cross_validation
from sklearn import metrics 
import time

import os
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

   

    return K
#############################################
    #linair kenrnal


def linear_kernel(X,n_components):
    K = X.dot(X.T)
   
    return K
#########################################
    #polynomial

def poly_kernel(X,n_components,P):
    #K = X.dot(X.T)## change
    A=1
    B=1
    K=((X.dot(X.T))+1)**P
    #K=(A<X, X> + B)**P
   
    return K

############################################################"
    #laplacien

def laplacian_kernel(X,gamma,n_components):
    
    #sq_dists = pdist(X, 'sqeuclidean')
    ma_dists=pdist(X,'cityblock')
    # Converting the pairwise distances into a symmetric MxM matrix.
    mat_ma_dists = squareform(ma_dists)

    # Computing the MxM kernel matrix.
    K = exp(-gamma * mat_ma_dists)
    
 
    return K

###############################kmeans plots

def kmeans_plots(X,y_kmeans,centroids):
    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

      
    plt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=200, alpha=0.5);
    plt.show()


##################################
    
    
data_benchmarks = [make_moons,make_circles,make_classification]


for data_set in data_benchmarks:
  print("the data set is ",data_set)
  X,y=data_set(n_samples=2500, random_state=613)  
  start=time.time()
   
  kmeans = cluster.KMeans(n_clusters=2)
  kmeans.fit(X)
  centroids1 = kmeans.cluster_centers_
  
  y_kmeans = kmeans.predict(X)
  print("plot the non kernel kmeans kmeans")
  kmeans_plots(X,y_kmeans,centroids1)
  end=time.time()
  print("the time spent on the current task is ",end - start)
  pause()
#############################apply Linear kernal ##################################"
  print("linear kernel")
  start=time.time()
  X_linear=linear_kernel(X,n_components=2)
   
  kmeans = cluster.KMeans(n_clusters=2)
  kmeans.fit(X_linear)
  centroids2 = kmeans.cluster_centers_
  
  y_kmeans = kmeans.predict(X_linear)
  kmeans_plots(X_linear,y_kmeans,centroids2)
  end=time.time()
  print("the time spent on the current task is ",end - start)
  pause()
############################apply the poly kernal#####################################
  for i in (2,3,4,5):
    start=time.time()
     
    print("poly kernal P is ",i)  
    X_poly=poly_kernel(X,n_components=2,P=i)
     
    kmeans = cluster.KMeans(n_clusters=2)
    kmeans.fit(X_poly)
    centroids3 = kmeans.cluster_centers_
    
    y_kmeans = kmeans.predict(X_poly)
    kmeans_plots(X_poly,y_kmeans,centroids3)
    end=time.time()
    print("the time spent on the current task is ",end - start)
    pause()
  
########################################################################
#####apply the rbf kernal

  print("RBF")
  for i in (1,2,3):
    print("rbf kernel with gamma set to ",i)  
    start=time.time()  
    X_rbf=rbf_kernel(X,gamma=i,n_components=5)
     
    kmeans = cluster.KMeans(n_clusters=2)
    kmeans.fit(X_rbf)
    centroids4 = kmeans.cluster_centers_
   
    y_kmeans = kmeans.predict(X_rbf)
    kmeans_plots(X_rbf,y_kmeans,centroids4)
    end=time.time()
    print("the time spent on the current task is ",end - start)
    pause()

  
  