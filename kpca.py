#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 11:34:00 2017

@author: Oussama
"""

from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
import numpy as np
from sklearn.preprocessing import StandardScaler
#import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

import time

import os
#import system

def pause():
    programPause = input("Press the <ENTER> key to continue...")





def basic_pca(X,n_components):
    cov_mat = np.cov(X)

    eig_vals, eig_vecs = eigh(cov_mat)
    
    #Keep eienvectors according to number of components with highest eigenvalues
    X_pca = np.column_stack((eig_vecs[:,-comp] for comp in range(1,
                              n_components+1)))
    return (X_pca)


###############################
#RBF kernal

def rbf_kernel(X, gamma, n_components):
    
    # Calculating the squared Euclidean distances for every pair of points
    # in the MxN dimensional dataset.
    sqdists = pdist(X, 'sqeuclidean')

    # Converting the pairwise distances into a symmetric MxM matrix.
    mat_sqdists = squareform(sqdists)

    # Computing the MxM kernel matrix.
    K = exp(-gamma * mat_sqdists)

    # Centering the symmetric NxN kernel matrix.
    N = K.shape[0]
    one_n = np.ones((N,N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

    # Obtaining eigenvalues in descending order with corresponding
    # eigenvectors from the symmetric matrix.
    eigenvals, eigenvecs = eigh(K)

    # Obtaining the i eigenvectors that corresponds to the i highest eigenvalues.
    X_pc = np.column_stack((eigenvecs[:,-i] for i in range(1,n_components+1)))

    return X_pc
#############################################
    #linear kenrnel


def linear_kernel(X,n_components):
    K = X.dot(X.T)
    N = K.shape[0]
    one_n = np.ones((N,N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

    # Obtaining eigenvalues in descending order with corresponding
    # eigenvectors from the symmetric matrix.
    eigenvals, eigenvecs = eigh(K)

    # Obtaining the i eigenvectors that corresponds to the i highest eigenvalues.
    X_pc = np.column_stack((eigenvecs[:,-i] for i in range(1,n_components+1)))
 
    return X_pc
#########################################
    #polynomial kernel

def poly_kernel(X,n_components,P):
    #K = X.dot(X.T)## change
    A=1
    B=1
    K=((X.dot(X.T))+1)**P
    #K=(A<X, X> + B)**P
    N = K.shape[0]
    one_n = np.ones((N,N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

    # Obtaining eigenvalues in descending order with corresponding
    # eigenvectors from the symmetric matrix.
    eigenvals, eigenvecs = eigh(K)

    # Obtaining the i eigenvectors that corresponds to the i highest eigenvalues.
    X_pc = np.column_stack((eigenvecs[:,-i] for i in range(1,n_components+1)))
 
    return X_pc

############################################################"
    #laplacien kernel

def laplacian_kernel(X,gamma,n_components):
    
    
    manhatten_dists=pdist(X,'cityblock')
    # Converting the pairwise distances into a symmetric MxM matrix.
    mat_manhatten_dists = squareform(manhatten_dists)

    # Computing the MxM kernel matrix.
    K = exp(-gamma * mat_manhatten_dists)
    
    N = K.shape[0]
    one_n = np.ones((N,N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

    # Obtaining eigenvalues in descending order with corresponding
    # eigenvectors from the symmetric matrix.
    eigenvals, eigenvecs = eigh(K)

    # Obtaining the i eigenvectors that corresponds to the i highest eigenvalues.
    X_pc = np.column_stack((eigenvecs[:,-i] for i in range(1,n_components+1)))
 
    return X_pc

#######################################################################""
 ##study the makemoons data set 
 
def comparaison_makemoons(X,y):
   plt.figure(figsize=(8,6))

   plt.scatter(X[y==0, 0], X[y==0, 1], color='red', alpha=0.5)
   plt.scatter(X[y==1, 0], X[y==1, 1], color='blue', alpha=0.5)
   plt.title('A nonlinear 2Ddataset')
   plt.ylabel('y coordinate')
   plt.xlabel('x coordinate')
   #name = 'moons' + ' original dataset'
   
   #filename='/home/dell1/Desktop/mlcourse/'+name+'.jpeg'
   
   #plt.savefig(filename,format='jpeg')     
   plt.show()
   #plt.close()
   pause()
                    #################################
   from sklearn.decomposition import PCA
   start=time.time()
   pca = PCA(n_components=2)
   X_spca = pca.fit_transform(X)
   
   plt.figure(figsize=(8,6))
   plt.scatter(X_spca[y==0, 0], X_spca[y==0, 1], color='red', alpha=0.5)
   plt.scatter(X_spca[y==1, 0], X_spca[y==1, 1], color='blue', alpha=0.5)

   plt.title('First 2 principal components after Linear PCA')
   plt.xlabel('PC1')
   plt.ylabel('PC2')
   #name ='moons' + ' 2PCA'
   #filename = '/home/dell1/Desktop/mlcourse/' + name+'.jpeg'
   #plt.savefig(filename,format='jpeg')     
   plt.show()
   #plt.close()
   

###########################################################################
   import numpy as np
    
   pca = PCA(n_components=1)
   X_spca = pca.fit_transform(X)
   
   plt.figure(figsize=(8,6))
   plt.scatter(X_spca[y==0, 0], np.zeros((50,1)), color='red', alpha=0.5)
   plt.scatter(X_spca[y==1, 0], np.zeros((50,1)), color='blue', alpha=0.5)
   #name ='moons'+ ' 1PCA'
   #filename = '/home/dell1/Desktop/mlcourse/' + name+'.jpeg'
   plt.title('First principal component after Linear PCA')
   plt.xlabel('PC1')
   #plt.savefig(filename,format='jpeg')     
   plt.show()
   #plt.close()
   end=time.time()
   print("the time spent on the current task is ",end - start)
   pause()

   ##############################################################
                        ## rbf kernel application
                        
   for i in (1,5,10,15,25):
     start=time.time()  
     X_pc = rbf_kernel(X, gamma=i, n_components=2)
     plt.figure(figsize=(8,6))
     plt.scatter(X_pc[y==0, 0], X_pc[y==0, 1], color='red', alpha=0.5)
     plt.scatter(X_pc[y==1, 0], X_pc[y==1, 1], color='blue', alpha=0.5)
     plt.title('First 2 principal components after RBF Kernel PCA with gamma set to %s'%i)
     #name ='moons'+ ' RBF_2PCA gamma'+str(i)
     #filename = '/home/dell1/Desktop/mlcourse/' + name+'.jpeg'
     #plt.savefig(filename,format='jpeg')  
     plt.xlabel('PC1')
     plt.ylabel('PC2')
     plt.show()
     #plt.close()
 

     plt.figure(figsize=(8,6))
     plt.scatter(X_pc[y==0, 0], np.zeros((50)), color='red', alpha=0.5)
     plt.scatter(X_pc[y==1, 0], np.zeros((50)), color='blue', alpha=0.5)

     plt.title('First principal component after RBF Kernel PCA with gamma set to %s'%i)    
     plt.xlabel('PC1')
     #name ='moons'+ ' RBF_1PCA gamma'+str(i)
     #filename = '/home/dell1/Desktop/mlcourse/' + name+'.jpeg'
     #plt.savefig(filename,format='jpeg')  
     plt.xlabel('PC1')
     plt.ylabel('PC2')
     plt.show()
     #plt.close()
     end=time.time()
     print("the time spent on the current task is ",end - start)
     pause()

########################################################################
     ### linear kernel application 
     
   X_pc = linear_kernel(X,n_components=2)
   start=time.time()
   plt.figure(figsize=(8,6))
   plt.scatter(X_pc[y==0, 0], X_pc[y==0, 1], color='red', alpha=0.5)
   plt.scatter(X_pc[y==1, 0], X_pc[y==1, 1], color='blue', alpha=0.5)

   plt.title('First 2 principal components after linear Kernel PCA') 
   plt.xlabel('PC1')
   plt.ylabel('PC2')
   
   #name ='moons'+ 'linear_2pca'
   #filename = '/home/dell1/Desktop/mlcourse/' + name+'.jpeg'
   #plt.savefig(filename,format='jpeg')
   plt.show()
   #plt.close()


   plt.figure(figsize=(8,6))
   plt.scatter(X_pc[y==0, 0], np.zeros((50)), color='red', alpha=0.5)
   plt.scatter(X_pc[y==1, 0], np.zeros((50)), color='blue', alpha=0.5)

   plt.title('First principal component after linear Kernel PCA')
   plt.xlabel('PC1')
  
   #name ='moons'+ ' linear_1pca '
   #filename = '/home/dell1/Desktop/mlcourse/' + name+'.jpeg'
   #plt.savefig(filename,format='jpeg')
   plt.show()
   #plt.close()
   end=time.time()
   
   print("time spent on the current task is ",end - start)
   pause()

################################################################
   ###poly kernel application 
   
   for i in (2,3,4,5):
     start=time.time()  
     X_pc = poly_kernel(X,n_components=2,P=i)

     plt.figure(figsize=(8,6))
     plt.scatter(X_pc[y==0, 0], X_pc[y==0, 1], color='red', alpha=0.5)
     plt.scatter(X_pc[y==1, 0], X_pc[y==1, 1], color='blue', alpha=0.5)

     plt.title('First 2 principal components after polynomial Kernel PCA with the degree set to %s'%i)
     
     plt.xlabel('PC1')
     plt.ylabel('PC2')
     
    # name ='moons'+ ' poly_2pca degree'+str(i)
    # filename = '/home/dell1/Desktop/mlcourse/' + name+'.jpeg'
    # plt.savefig(filename,format='jpeg')
     plt.show()
    # plt.close()

     plt.figure(figsize=(8,6))
     plt.scatter(X_pc[y==0, 0], np.zeros((50)), color='red', alpha=0.5)
     plt.scatter(X_pc[y==1, 0], np.zeros((50)), color='blue', alpha=0.5)

     plt.title('First principal component after polynimial  Kernel PCA with the degree set to %s'%i)

     plt.xlabel('PC1')
     
     #name ='moons'+ ' poly_1pca degree'+str(i)
     #filename = '/home/dell1/Desktop/mlcourse/' + name+'.jpeg'
     #plt.savefig(filename,format='jpeg')
     plt.show()
     #plt.close()
     end=time.time()
     print("time spent on the current task is ",end - start)
     pause()


########################################################################
     ### laplacian kernel application 
     
   for i in (1,5,10,15,25): 
       start=time.time()  
       X_pc = laplacian_kernel(X, gamma=i, n_components=2)
       plt.figure(figsize=(8,6))
       plt.scatter(X_pc[y==0, 0], X_pc[y==0, 1], color='red', alpha=0.5)
       plt.scatter(X_pc[y==1, 0], X_pc[y==1, 1], color='blue', alpha=0.5)

       plt.title('First 2 principal components after laplacian Kernel PCA with gamma set to %s'%i)
       plt.xlabel('PC1')
       plt.ylabel('PC2')
      
       #name ='moons'+ ' laplacian_2pca_gamma '+str(i)
       #filename = '/home/dell1/Desktop/mlcourse/' + name+'.jpeg'
       #plt.savefig(filename,format='jpeg')
       plt.show()
       #plt.close()

       plt.figure(figsize=(8,6))
       plt.scatter(X_pc[y==0, 0], np.zeros((50)), color='red', alpha=0.5)
       plt.scatter(X_pc[y==1, 0], np.zeros((50)), color='blue', alpha=0.5)

       plt.title('First principal component after laplacian Kernel PCA with gamma set to %s'%i)
       
       plt.xlabel('PC1')
       
       #name ='moons'+ ' laplacian_1pca_gamma '+str(i)
       #filename = '/home/dell1/Desktop/mlcourse/' + name+'.jpeg'
       #plt.savefig(filename,format='jpeg')
       plt.show()
       #plt.close()
       end=time.time()
       print("time spent on the current task is ",end - start)
       pause()
           
   return()
#######################################################################
   ###study the makecircles data set
   
def comparaison_makecircles(X,y):
   import numpy as np 
   plt.figure(figsize=(8,6))

   plt.scatter(X[y==0, 0], X[y==0, 1], color='red', alpha=0.5)
   plt.scatter(X[y==1, 0], X[y==1, 1], color='blue', alpha=0.5)
   plt.title('Concentric circles')
   plt.ylabel('y coordinate')
   plt.xlabel('x coordinate')
   #name ='circles'+ 'dataset'
   #filename = '/home/dell1/Desktop/mlcourse/' + name+'.jpeg'
   #plt.savefig(filename,format='jpeg')
   plt.show()
   #plt.close()

#######################################################################"
   from sklearn.decomposition import PCA

   
   import numpy as np
   start=time.time() 
   pca = PCA(n_components=2)
   X_spca = pca.fit_transform(X)

   plt.figure(figsize=(8,6))
   plt.scatter(X_spca[y==0, 0], np.zeros((500,1))+0.1, color='red', alpha=0.5)
   plt.scatter(X_spca[y==1, 0], np.zeros((500,1))-0.1, color='blue', alpha=0.5)
   plt.ylim([-15,15])
   
   plt.title('First principal component after Linear PCA')
   plt.xlabel('PC1')
   #name ='circles'+ '1pca'
   #filename = '/home/dell1/Desktop/mlcourse/' + name+'.jpeg'
   #plt.savefig(filename,format='jpeg')
   plt.show()
   #plt.close()
   end=time.time()
   print("the time spent on the current task is ",end - start)
   pause()
#########################################################################
      ## rbf kernel application 
      
   for i in (1,5,10,15,25):
     start=time.time()  
     X_pc = rbf_kernel(X, gamma=i, n_components=1)
     plt.figure(figsize=(8,6))  
     plt.scatter(X_pc[y==0, 0], np.zeros((500,1)), color='red', alpha=0.5)
     plt.scatter(X_pc[y==1, 0], np.zeros((500,1)), color='blue', alpha=0.5)
     plt.title('First principal component after RBF Kernel PCA with gamma set to %s'%i)    
     plt.xlabel('PC1')
     #name ='circles'+ 'rbf_1pca_gamma'+str(i)
     #filename = '/home/dell1/Desktop/mlcourse/' + name+'.jpeg'
     #plt.savefig(filename,format='jpeg')
     plt.show()
     #plt.close()
     end=time.time()
     print("the time spent on the current task is ",end - start)
     pause()

########################################################################
     ## linear kernel application 
     
   X_pc = linear_kernel(X,n_components=1)
   start=time.time()  
   plt.figure(figsize=(8,6))
   plt.scatter(X_pc[y==0, 0], np.zeros((500,1)), color='red', alpha=0.5)
   plt.scatter(X_pc[y==1, 0], np.zeros((500,1)), color='blue', alpha=0.5)
   plt.title('First principal component after linear Kernel PCA')
   plt.xlabel('PC1')
   #name ='circles'+ ' linear_1pca'
   #filename = '/home/dell1/Desktop/mlcourse/' + name+'.jpeg'
   #plt.savefig(filename,format='jpeg')
   plt.show()
   #plt.close()
   end=time.time()
   print("time spent on the current task is ",end - start)
   pause()

################################################################
   ###poly kernel application 
   
   for i in (2,3,4,5):
     start=time.time()  
     X_pc = poly_kernel(X,n_components=1,P=i)
     plt.figure(figsize=(8,6))
     plt.scatter(X_pc[y==0, 0], np.zeros((500,1)), color='red', alpha=0.5)
     plt.scatter(X_pc[y==1, 0], np.zeros((500,1)), color='blue', alpha=0.5)
     plt.title('First principal component after polynimial  Kernel PCA with the degree set to %s'%i)
     plt.xlabel('PC1')
    # name ='circles'+ ' poly_1pca_degree'+str(i)
    # filename = '/home/dell1/Desktop/mlcourse/' + name+'.jpeg'
    # plt.savefig(filename,format='jpeg')
     plt.show()
    # plt.close()
     end=time.time()
     print("time spent on the current task is ",end - start)
     pause()


########################################################################
     ###laplacian kernel application
     
   for i in (1,5,10,15,25): 
       start=time.time()  
       X_pc = laplacian_kernel(X, gamma=i, n_components=1)
       plt.figure(figsize=(8,6))
       plt.scatter(X_pc[y==0, 0], np.zeros((500,1)), color='red', alpha=0.5)
       plt.scatter(X_pc[y==1, 0], np.zeros((500,1)), color='blue', alpha=0.5)
       plt.title('First principal component after laplacian Kernel PCA with gamma set to %s'%i)
       plt.xlabel('PC1')
       #name ='circles'+ 'laplacian_1pca gammma'+str(i)
       #filename = '/home/dell1/Desktop/mlcourse/' + name+'.jpeg'
       #plt.savefig(filename,format='jpeg')
       plt.show()
       #plt.close()
       end=time.time()
       print("time spent on the current task is ",end - start)
       pause()
       
       
       
   return()   
######################################################################"   
###############study the swissroll
   
def comparaison_swissroll(X,y):
   fig = plt.figure(figsize=(7,7))
   ax = fig.add_subplot(111, projection='3d')
   ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.rainbow)
   plt.title('Swiss Roll in 3D')
   #name ='swissroll'+ 'dataset'
   #filename = '/home/dell1/Desktop/mlcourse/' + name+'.jpeg'
   #plt.savefig(filename,format='jpeg')
   plt.show()
   #plt.close()
   pause()
                 #############################normal pca
   start=time.time()               
   pca = PCA(n_components=2)
   X_spca = pca.fit_transform(X)

   plt.figure(figsize=(8,6))
   plt.scatter(X_spca[:, 0], X_spca[:, 1], c=y, cmap=plt.cm.rainbow)

   plt.title('First 2 principal components after Linear PCA')
   plt.xlabel('PC1')
   plt.ylabel('PC2')
   #name ='swissroll'+ '2PCA'
   #filename = '/home/dell1/Desktop/mlcourse/' + name+'.jpeg'
   #plt.savefig(filename,format='jpeg')
   plt.show()
   #plt.close()
               
                            
   import numpy as np
   pca = PCA(n_components=1)
   X_spca = pca.fit_transform(X)
   
   plt.figure(figsize=(8,6))
   plt.scatter(X_spca, np.zeros((800,1)), c=y, cmap=plt.cm.rainbow)
   plt.title('First principal component after Linear PCA')
   plt.xlabel('PC1')
   #name ='swissroll'+ '1PCA'
   #filename = '/home/dell1/Desktop/mlcourse/' + name+'.jpeg'
   #plt.savefig(filename,format='jpeg')
   plt.show()
   #plt.close()
   end=time.time()
   print("the time spent on the current task is ",end - start)
   pause()

###############################rbf kernel application######
   for i in (0.1,0.5,1,2):
     start=time.time()  
     X_pc = rbf_kernel(X, gamma=i, n_components=2)
     plt.figure(figsize=(8,6))
     plt.scatter(X_pc[:, 0], X_pc[:, 1], c=y, cmap=plt.cm.rainbow)

     plt.title('First 2 principal components after RBF Kernel PCA with gamma set to %s'%i)
   
     plt.xlabel('PC1')
     plt.ylabel('PC2')
    # name ='swissroll'+ '2PCA_rbf gamma'+str(i)
    # filename = '/home/dell1/Desktop/mlcourse/' + name+'.jpeg'
    # plt.savefig(filename,format='jpeg')
     plt.show()
    # plt.close()
    
                            
     plt.figure(figsize=(8,6))
     plt.scatter(X_pc[:,0], np.zeros((800,1)), c=y, cmap=plt.cm.rainbow)
 
     plt.title('First principal component after RBF Kernel PCA with gamma set to %s'%i)
     plt.xlabel('PC1')
     #name ='swissroll'+ '1PCA_rbf_gamma'+str(i)
     #filename = '/home/dell1/Desktop/mlcourse/' + name+'.jpeg'
     #plt.savefig(filename,format='jpeg')
     plt.show()
     #plt.close()
    
     end=time.time()
     print("the time spent on the current task is ",end - start)
     pause()

 #######################linear kernel application## 
   X_pc = linear_kernel(X,n_components=2)
   start=time.time()
   plt.figure(figsize=(8,6))
   plt.scatter(X_pc[:, 0], X_pc[:, 1], c=y, cmap=plt.cm.rainbow)
   plt.title('First 2 principal components after linear Kernel PCA')  
   plt.xlabel('PC1')
   plt.ylabel('PC2')
   #name ='swissroll'+ 'linear_2PCA'
   #filename = '/home/dell1/Desktop/mlcourse/' + name+'.jpeg'
   #plt.savefig(filename,format='jpeg')
   plt.show()
   #plt.close()
   
                              
   plt.figure(figsize=(8,6))
   plt.scatter(X_pc[:,0], np.zeros((800,1)), c=y, cmap=plt.cm.rainbow)
   plt.title('First principal component after linear Kernel PCA')
   plt.xlabel('PC1')
   #name ='swissroll'+ 'linear_2PCA'
   #filename = '/home/dell1/Desktop/mlcourse/' + name+'.jpeg'
   #plt.savefig(filename,format='jpeg')
   plt.show()
   #plt.close()
   end=time.time()
   print("time spent on the current task is ",end - start)
   pause()

################################################################
   ###poly kernel application 
   
   for i in (2,3,4,5):
     start=time.time()  
     X_pc = poly_kernel(X,n_components=2,P=i)
     plt.figure(figsize=(8,6))
     plt.scatter(X_pc[:, 0], X_pc[:, 1], c=y, cmap=plt.cm.rainbow)
     plt.title('First 2 principal components after poly Kernel PCA with the degree set to %s'%i)     
     plt.xlabel('PC1')
     plt.ylabel('PC2')
     #name ='swissroll'+ 'poly_2pca_degree'+str(i)
     #filename = '/home/dell1/Desktop/mlcourse/' + name+'.jpeg'
     #plt.savefig(filename,format='jpeg')
     plt.show()
     #plt.close()
     


     
     plt.figure(figsize=(8,6))
     plt.scatter(X_pc[:,0], np.zeros((800,1)), c=y, cmap=plt.cm.rainbow)    
     plt.title('First principal component after poly Kernel PCA with the degree set to %s'%i)
     plt.xlabel('PC1')
     #name ='swissroll'+ 'poly_1pca degree'+str(i)
     #filename = '/home/dell1/Desktop/mlcourse/' + name+'.jpeg'
     #plt.savefig(filename,format='jpeg')
     plt.show()
     #plt.close()    
     end=time.time()
     print("time spent on the current task is ",end - start)
     pause()


########################################################################
     ### laplacian kernel application 
     
   for i in (0.1,0.5,1,2): 
       start=time.time()  
       X_pc = laplacian_kernel(X, gamma=i, n_components=2)
       plt.figure(figsize=(8,6))
       plt.scatter(X_pc[:, 0], X_pc[:, 1], c=y, cmap=plt.cm.rainbow)

       plt.title('First 2 principal components after laplacian Kernel PCA with gamma set to %s'%i)       
       plt.xlabel('PC1')
       plt.ylabel('PC2')
       #name ='swissroll'+ 'laplacian_2pca_gamma'+str(i)
       #filename = '/home/dell1/Desktop/mlcourse/' + name+'.jpeg'
       #plt.savefig(filename,format='jpeg')
       plt.show()
       #plt.close()
       


       plt.figure(figsize=(8,6))
       plt.scatter(X_pc[:,0], np.zeros((800,1)), c=y, cmap=plt.cm.rainbow)
  
       plt.title('First principal component after laplacian Kernel PCA with gamma set to %s'%i)
       plt.xlabel('PC1')
       #name ='swissroll'+ 'laplacian_1pca_gamma'+str(i)
       #filename = '/home/dell1/Desktop/mlcourse/' + name+'.jpeg'
       #plt.savefig(filename,format='jpeg')
       plt.show()
       #plt.close()
       
       end=time.time()
       print("time spent on the current task is ",end - start)
       pause()
       
       
       
       
   return()   
   


import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from sklearn.datasets import make_circles
from sklearn.datasets import make_swiss_roll
import sklearn.datasets 

#####################make_moons##########################################
X, y = make_moons(n_samples=100, random_state=123)
print("the data set is make moons")
comparaison_makemoons(X,y)
pause()


######################make circles####################################
X, y = make_circles(n_samples=1000, random_state=123, noise=0.1, factor=0.2)
print("the data set is make circles")
comparaison_makecircles(X,y)
pause()


#######################make swiss roll#####################################
from mpl_toolkits.mplot3d import Axes3D

X, y = make_swiss_roll(n_samples=800, random_state=123)
print("the data set is swiss roll")
comparaison_swissroll(X,y)



















