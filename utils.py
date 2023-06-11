#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper functions and stuff behind the scenes

"""

import numpy as np
from scipy.stats import ortho_group
import matplotlib.pyplot as plt
from scipy.linalg import hadamard


def approx_orth_mat(d):      
    "Function to get a HD-product matrix for a faster approximate \
        implementation (see App. B.2 in paper)"
    (h1,h2,h3) = (hadamard(d),hadamard(d),hadamard(d))
    (d1,d2,d3)=(np.diag(np.sign(np.random.random(d)-0.5)),
                np.diag(np.sign(np.random.random(d)-0.5)),
                np.diag(np.sign(np.random.random(d)-0.5)))
    m = np.matmul(np.matmul(h3,d3),np.matmul(np.matmul(h2,d2),np.matmul(h1,d1)))
    m=m/np.sqrt(np.sum(m[0]**2))
    return m    


def get_true_acc(sigma,train_size,test_size,train,test,trainlab,testlab):
    "Get the nonparametric classification accuracy when we use the *exact* \
        (c.f. RF-approximated) kernel "
    true_weights=np.zeros((train_size,test_size))
    for i in range(train_size):
        x = train[i] * sigma
        for j in range(test_size):
            y=test[j] * sigma                  
            true_weights[i,j] = np.exp(-np.sum((x-y)**2)/2)
   
    true_acc = np.sum(np.argmax(np.matmul(true_weights.T,trainlab),axis=1)== \
                      np.argmax(testlab,axis=1))/test_size
    return true_weights,true_acc


def get_iid_features(N, sigma, data):
    "Returns *IIDRFs* for data.\
    - N is the number of features (/d) \
    - sigma is a scalar float that chooses thee kernel lengthscale, \
    - data is the vector of objects and their labels generated using load_data"

    (train, trainlab, valid_objs, valid_labels, test, testlab) = data
    d = np.shape(train)[1]
    train_size = np.shape(train)[0]
    test_size = np.shape(test)[0]
    
    #Initialise holders to store the vectors in
    ktrainv= np.zeros((train_size,N*d))
    ktestv= np.zeros((test_size,N*d))

    for n in range(N):
        random_weights = np.random.normal(0,1,(d,d))
        for i in range(train_size):
            x = train[i] * sigma
            wx = np.matmul(random_weights,x)
            ktrainv[i,n*d:(n+1)*d]=np.exp(wx)*np.exp(-np.dot(x,x))/np.sqrt(N*d)
        for i in range(test_size):
            x = test[i] * sigma
            wx = np.matmul(random_weights,x)
            ktestv[i,n*d:(n+1)*d]=np.exp(wx)*np.exp(-np.dot(x,x))/np.sqrt(N*d) 
            
    return ktrainv,ktestv

def get_orth_features(N, sigma,data,approx_R=False):
    "Returns *ORFs* for data.\
    - N is the number of features (/d) \
    - sigma is a scalar float that chooses thee kernel lengthscale, \
    - data is the vector of objects and their labels generated using load_data\
    - approx_R instructs whether to use HD-product in lieu of orthogonal matrix\
        drawn from Haar measure for a quicker approx implementation"
    
    (train, trainlab, valid_objs, valid_labels, test, testlab) = data
    d = np.shape(train)[1]
    train_size = np.shape(train)[0]
    test_size = np.shape(test)[0]
    
    ktraino= np.zeros((train_size,N*d))
    ktesto= np.zeros((test_size,N*d))
    
    
    for n in range(N):
        rand_orth = np.diag(np.sqrt(np.random.chisquare(d,size=d)))
        
        if approx_R == False:
            rot_matrix = ortho_group.rvs(d)
        else:
            rot_matrix = approx_orth_mat(d)
            
        for i in range(train_size):
            x = train[i] * sigma
            x_prime = np.matmul(rot_matrix,x)
            wx = np.matmul(rand_orth,x_prime)
            ktraino[i,n*d:(n+1)*d] = \
             np.exp(wx)*np.exp(-np.dot(x,x))/np.sqrt(N*d)
        for i in range(test_size):
            x = test[i] * sigma
            x_prime = np.matmul(rot_matrix,x)
            wx = np.matmul(rand_orth,x_prime)
            ktesto[i,n*d:(n+1)*d]=np.exp(wx)*np.exp(-np.dot(x,x))/np.sqrt(N*d)
            
    return ktraino,ktesto

def get_simp_features(N, sigma, data, approx_R=False):
    "Returns *SimRFs* for data.\
    - N is the number of features (/d) \
    - sigma is a scalar float that chooses thee kernel lengthscale, \
    - data is the vector of objects and their labels generated using load_data\
    - approx_R instructs whether to use HD-product in lieu of orthogonal matrix\
        drawn from Haar measure for a quicker approx implementation"
        
    (train, trainlab, valid_objs, valid_labels, test, testlab) = data
    d = np.shape(train)[1]
    train_size = np.shape(train)[0]
    test_size = np.shape(test)[0]
    
    simp_dir = np.diag(np.ones(d))/np.sqrt(2) - np.ones((d,d))*(1/((d-1) * 
         np.sqrt(2))) *(1 + 1/np.sqrt(d)) #begin by getting the proj directions
    simp_dir[d-1,:] = 1/np.sqrt(2 * d) * np.ones((d))
    simp_dir[:,d-1] = 0
    simp_dir = simp_dir / np.sqrt(np.sum(simp_dir[1,:]**2))
       
    ktrains= np.zeros((train_size,N*d))
    ktests= np.zeros((test_size,N*d))

    for n in range(N):
        rand_sim=np.matmul(np.diag(np.sqrt(np.random.chisquare(d,size=d))),simp_dir)
        
        if approx_R == False:
            rot_matrix = ortho_group.rvs(d)
        else:
            rot_matrix = approx_orth_mat(d)
            
        for i in range(train_size):
            x = train[i] * sigma
            x_prime = np.matmul(rot_matrix,x)
            wx = np.matmul(rand_sim,x_prime)
            ktrains[i,n*d:(n+1)*d]=np.exp(wx)*np.exp(-np.dot(x,x))/np.sqrt(N*d)
        for i in range(test_size):
            x = test[i] * sigma
            x_prime = np.matmul(rot_matrix,x)
            wx = np.matmul(rand_sim,x_prime)
            ktests[i,n*d:(n+1)*d]= \
            np.exp(wx)*np.exp(-np.dot(x,x))/np.sqrt(N*d)
                
    return ktrains,ktests

def get_simp_plus_features(N, sigma, data, approx_R=False, sim_plus_iter=5):
    "Returns *SimRFs+* for data.\
    - N is the number of features (/d) \
    - sigma is a scalar float that chooses thee kernel lengthscale, \
    - data is the vector of objects and their labels generated using load_data\
    - approx_R instructs whether to use HD-product in lieu of orthogonal matrix\
        drawn from Haar measure for a quicker approx implementation \
    - sim_plus_iter is number of iterations in vector direction adjustment, \
        see Eq. 17 in paper"
        
    (train, trainlab, valid_objs, valid_labels, test, testlab) = data
    d = np.shape(train)[1]
    train_size = np.shape(train)[0]
    test_size = np.shape(test)[0]
    
    simp_dir = np.diag(np.ones(d))/np.sqrt(2) - np.ones((d,d))*(1/((d-1) * 
         np.sqrt(2))) *(1 + 1/np.sqrt(d)) #begin by getting the proj directions
    simp_dir[d-1,:] = 1/np.sqrt(2 * d) * np.ones((d))
    simp_dir[:,d-1] = 0
    simp_dir = simp_dir / np.sqrt(np.sum(simp_dir[1,:]**2))
    
    ktrainsp= np.zeros((train_size,N*d))
    ktestsp= np.zeros((test_size,N*d))
    

    weights = np.diag(np.sqrt(np.random.chisquare(d,size=d)))


    simp_plus_dir = np.copy(simp_dir)   #doing weight-dependent coupling
    for iterate in range(sim_plus_iter):       
      for dcount in range(d):
          mask = np.diag(np.ones(d))
          mask[dcount,dcount]=0
          resultant = np.sum(np.matmul(weights,np.matmul(mask,simp_plus_dir)),axis=0)
          simp_plus_dir[dcount] = -resultant/np.sqrt(np.sum(resultant**2))      
    
    for n in range(N):
        rand_sim = np.matmul(weights,simp_plus_dir)
        
        if approx_R == False:
            rot_matrix = ortho_group.rvs(d)
        else:
            rot_matrix = approx_orth_mat(d)
            
            
        for i in range(train_size):
            x = train[i] * sigma
            x_prime = np.matmul(rot_matrix,x)
            wx = np.matmul(rand_sim,x_prime)
            ktrainsp[i,n*d:(n+1)*d] = np.exp(wx)* \
            np.exp(-np.dot(x,x))/np.sqrt(N*d)
        for i in range(test_size):
            x = test[i] * sigma
            x_prime = np.matmul(rot_matrix,x)
            wx = np.matmul(rand_sim,x_prime)
            ktestsp[i,n*d:(n+1)*d] = np.exp(wx)* \
            np.exp(-np.dot(x,x))/np.sqrt(N*d)
        
    return ktrainsp, ktestsp


 
    
def do_plot(means_record,stds_record, dataset_names, include_method,N):
    "Make a plot from the means and stds to reproduce Fig 8"
    
    means = means_record
    stdevs = stds_record
    labels = ['IIDRFs','ORFs','SimRFs','SimRFs+']
    linestyles = ['solid','dashed','dashdot','dotted']
    
    
    plt.figure(figsize=(12.5,12),dpi=70)
    
    for ds_index,name in enumerate(dataset_names):
        plt.subplot(4,3, ds_index+1)
        for i in range(4):
          if include_method[i]==True:
            plt.fill_between(N,means[ds_index,:,i] + stdevs[ds_index,:,i]
                            ,means[ds_index,:,i] - stdevs[ds_index,:,i],alpha=0.2)
            plt.plot(N,means[ds_index,:,i],label=labels[i],
                     linestyle=linestyles[i])
        plt.xscale('log')
        plt.xlabel('No. features (/d)')
        plt.ylabel('Accuracy')
        plt.title(name)
        #plt.plot((N[0],N[-1]),(true_acc,true_acc),linestyle= '--', 
        # color='black',label='exact kernel') #optionally add result from exact
    
        
    plt.subplot(4,3,9)
    for i in range(4):
      if include_method[i]==True:
        plt.plot(N,np.mean(means,axis=0)[:,i],label=labels[i],
                 linestyle=linestyles[i])
        plt.fill_between(N,np.mean(means,axis=0)[:,i] + 
                         np.mean(stdevs,axis=0)[:,i] ,np.mean(means,axis=0)[:,i] 
                         - np.mean(stdevs,axis=0)[:,i],alpha=0.2)
        #plt.plot((N[0],N[-1]),(np.mean(exact_record),np.mean(exact_record)),
        #linestyle= '--', color='black',label='exact kernel')
      plt.xscale('log')
      plt.xlabel('No. features (/d)')
      plt.ylabel('Accuracy')
      plt.title('average')
    
    plt.tight_layout()
    plt.legend(loc=(-1.23,-0.5),ncol=3)