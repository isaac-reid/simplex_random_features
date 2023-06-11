#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example script -- does nonparametric classification via kernel regression and 
generates Fig. 8 from the paper (as well as optional Simplex+)

"""

import numpy as np
from tqdm import tqdm
import os
import math
from data_loader import *
from utils import *


#os.chdir('/homes/ir337/Documents/simplex_random_features')
os.chdir('/Users/isaacreid/Documents/PhD/Year 1/simplex_random_features')

labels = ['IIDRFs','ORFs','SimRFs','SimRFs+'] #the 4 possible methods
include_method = [True,True,True,False] #indicator variable for whether to show
dataset_names = ['abalone','banknote','car','yeast','cmc',\
                 'nursery','wifi','chess']  #dataset names
names_with_types =  ['abalone.data', 'banknote.txt', 'car.data', 'yeast.data', \
                     'cmc.data', 'nursery.data', 'wifi.txt', 'chess.data']
    
best_sigmas = np.asarray([0.41331138, 1.06445901, 0.77656608, 0.90918797,
     0.48389667, 0.18788887, 0.21997651, 0.25754408]) #from validation dataset. 
  # See Fig. 10 in paper. Rescales data (or equivalently kernel lengthscale)
  # so we are in regime where a random feature approach will be effective so 
  # we can identify isolate effect of using simplex
  
approx_R = False  #do you want to use an orthogonal matrix exactly sampled from  
                 #Haar measure or approximately (k=3 Hadamard-product)?                 
use_restricted_dataset_sizes = True #optionally subsample big datasets for speed

# Here choose m, the number of random features (/d):
N = np.asarray(np.exp(np.linspace(np.log(1),np.log(10),4)),dtype=int) 
#N = [1,2]

N_var = 500 # number of times we do experiment for each m to get statistics
sim_plus_iter = 5 # hyperparam for number of iterations in SimRFs+ opt 
# (see Eq. 17 in paper) -- small is empirically OK

def get_all_accuracies(N,sigma,include_method,data):
    "Function to get all the random features, use them for kernel regression \
        and evaluate the accuracy of the result"
        
    (train, trainlab, valid_objs, valid_labels, test, testlab) = data
    d = np.shape(train)[1]
    train_size = np.shape(train)[0]
    test_size = np.shape(test)[0]
    
    #get all the features according to the different schemes
    (ktrainv,ktestv) = get_iid_features(N,sigma,data)
    (ktraino,ktesto) = get_orth_features(N,sigma,data,approx_R)
    (ktrains,ktests) = get_simp_features(N,sigma,data,approx_R)
    (ktrainsp,ktestsp) = get_simp_plus_features(N,sigma,data,approx_R,sim_plus_iter)
    
    
    #Do all the dot products, evaluating the approximations to the kernel \
    #Measure similarity between each train and test datapoint
    vweights =  np.matmul(ktrainv,ktestv.T)
    oweights =  np.matmul(ktraino,ktesto.T)
    sweights = np.matmul(ktrains,ktests.T)
    spweights = np.matmul(ktrainsp,ktestsp.T)

    return (np.sum(np.argmax(np.matmul(vweights.T,trainlab),axis=1)==\
                   np.argmax(testlab,axis=1))/test_size ,\
            np.sum(np.argmax(np.matmul(oweights.T,trainlab),axis=1)==\
                   np.argmax(testlab,axis=1))/test_size,\
            np.sum(np.argmax(np.matmul(sweights.T,trainlab),axis=1)==\
                   np.argmax(testlab,axis=1))/test_size, \
            np.sum(np.argmax(np.matmul(spweights.T,trainlab),axis=1)==\
                   np.argmax(testlab,axis=1))/test_size )


"Reproducing the kernel regression plots in Fog. 8"

means_record = np.zeros((len(dataset_names),len(N),4)) #holder mat for results
stds_record = np.zeros((len(dataset_names),len(N),4))
exact_record=[]   #keeps track of outcome with exact kernel evaluation

for ds_index,name in enumerate(dataset_names):
    print('********')
    print(name)

    data = load_data_onehot(names_with_types[ds_index])

    (train, trainlab, valid_objs, valid_labels, test, testlab) = data
    d = np.shape(train)[1]
    train_size = np.shape(train)[0]
    test_size = np.shape(test)[0] 


    if approx_R == True: #pad out the data if we need d to be a power of 2
            d_prime = 2**int(np.ceil(math.log(d,2)))
            padded_train = np.zeros((np.shape(data[0])[0],d_prime))
            padded_train[:,:d] = data[0]
            padded_test = np.zeros((np.shape(data[4])[0],d_prime))
            padded_test[:,:d] = data[4]
            data[0] = padded_train
            data[4]=padded_test
            d=d_prime


    sigma = best_sigmas[ds_index]

    if use_restricted_dataset_sizes == True:
      if name == 'nursery' or name=='chess': #in the interests of time...
          train=train[:1000]
          trainlab=trainlab[:1000]
          test=test[:100]
          testlab=testlab[:100]
          train_size = np.shape(train)[0]
          test_size = np.shape(test)[0]
    
    true_weights,true_acc = get_true_acc(sigma,train_size,test_size,train,\
                                         test,trainlab,testlab)
    
    exact_record.append(true_acc)

    means = np.zeros((len(N),4))
    stdevs = np.zeros((len(N),4))
    
    for j in range(len(N)):
        print(str(j+1) + ' of ' + str(len(N)))
        performance_log = np.zeros((N_var,4))
        for i in tqdm(range(N_var)):
            performance_log[i,:] = get_all_accuracies(N[j],sigma,\
                                                      include_method,data)
        means[j,:] = np.mean(performance_log,axis=0)
        stdevs[j,:] = np.sqrt(np.var(performance_log,axis=0)/N_var)
 
    means_record[ds_index,:,:] = means
    stds_record[ds_index,:,:] = stdevs


#Do a plot
do_plot(means_record,stds_record, dataset_names, include_method,N)