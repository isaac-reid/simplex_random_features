import requests
import numpy as np
from scipy.stats import ortho_group
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import matplotlib
from scipy.special import gamma,factorial
from scipy.optimize import minimize 
from scipy.linalg import hadamard
import math
from data_loader import *

#hyperparameters and that sort of thing
#os.chdir('/homes/ir337/Documents/simplex_random_features')
os.chdir('/Users/isaacreid/Documents/PhD/Year 1/simplex_random_features')

labels = ['IIDRFs','ORFs','SimRFs','SimRFs+'] #the 4 possible methods
include_method = [True,True,True,False] #indicator variable for whether to show
dataset_names = ['abalone','banknote','car','yeast','cmc',\
                 'nursery','wifi','chess']  #dataset names
    
names_with_types =  ['abalone.data', 'banknote.txt', 'car.data', 'yeast.data', 'cmc.data', 'nursery.data', 'wifi.txt', 'chess.data']   

    
best_sigmas = np.asarray([0.41331138, 1.06445901, 0.77656608, 0.90918797,
     0.48389667, 0.18788887, 0.21997651, 0.25754408]) #from validation dataset. 
  # can choose any positive real number and will observe differing performance

approx_R = False  #do you want to use an orthogonal matrix exactly sampled from  
                 #Haar measure or approximately (k=3 Hadamard-product)?
use_restricted_dataset_sizes = True #optionally subsample big datasets for speed

#m, number of random features (/d):
#N = np.asarray(np.exp(np.linspace(np.log(1),np.log(10),4)),dtype=int) 
N = [1] #single trial; not suitable for plotting but good for testing

N_var = 100 #number of times we do experiment for each m to get statistics

sim_plus_iter = 5 #hyperparam for num iter in SimRFs+ opt -- small is OK


def approx_orth_mat(d):       #Needed to implement approx
    (h1,h2,h3) = (hadamard(d),hadamard(d),hadamard(d))
    (d1,d2,d3)=(np.diag(np.sign(np.random.random(d)-0.5)),
                np.diag(np.sign(np.random.random(d)-0.5)),
                np.diag(np.sign(np.random.random(d)-0.5)))
    m = np.matmul(np.matmul(h3,d3),np.matmul(np.matmul(h2,d2),np.matmul(h1,d1)))
    m=m/np.sqrt(np.sum(m[0]**2))
    return m    


#Everything we need to load the data from UCI repo

def get_true_acc(sigma,train_size,test_size,train,test,trainlab,testlab):
    true_weights=np.zeros((train_size,test_size))
    for i in range(train_size):
        x = train[i] * sigma
        for j in range(test_size):
            y=test[j] * sigma                  
            true_weights[i,j] = np.exp(-np.sum((x-y)**2)/2)
   
    true_acc = np.sum(np.argmax(np.matmul(true_weights.T,trainlab),axis=1)== \
                      np.argmax(testlab,axis=1))/test_size
    return true_weights,true_acc


#a slightly monstruous function for doing the nonparametric classification

def get_all_accuracies(N,sigma,include_method,data):
    (train, trainlab, valid_objs, valid_labels, test, testlab) = data
    d = np.shape(train)[1]
    train_size = np.shape(train)[0]
    test_size = np.shape(test)[0]
    
    #IID case
    ktrainv= np.zeros((train_size,N*d))
    ktestv= np.zeros((test_size,N*d))
    if include_method[0]==True:
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
   
    #ORF case
    ktraino= np.zeros((train_size,N*d))
    ktesto= np.zeros((test_size,N*d))
    if include_method[1]==True:
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
           
    #SimRF case
    simp_dir = np.diag(np.ones(d))/np.sqrt(2) - np.ones((d,d))*(1/((d-1) * 
         np.sqrt(2))) *(1 + 1/np.sqrt(d)) #begin by getting the proj directions
    simp_dir[d-1,:] = 1/np.sqrt(2 * d) * np.ones((d))
    simp_dir[:,d-1] = 0
    simp_dir = simp_dir / np.sqrt(np.sum(simp_dir[1,:]**2))
       
    ktrains= np.zeros((train_size,N*d))
    ktests= np.zeros((test_size,N*d))

    if include_method[2]==True:
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

    #SimRFs+ case       
    ktrainsp= np.zeros((train_size,N*d))
    ktestsp= np.zeros((test_size,N*d))
    
    if include_method[3]==True:
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

#Now actually do it
means_record = np.zeros((len(dataset_names),len(N),4)) #holder mat for results
stds_record = np.zeros((len(dataset_names),len(N),4))
exact_record=[]   #keeps track of outcome with exact kernel evaluation

for ds_index,name in enumerate(dataset_names):
    print('********')
    print(name)

    data = load_data_onehot(names_with_types[ds_index])
    #data = list(load_uci_dataset(name,0))

    (train, trainlab, valid_objs, valid_labels, test, testlab) = data
    d = np.shape(train)[1]
    train_size = np.shape(train)[0]
    test_size = np.shape(test)[0]
    
    
    d = np.shape(data[0])[1]

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
    
    #plt.subplot(3,3, ds_index+1)
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
 
    labels = ['i.i.d.','orthogonal','simplex','simplex+']
    linestyles = ['solid','dashed','dashdot','dotted']
   
    means_record[ds_index,:,:] = means
    stds_record[ds_index,:,:] = stdevs


#Do a plot if we are sampling at more than one N (= m in paper)

if len(N)>0:
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

