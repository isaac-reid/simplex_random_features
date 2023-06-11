#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stuff for loading datasets and processing into train and test

"""
import os
import numpy as np

#os.chdir('/homes/ir337/Documents/simplex_random_features/datasets')
os.chdir('/Users/isaacreid/Documents/PhD/Year 1/simplex_random_features')


def onehot_encode(labels, n_classes):
  onehot_labels = np.zeros((len(labels), n_classes))
  onehot_labels[np.arange(len(labels)), labels.astype(int)] = 1
  return onehot_labels

train_prop = 0.95
test_ratio = 0.05

TEXT_FEAT_INDICES = {
    'abalone.data': [0],
    'banknote.txt': [],
    'car.data': list(range(6)),
    'yeast.data': [],
    'cmc.data': [],
    'nursery.data': list(range(8)),
    'wifi.txt': [],
    'chess.data': list(range(6))
}

SEPARATORS = {
    'abalone.data': ',',
    'banknote.txt': ',',
    'car.data': ',',
    'yeast.data': None,
    'cmc.data': ',',
    'nursery.data': ',',
    'wifi.txt': None,
    'chess.data': ','
}

def load_data(name):
    os.chdir('datasets')
    text = []
    with open(name) as f:
        for line in f:
            #do something with line
            text.append(line)
           
    data = []
    for i in range(len(text)):
        data.append(text[i].split('\n')[0].split(SEPARATORS[name]))
    data = np.asarray(data[:-1])
   
    if name == 'yeast.data':
        data = data[:,1:]
   
    holder = []
    for index,x in enumerate(data.T):
        if index in TEXT_FEAT_INDICES[name] or index == np.shape(data)[1]-1:
            holder.append(np.unique(x, return_inverse=True)[1])
        else:
            holder.append(x)    
   
    #data = [np.unique(x, return_inverse=True)[1] for x in data.T[TEXT_FEAT_INDICES[name]]]
    data = list(np.asarray(holder,dtype=float).T)
    data = np.random.permutation(data)
    data=np.asarray(data,dtype=float)
    #data = np.asarray(data,dtype = int)
    labels = data[:,-1]
    data = data[:,:-1]
    classes = len(np.unique(labels))
   
    train_objs = data[: int(train_prop * len(data))]
    test_objs = data [ int(train_prop * len(data)):]
   
    mean = train_objs.mean(axis=0, keepdims=True)
    std = train_objs.std(axis=0, keepdims=True)
    train_objs = (train_objs - mean) / std
    test_objs = (test_objs - mean) / std
   
    labels  =onehot_encode(labels,classes)
    train_labels = labels[: int(train_prop * len(labels))]
    test_labels = labels [ int(train_prop * len(labels)):]
   
    data = (train_objs, train_labels, test_objs, test_labels, test_objs, test_labels)
    os.chdir('..')
    return data

def load_data_onehot(name):
    rng = np.random.default_rng(42)
    os.chdir('datasets')
    text = []
    with open(name) as f:
        for line in f:
            #do something with line
            text.append(line)
           
    data = []
    for i in range(len(text)):
        data.append(text[i].split('\n')[0].split(SEPARATORS[name]))
    data = np.asarray(data[:-1])
   
    if name == 'yeast.data':
        data = data[:,1:]
   
    data = data.T
    
    n_cols = len(data)
    text_data = [data[i] for i in TEXT_FEAT_INDICES[name] + [n_cols - 1]]
    num_data = [data[i] for i in range(n_cols) if i not in \
                TEXT_FEAT_INDICES[name] + [n_cols - 1]]
        
    cls_counts = [len(np.unique(x)) for x in text_data]
    text_data = [np.unique(x, return_inverse=True)[1] for x in text_data]
    text_data = [onehot_encode(x, n) for x, n in zip(text_data, cls_counts)]

    text_data = list(zip(*text_data))
    num_data = list(zip(*num_data))
    
    if len(num_data) == 0:
      num_data = [() for _ in text_data]

    objs = []
    labels = []

    for text_feats, num_feats in zip(text_data, num_data):
      float_num_feats = tuple(float(x) for x in num_feats)
      obj = np.concatenate((float_num_feats,) + text_feats[:-1])
      label = text_feats[-1]
      objs.append(obj)
      labels.append(label)
      
    objs = np.array(objs)
    labels = np.array(labels)

    data_perm = rng.permutation(len(objs))
    objs = objs[data_perm]
    labels = labels[data_perm]
    
    train_size = int(len(objs) * (1 - test_ratio))
    test_size = int(len(objs) * test_ratio)
    
    train_objs = objs[:train_size]
    train_labels = labels[:train_size]
    test_objs = objs[-test_size:]
    test_labels = labels[-test_size:]

    mean = train_objs.mean(axis=0, keepdims=True)
    std = train_objs.std(axis=0, keepdims=True)
    train_objs = (train_objs - mean) / std
    test_objs = (test_objs - mean) / std
   
    data = (train_objs, train_labels, test_objs, test_labels, test_objs, test_labels)
    os.chdir('..')
    return data