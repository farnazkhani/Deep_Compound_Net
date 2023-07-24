#Deep_Compound_Net
#!/usr/bin/env python
# coding: utf-8

# In[ ]:
from __future__ import print_function
from google.colab import drive
import sys
sys.path.insert(0,' ')

import sys
sys.path.append('/integrating.py')
sys.path.insert(0,'/integrating.py')
import pickle
import time, argparse, gc, os
import cupy as cp
import numpy as np
import integrateMV as MV

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Reporter, report, report_scope
from chainer import Link, Chain, ChainList, training
from chainer.datasets import tuple_dataset
from chainer.training import extensions
from chainer.optimizer_hooks import WeightDecay


import numpy as np
import keras
import tensorflow as tf
import random as rn
from keras import backend as K
from itertools import product
from keras.models import Model
from keras.preprocessing import sequence
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D, GlobalAveragePooling1D
from keras.layers import Conv2D, GRU
from keras.layers import Input, Embedding, LSTM, Dense, TimeDistributed, Masking, RepeatVector, merge, Flatten,Lambda, TimeDistributed, LSTM, Add
from keras.models import Model
from keras.layers import Bidirectional
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import optimizers, layers
import sys, pickle, os
import math, json, time
from keras.regularizers import l2
import decimal
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from random import shuffle
from copy import deepcopy
from sklearn import preprocessing
import sys, re, math, time
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import json
import pickle
import collections
from collections import OrderedDict
from matplotlib.pyplot import cm
import argparse
from sklearn import metrics
#-------------------------------------------------------------
 # featurevector size
plensize= 20
atomInfo = 21
structInfo = 21
lensize= atomInfo + structInfo
#-------------------------------------------------------------
path_data=' '
class o:
     def __init__(self):

        START = time.time()

        import argparse as arg
        args ={'gpu','batchsize'}
        
        self.atomsize = 20
        self.gpu = 0
        self.batchsize = 100
        self.epoch = 150
        self.s1 = 1
        self.sa1 = 1
        self.s2 = 1
        self.sa2 = 1
        self.s3 = 1 
        self.sa3 = 1
        self.j1 = 6
        self.pf1 = 100
        self.ja1 = 2
        self.j2 = 6
        self.pf2 = 100
        self.ja2 = 2
        self.j3 = 6
        self.pf3 = 100
        self.ja3 = 2
        self.n_hid3 = 100
        self.n_hid4 = 100
        self.n_hid5 = 100
        self.n_out = 1
        self.prosize = 5762
        self.k1 = 4
        self.st1 = 1
        self.f1 = 100
        self.k2 = 4
        self.st2 = 1
        self.k3 = 4
        self.st3 = 1
        self.f3 = 100
        self.k4 =4
        self.st4 = 1
        self.k5 =4
        self.st5 = 1
        self.f5 = 100
        self.k6 = 4
        self.st6 = 1
        self.n_out_com = 1
        self.n_hid6 = 100
        self.n_hid7 = 100
        self.frequency=100
args = o()   

    
print(args.gpu)
print('GPU: ', args.gpu)
print('# Minibatch-size: ', args.batchsize)
print('')
#-------------------------------
# GPU check
xp = np
for j in range(5):
    print('GPU mode')
    #xp = cp
    kfold = np.roll(np.arange(5),j)
    #-------------------------------
    # Loading datasets
    for i in range(5):
        print('Making Training dataset...')

        file_interactions=xp.load(path_data+'dataset_hard'+'/cv_'+str(kfold[i])+'/train_interaction.npy')
        print('Loading labels: train_interaction.npy')
        cID = xp.load(path_data+'dataset_hard'+'/cv_'+str(kfold[i])+'/train_chemIDs.npy')
        print('Loading chemIDs: train_chemIDs.npy')
        with open(path_data+'dataset_hard'+'/cv_'+str(kfold[i])+'/train_proIDs.txt') as f:
            pID = [s.strip() for s in f.readlines()]
        print('Loading proIDs: train_proIDs.txt')
        n2v_c, n2v_p = [], []
        with open(path_data+'/data_multi/modelpp.pickle', mode='rb') as f:
            modelpp = pickle.load(f)
        with open(path_data+'/data_multi/modelcc.pickle', mode='rb') as f:
            modelcc = pickle.load(f)
        for j in cID:
            n2v_c.append(modelcc.wv[str(j)])
        for k in pID:
            n2v_p.append(modelpp.wv[k])
        interactions = xp.asarray(file_interactions, dtype='int32').reshape(-1,args.n_out)
        n2vc = np.asarray(n2v_c, dtype='float32').reshape(-1,128)
        n2vp = np.asarray(n2v_p, dtype='float32').reshape(-1,128)
        #reset memory
        del n2v_c, n2v_p, cID, pID, modelcc, modelpp, file_interactions
        gc.collect()
        import pandas as pd
        file_smiles=xp.load(path_data+'smiles_mol'+str(kfold[i])+'.npy')  
        print('Loading smiles: train_recompound.npy', flush=True)
        
        smiles = xp.asarray(file_smiles, dtype='float32')#.reshape(-1,1,args.atomInfo,lensize)

        file_sequences=xp.load(path_data+'dataset_hard'+'/cv_'+str(kfold[i])+'/train_reprotein.npy')
        print('Loading sequences: train_reprotein.npy', flush=True)
        sequences = xp.asarray(file_sequences, dtype='float32')
        # reset memory
        del file_smiles
        gc.collect()
       
        del file_sequences
        gc.collect()
        
        print(interactions.shape, smiles.shape, sequences.shape, n2vc.shape, n2vp.shape, flush=True)

        print('Now concatenating...', flush=True)
        print(smiles.shape)
        print(sequences.shape)
        print(n2vc.shape)
        print(n2vp.shape)
        print(interactions.shape)
        n = int(0.8 * smiles.shape[0])
        if i==0:
           Drug_input = Input(shape=(100,), dtype='int32',name='drug_input') 
           Protein_input = Input(shape=(2000,), dtype='int32',name='protein_input')
           Drug_Drug_input = Input(shape=(128,), dtype='float32',name='drug_drug_input') 
           Protein_Protein_input = Input(shape=(128,), dtype='float32',name='protein_protein_input')

           num_filters = 64
           smiles_filter_lengths = 6
           encode_smiles = Embedding(input_dim=64+1, output_dim = 128, input_length=100,name='smiles_embedding')(Drug_input) 
           encode_smiles = Conv1D(filters=num_filters, kernel_size=smiles_filter_lengths,  activation='relu', padding='valid',  strides=1, name='conv1_smiles')(encode_smiles)
           encode_smiles = Conv1D(filters=num_filters*2, kernel_size=smiles_filter_lengths,  activation='relu', padding='valid',  strides=1, name='conv2_smiles')(encode_smiles)
           encode_smiles = Conv1D(filters=128, kernel_size=smiles_filter_lengths,  activation='relu', padding='valid',  strides=1, name='conv3_smiles')(encode_smiles)

           num_filters=32
           protein_filter_lengths = 12
           encode_protein = Embedding(input_dim=20+1, output_dim = 128, input_length=2000, name='protein_embedding')(Protein_input)
           encode_protein = Conv1D(filters=num_filters, kernel_size=protein_filter_lengths,  activation='relu', padding='valid',  strides=1, name='conv1_prot')(encode_protein)
           encode_protein = Conv1D(filters=num_filters*2, kernel_size=protein_filter_lengths,  activation='relu', padding='valid',  strides=1, name='conv2_prot')(encode_protein)
           encode_protein = Conv1D(filters=128, kernel_size=protein_filter_lengths,  activation='relu', padding='valid',  strides=1, name='conv3_prot')(encode_protein)
    
           encode_protein = GlobalMaxPooling1D()(encode_protein)
           encode_smiles = GlobalMaxPooling1D()(encode_smiles)

           drug_concate = keras.layers.concatenate([encode_smiles, Drug_Drug_input], axis=-1) 
           prot_concate = keras.layers.concatenate([encode_protein, Protein_Protein_input], axis=-1) 
           encode_interaction = keras.layers.concatenate([drug_concate, prot_concate], axis=-1) 

           # Fully connected 
           FC1 = Dense(1024, activation='relu', name='dense1')(encode_interaction)
           FC2 = Dropout(0.1)(FC1)
           FC2 = Dense(1024, activation='relu', name='dense2')(FC2)
           FC2 = Dropout(0.1)(FC2)
           FC2 = Dense(512, activation='relu', name='dense3')(FC2)
           # And add a logistic regression on top
           predictions = Dense(1, kernel_initializer='normal', name='dense4')(FC2) # if you want train model for active    /inactive set activation='sigmoid'

           model = Model(inputs=[Drug_input, Protein_input, Drug_Drug_input, Protein_Protein_input], outputs=[predictions])

           print('pattern: ', i, flush=True)
           args.output=path_data+'output'
           output_dir = args.output+'/'+'smilesN2vc_mSGD'+'/'+'pattern'+str(i)+'/'

           #-------------------------------
           #reset memory again
           #del  sequences, interactions, smiles, n2vc, n2vp
           gc.collect()
           learning_rate = 0.001
           METRICS = [ 
            keras.metrics.BinaryAccuracy(name='accuracy'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc'),
            keras.metrics.AUC(name='prc', curve='PR') # precision-recall curve
            ]
           adam=Adam(lr=learning_rate)
           model.compile(optimizer=adam, loss='mean_squared_error', metrics=['accuracy']) 
           es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
        if i<4:
            model.fit(([np.array(smiles[:n,:]),np.array(sequences[:n,:]), np.array(n2vc[:n,:]), np.array(n2vp[:n,:])]), np.array(interactions[:n,:]), batch_size=256, epochs=100, shuffle=False, callbacks=[es], validation_data=(([np.array(smiles[n:,:]),np.array(sequences[n:,:]), np.array(n2vc[n:,:]), np.array(n2vp[n:,:])]), np.array(interactions[n:,:])))
        else:
            pred_score=model.predict([np.array(smiles),np.array(sequences), np.array(n2vc), np.array(n2vp)])
            pred_score = np.array(pred_score).reshape(-1,1)
            pred = 1*(pred_score >=0.5)
            shape_interactions = np.array(interactions).reshape(-1, 1)
            count_TP= np.sum(np.logical_and(shape_interactions == pred, pred == 1)*1)
            count_FP = np.sum(np.logical_and(shape_interactions != pred, pred == 1)*1)
            count_FN = np.sum(np.logical_and(shape_interactions != pred, pred == 0)*1)
            count_TN = np.sum(np.logical_and(shape_interactions == pred, pred == 0)*1)

            Accuracy = (count_TP + count_TN)/(count_TP+count_FP+count_FN+count_TN)
            Sepecificity = count_TN/(count_TN + count_FP)
            Precision = count_TP/(count_TP+count_FP)
            Recall = count_TP/(count_TP+count_FN)
            Fmeasure = 2*Recall*Precision/(Recall+Precision)
            B_accuracy = (Sepecificity+Recall)/2
 
            print(count_TP,count_FN,count_FP,count_TN,Accuracy,B_accuracy,Sepecificity,Precision,Recall,Fmeasure, sep="\t")
            
            AUC = metrics.roc_auc_score(shape_interactions, pred_score, average = 'weighted')
            AUPR = metrics.average_precision_score(shape_interactions, pred_score, average = 'weighted')

            print(count_TP,count_FN,count_FP,count_TN,Accuracy,B_accuracy,Sepecificity,Precision,Recall,Fmeasure,AUC,AUPR, sep="\t")
        
        
        gc.collect()


