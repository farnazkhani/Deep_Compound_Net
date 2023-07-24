#integrateMV
import time, argparse, gc, os

import numpy as np
import cupy as cp
from cupy import _core


from rdkit import Chem

#from feature import *
#import SCFPfunctions as Mf
import pickle

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

#-------------------------------------------------------------
    #Network definition
class CNN(chainer.Chain):

    def __init__(self, prosize, plensize, batchsize, s1, sa1, s2, sa2, s3, sa3, j1, pf1, ja1, j2, pf2, ja2, j3, pf3, ja3, n_hid3, n_hid4, n_hid5, n_out,
                 atomsize, lensize, k1, st1, f1, k2, st2, k3, st3, f3, k4, st4,k5,st5,f5,k6,st6 ,n_hid, n_out_com ):

        # prosize, plensize_20 = size of protein one hot feature matrix
        # j1, s1, pf1 = window-size, stride-step, No. of filters of first protein-CNN convolution layer
        # ja1, sa1 = window-size, stride-step of first protein-CNN average-pooling layer
        # j2, s2, pf2 = window-size, stride-step, No. of filters of second protein-CNN convolution layer
        # ja2, sa2 = window-size, stride-step of second protein-CNN average-pooling layer
        # j3, s3, pf3 = window-size, stride-step, No. of filters of third protein-CNN convolution layer
        # ja3, sa3 = window-size, stride-step of third protein-CNN average-pooling layer
        
        # atomsize, lenseize = size of feature matrix
        # k1, st1, f1 = window-size, stride-step, No. of filters of first convolution layer
        # k2, st2 = window-size, stride-step of first max-pooling layer
        # k3, st3, f3 = window-size, stride-step, No. of filters of second convolution layer
        # k4, st4 = window-size, stride-step of second max-pooling layer
        # k5,st5,f5 = window-size, stride-step, No. of filters of third convolutinal layer
        # k6,st6 = window-size, stride-step of third max-pooling layer

        super(CNN, self).__init__(
            conv1_pro=L.Convolution2D(pf1, (j1, plensize), stride=s1, pad = (j1//2,0)),
            bn1_pro=L.BatchNormalization(pf1),
            conv2_pro=L.Convolution2D(pf1, pf2, (j2, 1), stride=s2, pad = (j2//2,0)),
            bn2_pro=L.BatchNormalization(pf2),
            conv3_pro=L.Convolution2D(pf2, pf3, (j3, 1), stride=s3, pad = (j3//2,0)),
            bn3_pro=L.BatchNormalization(pf3),
            fc4=L.Linear(None, n_hid4),
            fc5=L.Linear(None, n_hid5),
            fc3_pro=L.Linear(None, n_hid3),
            fc4_pro=L.Linear(None, n_hid4),
            fc5_pro=L.Linear(None, n_hid5),
            fc6=L.Linear(None, n_out),
            fc_concat_pro=L.Linear(None, 100),
            fc_concat_com=L.Linear(None, 100),
            conv1_com=L.Convolution2D(f1,(k1, lensize), stride=st1, pad = (k1//2,0)),
            bn1_com=L.BatchNormalization(f1),
            conv2_com=L.Convolution2D(f1, f3, (k3, 1), stride=st3, pad = (k3//2,0)),
            bn2_com=L.BatchNormalization(f3),
            conv3_com=L.Convolution2D(f3,f5,(k5,1),stride=st5,pad=(k5//2,0)),
            bn3_com=L.BatchNormalization(f5),
            fc6_com=L.Linear(None, n_hid),
            fc6_z=L.Linear(None, n_hid),
            bn4_com=L.BatchNormalization(n_hid),
            fc7=L.Linear(None, n_out_com),
            fc7_pro=L.Linear(None, n_out_com)
        )
        self.n_hid3, self.n_hid4, self.n_hid5, self.n_out = n_hid3, n_hid4, n_hid5, n_out
        self.prosize, self.plensize = prosize, plensize
        self.s1, self.sa1, self.s2, self.sa2, self.s3, self.sa3 = s1, sa1, s2, sa2, s3, sa3
        self.j1, self.ja1, self.j2, self.ja2, self.j3, self.ja3 = j1, ja1, j2, ja2, j3, ja3

        self.m1 = (self.prosize+(self.j1//2*2)-self.j1)//self.s1+1
        self.m2 = (self.m1+(self.ja1//2*2)-self.ja1)//self.sa1+1
        self.m3 = (self.m2+(self.j2//2*2)-self.j2)//self.s2+1
        self.m4 = (self.m3+(self.ja2//2*2)-self.ja2)//self.sa2+1
        self.m5 = (self.m4+(self.j3//2*2)-self.j3)//self.s3+1
        self.m6 = (self.m5+(self.ja3//2*2)-self.ja3)//self.sa3+1
        
        
        self.n_hid6, self.n_hid7,self.n_hid5, self.n_out_com = n_hid, n_hid, n_hid5, n_out_com
        self.atomsize, self.lensize, self.n_out = atomsize, lensize, n_out
        self.k1, self.st1, self.f1, self.k2, self.st2, self.k3, self.st3, self.f3, self.k4, self.st4,self.k5,self.st5,self.f5,self.k6,self.st6 = k1, st1, f1, k2, st2, k3, st3, f3, k4, st4,k5, st5, f5, k6, st6
        self.l1 = (self.atomsize+(self.k1//2*2)-self.k1)//self.st1+1
        self.l2 = (self.l1+(self.k2//2*2)-self.k2)//self.st2+1
        self.l3 = (self.l2+(self.k3//2*2)-self.k3)//self.st3+1
        self.l4 = (self.l3+(self.k4//2*2)-self.k4)//self.st4+1
        self.l5=(self.l4+(self.k5//2*2)-self.k5)//self.st5+1
        self.l6=(self.l5+(self.k6//2*2)-self.k6)//self.st6+1

  

    def __call__(self, smiles, sequences, n2vc, n2vp, interactions):
        print("integrateeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
        print(n2vc.shape)
        print(n2vp.shape)
        smiles = np.transpose(smiles, (3,2,1,0))
        sequences = np.transpose(sequences, (3,2,1,0))
        z = self.cos_similarity(smiles, sequences, n2vc, n2vp)
        ZZ = self.fc6_z(z)
        print('type0')
        #print(type(Z))
        print(type(interactions))
        loss = F.sigmoid_cross_entropy(ZZ, interactions.transpose)#F.get_item(interactions.transpose,None), F.get_item(Z,None) )
        #accuracy = F.binary_accuracy(Z, interactions.transpose)
        #report({'loss': loss, 'accuracy': accuracy}, self)
        #loss = 1.
        return loss
        
    
    def predict_pro(self, seq):
        h = F.dropout(F.leaky_relu(self.bn1_pro(self.conv1_pro(seq))), ratio=0.2) # 1st conv
        h = F.average_pooling_2d(h, (self.ja1,1), stride=self.sa1, pad=(self.ja1//2,0)) # 1st pooling
        h = F.dropout(F.leaky_relu(self.bn2_pro(self.conv2_pro(h))), ratio=0.2) # 2nd conv
        h = F.average_pooling_2d(h, (self.ja2,1), stride=self.sa2, pad=(self.ja2//2,0)) # 2nd pooling
        h = F.dropout(F.leaky_relu(self.bn3_pro(self.conv3_pro(h))), ratio=0.2) # 3rd conv
        h = F.average_pooling_2d(h, (self.ja3,1), stride=self.sa3, pad=(self.ja3//2,0)) # 3rd pooling
        h_pro = F.max_pooling_2d(h, (self.m6,1)) # grobal max pooling, fingerprint
        #print(h_pro.shape)
        h_pro = F.dropout(F.leaky_relu(self.fc3_pro(h_pro)), ratio=0.2)# fully connected_1
        #print(h_pro.shape)
        return self.fc5_pro(h_pro)
    
   
    
    def predict_com(self,smiles):
        h = F.leaky_relu(self.bn1_com(self.conv1_com(smiles))) # 1st conv
        h = F.average_pooling_2d(h, (self.k2,1), stride=self.st2, pad=(self.k2//2,0)) # 1st pooling
        h = F.leaky_relu(self.bn2_com(self.conv2_com(h))) # 2nd conv
        h = F.average_pooling_2d(h, (self.k4,1), stride=self.st3, pad=(self.k4//2,0)) # 2nd pooling
        h=  F.leaky_relu(self.bn3_com(self.conv3_com(h))) #3th conv
        h = F.average_pooling_2d(h, (self.k6,1), stride=self.st3, pad=(self.k6//2,0))
        h = F.max_pooling_2d(h, (self.l4,1)) # grobal max pooling, fingerprint
        h = self.fc6(h) # fully connected
        #sr = 0.00001* cp.mean(cp.log(1 + h.data * h.data)) # sparse regularization
        h = F.leaky_relu(self.bn3_com(h))
        return self.fc5(h)#, sr


    def cos_similarity(self, smiles, seq, n2c, n2p):
        x_protein = self.predict_pro(seq)
        x_compound = self.predict_com(smiles)
        print('similarity')
        print(x_compound)
        print(n2c.shape)
        #x_compound = self.fc_concat_com(F.concat((x_compound[0], n2c)))
        x_compound = F.dropout(F.leaky_relu(x_compound), ratio=0.2)
        x_compound = F.dropout(F.leaky_relu(self.fc7(x_compound)), ratio=0.2)
        #x_protein = self.fc_concat_pro(F.concat((x_protein, n2p)))
        x_protein = F.dropout(F.leaky_relu(x_protein), ratio=0.2)
        x_protein = F.dropout(F.leaky_relu(self.fc7_pro(x_protein)), ratio=0.2)
        #print(x_protein.shape)
        #print(x_compound.shape)

        y = x_compound * x_protein

        return y
