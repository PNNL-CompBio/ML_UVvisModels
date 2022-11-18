#!/usr/bin/env python3

# This script will predict a saved model. Note -- we assume the saved model is GPU-enabled.
# Input -- saved model file, test file. Output -- CSV with <SMILE>,<True>,<Pred>
# Assumption number 1 -- we assume that the saved model's parameters can be parsed from the model's filename
#                        This is the case if you use the --savemodel feature from train.py
# Assumption number 2 -- We assume that the model you are using is for the GPU.

import os
import sys
import pandas as pd
import torch
import numpy as np

#we assume that you are running the model from the main section of this github repository
sys.path.append(os.getcwd())
sys.path.append('src')

import argparse
import time
from transformer import make_model
from data_utils import load_data_from_df, construct_loader
import pickle
import re

def parse_model_options(filename):
        '''
        This function parses out the the following from the filename:
                dropout                    | _drop{#}_
                lambda_distance            | _ldist{#}_
                lanmbda_attention          | _lattn{#}_
                number of dense layers     | _Ndense{#}_
                heads                      | _heads{#}_
                dimension of model         | _dmodel{#}_
                number of stacked layers   | _nsl{#}_

        --ASSUMPTIONS--
                Assumes that the file's name has the convention specified above!

        --Returns--
                the same order
        '''

        drop=float(re.search(r'drop(\d+\.?\d*)',filename).group(1))
        ldist=float(re.search(r'ldist(\d+\.?\d*)',filename).group(1))
        lattn=float(re.search(r'lattn(\d+\.?\d*)',filename).group(1))
        nDense=int(re.search(r'Ndense(\d+\.?\d*)',filename).group(1))
        heads=int(re.search(r'heads(\d+\.?\d*)',filename).group(1))
        dmodel=int(re.search(r'dmodel(\d+\.?\d*)',filename).group(1))
        nsl=int(re.search(r'nsl(\d+\.?\d*)',filename).group(1))

        return drop,ldist,lattn,nDense,heads,dmodel,nsl

parser=argparse.ArgumentParser(description='Predict MAT model on a given test set')
parser.add_argument('-m','--model',type=str,required=True,help='Trained torch model file')
parser.add_argument('-i','--input',type=str,required=True,help='File to evaluate. Assumed format is "<SMILE>,<solubility>"')
parser.add_argument('-o','--output',type=str,required=True,help='File for Predictions. Format is "<SMILE>,<True>,<Predicted>"')
parser.add_argument('--stats',default=False,action='store_true',help='Flag to print the R2, RMSE, and the time to perform the evaluation.')
parser.add_argument('--twod',default=False,action='store_true',help='Flag to use 2D conformers for distance matrix.')
args=parser.parse_args()

if args.stats:
        start=time.time()

#loading the data
X,gold=load_data_from_df(args.input,one_hot_formal_charge=True,use_data_saving=True,two_d_only=args.twod)
data_loader=construct_loader(X,gold,batch_size=8,shuffle=False)

if args.stats:
        print('Loading data time:',time.time()-start)
        start=time.time()

#constructing the model
drop,ldist,lattn,nDense,heads,dmodel,nsl=parse_model_options(args.model)

d_atom=X[0][0].shape[1]
model_params={
        'd_atom': d_atom,
    'd_model': dmodel,
    'N': nsl,
    'h': heads,
    'N_dense': nDense,
    'lambda_attention': lattn, 
    'lambda_distance': ldist,
    'leaky_relu_slope': 0.1, 
    'dense_output_nonlinearity': 'relu', 
    'distance_matrix_kernel': 'exp', 
    'dropout': drop,
    'aggregation_type': 'mean'
}

model=make_model(**model_params)
model.load_state_dict(torch.load(args.model))
model.cuda()
model.eval()

if args.stats:
        print('Model construction time:',time.time()-start)
        start=time.time()

##getting the predictions
preds=np.array([])
ys=np.array([])
for batch in data_loader:
        adjacency_matrix, node_features,distance_matrix,y=batch
        batch_mask=torch.sum(torch.abs(node_features),dim=-1) != 0
        pred=model(node_features,batch_mask,adjacency_matrix,distance_matrix,None)
        preds=np.append(preds,pred.tolist())
        ys=np.append(ys,y.tolist())

if args.stats:
        print('Model Prediction time:',time.time()-start)
        print('RMSE:',np.sqrt(np.mean((preds-ys)**2)))
        print('R2:',np.corrcoef(preds,ys)[0][1]**2)

with open(args.output,'w') as outfile:
        outfile.write('smile,true | pred\n')
        lines=open(args.input).readlines()
        lines=lines[1:]

#        preds=preds.tolist()
        assert len(lines)==len(preds)/180
        preds = np.split(preds,94)
#        print('preds: ',preds)
        i = 0 
        j=0
        for x in lines:
                i += 1  
        for x in preds:
                j += 1
                print(x)
#### DEBUG: See if the lines and preds match up 
#        print('Finished i: ', i)
#        print('Finished j: ', j)
####
        check_list = isinstance(preds,list)
        print(check_list)
        
#        preds = [str(element) for element in preds]
#        preds = ",".join(preds)

        preds = [l.tolist() for l in preds]

        for x,y in zip(lines,preds):
                outfile.write(str(x)+'|'+str(y)+'\n')
