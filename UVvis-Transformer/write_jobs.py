#!/usr/bin/env python3

#This script will generate a text file containing the training jobs/sweeps for wandb in a grid search of the options specified.

#NOTE -- this script will make all combinations -- some of which might fail during actual training. train.py handle the sanitization of the inputs.

import argparse,itertools

parser=argparse.ArgumentParser(description="Create Training Jobs for hyper parameter sweeps")

#options that are the same for each run in the sweep
parser.add_argument('--trainfile',type=str,default='',help='PATH to the training file you wish to use. For use with --testfile. Note this option supercedes --prefix and --fold.')
parser.add_argument('--testfile',type=str,default='',help='PATH to the testing file you wish to use. For use with --trainfile.')
parser.add_argument('--prefix',type=str,default='',help='PREFIX to the CCV training and testing files you wish to use. Assumed to follow <prefix>_<train|test><fold>.csv. Requires --fold.')
parser.add_argument('--fold',type=str,default='',help='Fold for the CCV datafiles. Used in conjunction with --prefix.')
parser.add_argument('--datadir',type=str,default='sweep',help='Absolute path to where the output of the training will be placed. Defaults to sweep')
parser.add_argument('--savemodel',action='store_true',help='Flag to have the training save the final weights of the model')
parser.add_argument('-e','--epochs',default=100,help='Maximum number of epochs to run the training for. Defaults to 100.')
parser.add_argument('--lr',type=float,default=0.04,help='Learning rate for the given sweep. Defaults to 0.04.')
parser.add_argument('--loss',type=str,default='huber',help='Loss function to be used for training. Defaults to huber. Must be in [mse,mae,huber].')


#variable options
parser.add_argument('--dropout',type=float,default=[0,0.1],nargs='+',help='Applying Dropout to model weights when training. Accepts any number of arguments. Defaults to [0,0.1].')
parser.add_argument('--ldist',type=float,default=[0,0.33],nargs='+',help='Lambda for model attention to the distance matrix. Accepts any number of arguments. Defaults to [0,0.33]')
parser.add_argument('--lattn',type=float,default=[0.25, 0.33, 0.5],nargs='+',help='Lambda for model attention to the attention matrix. Accepts any number of arguments. Defaults to [0.25,0.33,0.5]')
parser.add_argument('--ndense',type=int,default=[1],nargs='+',help='Number of Dense blocks in FeedForward section. Accepts any number of arguments. Defaults to 1')
parser.add_argument('--heads',type=int,default=[16,32,8,4,2],nargs='+',help='Number of attention heads in MultiHeaded Attention. Accepts any number of arguments. **Needs to evenly divide dmodel** Defaults to [16,32,8,4,2].')
parser.add_argument('--dmodel',type=int,default=[1024,512,256,128,64,32,16,8,4,2],nargs='+',help='Dimension of the hidden layer for the model. Accepts any number of arguments. Defaults to [1024,512,256,128,64,32,16,8,4,2].')
parser.add_argument('--nstacklayers',type=int,default=[16,8,6,4,2],nargs='+',help='Number of stacks in the Encoder layer. Accepts any number of arguments. Defaults to [16,8,6,4,2]')
parser.add_argument('--seed',type=int,default=[420],nargs='+',help='Random seed for training the models. Accepts any number of arguments. Defaults to 420.')
parser.add_argument('--dynamic',type=int,default=[0],nargs='+',help='If set, the maximum number of epochs a model can not improve on the training set before stopping training. Defaults to not being set. Can accept any number of arguments.')

#additional add on options
parser.add_argument('--twod',action='store_true',help='Flag to only use 2D conformers for making the distance matrix.')
parser.add_argument('--wandb',default=None,help='Project name for weights and biases integration.')
parser.add_argument('--cpu',action='store_true',help='Flag to use CPU models for the train.py job.')
parser.add_argument('-o','--outname',default='grid_sweep.cmds',help='Output filename. Defaults to grid_sweep.cmds')
args=parser.parse_args()

#Check that the parameters work together.
if args.trainfile:
    assert (bool(args.testfile)), 'Need to set --testfile when trainfile is set.'

if args.prefix:
    assert (bool(args.fold)), 'Need to set --fold when --prefix is set.'

#create the grid of the specified parameters
combos=itertools.product(args.dropout,args.ldist,args.lattn,args.ndense,args.heads,args.dmodel,args.nstacklayers,args.seed,args.dynamic)
with open(args.outname,'w') as outfile:
    for c in combos:
        drop,lam_dist,lam_attn,nden,head,dim,nsl,s,dyn=c
        if args.trainfile:
            preamble=f'python3 train.py --trainfile {args.trainfile} --testfile {args.testfile}'
        else:
            preamble=f'python3 train.py --prefix {args.prefix} --fold {args.fold}'
        sent=f'{preamble} --datadir {args.datadir} --epochs {args.epochs} --lr {args.lr} --loss {args.loss} --dropout {drop} --ldist {lam_dist} --lattn {lam_attn} --Ndense {nden} --heads {head} --dmodel {dim} --nstacklayers {nsl} --seed {s} --dynamic {dyn}'
        if args.twod:
            sent+=' --twod'
        if args.wandb:
            sent+=f' --wandb {args.wandb}'
        if args.cpu:
            sent+=' --cpu'

        outfile.write(sent+'\n')
