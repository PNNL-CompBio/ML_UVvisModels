"""Retrain the G4MP2 modle on all available data"""

from schnetpack.data import AtomsLoader, StatisticsAccumulator
from schnetpack.train import hooks, Trainer
from schnetpack import nn
from tempfile import TemporaryDirectory
from datetime import datetime
from math import ceil
import pickle as pkl
import numpy as np
import argparse
import logging
import shutil
import torch
import json
import sys
import os
import rdkit
from schnetpack.atomistic import Atomwise, AtomisticModel, DeltaLearning
from schnetpack.representation import SchNet

# Hard-coded options
batch_size = 5
validation_frac = 0.05
chkp_interval = lambda n: ceil(100000 / n)
lr_patience = lambda n: ceil(25 * 100000 / n)
lr_start = 1e-3
lr_decay = 0.5
lr_min = 1e-6

max_size = 117232
n_atom_basis = 256
n_filters = 256
cutoff=5
n_gaussians=25
max_z=35

if __name__ == "__main__":
    # Make the parser
    parser = argparse.ArgumentParser()

    # Add the arguments to the parser
    parser.add_argument('-d', '--device', help='Name of device to train on', type=str, default='cpu')
    parser.add_argument('-n', '--num-workers', help='Number of workers to use for data loaders',
                        type=int, default=4)
    parser.add_argument('--copy', help='Create a copy of the training SQL to a local folder',
                        type=str, default=None)
    
    # Parse the arguments
    args = parser.parse_args()

    # Configure the logger
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('schnet-delta')

    logger.info('Training model on {}'.format(args.device))

    # Open the right training data subset

#    with open('UV_Expt_DFT.pkl', 'rb') as fp
    with open('train_dataset.pkl', 'rb') as fp:
        full_train_data = pkl.load(fp)

    # Get the total training set size
    logger.info('Loaded the training data: Size = {}'.format(len(full_train_data)))

    # Make sure model hasn't finished training already
    net_dir = os.path.join('model')  # Path to this network
    work_dir = net_dir  # Path to the specific run
    if os.path.isfile(os.path.join(net_dir, 'atomref.npy')):
            atomref = np.load(os.path.join(net_dir, 'atomref.npy'))
    if os.path.isfile(os.path.join(work_dir, 'finished')):
        logger.info('Model has already finished training')
#        exit()

    # Loading the model
    #model = torch.load(os.path.join(net_dir, 'architecture.pth'))
    reps = SchNet(n_atom_basis=n_atom_basis, n_filters=n_filters, n_interactions=6,
              cutoff=5, n_gaussians=n_gaussians, max_z=max_z)
    output = Atomwise(256, n_out=181, atomref=atomref,
                 mean=torch.Tensor([0]*(181)),
                 stddev=torch.Tensor([1]*(181)),
                 train_embeddings=True)
    model = AtomisticModel(reps, output)
    #print(model)

    with open(os.path.join(net_dir, 'options.json')) as fp:
        options = json.load(fp)
    logger.info('Loaded model from {}'.format(os.path.join(net_dir)))

    # Load in the training data
    split_file = os.path.join(net_dir, 'split.npz')
    #print("split file",split_file)
    with TemporaryDirectory(dir=args.copy) as td:

        # If desired, copy the SQL file to a temporary directory
        if args.copy is not None:
            new_sql_path = os.path.join(td, os.path.basename(full_train_data.dbpath))
            shutil.copyfile(full_train_data.dbpath, new_sql_path)
            full_train_data.dbpath = new_sql_path
            logger.info('Copied db path to: {}'.format(new_sql_path))

        # Get the training and validation size
        validation_size = int(len(full_train_data) * validation_frac)
        train_size = len(full_train_data) - validation_size
        train_data, valid_data, _ = full_train_data.create_splits(train_size, validation_size)#,split_file)
        train_load = AtomsLoader(train_data, shuffle=True,
                                 num_workers=args.num_workers, batch_size=batch_size)
        valid_load = AtomsLoader(valid_data, num_workers=args.num_workers, batch_size=batch_size)
        logger.info('Made training set loader. Workers={}, Train Size={}, '
                    'Validation Size={}, Batch Size={}'.format(
            args.num_workers, len(train_data), len(valid_data), batch_size))

        # Update the mean and standard deviation of the dataset
        atomref = None
        if os.path.isfile(os.path.join(net_dir, 'atomref.npy')):
            atomref = np.load(os.path.join(net_dir, 'atomref.npy'))
        if options.get('delta', None) is not None:
            # Compute the stats for the delta property
            delta_prop = options['delta']
            statistic = StatisticsAccumulator(batch=True)
            for d in train_load:
                #print("*******",d)
                d['delta_temp'] =  d[delta_prop]
                train_load._update_statistic(True, atomref, 'delta_temp', d, statistic)
            mean, std = statistic.get_statistics()
            mean = (mean,)  # Make them a tuple
            std = (std,)
            logger.info('Computed statistics for delta-learning model')
        else:
            if atomref is not None:
                mean, std = zip(*[train_load.get_statistics(x, per_atom=True, atomrefs=ar[:, None])
                                  for x, ar in zip(options['output_props'], atomref.T)])
            else:
                mean, std = zip(*[train_load.get_statistics(x, per_atom=True)
                                  for x in options['output_props']])
        model.output_modules.standardize = nn.base.ScaleShift(torch.cat(mean), torch.cat(std))
        model.to(args.device)

        # Make the loss function, optimizer, and hooks -> Add them to a trainer
        def loss(b, p):
            print("b=====",b)
            print("p=====",p)
            #print("p['y'], y_true",p['y'].shape, y_true.shape)
            y_true = torch.stack(tuple(torch.squeeze(b[s]) for s in options['output_props']), 1)
            print("p['y'], y_true",p['y'].shape, y_true.shape)
            y_true = y_true.permute(0, 2, 1)[:, :, -1]
            #print("p['y'], y_true",p['y'].shape, y_true.shape)
            return torch.nn.functional.mse_loss(p['y'], y_true)

        #  Get only the fittable parameters
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        opt = torch.optim.Adam(trainable_params, lr=1e-4)

        hook_list = [hooks.CSVHook(work_dir, []),
                     hooks.ReduceLROnPlateauHook(opt, patience=lr_patience(len(train_data)),
                                                 factor=lr_decay, min_lr=lr_min,
                                                 stop_after_min=True)]
        logger.info('Created loss, hooks, and optimizer')
        trainer = Trainer(work_dir, model, loss, opt, train_load, valid_load, map_location=args.device,
                          hooks=hook_list, checkpoint_interval=chkp_interval(len(train_data)))

        # Run the training
        logger.info('Started training')
        sys.stdout.flush()
        trainer.train(args.device)

        # Mark training as complete
        with open(os.path.join(work_dir, 'finished'), 'w') as fp:
            print(str(datetime.now()), file=fp)
        logger.info('Training finished')
