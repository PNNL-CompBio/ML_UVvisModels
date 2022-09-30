from __future__ import division
import gc
import os
import sys
import copy
from os import path
import pdb
import cPickle as pickle
import logging
import gzip
import timeit
import numpy as np
import lasagne
import theano.tensor as T
import theano
from utils import load_qm7b_data, load_oqmd_data, feature_expand
from sklearn.model_selection import train_test_split
from datetime import datetime
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from util import get_logger
# from model.deeptensor import orig_model
# import model.deeptensor as model_factory
import model as model_factory
import click

theano.config.floatX = 'float32'

# loggger = None

def train_and_get_error(train_data, valid_data, test_data, Estd, Emean, num_species,
            max_mol_size, values_to_predict, max_epochs, batch_size,num_interaction_passes,
            num_hidden_neurons,c_len,num_dist_basis, cost, model_name, learn_rate, earlystop_epochs,
            logger,check_every=2,**kwargs):
        # training_data,valid_data,test_data, trainin_hyperparams,model_name, model_hyperparams,
        #                 num_train_batches,num_train_samples,batch_size = 100,max_epochs=10000,
        #                 c_len = 30,num_hidden_neurons = 60,num_interaction_passes = 2,values_to_predict=-1
        #                 ):

    """
    param check_every      : Check the validation and test error every, these many epochs.
    param earlystop_epochs : If the validation error has not reduced in these many epochs
                             the training stops. It is decremented by 'check_every' depending
                             on the 'cost'. It is reset to its initial value if error reduces.
    """
    Z_train, D_train, y_train = train_data
    Z_val, D_val, y_val = valid_data
    Z_test, D_test, y_test    = test_data

    rng = np.random.RandomState(4)
    np.random.seed(1)

    try:
        logger.info("Trying to load model '%s'" % model_name)
        model = getattr(model_factory, model_name)
    except AttributeError as e:
        logger.error("Model not found. %s" % e)
        RuntimeError("Couldn't find model %s in the model_factory" % model_name)
    else:
        logger.info("Loaded '%s'" % model_name)
    
    f_train, f_eval_test, f_test, l_out =  model(Emean, Estd, max_mol_size,
                num_dist_basis, c_len, num_species,num_interaction_passes, 
                num_hidden_neurons, values_to_predict, cost, **kwargs)
        
    start_time = timeit.default_timer()

    lowest_test_mae = np.inf
    lowest_test_rmse = np.inf
    lowest_test_error = np.inf
    mu_max = None#np.max(D_train)+1

    D_val_fe = feature_expand(D_val, num_dist_basis, mu_max=mu_max)
    D_test_fe = feature_expand(D_test, num_dist_basis,mu_max=mu_max)

    logger.info("Expanded Test and Val.")

    num_train_samples = Z_train.shape[0]
    logger.info(num_train_samples)
    num_train_batches = num_train_samples // batch_size

    earlystop_epoch_counter = copy.deepcopy(earlystop_epochs)

    for epoch in range(max_epochs):
        # Randomly permute training data
        rand_perm = rng.permutation(Z_train.shape[0])
        Z_train_perm = Z_train[rand_perm]
        D_train_perm = D_train[rand_perm]
        y_train_perm = y_train[rand_perm]

        # logger.info("Shuffled train data")
        if learn_rate is None:
            if epoch < 10:
            #if epoch < 50:
                learning_rate = 0.01
            elif epoch < 500:
                learning_rate = 0.001
            elif epoch < 5000:
                learning_rate = 0.0001
            else:
                learning_rate = 0.00001
        else:
            learning_rate = learn_rate

        train_cost = 0
        for batch in range(num_train_batches):
            train_cost += f_train(
                    Z_train_perm[batch*batch_size:((batch+1)*batch_size)],
                    feature_expand(D_train_perm[batch*batch_size:((batch+1)*batch_size)], num_dist_basis, mu_max=mu_max),
                    y_train_perm[batch*batch_size:((batch+1)*batch_size)],
                    learning_rate
                    )
            #print("miniBatch %d of %d done." % (batch, num_train_batches))
        train_cost = float(train_cost) / float(num_train_batches)
        # logger.info("Got training cost.")

        if (epoch % check_every) == 0:
            y_pred = f_eval_test(Z_val, D_val_fe)
            val_errors = y_pred-y_val

            with open( 'predicted_spectra.txt', 'w') as f:
                for item in y_pred:
                    print >> f, item

            with open( 'given_spectra.txt', 'w') as f:
                for item in y_val:
                    print >> f, item


            # logger.info("Got Y val_pred")

            y_pred = f_eval_test(Z_test, D_test_fe)
            test_errors = y_pred-y_test
            test_cost = f_test(Z_test, D_test_fe, y_test)
            
            rmse = lambda x: np.sqrt(np.square(x).mean())
            mae  = lambda x: np.abs(x).mean()

            logger.info("VAL MAE:  %5.2f kcal/mol TEST MAE:  %5.2f kcal/mol" %
                    (mae(val_errors), mae(test_errors)))
            logger.info("VAL RMSE: %5.2f kcal/mol TEST RMSE: %5.2f kcal/mol" %
                    (rmse(val_errors), rmse(test_errors)))
            

            all_params = lasagne.layers.get_all_param_values(l_out)
            with gzip.open('results/model_epoch%d.pkl.gz' % (epoch), 'wb') as f:
                pickle.dump(all_params, f, protocol=pickle.HIGHEST_PROTOCOL)

            new_test_mae = mae(test_errors)
            if  new_test_mae < lowest_test_mae:
                lowest_test_mae = new_test_mae
                logger.info("Found best test MAE : {}".format(lowest_test_mae))
                np.savez("Y_test_pred_best_mae.npz", Y_test_pred = y_pred)
                if cost == "mae":
                    # Only update early stoppping counter if we are interested in this cost
                    earlystop_epoch_counter = copy.deepcopy(earlystop_epochs)
                    lowest_test_error = lowest_test_mae

            new_test_rmse = rmse(test_errors)
            if new_test_rmse < lowest_test_rmse:
                lowest_test_rmse = new_test_rmse
                logger.info("Found best test RMSE : {}".format(lowest_test_rmse))
                np.savez("Y_test_pred_best_rmse.npz", Y_test_pred = y_pred)
                if cost == "rmse":
                    # Only update early stoppping counter if we are interested in this cost
                    earlystop_epoch_counter = copy.deepcopy(earlystop_epochs)
                    lowest_test_error = lowest_test_rmse

            end_time = timeit.default_timer()

            test_error = np.sqrt(test_cost)
            train_error = np.sqrt(train_cost)
            logger.debug("Time %4.1f, Epoch %4d, train_cost=%5g, test_error=%5g" % (end_time - start_time, epoch, train_error, test_error))
            start_time = timeit.default_timer()

            # update and check the early stop counter.
            earlystop_epoch_counter -= check_every
            if earlystop_epoch_counter <= 0:
                break

    # return test_error
    return lowest_test_error

def get_data(path_to_xyz_file, path_to_targets_file, logger, num_dist_basis=40, 
            expand_features=False,train_split=0.9, valid_test_split=0.5, **kwargs):

    Z, D, y, num_species = load_qm7b_data(num_dist_basis, dtype=theano.config.floatX,
                                             xyz_file=path_to_xyz_file,expand_features=expand_features)
    # The y values are free_energies but we want to predict 16 energies / spectrum so 
    # we load them next.

    # Largest interatomic distance in dataset
    mu_max = np.max(D)
    max_mol_size = Z.shape[1]

    # We predict values in the targets file
    # the targets file could be a txt file or an npz file.
    try:
        # Try loading the file as a txt file.
        y = np.loadtxt(path_to_targets_file).astype(np.float32)
    except (ValueError,UnicodeDecodeError) as e:
        # Not a txt file, Try loading the file as an npz file.
        # UnicodeDecodeError in python3.6 NumPy (1.11.3)
        # ValueError in python2.7 NumPy (1.12.1)
        data_target = np.load(path_to_targets_file)
        assert len(data_target.files) == 1, "There appear to be more than one variable in the targets npz file: {}. There must be only one.".format(data_target.files)
        key = data_target.files[0]
        logger.info("Using the target {} from the targets npz file.".format(key))
        y = data_target[key].astype(np.float32)

    values_to_predict = y.shape[1]

    # Split data for test and training
    Z_train, Z_test, D_train, D_test, y_train, y_test = train_test_split(
            Z, D, y, test_size=1.0-train_split, random_state=0)

    Z_test, Z_val, D_test, D_val, y_test, y_val = train_test_split(
            Z_test, D_test, y_test, test_size=valid_test_split, random_state=0)

    # print([len(_) for _ in (y_train,y_val,y_test)])
    # Compute mean and standard deviation of per-atom-energy
    Z_train_non_zero = np.count_nonzero(Z_train, axis=1)
    Z_train_non_zero = np.expand_dims(Z_train_non_zero,axis=1)

    Estd = np.std(y_train / Z_train_non_zero, axis=0) # y values originally were free energies, they would be more when there are more atoms in the molecule, hence division scales them to be energy per atom.
    Emean = np.mean(y_train / Z_train_non_zero, axis=0) # axis needs to be specified so that we get mean and std per energy/spectrum value (i.e. dimension in y) doesn't affect when y just a scalar, i.e. free energy

    np.savez("X_vals.npz", Z_train=Z_train, Z_test=Z_test, Z_val=Z_val,D_train=D_train, D_test=D_test, D_val=D_val)
    np.savez("Y_vals.npz", Y_test=y_test, Y_train=y_train, Y_val=y_val, Y_mean=Emean, Y_std=Estd)

    return_dict = {
                "train_data": (Z_train, D_train, y_train),
                "valid_data": (Z_val, D_val, y_val),
                "test_data" : (Z_test, D_test, y_test),
                "Estd":Estd, "Emean":Emean, "num_species":num_species,
                "max_mol_size":max_mol_size,"values_to_predict": values_to_predict,
                "mu_max":mu_max,
                "num_species": num_species,
            }

    return return_dict

def main(**params):
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    parent_dir = os.getcwd()
    mydir = os.path.join(os.getcwd(),timestamp)
    os.makedirs(mydir)
    os.chdir(mydir)
    os.mkdir("results")
    
    app_name="DeepTensor"
    #global logger;
    # logger,fh,ch = get_logger(app_name=app_name, logfolder=mydir, fname="log_file_{}.log".format(timestamp))
    logger = get_logger(app_name=app_name, logfolder=mydir)
    logger.info("GOT LOGGER !")



    # save hyparameters
    np.savez("results/hyperparams.npz",  **params)
    data = get_data(logger=logger,**params)
    print("GOT DATA !")
    
    params_and_data = params.copy()
    params_and_data.update(data)
    
    error = train_and_get_error(logger=logger,**params_and_data)
    print("Test error :", error)
    
    #clean up
    os.chdir(parent_dir)
    return error

@click.command()
@click.option('--clen', default=30, help="The atomic descriptor 'C' used in Deep Tensor Neural Networks. Shutt et.al. 2017")
@click.option('--batch_size', default=100, help="The training batch size.")
@click.option('--num_neu_1', default=100, help="The number of neurons in the penultimate layer in DTNN. (Used only if model accepts this param)")
@click.option('--num_neu_2', default=200, help="The number of neurons in last layer in DTNN. (Used only if model accepts this param)")
@click.option('--model_name', default="orig_model", help="Name of the DTNN model")
@click.option('--learn_rate', type=click.FLOAT, default=None, help="The adam learning rate, if not set then a separate scheme is used.")
@click.option('--max_epochs', default=10000, help="Maximum number of training epochs.")
@click.option('--earlystop_epochs', default=100, help="If the validation error doesn't improve in these many epochs, the training is terminated.")
@click.argument('path_to_xyz_file')#,help="Path to the XYZ file.")
@click.argument('path_to_targets_file')#,help="Path to the energies or spectrum files.")
def get_params_and_goto_main(path_to_xyz_file, path_to_targets_file, model_name, 
        learn_rate=0.00001, earlystop_epochs=100, clen=30, numHiddenNeurons=60,
        numInteracPasses=2, numDistBasis=40, batch_size=100,
        trainSplit=0.9, valid_test_split=0.5, max_epochs=10000, cost="rmse",
        num_neu_1=100, num_neu_2=200):
    
    params = {
            "c_len":clen, 
            "num_hidden_neurons": numHiddenNeurons, 
            "num_interaction_passes":numInteracPasses,
            "num_dist_basis":numDistBasis, 
            "path_to_xyz_file":path_to_xyz_file, 
            "path_to_targets_file":path_to_targets_file, 
            "train_split" : trainSplit,
            "valid_test_split" : valid_test_split,
            "max_epochs" :  max_epochs,
            "batch_size" : batch_size,
            "cost" : cost,
            "model_name" : model_name,
            "learn_rate" : learn_rate,
            "earlystop_epochs" : earlystop_epochs,
            "num_neu_1" : num_neu_1,
            "num_neu_2" : num_neu_2
            }
    main(**params)



if __name__ == "__main__":
    get_params_and_goto_main() 
