import gc
import os
import sys
from os import path
import pdb
import cPickle as pickle
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
from dtnn_layers import SwitchLayer, MaskLayer, SumMaskedLayer, RecurrentLayer

theano.config.floatX = 'float32'


mydir = os.path.join(os.getcwd(),datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
os.makedirs(mydir)
os.chdir(mydir)
os.mkdir("results")

app_name=""
try:
    app_name = sys.argv[0]
except IndexError as e:
    app_name="DeepTensor"

logger = get_logger(app_name=app_name, logfolder=mydir)

path_to_xyz_file = ""
try:
    path_to_xyz_file = sys.argv[1]
except IndexError as e:
    logger.critical("PATH TO XYZ FILE NOT DEFINED !!!")

# The following targets file could be
# 16 properties or 300 values of the spectrum.
# The neural network output dimensions are determined from the
# second dimension of the targets tensor once loaded. 
# (The first dimension is the datapoints.)

try:
    path_to_targets_file = sys.argv[2]
    logger.info("Predicting targets {}".format(path_to_targets_file))
except IndexError as e :
    # Predict free energies instead.
    path_to_targets_file = None
    logger.info("Predicting free energies [No targets file provided].")

remove5koutliers = False
try:
    if "remove5koutliers" in sys.argv[3:]:
        remove5koutliers = True
        logger.info("REMOVING 5k outliers.")
except IndexError as e :
    pass
finally:
    if remove5koutliers is False:
        logger.info("NOT Removing 5k outliers.")

mae_cost = False
try:
    if "mae_cost" in sys.argv[3:]:
        mae_cost = True
        logger.info("Using MAE as the training and test cost.")
except IndexError as e:
    pass
finally:
    if mae_cost is False:
        logger.info("Using MSE as the training and test cost.")

num_dist_basis = 40
try:
    if "num_dist_basis" in sys.argv[3:]:
        # next entry is # of basis
        num_basis_idx = sys.argv.index("num_dist_basis")+1
        num_dist_basis = int(sys.argv[num_basis_idx])
except IndexError as e:
    # There are no [3:] indices in sys.argv
    pass
finally:
    logger.info("Number of distance basis: %d" % num_dist_basis)



def main():
    rng = np.random.RandomState(4)
    np.random.seed(1)

    # Define model parameters
    # num_dist_basis = 40 # defined at the top
    c_len = 30
    num_hidden_neurons = 60
    num_interaction_passes = 2
    values_to_predict = 1

    # Load data
    Z, D, y, num_species = load_qm7b_data(num_dist_basis, dtype=theano.config.floatX,
                                             xyz_file=path_to_xyz_file,expand_features=False)
    # NOTE!!!  D is not feature_expanded expand_features = False

    #Z, D, y, num_species = load_oqmd_data(num_dist_basis, dtype=theano.config.floatX, filter_query="natoms<10,computation=standard")
    max_mol_size = Z.shape[1]

    if path_to_targets_file is not None:
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

    if remove5koutliers:
        assert y.shape[1] == 16, "Y.shape[1] != 16. Remove 5k outliers only useful for energy files."
        from get_idxs_to_keep import get_idxs_to_keep
        idxs = get_idxs_to_keep(path_to_targets_file)
        Z = Z[idxs,:]
        D = D[idxs,:]
        y = y[idxs,:]

    # Split data for test and training
    Z_train, Z_test, D_train, D_test, y_train, y_test = train_test_split(
            Z, D, y, test_size=0.1, random_state=0)

    Z_test, Z_val, D_test, D_val, y_test, y_val = train_test_split(
            Z_test, D_test, y_test, test_size=0.5, random_state=0)

    print([len(_) for _ in (y_train,y_val,y_test)])
    # Compute mean and standard deviation of per-atom-energy
    Z_train_non_zero = np.count_nonzero(Z_train, axis=1)

    if path_to_targets_file is not None:
        Z_train_non_zero = np.expand_dims(Z_train_non_zero,axis=1)

    Estd = np.std(y_train / Z_train_non_zero, axis=0) # y values originally were free energies, they would be more when there are more atoms in the molecule, hence division scales them to be energy per atom.
    Emean = np.mean(y_train / Z_train_non_zero, axis=0) # axis needs to be specified so that we get mean and std per energy/spectrum value (i.e. dimension in y) doesn't affect when y just a scalar, i.e. free energy

    np.savez("X_vals.npz", Z_train=Z_train, Z_test=Z_test, Z_val=Z_val,D_train=D_train, D_test=D_test, D_val=D_val)
    np.savez("Y_vals.npz", Y_test=y_test, Y_train=y_train, Y_val=y_val, Y_mean=Emean, Y_std=Estd)

    sym_Z = T.imatrix()
    sym_D = T.tensor4()
    sym_y = T.vector()
    if path_to_targets_file is not None:
        sym_y = T.fmatrix()
    sym_learn_rate = T.scalar()

    l_in_Z = lasagne.layers.InputLayer((None, max_mol_size))
    l_in_D = lasagne.layers.InputLayer((None, max_mol_size, max_mol_size, num_dist_basis))
    l_mask = MaskLayer(l_in_Z)
    l_c0 = SwitchLayer(l_in_Z, num_species, c_len, W=lasagne.init.Uniform(1.0/np.sqrt(c_len)))

    l_cT = RecurrentLayer(l_c0, l_in_D, l_mask, num_passes=num_interaction_passes, num_hidden=num_hidden_neurons)

    # Compute energy contribution from each atom
    # l_atom1 = lasagne.layers.DenseLayer(l_cT, 15, nonlinearity=lasagne.nonlinearities.tanh, num_leading_axes=2) # outdim (-1, 23, 15)
    l_atom1 = lasagne.layers.DenseLayer(l_cT, 100, nonlinearity=lasagne.nonlinearities.tanh, num_leading_axes=2) # outdim (-1, 23, 15)
    l_atom1 = lasagne.layers.DenseLayer(l_atom1, 100, nonlinearity=lasagne.nonlinearities.tanh, num_leading_axes=2) # outdim (-1, 23, 15)
    l_atom1 = lasagne.layers.DenseLayer(l_atom1, 100, nonlinearity=lasagne.nonlinearities.tanh, num_leading_axes=2) # outdim (-1, 23, 15)
    l_atom2 = lasagne.layers.DenseLayer(l_atom1, values_to_predict, nonlinearity=None, num_leading_axes=2) # outdim (-1, 23, values_to_predict)
    if path_to_targets_file is None:
        l_atom2 = lasagne.layers.FlattenLayer(l_atom2, outdim=2) # Flatten singleton dimension # outdim (-1, 23)
        # but if path_to_targets is not None, then we don't want to flatten since we want to get outputs (energies, or spectrum values) for each atom: ie. we want outdim (-1, 23, values_to_predict)
    l_atomE = lasagne.layers.ExpressionLayer(l_atom2, lambda x: (x*Estd+Emean)) # Scale and shift by mean and std deviation
    if path_to_targets_file is not None:
        l_mask = lasagne.layers.ReshapeLayer(l_mask, ([0], [1], 1)) # add an extra dimension so that l_atomE (-1, 23, 16) l_mask "after reshape" (-1, 23, 1) can be multiplied
    l_out = SumMaskedLayer(l_atomE, l_mask) # TODO : BUG HERE.

    params = lasagne.layers.get_all_params(l_out, trainable=True)
    for p in params:
        logger.debug("%s, %s" % (p, p.get_value().shape))

    out_train = lasagne.layers.get_output(l_out, {l_in_Z: sym_Z, l_in_D: sym_D}, deterministic=False)
    out_test = lasagne.layers.get_output(l_out, {l_in_Z: sym_Z, l_in_D: sym_D}, deterministic=True)
    if mae_cost is True:
        cost_train = T.mean(np.abs(out_train-sym_y))
        cost_test = T.mean(np.abs(out_test-sym_y))
        logger.info("Used MAE cost")
    else:
        cost_train = T.mean(lasagne.objectives.squared_error(out_train, sym_y))
        cost_test = T.mean(lasagne.objectives.squared_error(out_test, sym_y))
        logger.info("Used MSE cost")


    updates = lasagne.updates.adam(cost_train, params, learning_rate=sym_learn_rate)


    f_train = theano.function(
            inputs = [sym_Z, sym_D, sym_y, sym_learn_rate],
            outputs = cost_train,
            updates = updates
            )

    f_eval_test = theano.function(
            inputs = [sym_Z, sym_D],
            outputs = out_test
            )

    f_test = theano.function(
            inputs = [sym_Z, sym_D, sym_y],
            outputs = cost_test,
            )

    # Define training parameters
    batch_size = 100
    num_train_samples = Z_train.shape[0]
    num_train_batches = num_train_samples // batch_size
    max_epochs=10000
    start_time = timeit.default_timer()

    lowest_test_mae = np.inf
    lowest_test_rmse = np.inf
    mu_max = None#np.max(D_train)+1
    # saving other variables for evaluating the model later.
    np.savez("results/hyperparams.npz", max_mol_size=max_mol_size, 
            values_to_predict=values_to_predict, Estd=Estd, Emean=Emean,
            c_len=c_len, num_hidden_neurons=num_hidden_neurons,
            num_interaction_passes=num_interaction_passes,
            num_species = num_species,
            num_dist_basis = num_dist_basis, mu_max=mu_max)

    D_val_fe = feature_expand(D_val, num_dist_basis, mu_max=mu_max)
    D_test_fe = feature_expand(D_test, num_dist_basis,mu_max=mu_max)

    for epoch in range(max_epochs):
        # Randomly permute training data
        rand_perm = rng.permutation(Z_train.shape[0])
        Z_train_perm = Z_train[rand_perm]
        D_train_perm = D_train[rand_perm]
        y_train_perm = y_train[rand_perm]

        if epoch < 100:
        #if epoch < 50:
            learning_rate = 0.01
        elif epoch < 500:
            learning_rate = 0.001
        elif epoch < 3000:
            learning_rate = 0.0001
        else:
            learning_rate = 0.00001

        train_cost = 0
        for batch in range(num_train_batches):
            train_cost += f_train(
                    Z_train_perm[batch*batch_size:((batch+1)*batch_size)],
                    feature_expand(D_train_perm[batch*batch_size:((batch+1)*batch_size)], num_dist_basis, mu_max=mu_max),
                    y_train_perm[batch*batch_size:((batch+1)*batch_size)],
                    learning_rate
                    )
            #print("miniBatch %d of %d done." % (batch, num_train_batches))
        train_cost = train_cost / num_train_batches

        if (epoch % 2) == 0:
            # D_train_fe = feature_expand(D_train, num_dist_basis)
            # y_pred = f_eval_test(Z_train, D_train_fe)
            # train_errors = y_pred-y_train
            # del D_train_fe
            # gc.collect()

            y_pred = f_eval_test(Z_val, D_val_fe)
            val_errors = y_pred-y_val
            # del D_val_fe
            # gc.collect()
            train_errors = val_errors 

            y_pred = f_eval_test(Z_test, D_test_fe)
            test_errors = y_pred-y_test
            test_cost = f_test(Z_test, D_test_fe, y_test)
            # del D_test_fe
            # gc.collect()

            #logger.info("TRAIN MAE:  %5.2f kcal/mol TEST MAE:  %5.2f kcal/mol" %
            logger.info("VAL MAE:  %5.2f kcal/mol TEST MAE:  %5.2f kcal/mol" %
                    (np.abs(train_errors).mean(), np.abs(test_errors).mean()))
            #logger.info("TRAIN RMSE: %5.2f kcal/mol TEST RMSE: %5.2f kcal/mol" %
            logger.info("VAL RMSE: %5.2f kcal/mol TEST RMSE: %5.2f kcal/mol" %
                    (np.sqrt(np.square(train_errors).mean()), np.sqrt(np.square(test_errors).mean())))

            all_params = lasagne.layers.get_all_param_values(l_out)
            with gzip.open('results/model_epoch%d.pkl.gz' % (epoch), 'wb') as f:
                pickle.dump(all_params, f, protocol=pickle.HIGHEST_PROTOCOL)
            new_test_mae = np.abs(test_errors).mean()
            if  new_test_mae < lowest_test_mae:
                lowest_test_mae = new_test_mae
                logger.info("Found best test MAE : {}".format(lowest_test_mae))
                np.savez("Y_test_pred_best_mae.npz", Y_test_pred = y_pred)

            new_test_rmse = np.sqrt(np.square(test_errors).mean())
            if new_test_rmse < lowest_test_rmse:
                lowest_test_rmse = new_test_rmse
                logger.info("Found best test RMSE : {}".format(lowest_test_rmse))
                np.savez("Y_test_pred_best_rmse.npz", Y_test_pred = y_pred)

        #if (epoch % 2) == 0:
            # test_cost = f_test(Z_test, D_test, y_test)
            end_time = timeit.default_timer()

            logger.debug("Time %4.1f, Epoch %4d, train_cost=%5g, test_error=%5g" % (end_time - start_time, epoch, np.sqrt(train_cost), np.sqrt(test_cost)))
            start_time = timeit.default_timer()

    print("Execution complete. Save the Y values")

if __name__ == "__main__":
    main()
