import os
import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
import logging
import numpy as np
import lasagne
import theano.tensor as T
import theano
from dtnn_layers import SwitchLayer, MaskLayer, SumMaskedLayer, RecurrentLayer

logger = logging.getLogger(__name__)

def model_neu1_neu2(Emean, Estd, max_mol_size, num_dist_basis, c_len, num_species,
        num_interaction_passes, num_hidden_neurons, values_to_predict,cost, num_neu_1, num_neu_2,**kwargs):

    # path to targets_file is not NONE
    sym_Z = T.imatrix()
    sym_D = T.tensor4()
    sym_y = T.fmatrix()
    sym_learn_rate = T.scalar()

    l_in_Z = lasagne.layers.InputLayer((None, max_mol_size))
    l_in_D = lasagne.layers.InputLayer((None, max_mol_size, max_mol_size, num_dist_basis))
    l_mask = MaskLayer(l_in_Z)
    l_c0 = SwitchLayer(l_in_Z, num_species, c_len, W=lasagne.init.Uniform(1.0/np.sqrt(c_len)))

    l_cT = RecurrentLayer(l_c0, l_in_D, l_mask, num_passes=num_interaction_passes, num_hidden=num_hidden_neurons)

    # Compute energy contribution from each atom
    # l_atom1 = lasagne.layers.DenseLayer(l_cT, 15, nonlinearity=lasagne.nonlinearities.tanh, num_leading_axes=2) # outdim (-1, 23, 15)
    l_atom1 = lasagne.layers.DenseLayer(l_cT, num_neu_1, nonlinearity=lasagne.nonlinearities.tanh, num_leading_axes=2) # outdim (-1, 23, 15)
    l_atom1 = lasagne.layers.DenseLayer(l_atom1, num_neu_2, nonlinearity=lasagne.nonlinearities.tanh, num_leading_axes=2) # outdim (-1, 23, 15)
    l_atom2 = lasagne.layers.DenseLayer(l_atom1, values_to_predict, nonlinearity=None, num_leading_axes=2) # outdim (-1, 23, values_to_predict)
    l_atomE = lasagne.layers.ExpressionLayer(l_atom2, lambda x: (x*Estd+Emean)) # Scale and shift by mean and std deviation
    l_mask = lasagne.layers.ReshapeLayer(l_mask, ([0], [1], 1)) # add an extra dimension so that l_atomE (-1, 23, 16) l_mask "after reshape" (-1, 23, 1) can be multiplied
    l_out = SumMaskedLayer(l_atomE, l_mask)

    params = lasagne.layers.get_all_params(l_out, trainable=True)
    for p in params:
        logger.debug("%s, %s" % (p, p.get_value().shape))

    out_train = lasagne.layers.get_output(l_out, {l_in_Z: sym_Z, l_in_D: sym_D}, deterministic=False)
    out_test = lasagne.layers.get_output(l_out, {l_in_Z: sym_Z, l_in_D: sym_D}, deterministic=True)
    if cost == "mae":
        cost_train = T.mean(np.abs(out_train-sym_y))
        cost_test = T.mean(np.abs(out_test-sym_y))
        logger.info("Used MAE cost")
    elif cost == "rmse":
        cost_train = T.mean(lasagne.objectives.squared_error(out_train, sym_y))
        cost_test = T.mean(lasagne.objectives.squared_error(out_test, sym_y))
        logger.info("Used MSE cost")
    else:
        raise ValueError("unknown cost function {}".format(cost))


    updates = lasagne.updates.adam(cost_train, params, learning_rate=sym_learn_rate)


    f_train = theano.function(
            inputs = [sym_Z, sym_D, sym_y, sym_learn_rate],
            outputs = cost_train,
            updates = updates
            )

    os.environ["THEANO_FLAGS"]="device=gpu"
    f_eval_test = theano.function(
            inputs = [sym_Z, sym_D],
            outputs = out_test
            )

    f_test = theano.function(
            inputs = [sym_Z, sym_D, sym_y],
            outputs = cost_test,
            )

    return f_train, f_eval_test, f_test, l_out

def model_neu1_neu2_with_noise(Emean, Estd, max_mol_size, num_dist_basis, c_len, num_species,
        num_interaction_passes, num_hidden_neurons, values_to_predict,cost, num_neu_1, num_neu_2, noise_std=0.1, **kwargs):

    # path to targets_file is not NONE
    sym_Z = T.imatrix()
    sym_D = T.tensor4()
    sym_y = T.fmatrix()
    sym_learn_rate = T.scalar()

    l_in_Z = lasagne.layers.InputLayer((None, max_mol_size))
    l_in_D = lasagne.layers.InputLayer((None, max_mol_size, max_mol_size, num_dist_basis))
    l_in_D = lasagne.layers.GaussianNoiseLayer(l_in_D, sigma=noise_std)
    l_mask = MaskLayer(l_in_Z)
    l_c0 = SwitchLayer(l_in_Z, num_species, c_len, W=lasagne.init.Uniform(1.0/np.sqrt(c_len)))

    l_cT = RecurrentLayer(l_c0, l_in_D, l_mask, num_passes=num_interaction_passes, num_hidden=num_hidden_neurons)

    # Compute energy contribution from each atom
    # l_atom1 = lasagne.layers.DenseLayer(l_cT, 15, nonlinearity=lasagne.nonlinearities.tanh, num_leading_axes=2) # outdim (-1, 23, 15)
    l_atom1 = lasagne.layers.DenseLayer(l_cT, num_neu_1, nonlinearity=lasagne.nonlinearities.tanh, num_leading_axes=2) # outdim (-1, 23, 15)
    l_atom1 = lasagne.layers.DenseLayer(l_atom1, num_neu_2, nonlinearity=lasagne.nonlinearities.tanh, num_leading_axes=2) # outdim (-1, 23, 15)
    l_atom2 = lasagne.layers.DenseLayer(l_atom1, values_to_predict, nonlinearity=None, num_leading_axes=2) # outdim (-1, 23, values_to_predict)
    l_atomE = lasagne.layers.ExpressionLayer(l_atom2, lambda x: (x*Estd+Emean)) # Scale and shift by mean and std deviation
    l_mask = lasagne.layers.ReshapeLayer(l_mask, ([0], [1], 1)) # add an extra dimension so that l_atomE (-1, 23, 16) l_mask "after reshape" (-1, 23, 1) can be multiplied
    l_out = SumMaskedLayer(l_atomE, l_mask)

    params = lasagne.layers.get_all_params(l_out, trainable=True)
    for p in params:
        logger.debug("%s, %s" % (p, p.get_value().shape))

    out_train = lasagne.layers.get_output(l_out, {l_in_Z: sym_Z, l_in_D: sym_D}, deterministic=False)
    out_test = lasagne.layers.get_output(l_out, {l_in_Z: sym_Z, l_in_D: sym_D}, deterministic=True)
    if cost == "mae":
        cost_train = T.mean(np.abs(out_train-sym_y))
        cost_test = T.mean(np.abs(out_test-sym_y))
        logger.info("Used MAE cost")
    elif cost == "rmse":
        cost_train = T.mean(lasagne.objectives.squared_error(out_train, sym_y))
        cost_test = T.mean(lasagne.objectives.squared_error(out_test, sym_y))
        logger.info("Used MSE cost")
    else:
        raise ValueError("unknown cost function {}".format(cost))


    updates = lasagne.updates.adam(cost_train, params, learning_rate=sym_learn_rate)


    f_train = theano.function(
            inputs = [sym_Z, sym_D, sym_y, sym_learn_rate],
            outputs = cost_train,
            updates = updates
            )

    os.environ["THEANO_FLAGS"]="device=gpu"
    f_eval_test = theano.function(
            inputs = [sym_Z, sym_D],
            outputs = out_test
            )

    f_test = theano.function(
            inputs = [sym_Z, sym_D, sym_y],
            outputs = cost_test,
            )

    return f_train, f_eval_test, f_test, l_out



if __name__ == "__main__":
    # write tests here
    pass
