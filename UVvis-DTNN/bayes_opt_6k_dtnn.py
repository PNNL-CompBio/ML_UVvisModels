import pprint
import numpy as np
from numpy.random import seed
import GPy
import GPyOpt
from train import main
import matplotlib
matplotlib.use('Agg')

data_path = "/m/home/home0/00/ghoshk1/data/Desktop/Thesis/data/Annika_new/5k_16_opt_eV"
path_to_xyz_file = data_path + "/data.xyz"
path_to_targets_file = data_path + "/energies.txt"
max_epochs = 500
model_name = "model_neu1_neu2_with_noise"

def optimize():
    bounds = [
            {'name':'c_len', 'type':'discrete', 'domain':range(20,41)}, # 20
            {'name':'learn_rate', 'type': 'discrete', 'domain':range(2,7)}, # 5 
            {'name':'num_neu_1', 'type':'discrete', 'domain': np.arange(100, 601 ,50)}, # 11
            {'name':'num_neu_2', 'type':'discrete', 'domain': np.arange(100, 601 ,50)}, # 11
            {'name':'batch_size', 'type':'discrete', 'domain': np.arange(20, 201, 5)},  # 37
            ]

    def wrapper_function(gpy_input):
        print("Params : {}".format(gpy_input))
        c_len, learn_rate, num_neu_1, num_neu_2, batch_size = np.int32(gpy_input[0])
        learn_rate = np.float32(10.0**(-learn_rate))

        params = {
        "c_len":c_len, 
        "num_hidden_neurons": 60, 
        "num_interaction_passes":2,
        "num_dist_basis":40, 
        "path_to_xyz_file":path_to_xyz_file, 
        "path_to_targets_file":path_to_targets_file, 
        "train_split" : 0.9,
        "valid_test_split" : 0.5,
        "max_epochs" :  max_epochs,
        "batch_size" : batch_size,
        "cost" : "rmse",
        "model_name" : model_name,
        "learn_rate" : learn_rate,
        "earlystop_epochs" : 100,
        "num_neu_1" : num_neu_1,
        "num_neu_2" : num_neu_2
        }
        pprint.pprint(params)
        error = main(**params)
        return error 

    bayes_opt = GPyOpt.methods.BayesianOptimization(wrapper_function,
            domain = bounds,
            acquisition_type='EI',
            exact_feval=False
            )
    bayes_opt.run_optimization(max_iter=10)
    print("Suggested sample: {}".format(bayes_opt.suggested_sample))
    bayes_opt.plot_acquisition("acquisition_200epoch.png")
    bayes_opt.plot_convergence("convergence_200epoch.png")

if __name__ == "__main__":
    seed(1234)
    optimize()
