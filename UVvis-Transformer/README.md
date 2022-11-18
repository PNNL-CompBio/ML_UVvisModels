# SolTranNet Paper
The data sets and scripts used to generate the figures for [SolTranNet](https://pubs.acs.org/doi/10.1021/acs.jcim.1c00331).

SolTranNet is a fork of the Molecule Attention Transformer, whose implementation can be found [here](https://https://github.com/gnina/SolTranNet).

## Requirements
 - PyTorch 1.7 -- compiled with CUDA
 - pandas 0.25.3+
 - RDKit 2020.03.1dev1
 - CUDA 10.1 or 10.2

## Recreating the Figures for the Paper

1) We have provided the predictions that we used to generate the figures in soltrannet_data.tar.gz

NOTE -- we provided 2 versions of each training data file -- 1 for 2D and 1 for 3D. They are identical in content except for the name. This is because during training, the first time a datafile is loaded, the resulting embeddings get saved, which will differ if a 2D or 3D conformer is used to generate the embedding.

```
tar -xzf soltrannet_data.tar.gz
```

2) After extracting, you can follow along with Making_figures.ipynb  for the code we used to generate our Figures and Tables in the paper.

## Re-training Linear ML models

1) Install the qsar-tools repository from [here](https://github.com/dkoes/qsar-tools)

2) After qsar-tools is installed, use the trainlinearmodel.py script there to train a new version of the model.

NOTE -- we trained LASSO, elastic, PLS, and ridge models and have provided the requisite fingerprints for each fold of our scaffold-CCV split of AqSolDB, the full AqSolDB, and our Independent Set.

```
python3 $QSARTOOLSDIR/trainlinearmodel.py -o data/training_data/linear_ml/full_aqsol_rdkit2048fp.model --lasso --maxiter 100000 data/training_data/linear_ml/full_aqsol_rdkit2048fp.gz
```

3) After that model has finished training, you can use it to produce a predictions file:

```
python3 $QSARTOOLSDIR/applylinearmodel.py data/training_data/linear_ml/full_aqsol_rdkit2048fp.model data/training_data/linear_ml/independent_sol_rdkit2048fp.gz > data/training_data/linear_ml/full_aqsol_lasso_rdkit2048fp_ind_test.predictions
```

## Recreating our SolTranNet sweeps

Our sweeps assume that you have compiled pytorch with CUDA enabled, and have a GPU to utilize.

1) We provide a way to generate jobs for a grid search with write_jobs.py. For the CCV splits, each fold would be run through the script as shown below. 

NOTE -- The default arguments will recreate our architecture sweep. See the --help options to explore all available options to you. If you want a 3D version, run again but leave out the --twod option.

NOTE -- If you want to train a full model, use the --trainfile and --testfile options to write_jobs.py. This will change your output from training. Be careful not to overwrite!

```
python3 write_jobs.py --prefix data/training_data/aqsol_scaf_2d --fold 0 --twod --outname grid_search_2D_ccv0_training.cmds
```

NOTE -- if you have a weights and biases account, you can setup a sweep there and pass it as an argument into write_jobs.py to log the results there as well.

2) Each line of the output of (1) will result in training 1 version of SolTranNet using train.py.

NOTE -- predict.py REQUIRES the model name of the saved model from (1) in order to function properly! DO NOT CHANGE THE NAMES OF THE MODELS

```
python3 train.py --help
```

As an example, we will asssume that the following line going forward.

```
python3 train.py --prefix data/training_data/aqsol_scaf_2d --fold 0 --datadir sweep --epochs 100 --lr 0.04 --loss huber --dropout 0 --ldist 0 --lattn 0.25 --Ndense 1 --heads 16 --dmodel 1024 --nstacklayers 16 --seed 420 --dynamic 0 --twod
```

3) After training is complete, you will need to run predict.py with the corresponding test set file, and trained model weights. The model weights will be saved where you specify with --datadir in (2).

To see all of the options available to you, run the following:

```
python3 predict.py --help
```

Continuing our example from earlier, there should be a sweep directory with some files in it as the result of training. The weights file is the .model file in said directory, with the following format:
```
{datadir}/{prefix}_{fold}_drop{dropout}_ldist{ldist}_lattn{lattn}_Ndense{Ndense}_heads{heads}_dmodel{dmodel}_nsl{nstacklayers}_epochs{epochs}_dyn{dynamic}_seed{seed}_trained.model
```

NOTE -- If your training job utilized the --twod option, you MUST pass it again here.

```
python3 predict.py -m sweep/aqsol_scaf_2d_0_drop0_ldist0_lattn0.25_Ndense1_heads16_dmodel1024_nsl16_epochs100_dyn0_seed420_trained.model -i data/training_data/aqsol_scaf_2d_test0.csv -o my_model_fold0.predictions --twod --stats
```

This will output a predictions file, as well as display the Person R-squared and RMSE of the predicted values.
