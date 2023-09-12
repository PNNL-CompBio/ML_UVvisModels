# UVvis-MPNN Scripts

This directory contains the files necessary to run the 4 variations of the UVvis-MPNN model. They are broken into the following:

1. 3D_distance_plus_spectra :: Model that was trained on 3D distances between atoms and DFT UV spectra as features
2. 3D_only :: Model that was trained on 3D distances between atoms as features
3. original :: Model that was trained only on the 2D SMILES strings as features
4. original_plus_spectra :: Model that was trained on 2D SMILES and DFT UV spectra as features

The pre-trained weights for these models are located in their respective directories. Based on the architecture of `Chemprop`, 10 model weights are ensembled and used for prediction.

The method of prediction is straight forward, once in the respective models main folder:

```bash
python predict.py --test_path smiles.csv --features_path features.csv --checkpoint_dir model_checkpoints --preds_path uv_preds.csv
```

**Note**: This `--features_path` is for the models that incorporate DFT spectra as a training feature (`_plus_spectra` models). It is not explicity needed, but may improve prediction results.

The SMILES input, `smiles.csv` should be a csv list of Strings with the header: `smiles`. The Features input `features.csv` should be csv list of absorption values corresponding to the provided SMILES with the header being some index corresponding to the wavelength (i.e, 220->400). 


This input organization follows the training process, which takes a csv of smiles and absorption values and then splits them into 2 separate files.

```
smiles, 1, 2, 3, ..., 181
CCC, 0.5, 0.3, 0.2, ..., 1
```

and after splitting out the smiles column:

```console
total_data.csv
├── smiles.csv
└── features.csv
```

## Scaling

After running the inference method of the UVvis-MPNN, you will probably need one last scaling of the values between 0-1. There is a provided script `spectra_scaling.py` to run a min-max scaling over the `uv_preds.csv` file. This will ensure smooth plotting and interpretable results.