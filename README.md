# Machine Learning Models for the Prediction of UV-Visible Spectra

This repository contains code for our recent [paper](https://doi.org/10.1021/acs.jcim.2c01662).

In the current work, we performed landscape assessment on computational tools to devise a method for predicting accurate UV-Vis spectra to accelerate the hit identification in drug design. Below is the representation of the UVvis-MPNN model, one of the best model. The inputs to the model are the 3D optimized coordinates for the molecule, the QM predicted spectra, and the SMILES of the molecule to train on. The output of the model is the most accurate prediction of the UV spectra.

![image-1.png](./image-1.png)
This repository contains 4 models for the purposes of predicting UV-Visible spectra. Each model uses compounds with Quantum mechanics predicted and experimentally measured UV-Vis spectra to train. Each model is adapted from a previously existing ML framework for learning patterns from molecules and predicted QM properties.

1. UVvis-DTNN, adapted from [Ghosh, et al.](https://doi.org/10.1002/advs.201801367)
2. UVvis-MPNN, adapted from [chemprop-ir](https://github.com/gfm-collab/chemprop-IR)
3. UVvis-SchNet adapted from [SchNet Delta](https://github.com/globus-labs/g4mp2-atomization-energy)
4. UVvis-Transformer, adapted from [SolTranNet](https://github.com/gnina/SolTranNet)

Each model has it's own training workflow, therefore follow the README.md located in each main method directory can be used to run the model. 

# References

[1]Ghosh, K.; Stuke, A.; Todorović, M.; Jørgensen, P. B.; Schmidt, M. N.; Vehtari, A.;
Rinke, P. Deep Learning Spectroscopy: Neural Networks for Molecular Excitation Spec-
tra. Advanced Science 2019, 6, 1801367.

[2] McGill, C.; Forsuelo, M.; Guan, Y.; Green, W. H. Predicting Infrared Spectra with Mes-
sage Passing Neural Networks. Journal of Chemical Information and Modeling 2021,
18

[3]Ward, L.; Blaiszik, B.; Foster, I.; Assary, R. S.; Narayanan, B.; Curtiss, L. Machine
Learning Prediction of Accurate Atomization Energies of Organic Molecules from Low-
Fidelity Quantum Chemical Calculations. 2019,

[4]Francoeur, P. G.; Koes, D. R. SolTranNet–A Machine Learning Tool for Fast Aqueous
Solubility Prediction. Journal of Chemical Information and Modeling 2021, 61, 2530–
2536.

# How to Cite:

```bibtex
@article{McNaughton2023,
author = {McNaughton, Andrew D. and Joshi, Rajendra P. and Knutson, Carter R. and Fnu, Anubhav and Luebke, Kevin J. and Malerich, Jeremiah P. and Madrid, Peter B. and Kumar, Neeraj},
doi = {10.1021/acs.jcim.2c01662},
file = {:Users/mcna892/Library/Application Support/Mendeley Desktop/Downloaded/McNaughton et al. - 2023 - Machine Learning Models for Predicting Molecular UV–Vis Spectra with Quantum Mechanical Properties.pdf:pdf},
issn = {1549-9596},
journal = {Journal of Chemical Information and Modeling},
month = {mar},
number = {5},
pages = {1462--1471},
title = {{Machine Learning Models for Predicting Molecular UV–Vis Spectra with Quantum Mechanical Properties}},
url = {https://pubs.acs.org/doi/10.1021/acs.jcim.2c01662},
volume = {63},
year = {2023}
}
```
