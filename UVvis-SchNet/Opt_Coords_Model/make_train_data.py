"""Utilities associated with using SchNetPack"""

import os
import pickle as pkl
import re
import tempfile
from random import shuffle

import numpy as np
from ase.io.extxyz import read_xyz
from ase.units import Debye, Bohr, Hartree
from schnetpack.data import AtomsData

A = "rotational_constant_A"
B = "rotational_constant_B"
C = "rotational_constant_C"
mu = "dipole_moment"
alpha = "isotropic_polarizability"
homo = "homo"
lumo = "lumo"
gap = "gap"
r2 = "electronic_spatial_extent"
zpve = "zpve"
U0 = "energy_U0"
U = "energy_U"
H = "enthalpy_H"
G = "free_energy"
Cv = "heat_capacity"
g4mp2_0k = 'g4mp2_0k'
u0 = 'u0'

available_properties = [
            A,
            B,
            C,
            mu,
            alpha,
            homo,
            lumo,
            gap,
            r2,
            zpve,
            U0,
            U,
            H,
            G,
            Cv,
            g4mp2_0k,
            u0

        ]

units = [
            1.0,
            1.0,
            1.0,
            Debye,
            Bohr ** 3,
            Hartree,
            Hartree,
            Hartree,
            Bohr ** 2,
            Hartree,
            Hartree,
            Hartree,
            Hartree,
            Hartree,
            1.0
        ]
units_dict = dict(zip(available_properties, units))

def make_schnetpack_data(dbpath,overwrite=True):
    """Convert a Pandas dictionary to a SchNet database
    Args:
        dataset (pd.DataFrame): Dataset to convert
        dbpath (string): Path to database to be saved
        properties ([string]): List of properties to include in the dataset
        conformers (str): Name of column with conformers as xyz
        xyz_col (string): Name of the column with the XYZ data
        overwrite (True): Whether to overwrite the database
    """

    # If needed, delete the previous database
    raw_path = '/qfs/projects/MulCME/Rajendra/darpa/MMPI_set3/TRAIN_TEST_VALIDATION/TRAIN_TEST_DATA/final_training_set'
    ordered_files = sorted(
        os.listdir(raw_path),key=lambda x: (int(re.sub("\D", "", x)), x))

    shuffle(ordered_files)
    # print(ordered_files)

    all_atoms = []
    all_properties = []

    irange = np.arange(len(ordered_files), dtype=np.int)
    tmpdir = tempfile.mkdtemp("gdb9")

    for i in irange:
        xyzfile = os.path.join(raw_path, ordered_files[i])
        properties = {}
        tmp = os.path.join(tmpdir, "tmp.xyz")
        print(xyzfile)
        with open(xyzfile, "r") as f:
            lines = f.readlines()
            l = lines[1].split()[2:]
            for pn, p in zip(available_properties, l):
                if pn == 'g4mp2_0k' or pn == 'u0':
                    continue
                else:
                    properties[pn] = np.array([float(p) * units_dict[pn]])
            u0 = lines[-1].strip().split("\t")[0]
            g4mp2_0k = lines[-2].strip().split("\t")[0]
            u0 = [float(i) for i in u0.split()]
            g4mp2_0k = [float(i) for i in g4mp2_0k.split()]
            properties['g4mp2_0k'] = np.asarray(g4mp2_0k, np.float32)
            properties['u0'] = np.asarray(u0, np.float32)
            with open(tmp, "wt") as fout:
                for line in lines:
                    fout.write(line.replace("*^", "e"))

        with open(tmp, "r") as f:
            ats = list(read_xyz(f, 0))[0]
        all_atoms.append(ats)
        all_properties.append(properties)
    if os.path.exists(dbpath) and overwrite:
        os.unlink(dbpath)

    # Convert all entries to ase.Atoms objects
    # atoms = dataset[xyz_col].apply(lambda x: read_xyz(StringIO(x)).__next__())
    #
    # # Every column besides the training set will be a property
    # prop_cols = set(properties).difference([xyz_col])
    # property_list = [dict(zip(prop_cols, [np.atleast_1d(row[p]) for p in prop_cols])) for i, row in
    #                  dataset.iterrows()]
    #
    # # Add conformers to the property list, but it isn't a required property when loading entries
    # if conformers is not None:
        #     for d, c in zip(property_list, dataset[conformers]):
    #         d['conformers'] = np.atleast_1d(c)

    # Initialize the object
    db = AtomsData(dbpath, required_properties=available_properties)

    # Add every system to the db object
    db.add_systems(all_atoms, all_properties)
    return db


if __name__ == '__main__':
    db = make_schnetpack_data('train_dataset.db')
    with open('train_dataset.pkl', 'wb') as fp:
        pkl.dump(db, fp)
