import re
import numpy as np
import itertools
from ase.neighborlist import NeighborList
from elements import ELEMENTS

class Molecule():
    def __init__(self, num_atoms):
        self.energy = None
        self.z = np.zeros(num_atoms, dtype=np.int32)
        self.coord = np.zeros((num_atoms, 3))

def load_xyz_file(filename):
    STATE_READ_NUMBER = 0
    STATE_READ_COMMENT = 1
    STATE_READ_ENTRY = 2

    numeric_const_pattern = r"""
        [-+]? # optional sign
        (?:
            (?: \d* \. \d+ ) # .1 .12 .123 etc 9.1 etc 98.1 etc
            |
            (?: \d+ \.? ) # 1. 12. 123. etc 1 12 123 etc
        )
        # followed by optional exponent part if desired
        (?: [Ee] [+-]? \d+ ) ?
        """

    rx = re.compile(numeric_const_pattern, re.VERBOSE)

    state = STATE_READ_NUMBER
    entries_read = 0
    descriptors = []
    cur_desc = None

    with open(filename, "r") as f:
        for line_no, line in enumerate(f):
            if state == STATE_READ_NUMBER:
                entries_to_read = int(line)
                cur_desc = Molecule(entries_to_read)
                entries_read = 0
                state = STATE_READ_COMMENT
            elif state == STATE_READ_COMMENT:
                floats = rx.findall(line)
                if len(floats) > 0:
                    cur_desc.energy = floats[0]
                else:
                    cur_desc.energy = None
                state = STATE_READ_ENTRY
            elif state == STATE_READ_ENTRY:
                parts = line.split()
                assert len(parts) == 4
                atom = parts[0]
                el = ELEMENTS[atom]
                floats = map(float, parts[1:])
                cur_desc.coord[entries_read,:] = np.array(floats)
                cur_desc.z[entries_read] = el.number
                entries_read += 1
                if entries_read == cur_desc.z.size:
                    descriptors.append(cur_desc)
                    state = STATE_READ_NUMBER
            else:
                raise Exception("Invalid state")
    return descriptors

def load_oqmd_data(num_dist_basis, dtype=float, cutoff=2.0, filter_query=None):
    """ Returns tuple (Z, D, y, num_species)
    where
        Z is a matrix, each row in Z corresponds to a molecule and the elements
        are atomic numbers
        D is a 4D tensor giving the Gaussian feature expansion of distances
            between atoms
        y is the energy of each molecule in kcal/mol
        num_species is the integer number of different atomic species in Z """
    import ase.db
    con = ase.db.connect("oqmd_all_entries.db")
    sel = filter_query

    # Create sorted list of species
    all_species = sorted(list(reduce(lambda x,y: x.union(set(y.numbers)), con.select(sel), set())))
    # Insert dummy species 0
    all_species.insert(0,0)
    # Create mapping from species to species index
    species_map = dict([(x,i) for i,x in enumerate(all_species)])

    # Find max number of atoms
    max_atoms = 0
    num_molecules = 0
    for row in con.select(sel):
        num_molecules += 1
        atoms = row.toatoms()
        max_atoms = max(max_atoms, len(atoms))

    # Gaussian basis function parameters
    mu_min = -1
    mu_max = 2*cutoff+1
    step = (mu_max-mu_min)/num_dist_basis
    sigma = step
    print "Number of molecules %d, max atoms %d" % (num_molecules, max_atoms)
    k = np.arange(num_dist_basis)
    D = np.zeros((num_molecules, max_atoms, max_atoms, num_dist_basis), dtype=dtype)
    Z = np.zeros((num_molecules, max_atoms), dtype=np.int8)
    y = np.zeros(num_molecules, dtype=dtype)

    for i_mol, row in enumerate(con.select(sel)):
        print "feature expansion %10d/%10d\r" %(i_mol+1, num_molecules),
        atoms = row.toatoms()
        y[i_mol] = row.total_energy
        Z[i_mol,0:len(row.numbers)] = map(lambda x: species_map[x], row.numbers)
        assert np.all(atoms.get_atomic_numbers() == row.numbers)
        neighborhood = NeighborList([cutoff]*len(atoms), self_interaction=False, bothways=False)
        neighborhood.build(atoms)
        for ii in range(len(atoms)):
            neighbor_indices, offset = neighborhood.get_neighbors(ii)
            for jj, offs in zip(neighbor_indices, offset):
                ii_pos = atoms.positions[ii]
                jj_pos = atoms.positions[jj] + np.dot(offs, atoms.get_cell())
                dist = np.linalg.norm(ii_pos-jj_pos)
                d_expand = np.exp( -((dist-(mu_min+k*step))**2.0)/(2.0*sigma**2.0))
                D[i_mol, ii, jj, :] += d_expand
                if jj != ii:
                    D[i_mol, jj, ii, :] += d_expand
    return Z, D, y, len(all_species)

def feature_expand(X, num_basis,mu_max=None):
    """ Expand distance matrices in uniform grid of Gaussians """

    mu_min = -1
    step = 0.2
    sigma = step
    if mu_max is not None:
        step = (mu_max - mu_min)/num_basis
        sigma = step

    #print("feature_expand step = ", step)
    k = np.arange(num_basis, dtype=X.dtype)

    X = np.expand_dims(X, -1)

    return np.exp( -((X-(mu_min+k*step))**2.0)/(2.0*sigma**2.0))

def load_qm7b_data(num_dist_basis, dtype=float, xyz_file="../qm7b.xyz", expand_features=True):
    """ Returns tuple (Z, D, y, num_species)
    where
        Z is a matrix, each row in Z corresponds to a molecule and the elements
        are atomic numbers
        D is a 4D tensor giving the Gaussian feature expansion of distances
            between atoms
        y is the energy of each molecule in kcal/mol
        num_species is the integer number of different atomic species in Z """
    # Load xyz file
    ## descriptors = load_xyz_file(xyz_file)
    ## xyz_file = "/u/00/ghoshk1/unix/Desktop/Thesis/data/5k_tmp/QMnew_16.xyz"
    descriptors = load_xyz_file(xyz_file)

    # Find max number of atoms
    max_atoms = max([len(d.z) for d in descriptors])
    # Create sorted list of species
    all_species = sorted(list(reduce(lambda x,y: x.union(set(y.z)), descriptors, set())))
    # Insert dummy species 0
    all_species.insert(0,0)
    # Create mapping from species to species index
    species_map = dict([(x,i) for i,x in enumerate(all_species)])

    Z = np.zeros((len(descriptors), max_atoms), dtype = np.int8)
    D = np.zeros((len(descriptors), max_atoms, max_atoms), dtype = dtype)
    y = np.zeros(len(descriptors), dtype = dtype)

    for i, desc in enumerate(descriptors):
        nz = np.size(desc.z)
        Z[i,0:nz] = map(lambda x: species_map[x], desc.z)
        y[i] = desc.energy
        for (ii, jj) in itertools.combinations(range(nz), 2):
            dist = np.linalg.norm(desc.coord[ii]-desc.coord[jj])
            D[i,ii,jj] = D[i,jj,ii] = dist

    return Z, feature_expand(D, num_dist_basis) if expand_features is True else D,\
            y, len(all_species)

if __name__ == "__main__":
    # desc = load_xyz_file("qm7b.xyz")
    # for d in desc:
    #     print d.energy, d.z, d.coord
    load_oqmd_data(filter_query="natoms<10,computation=standard")
