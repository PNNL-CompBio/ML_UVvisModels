from rdkit import Chem
from rdkit.Chem import AllChem
import torch


data = open('./UV_chemprop_Data.csv', 'r')

for lines in data.readlines():
	smile = lines.split(',')[0].rstrip()
	try:
		mol = Chem.MolFromSmiles(smile)
		AllChem.EmbedMolecule(mol)
		mol = Chem.AddHs(mol)
		conf = mol.GetConformer()
		print(lines.rstrip())
	except:
		pass
