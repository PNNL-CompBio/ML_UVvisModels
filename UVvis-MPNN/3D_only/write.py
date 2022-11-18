import rdkit
from rdkit import Chem

data = open('UV_chemprop_Data.csv', 'r')

for lines in data.readlines():
	line = lines.split(',')
	smiles = line[0]#.rstrip()
	print(smiles, Chem.MolFromSmiles(smiles))
