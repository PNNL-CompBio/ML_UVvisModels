"""Utility script for scaling the predicted spectra of the UVvis-MPNN method"""
import numpy as np
import pandas as pd
import sys

def scale_spectra(data):
    for i, row in data.iterrows():
        scaled_row = (row.values - np.min(row.values)) / (np.max(row.values) - np.min(row.values))
        data.loc[i] = scaled_row

    return data

if __name__ == "__main__":

    raw_spectra = pd.read_csv(sys.argv[1],index_col=0)
    
    scaled_spectra = scale_spectra(raw_spectra)

    scaled_spectra.to_csv('scaled_spectra.csv')
