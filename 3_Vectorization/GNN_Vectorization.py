#!/usr/bin/env python
# coding: utf-8
# Author: Yuemin Bian

import numpy as np
import pandas as pd
import argparse
import pickle
from data_utils import MoleculeDataset  

# # Parse input

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', required=True, help = 'A txt file containing smiles and tags')
args = parser.parse_args()


# # Make functions

def read_cmpds(name):
    print("Read input compounds file")
    df=pd.read_csv(name)
    original=df.shape[0]
    df['canonical_smiles'].replace('', np.nan, inplace=True)
    df.dropna(subset=['canonical_smiles'], inplace=True)
    recognized=df.shape[0]
    diff=original-recognized
    print(str(recognized) + ' compounds recognized. ' + str(diff) + ' unrecognized SMILES are dropped.')
    df.reset_index(drop=True, inplace=True)
    df=df['canonical_smiles']
    return df

def GNN_Vectorization(data_file):
    file_prefix = data_file.split('.')[0]
    output_file = f"{file_prefix}.pkl"
    
    dataset = MoleculeDataset(data_file)
    
    with open(output_file, 'wb') as f:
        pickle.dump(dataset, f)

    
    return output_file

# # Use functions

GNN_Vectorization(args.input)
