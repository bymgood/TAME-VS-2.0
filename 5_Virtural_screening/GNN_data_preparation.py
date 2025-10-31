#!/usr/bin/env python
# coding: utf-8
# Author: Yuemin Bian

print('Module of GNN data preparation')

import os
import pandas as pd
import pickle
import argparse
from data_utils import *

# # Parse input

parser = argparse.ArgumentParser()
parser.add_argument('-i','--input', help="data file used for virtual screening", required=True)
parser.add_argument('-s','--smiles', help = "Which column is the SMILES column? Starting from 1", required=True)
parser.add_argument('-c','--cmpd_id', help = "Which column is the compound id column? Starting from 1", required=True)
parser.add_argument('-f','--file_name', help = "A name to save out files...", required=True)
args = parser.parse_args()


# # Make functions

def read_cmpds(name,smi_col, id_col):
    print("Read input library file")
    df=pd.read_csv(name)
    smiles=df.iloc[:,smi_col]
    comp_id=df.iloc[:,id_col]
    return smiles, comp_id

def process_data():

    gnn_dataset = MoleculeDataset(f"{new_name}")

    with open(args.file_name, 'wb') as f:
        pickle.dump(gnn_dataset, f)


# # Use functions

smiles, comp_id=read_cmpds(args.input, (int(args.smiles)-1), (int(args.cmpd_id)-1))
df_data = pd.DataFrame({'smiles': smiles, 'comp_id': comp_id})
base_name = os.path.splitext(os.path.basename(args.input))[0]
base_name = base_name.replace("_compounds_collection", "")
new_name = f"{base_name}_gnn_preparation.csv"
df_data.to_csv(new_name, index=False, header=False)

process_data()
with open(args.file_name, 'rb') as f: 
    data = pickle.load(f)
num_compounds = len(data)
print(f"Number of compounds in the dataset: {num_compounds}")
print("Data processing complete.")
