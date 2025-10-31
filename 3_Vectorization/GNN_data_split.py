#!/usr/bin/env python
# coding: utf-8
# Author: Yuemin Bian

import os
import numpy as np
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split

# # Parse input

parser = argparse.ArgumentParser()
parser.add_argument('--input_inactives', required=True, 
                        help = "The compound list for FP calculation. Or simply the output file from Module 2")
parser.add_argument('--input_actives', required=True, 
                        help = "The compound list for FP calculation. Or simply the output file from Module 2")
parser.add_argument('-u', required=True, help = "The uniprot id of your target")
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

def split(active_csv, inactive_csv, random_state=42):

    active_df = pd.read_csv(active_csv)
    active_df['label'] = 1

    inactive_df = pd.read_csv(inactive_csv)
    inactive_df['label'] = 0

    data = pd.concat([active_df, inactive_df], ignore_index=True)

    train_data, val_data = train_test_split(
        data[['canonical_smiles', 'label']],
        test_size=0.2,
        stratify=data['label'],
        random_state=random_state
    )

    train_data.to_csv(args.u+'_gnn_train_data.csv', index=False, header=False)
    val_data.to_csv(args.u+'_gnn_val_data.csv', index=False, header=False)
    return train_data, val_data

# # Use functions

df_inactives=read_cmpds(args.input_inactives)
df_actives=read_cmpds(args.input_actives)
inactives_base_name = os.path.splitext(os.path.basename(args.input_inactives))[0]
actives_base_name = os.path.splitext(os.path.basename(args.input_actives))[0]
inactives_base_name = inactives_base_name.replace("_compounds_collection", "")
actives_base_name = actives_base_name.replace("_compounds_collection", "")
inactives_new_name = f"{inactives_base_name}_smiles.csv"
actives_new_name = f"{actives_base_name}_smiles.csv"
df_inactives.to_csv(inactives_new_name, index=False)
df_actives.to_csv(actives_new_name, index=False)

train_data, val_data = split(
    active_csv=actives_new_name,
    inactive_csv=inactives_new_name,
    random_state=42
)

