#!/usr/bin/env python
# coding: utf-8
# Author: Yuemin Bian

print('Module of library preparation')

import sys
import numpy as np
import pandas as pd
import argparse
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from rdkit.Chem.AtomPairs import Pairs
from rdkit.Chem.AtomPairs import Torsions
#np.set_printoptions(threshold=sys.maxsize)

# # Parse input

parser = argparse.ArgumentParser()
parser.add_argument('-i','--input', help = "The library file to be prepared", required=True)
parser.add_argument('-s','--smiles', help = "Which column is the SMILES column? Starting from 1", required=True)
parser.add_argument('-c','--cmpd_id', help = "Which column is the compound id column? Starting from 1", required=True)
parser.add_argument('-t','--fp_type', help = "Optional. Select from Morgan, AtomPair, Topological, and MACCS. Default=Morgan", required=False)
parser.add_argument('-n','--number_of_bits', help = "Optional. The number of bits for Morgan, AtomPair and Topological FPs. Default=1024", required=False)
parser.add_argument('-k','--chunksize', help="Optional. Number of rows per chunk", default=50000, required=False, type=int)
parser.add_argument('-f','--file_name', help = "A name to save out files...", required=True)
args = parser.parse_args()


# # Make functions

def read_cmpds(name, smi_col, id_col, chunksize=None):
    print("Reading input library file...")
    if chunksize:
        for chunk in pd.read_csv(name, chunksize=chunksize):
            smiles_chunk = chunk.iloc[:, smi_col]
            id_chunk = chunk.iloc[:, id_col]
            yield smiles_chunk, id_chunk
    else:
        df = pd.read_csv(name)
        smiles = df.iloc[:, smi_col]
        cmpd_id = df.iloc[:, id_col]
        yield smiles, cmpd_id

def get_morganfp(df,bits):
    print("Calcuating Morgan FP...")
    bit_fp=[]
    count=0
    invalid_smiles = []
    for i in df:
        m = Chem.MolFromSmiles(i)
        n = AllChem.GetMorganFingerprintAsBitVect(m,2,bits)
        fp=np.zeros((1,))
        DataStructs.ConvertToNumpyArray(n, fp)
        bit_fp.append(fp)
        count = count+1
        print (count)
    fp_df = pd.DataFrame(bit_fp, index=df.index).astype(int)    
    cmpd_fp = pd.concat([pd.DataFrame(df), fp_df], axis=1)
    print('FP calcuated')
    return cmpd_fp

def get_AtomPairfp(df,bits):
    print("Calcuating Atom Pair FP...")
    bit_fp=[]
    for i in df:
        m = Chem.MolFromSmiles(i)
        n = Pairs.GetHashedAtomPairFingerprint(m,bits)
        fp=np.zeros((1,))
        DataStructs.ConvertToNumpyArray(n, fp)
        bit_fp.append(fp)
    fp_df = pd.DataFrame(bit_fp, index=df.index).astype(int)    
    cmpd_fp = pd.concat([pd.DataFrame(df), fp_df], axis=1)
    return cmpd_fp

def get_TopologicalTorsionfp(df,bits):
    print("Calcuating Topological FP...")    
    bit_fp=[]
    for i in df:
        m = Chem.MolFromSmiles(i)
        n = Torsions.GetHashedTopologicalTorsionFingerprint(m,bits)
        fp = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(n, fp)
        bit_fp.append(fp)
    fp_df = pd.DataFrame(bit_fp, index=df.index).astype(int)    
    cmpd_fp = pd.concat([pd.DataFrame(df), fp_df], axis=1)
    return cmpd_fp

def get_MACCS(df):
    print("Calcuating MACCS FP...")    
    bit_fp=[]
    for i in df:
        m = Chem.MolFromSmiles(i)
        n = MACCSkeys.GenMACCSKeys(m)
        fp = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(n, fp)
        bit_fp.append(fp)
    fp_df = pd.DataFrame(bit_fp, index=df.index).astype(int)    
    cmpd_fp = pd.concat([pd.DataFrame(df), fp_df], axis=1)
    return cmpd_fp


def write_out(table,name):
    print('In progress...')
    file_name=str(name)
    table.to_csv(file_name,index=False)
    print ("Calculation finished. Files saved to the disk.")
    return

# # Use functions
smiles_col = int(args.smiles) - 1
id_col = int(args.cmpd_id) - 1
fp_type = args.fp_type if args.fp_type else 'Morgan'
bits = int(args.number_of_bits) if args.number_of_bits else 1024
all_data = []

for chunk_smiles, chunk_id in read_cmpds(args.input, smiles_col, id_col, chunksize=args.chunksize):
    if fp_type == 'Morgan':
        chunk_fp = get_morganfp(chunk_smiles, bits)
    elif fp_type == 'AtomPair':
        chunk_fp = get_AtomPairfp(chunk_smiles, bits)
    elif fp_type == 'Topological':
        chunk_fp = get_TopologicalTorsionfp(chunk_smiles, bits)
    elif fp_type == 'MACCS':
        chunk_fp = get_MACCS(chunk_smiles, bits)
    else:
        chunk_fp = get_morganfp(chunk_smiles, bits)

    chunk_result = pd.concat([chunk_id, chunk_fp], axis=1, join='inner')
    all_data.append(chunk_result)

print('Preparing write out...')
final_df = pd.concat(all_data, ignore_index=True)
final_df.to_csv(args.file_name, index=False)
