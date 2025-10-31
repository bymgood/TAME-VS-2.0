#!/usr/bin/env python
# coding: utf-8
# Author: Yuemin Bian

print('Module 5: Virtual screening')

import pandas as pd
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import gc

from scipy import interp
from itertools import cycle
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

# # Parse input

parser = argparse.ArgumentParser()
parser.add_argument('-m','--model', help = 'Load the prepared machine learning model', required=True)
parser.add_argument('-t','--model_type', help = 'Specify the algorithm of model to be loaded - choose from MLP and RF', required=True)
parser.add_argument('-s','--screen_set', help='load the compound set to screen', required=True)
parser.add_argument('-f','--file_name', help="filename to save the result to", required=True)
parser.add_argument('-k', '--chunksize', help="Optional. Number of rows per chunk", default=100000, required=False, type=int)
args = parser.parse_args()

# # Make functions

def read_model(name):
    model = pickle.load(open(name, 'rb'))
    print (name + " loaded")
    return model

def read_library(name, chunksize=None):
    if chunksize:
        return pd.read_csv(name, header=0, chunksize=chunksize)
    else:
        df = pd.read_csv(name, header=0)
        print('screen set loaded')
        print('screen set includes '+str(df.shape[0])+' compounds')
        return df


def screening(model, screen_set, name, chunksize=None):
    print('In progress of screening...')
    output_file = args.file_name + '.csv'
    first = True
    # Chunked or single
    iterator = screen_set if hasattr(screen_set, '__iter__') and not isinstance(screen_set, pd.DataFrame) else [screen_set]
    for chunk in iterator:
        df = chunk if isinstance(chunk, pd.DataFrame) else chunk
        comp_id = df.iloc[:, 0]
        smiles = df.iloc[:, 1]
        fps = df.iloc[:, 2:]
        probas = model.predict_proba(fps)[:, 1]
        out = pd.DataFrame({
            'comp_id': comp_id,
            'smiles': smiles,
            f'{name}_prediction_score': probas
        })
        if first:
            out.to_csv(output_file, index=False, mode='w')
            first = False
        else:
            out.to_csv(output_file, index=False, mode='a', header=False)
        # cleanup
        del df, comp_id, smiles, fps, probas, out
        gc.collect()
    return output_file

def write_out(outcome, name):
    file_name = str(name)
    print(f"Done. The library file with scores is saved to the disk: {outcome}")
    return

# # Use functions

model = read_model(args.model)
screen_set = read_library(args.screen_set, chunksize=args.chunksize)
out_file = screening(model, screen_set, args.model_type, chunksize=args.chunksize)
write_out(out_file, args.file_name)

