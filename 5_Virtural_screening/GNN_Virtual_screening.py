#!/usr/bin/env python
# coding: utf-8
# Author: Yuemin Bian

print('Module 5_2: GNN_Virtual screening')
import torch
import pickle
import pandas as pd
import argparse
from torch_geometric.loader import DataLoader
from tqdm import tqdm

# # Parse input

parser = argparse.ArgumentParser()
parser.add_argument('-s','--screen_set', help='load the compound set to screen', required=True)
parser.add_argument('-f','--file_name', help="filename to save the result to", required=True)
args = parser.parse_args()

# # Make functions

def score_model(model, data_loader):
    model.eval()
    scores = []
    smiles_list = []
    id_list = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Scoring"):
            batch = batch.to('cuda')
            out = model(batch)
            batch_scores = torch.exp(out)[:, 1].cpu().numpy()
            scores.extend(batch_scores)
            smiles_list.extend(batch.smiles) 
            id_list.extend(batch.cmpd_id)
    return smiles_list, id_list, scores

def save_sorted_scores(smiles_list, id_list, scores, output_csv_path):

    df = pd.DataFrame({
        'comp_id': id_list,
        'smiles': smiles_list,
        'GNN_prediction_score': scores,
    })


    df.to_csv(output_csv_path, index=False)
    print(f"scores saved to {output_csv_path}")

model_path = 'gnn_model.pth'
model = torch.load(model_path, weights_only=False)
model.eval()
dataset_path = args.screen_set
with open(dataset_path, 'rb') as f:
    dataset = pickle.load(f)
data_loader = DataLoader(dataset, batch_size=512, num_workers=8)

smiles_list, id_list, scores = score_model(model, data_loader)

output_csv_path = args.file_name+'.csv'
save_sorted_scores(smiles_list, id_list, scores, output_csv_path)
