#!/usr/bin/env python
# coding: utf-8
# Author: Yuemin Bian

print('Module 7: Data processing')

import sys
import numpy as np
import pandas as pd
np.set_printoptions(threshold=sys.maxsize)
import argparse

# # Parse input

def parse_weights(s: str):
    """
    Parse a weight string in the format mlp:rf:gnn (e.g. "1:2:3") into a tuple of floats.
    """
    parts = s.split(':')
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "Weight format must be mlp:rf:gnn, for example '1:2:3'"
        )
    try:
        return tuple(map(float, parts))
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Each weight must be a number"
        )
parser = argparse.ArgumentParser()
parser.add_argument('-r','--result_rf', help = 'The outcome rf file from the module 6', required=True)
parser.add_argument('-m','--result_mlp', help = 'The outcome mlp file from the module 6', required=True)
parser.add_argument('-g','--result_gnn', help = 'The outcome gnn file from the module 6', required=True)
parser.add_argument('-f','--file_name', help='The output file name', required=True)
parser.add_argument('-p', '--proportions', help="Weights for the three models in mlp:rf:gnn format (e.g. '1:1:1')", type=parse_weights, default='1:1:1', required=False)
args = parser.parse_args()

# # Make functions

def read_screen_result(name):
    print ('Read '+ name +'...')
    outcome = pd.read_csv(name, header=0)
    return outcome

def add_rank(outcome,model_type):
    i=1
    rank=[]
    while i < len(outcome)+1:
        rank.append(i)
        i+=1
    outcome[model_type+'_rank']=rank
    return outcome

def merge_three(mlp,rf,gnn):
    print ('Merge three files...')
    mlp = mlp[['comp_id','MLP_prediction_score','mlp_rank']]
    combined = rf.merge(mlp,on='comp_id')
    gnn = gnn[['comp_id', 'GNN_prediction_score', 'gnn_rank']]
    combined = combined.merge(gnn, on='comp_id')
    combined = combined[['comp_id', 'smiles', 'MolWt', 'MolLogP', 'NumHAcceptors', 'NumHDonors', 'NumRotatableBonds', 'TPSA', 'QED', 'MLP_prediction_score', 'mlp_rank', 'RF_prediction_score', 'rf_rank', 'GNN_prediction_score', 'gnn_rank']]
    return combined

def normalize_scores(combined):
    norm_rf = (combined['RF_prediction_score']-combined['RF_prediction_score'].min())/(combined['RF_prediction_score'].max()-combined['RF_prediction_score'].min())
    norm_mlp = (combined['MLP_prediction_score']-combined['MLP_prediction_score'].min())/(combined['MLP_prediction_score'].max()-combined['MLP_prediction_score'].min())
    norm_gnn = (combined['GNN_prediction_score']-combined['GNN_prediction_score'].min())/(combined['GNN_prediction_score'].max() - combined['GNN_prediction_score'].min())
    combined['norm_mlp_score'] = norm_mlp
    combined['norm_RF_score'] = norm_rf
    combined['norm_gnn_score'] = norm_gnn
    return combined

def ensemble_score(normalized_combined, w_mlp, w_rf, w_gnn):
    print ('Add ensemble ranking as well...')
    total = w_mlp + w_rf + w_gnn
    w_mlp /= total
    w_rf /= total
    w_gnn /= total
    i=0
    ensemble=[]
    while i < len(normalized_combined):
        row = normalized_combined.iloc[i]
        mlp_rank = row['mlp_rank']
        rf_rank  = row['rf_rank']
        gnn_rank = row.get('gnn_rank', 0)

        score = mlp_rank  * w_mlp \
              + rf_rank * w_rf \
              + gnn_rank * w_gnn
        score = round(score, 1)
        ensemble.append(score)
        i += 1
    normalized_combined['ensemble_rank']=ensemble
    normalized_combined = normalized_combined.sort_values(by=['ensemble_rank'],ascending=True)
    normalized_combined['rank'] = normalized_combined['ensemble_rank'].rank(method='min', ascending=True)
    return normalized_combined

def top_1_percent_combined(normalized_combined_ensemble):
    print ('Select non-duplicated top 1% VS hits from MLP, RF, GNN, and ensemble...')
    length = int(len(normalized_combined_ensemble)/100)
    normalized_combined_ensemble_1 = normalized_combined_ensemble.iloc[:length]
    mlp_1 = normalized_combined_ensemble.sort_values(by=['mlp_rank'],ascending=True).iloc[:length]
    rf_1 = normalized_combined_ensemble.sort_values(by=['rf_rank'],ascending=True).iloc[:length]
    gnn_1 = normalized_combined_ensemble.sort_values(by=['gnn_rank'],ascending=True).iloc[:length]
    frames= [normalized_combined_ensemble_1, mlp_1, rf_1, gnn_1]
    percent_1_combined = pd.concat(frames)
    percent_1_combined = percent_1_combined.drop_duplicates(subset=['comp_id'])
    
    return percent_1_combined
    
def top_1_ten_thousandth_percent_combined(normalized_combined_ensemble):
    print ('Select non-duplicated top 1% VS hits from MLP, RF, GNN, and ensemble...')
    length = int(len(normalized_combined_ensemble)/10000)
    normalized_combined_ensemble_1 = normalized_combined_ensemble.iloc[:length]
    mlp_1 = normalized_combined_ensemble.sort_values(by=['mlp_rank'],ascending=True).iloc[:length]
    rf_1 = normalized_combined_ensemble.sort_values(by=['rf_rank'],ascending=True).iloc[:length]
    gnn_1 = normalized_combined_ensemble.sort_values(by=['gnn_rank'],ascending=True).iloc[:length]
    frames= [normalized_combined_ensemble_1, mlp_1, rf_1, gnn_1]
    percent_1_ten_thousandth_combined = pd.concat(frames)
    percent_1_ten_thousandth_combined = percent_1_ten_thousandth_combined.drop_duplicates(subset=['comp_id'])
    
    return percent_1_ten_thousandth_combined

def write_out_top_1(outcome,name):
    file_name=str(name)
    outcome.to_csv(file_name+'_top_1_percent_VS_hits'+'.csv',index=False)
    print ("Done. Top selected compounds are written to the disk")
    return

def write_out_top_1_ten_thousandth(outcome,name):
    file_name=str(name)
    outcome.to_csv(file_name+'_top_1_ten_thousandth_percent_VS_hits'+'.csv',index=False)
    print ("Done. Top selected compounds are written to the disk")
    return

def write_out_all(outcome,name):
    file_name=str(name)
    outcome.to_csv(file_name+'_VS_all'+'.csv',index=False)
    print ("Done. All screened compounds are written to the disk")
    return

# # Use functions
# Load in outcome
mlp=read_screen_result(args.result_mlp)
rf=read_screen_result(args.result_rf)
gnn=read_screen_result(args.result_gnn)

mlp = add_rank(mlp,'mlp')
rf = add_rank(rf,'rf')
gnn = add_rank(gnn,'gnn')
combined=merge_three(mlp,rf,gnn)
w_mlp, w_rf, w_gnn = args.proportions
combined_ensemble=ensemble_score(combined, w_mlp=w_mlp, w_rf=w_rf, w_gnn=w_gnn)
outcome_top_1 = top_1_percent_combined(combined_ensemble)
outcome_top_1_ten_thousandth = top_1_ten_thousandth_percent_combined(combined_ensemble)

write_out_top_1(outcome_top_1,args.file_name)
write_out_top_1_ten_thousandth(outcome_top_1_ten_thousandth,args.file_name)
write_out_all(combined_ensemble,args.file_name)

