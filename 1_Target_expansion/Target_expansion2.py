#!/usr/bin/env python
# coding: utf-8
# Author: Yuemin Bian

print('Module 1: Target expansion')

from requests import get, post
from time import sleep
from unipressed import IdMappingClient
import numpy as np
import argparse
import sys
import tarfile
import pandas as pd
import csv
import time
import requests

# # Parse input

parser = argparse.ArgumentParser()
parser.add_argument('-i','--input', help = "Entering the PDB ID to start", required=True)
parser.add_argument('-f','--file_name', help = "A name to save out files...", required=True)
args = parser.parse_args()

# # Make functions

def get_pdb_file(uniprot_id):


    url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v6.pdb"

    try:
        response = requests.get(url, verify=False)
        if response.status_code == 200:
            pdb_content = response.text

            with open(f"{uniprot_id}.pdb", "w") as f:
                f.write(pdb_content)
            print(f"PDB file has been saved as {uniprot_id}.pdb")
        else:
            print(f"Failed to find the PDB file for UniProt ID {uniprot_id}. Status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while trying to download the PDB file: {e}")

def process_foldseek_query(fquery, databases_list):
    # submit a new job
    with open(fquery, 'r') as pdb_file:
        pdb_content = pdb_file.read()
        ticket = post('https://search.foldseek.com/api/ticket', 
                      files={'q': (pdb_content, pdb_content, 'application/octet-stream')},
                      data={
                        'mode' : 'tmalign',
                        'database[]' : databases_list,
                    }).json()

    # poll until the job was successful or failed
    repeat = True
    while repeat:
        status = get('https://search.foldseek.com/api/ticket/' + ticket['id']).json()
        if status['status'] == "ERROR":
            # handle error
            sys.exit(0)

        # wait a short time between poll requests
        sleep(10)
        repeat = status['status'] != "COMPLETE"

    # get results in JSON format
    result = get('https://search.foldseek.com/api/result/' + ticket['id'] + '/0').json()
    
    # download result to file
    download = get('https://search.foldseek.com/api/result/download/' + ticket['id'], stream=True)
    result_filename = 'result_' + databases_list[0] + '.tar.gz'
    with open(result_filename, 'wb') as fd:
        for chunk in download.iter_content(chunk_size=128):
            fd.write(chunk)
   
    with tarfile.open(result_filename, "r:gz") as tar:
        tar.extract("alis_" + databases_list[0] + ".m8")
 
    df = pd.read_csv(
        'alis_' + databases_list[0] + '.m8',
        sep='\t',            
        header=None,         
        dtype=str,            
        quoting=csv.QUOTE_NONE,   
        engine='python'           
    )

    return df



def process_afdb_database(fquery):
    databases = ['afdb-swissprot']
    df = process_foldseek_query(fquery, databases)

    columns_to_extract = [0, 1, 10, 11, 12, 20]
    subset_df = df.iloc[:, columns_to_extract]

    subset_df.columns = ['job', 'uniprot id', 'prob', 'TM-score', 'score', 'species']
    subset_df.loc[:, 'prob'] = pd.to_numeric(subset_df['prob'], errors='coerce')
    subset_df.loc[:, 'TM-score'] = pd.to_numeric(subset_df['TM-score'], errors='coerce')
    subset_df.loc[:, 'score'] = pd.to_numeric(subset_df['score'], errors='coerce')
    
    filtered_df = subset_df[subset_df['species'].str.contains('Homo sapiens', na=False) &
    (subset_df['prob'] > 0.8) &
    (subset_df['TM-score'] > 0.70) &
    (subset_df['score'] > 80)]
    filtered_df.loc[:, 'uniprot id'] = filtered_df['uniprot id'].apply(lambda x: x[x.find('-F1')-6:x.find('-F1')])

    filtered_df.to_csv(args.input+'_afdb.csv', index=False)
    unique_df = filtered_df.drop_duplicates(subset='uniprot id', keep='first')
    unique_df.to_csv(args.input+'_unique_uniprot_afdb.csv', index=False)


def combine_results():
    df1 = pd.read_csv(args.file_name+'_seq.csv')
    df2 = pd.read_csv(args.file_name+'_unique_uniprot_afdb.csv')

    combined_df = pd.concat([df1['uniprot id'], df2['uniprot id']])

    unique_df = combined_df.drop_duplicates()
    
    unique_df = unique_df.dropna(how='all')

    unique_df.to_csv(args.file_name+'.csv', index=False)

# # Use functions
uniprot_id = args.input
get_pdb_file(uniprot_id)
fquery = args.input+'.pdb'
process_afdb_database(fquery)
combine_results()




