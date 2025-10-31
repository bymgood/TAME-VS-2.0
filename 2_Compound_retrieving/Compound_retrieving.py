#!/usr/bin/env python
# coding: utf-8
# Author: Yuemin Bian

print('Module 2: Compound collection')

import argparse
import sqlite3
from chembl_webresource_client.new_client import new_client
from chembl_webresource_client.settings import Settings
import pandas as pd
import numpy as np
#pd.set_option('display.max_columns', None)

# # Parse input

parser = argparse.ArgumentParser()
parser.add_argument('-i','--input', help = "The target list for compound collection. Or simply the generated file from the Module 1", required=True)
parser.add_argument('-f','--file_name', help = "A name to save out files...", required=True)
args = parser.parse_args()

# # Make functions

def uniprot_id_chembl_id(uniprot_id):
    print('Transfering the uniprot id to the chembl id...')

    target = new_client.target
    uniprot_id = uniprot_id
    res = target.filter(target_components__accession=uniprot_id)

    res = pd.DataFrame(res)
    return res

def protein_chembl_id(res):
    i=0
    target_chembl_id = None
    while i < res.shape[0]:
        if res['target_type'][i] == 'SINGLE PROTEIN':
            target_chembl_id = res['target_chembl_id'][i]
        i=i+1

    return target_chembl_id

def get_compounds(target_chembl_id: str):
    sql = """
    SELECT
        a.*,                          -- activities 
        asy.*,                        -- assays 
        asy.chembl_id    AS assay_chembl_id,
        md.chembl_id     AS molecule_chembl_id,
        td.chembl_id     AS target_chembl_id,
        doc.chembl_id    AS doc_chembl_id,
        md.pref_name     AS compound_name,
        cs.canonical_smiles,
        cs.standard_inchi,
        cs.standard_inchi_key,
        doc.pubmed_id,
        doc.doi,
        doc.title        AS doc_title,
        td.pref_name     AS target_name
    FROM activities a
    JOIN assays        asy ON a.assay_id = asy.assay_id
    JOIN molecule_dictionary md ON a.molregno = md.molregno
    JOIN compound_structures cs ON md.molregno = cs.molregno
    JOIN docs         doc ON asy.doc_id = doc.doc_id
    JOIN target_dictionary td   ON asy.tid = td.tid
    WHERE td.chembl_id = ?
    """
    with sqlite3.connect('/home/bianlab/Documents/chembl_35.db') as conn:
        df = pd.read_sql_query(sql, conn, params=[target_chembl_id])
    print(f'{len(df)} rows returned for assay chembl_id = {target_chembl_id}')
    return df

def analysis_to_collected_cmpds(combined):
    test_type=[]
    number=[]
    assay=pd.DataFrame()
    for i in pd.unique(combined['standard_type']):
        test_type.append(i)
        number.append(combined.loc[combined['standard_type'] == i].shape[0])
        #print (i, combined.loc[combined['standard_type'] == i].shape[0])
    assay['type']=test_type
    assay['count']=number
    assay = assay.sort_values('count', ascending=False, ignore_index=True)
    print(str(assay.shape[0])+' values types')
    print(assay.iloc[0,0] + ' has the most record(s) - ' + str(assay.iloc[0,1]) + ' records')
    return assay

def remove_dup(combined, assay):
    keep = ['IC50', 'EC50', 'Ki', 'Kd', 'Inhibition', 'Activity']
    combined = combined[combined['standard_type'].isin(keep)].copy()
    combined = combined.dropna(subset=['standard_value'])
    combined['standard_value'] = pd.to_numeric(combined['standard_value'], errors='coerce')

    aggregated_rows = []

    for (mol, stype), group in combined.groupby(['molecule_chembl_id', 'standard_type']):
        if stype in ['IC50', 'EC50', 'Ki', 'Kd']:
            idx = group['standard_value'].idxmin()
        elif stype in ['Inhibition', 'Activity']:
            idx = group['standard_value'].idxmax()
        else:
            continue
        aggregated_rows.append(combined.loc[idx].to_dict()) 

    combined = pd.DataFrame(aggregated_rows).reset_index(drop=True)
    return combined

def active_cmpds(training_data, cutoff_nM=1000, cutoff_percent=50):
    df = training_data.copy()
    df["standard_value"] = pd.to_numeric(df["standard_value"], errors="coerce")

    mask_q = (df["standard_type"].isin(["IC50", "EC50", "Ki", "Kd"])) & (df["standard_value"] <= cutoff_nM)

    mask_inhib = (df["standard_type"] == "Inhibition") & (df["standard_value"] >= cutoff_percent)

    mask_activity = (df["standard_type"] == "Activity") & (df["standard_value"] >= cutoff_percent)

    training_data_active = df[mask_q | mask_inhib | mask_activity]
    print(f"{training_data_active.shape[0]} active compounds collected")
    return training_data_active


def inactive_cmpds(training_data, cutoff_nM=1000, cutoff_percent=50):
    df = training_data.copy()
    df["standard_value"] = pd.to_numeric(df["standard_value"], errors="coerce")

    mask_q = (df["standard_type"].isin(["IC50", "EC50", "Ki", "Kd"])) & (df["standard_value"] > cutoff_nM)

    mask_inhib = (df["standard_type"] == "Inhibition") & (df["standard_value"] < cutoff_percent)

    mask_activity = (df["standard_type"] == "Activity") & (df["standard_value"] < cutoff_percent)

    training_data_inactive = df[mask_q | mask_inhib | mask_activity]
    print(f"{training_data_inactive.shape[0]} inactive compounds collected")
    return training_data_inactive

def write_out(table,name):
    file_name=str(name)
    table.to_csv(file_name+'.csv',index=False)
    print ("Collected info is written to the disk.")
    return

# # Use functions

df=pd.read_csv(args.input)
combined=pd.DataFrame()
print (args.input + ' is recognized')

for uniprot_id_one in df['uniprot id']:
    res = uniprot_id_chembl_id(uniprot_id_one)
    if len(res) > 0:
        chembl_id = protein_chembl_id(res)
        if chembl_id is not None: 
            table = get_compounds(chembl_id)
            combined = pd.concat([combined, table], ignore_index=True)
        else:
            print(f"No valid ChEMBL ID found for UniProt ID: {uniprot_id_one}")
    
write_out(combined,args.file_name)

assay=analysis_to_collected_cmpds(combined)
write_out(assay,'assay_info_' + args.file_name)

training_data=remove_dup(combined,assay)
actives=active_cmpds(training_data)
write_out(actives,'actives_' + args.file_name)
inactives=inactive_cmpds(training_data)
write_out(inactives,'inactives_' + args.file_name)
