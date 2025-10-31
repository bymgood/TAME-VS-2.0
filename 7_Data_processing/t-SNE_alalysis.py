#!/usr/bin/env python
# coding: utf-8
# Author: Yuemin Bian
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


# # Parse input
parser = argparse.ArgumentParser()
parser.add_argument('-a','--analysis_number', help = "The analysis number to be used for the t-SNE analysis. Options: 'top_1_ten_thousandth_percent' or 'top_1_percent'", required=True)
parser.add_argument('-u','--uniprot_id', help='The uniprot id of protein', type=str, required=True)
args = parser.parse_args()

# # Make functions
def prepare_tsne_data():
    training_set_property = pd.read_csv(
        f"{args.uniprot_id}_target_driven_VS_VS_all.csv",
        header=0,
        usecols=range(2, 9)
    )

    analysis_number = args.analysis_number
    if analysis_number == "top_1_ten_thousandth_percent":
        training_set_property_sampled = training_set_property.iloc[::100, :]
    elif analysis_number == "top_1_percent":
        training_set_property_sampled = training_set_property
    else:
        raise ValueError("Please select your analysis number: 'top_1_ten_thousandth_percent' or 'top_1_percent'")

    top_property = pd.read_csv(
        f"{args.uniprot_id}_target_driven_VS_{args.analysis_number}_VS_hits.csv",
        header=0,
        usecols=range(2, 9)
    )
    actives_property = pd.read_csv(
        f"{args.uniprot_id}_actives_properties_calculated.csv",
        header=0,
        usecols=range(1, 8)
    )
    inactives_property = pd.read_csv(
        f"{args.uniprot_id}_inactives_properties_calculated.csv",
        header=0,
        usecols=range(1, 8)
    )


    training_set_property_sampled['Source'] = 'Screening_library'
    top_property['Source'] = 'Top'
    actives_property['Source'] = 'Actives'
    inactives_property['Source'] = 'Inactives'


    combined_data = pd.concat([
        training_set_property_sampled,
        top_property,
        actives_property,
        inactives_property
    ], axis=0, ignore_index=True)


    features = combined_data.iloc[:, :7]
    labels = combined_data['Source']


    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(features)


    tsne = TSNE(n_components=2, perplexity=30, n_iter=3000, random_state=42)
    embedded_data = tsne.fit_transform(normalized_features)


    df_embedded = pd.DataFrame(embedded_data, columns=['Dim_1', 'Dim_2'])
    df_embedded['Source'] = labels.values

    return df_embedded 


def plot_tsne(df_embedded, output_filename="t-SNE_analysis.png"):

    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 12
    
    plt.rc('font', size=SMALL_SIZE)  
    plt.rc('axes', titlesize=MEDIUM_SIZE) 
    plt.rc('axes', labelsize=SMALL_SIZE) 
    plt.rc('xtick', labelsize=SMALL_SIZE)  
    plt.rc('ytick', labelsize=SMALL_SIZE)  
    plt.rc('legend', fontsize=SMALL_SIZE)  
    plt.rc('figure', titlesize=BIGGER_SIZE)
    

    fig, ax = plt.subplots(figsize=(8, 6))
    

    custom_palette = {
        'Screening_library': '#E6D7EC', 
        'Top': '#FE3B69',  
        'Actives': '#ffc75f',  
        'Inactives': '#0081cf',  
    }
    
    scatter_plots = []
    

    categories = ['Screening_library', 'Top', 'Actives', 'Inactives']
    
    for category in categories:
        category_df = df_embedded[df_embedded['Source'] == category]
        scatter = ax.scatter(
            category_df['Dim_1'],
            category_df['Dim_2'],
            c=category_df['Source'].map(custom_palette),
            label=category,
            s=1,
            alpha=0.8,
        )
        scatter_plots.append(scatter)
    

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("t-SNE of Physicochemical Properties – Top Scaffold", pad=15)
    

    legend = ax.legend(loc='upper right', frameon=True)
    legend.get_frame().set_alpha(0.6)
    legend.get_frame().set_edgecolor('gray')
    

    ax.minorticks_on()
    ax.grid(which='minor', linestyle=':', linewidth=0.5, alpha=0.5)
    

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    
# # Use functions

df_embedded = prepare_tsne_data()
plot_tsne(df_embedded)
print('t-SNE visualization plot has been generated.')

