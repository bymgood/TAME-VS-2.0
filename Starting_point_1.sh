# Target-driven ML-enabled VS

echo Enter the uniprot id for the target of interest
read uniprot_ID
echo The uniprot id you entered is $uniprot_ID
echo Enter the working directory of this bash file of the screening package
read directory_1
echo The working directory is $directory_1
echo "Please select the database you would like to use for virtual screening:"
echo "1. 50K Enamine diversity Library"
echo "2. 4M Enamine Screening Library"
echo "3. Drugbank Screening Library"
echo "4. Enamine Fragment Collection Library"
echo "5. Enamine Covalent Compounds Library"
echo "6. Enamine Macrocycles Collection Library"
read -p "Enter your choice (1, 2, 3, 4 ,5, or 6): " database_choice
if [ "$database_choice" == "1" ]; then
    echo "You have selected the 50K Diversity Library."
    database_size="50K"
    database_name="Enamine_diversity_50K"
    alalysis_number='top_1_percent'
elif [ "$database_choice" == "2" ]; then
    echo "You have selected the 4M Compound Library."
    database_size="4M"
    database_name="Enamine_Screening_Collection_4M"
    alalysis_number='top_1_ten_thousandth_percent'
elif [ "$database_choice" == "3" ]; then  
    echo "You have selected the Drugbank Screening Library."
    database_size="Drugbank"  
    database_name="Drugbank_screening_library"
    alalysis_number='top_1_percent'  
elif [ "$database_choice" == "4" ]; then 
    echo "You have selected the Enamine Fragment Collection."
    database_size="Fragment"  
    database_name="Enamine_Fragment_Collection"
    analysis_number='top_1_percent'  
elif [ "$database_choice" == "5" ]; then  
    echo "You have selected the Enamine Covalent Compounds."
    database_size="Covalent"  
    database_name="Enamine_Covalent_Compounds"
    analysis_number='top_1_percent'  
elif [ "$database_choice" == "6" ]; then 
    echo "You have selected the Enamine Macrocycles Collection."
    database_size="Macrocycles"  
    database_name="Enamine_Macrocycles_Collection"
    analysis_number='top_1_percent'  
else
    echo "Invalid choice. Please enter 1, 2, 3, 4, 5, or 6."
    exit 1
fi
echo Let us get it started.

# Module 1
cd $directory_1/1_Target_expansion/
python Target_expansion.py -i $uniprot_ID -f $uniprot_ID -p 0.4
python Target_expansion2.py -i $uniprot_ID -f $uniprot_ID 
cp $uniprot_ID'.csv' $directory_1/2_Compound_retrieving/

# Module 2
cd $directory_1/2_Compound_retrieving/
python Compound_retrieving.py -i $uniprot_ID'.csv' -f $uniprot_ID'_compounds_collection'
cp *compounds_collection* $directory_1/3_Vectorization/

# Module 3
cd $directory_1/3_Vectorization/
python Vectorization.py -i 'actives_'$uniprot_ID'_compounds_collection.csv' -a active -f 'actives_'$uniprot_ID'_compounds_collection_fp'
python Vectorization.py -i 'inactives_'$uniprot_ID'_compounds_collection.csv' -a inactive -f 'inactives_'$uniprot_ID'_compounds_collection_fp'

python GNN_data_split.py --input_inactives 'inactives_'$uniprot_ID'_compounds_collection.csv' --input_actives 'actives_'$uniprot_ID'_compounds_collection.csv' -u $uniprot_ID
python GNN_Vectorization.py -i $uniprot_ID'_gnn_train_data.csv'
python GNN_Vectorization.py -i $uniprot_ID'_gnn_val_data.csv'

cp *fp* $directory_1/4_ML_modeling_training/
cp *fp* $directory_1/6_Post_VS_analysis/
cp *train_data.pkl* $directory_1/4_GNN_modeling_training/
cp *val_data.pkl* $directory_1/4_GNN_modeling_training/

# Module 4
cd $directory_1/4_ML_modeling_training/
python ML_model_training.py -a actives* -i inactives* -f $uniprot_ID
cp *.sav $directory_1/5_Virtural_screening/

cd $directory_1/4_GNN_modeling_training/
python GNN_modeling_training.py -u $uniprot_ID
cp *.pth $directory_1/5_Virtural_screening/

# Module 5
cd $directory_1/5_Virtural_screening/
python Virtual_screening.py -m *MLP.sav -t MLP -s $database_name'_morgan_1024_FP.csv' -f $uniprot_ID'_'$database_size'_VS_MLP'
python Virtual_screening.py -m *random_forest.sav -t RF -s $database_name'_morgan_1024_FP.csv' -f $uniprot_ID'_'$database_size'_VS_RF'
python GNN_Virtual_screening.py -s $database_name'_gnn_dataset.pkl' -f $uniprot_ID'_'$database_size'_VS_GNN'
cp *VS* $directory_1/6_Post_VS_analysis/

# Module 6
cd $directory_1/6_Post_VS_analysis/
python Post_VS_analysis.py -r *MLP.csv -a actives* -i inactives* -u $uniprot_ID
python Post_VS_analysis.py -r *RF.csv -a actives* -i inactives* -u $uniprot_ID
python Post_VS_analysis.py -r *GNN.csv -a actives* -i inactives* -u $uniprot_ID
cp *properties_calculated* $directory_1/7_Data_processing/

# Module 7
cd $directory_1/7_Data_processing/
python Data_processing.py -r *RF* -m *MLP* -g *GNN* -f $uniprot_ID'_target_driven_VS'
python Pharm2D_alalysis_butina.py -i $uniprot_ID'_target_driven_VS_'$alalysis_number'_VS_hits.csv'



