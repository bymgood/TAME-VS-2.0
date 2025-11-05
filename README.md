# Target-driven-ML-enabled-VS 2.0 (TAME-VS 2.0)
Target-driven machine-learning-enabled virtual screening 2.0 is a machine learning tool developed to accelerate the early-stage hit identification. In this repository, you can find all you need to launch virtual screening against your target protein with different types of input information:
- Mimimum info: a single uniport ID of your target protein (starting point 1); 
- Intermediate info: Compound datasets containing active and inactive molecules against homologies of your target of interest (starting point 2);
- Advanced info: Compound datasets containing active molecules directly against your target of interest (starting point 3).    

# About this repository
The Starting_point[1-3].sh are the production scripts for launching vritual screening with different types of inputs. Clone the entire repository to your local machine prior to start.

##  Prerequisites

1. **Install CUDA**

   Download and install the appropriate version of CUDA for your system:  
   [NVIDIA CUDA Toolkit Downloads](https://developer.nvidia.com/cuda-12-4-0-download-archive)

2. **Install PyTorch compatible with your CUDA version**

   Choose the matching PyTorch version for your CUDA environment here:  
   [PyTorch Installation Guide](https://pytorch.org/get-started/previous-versions/)

3. **Set up a Conda virtual environment (Python 3.9)**

   ```bash
   conda create -n TAME_VS2 python=3.9
   conda activate TAME_VS2

4. Install dependencies
```bash
pip install -r requirements.txt
```

# Set up the built-in ChEMBL database
Download the **ChEMBL 36 SQLite database** from the official EBI FTP server:

> **Download link:**  
> [https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_36/chembl_36_sqlite.tar.gz](https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_36/chembl_36_sqlite.tar.gz)

After downloading, extract the file:
```bash
tar -xvzf chembl_36_sqlite.tar.gz
```


# Preparing molecular fingerprints and molecular graph for virtual screening
To use the included Enamine 50k compound library for final ML virtual screening, please run the followig command from **5_Virtural_screening**
```bash
python Library_preparation.py -i Enamine_diversity_50K.csv -s 1 -c 2  -f Enamine_diversity_50K_morgan_1024_FP
python GNN_data_preparation.py -i Enamine_diversity_50K.csv -s 1 -c 2  -f Enamine_diversity_50K_only_gnn_dataset
```
In March 2024, we added a convert.py file to module 5_Virtual_screening. This convert.py can be used to convert a customized .sdf chemical library into the .csv format. Then the Library_preparation.py should be able to be used for fingerprints calculations.

# Run target-driven machine-learning-enabled VS
The following example uses starting point 1 as an example.
1. Search for the uniport ID of your target proteins (e.g. P09238);
2. Launch the ```Starting_point_1.sh``` script and provide uniport ID and working directory;
```bash
bash Starting_point_1.sh
```
