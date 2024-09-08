# Created by Maximilian Beckers, December 2021, initial Code by Franziska Kruger et al. J. Chem. Inf. Model. 2020, 60, 6, 2888–2902

import numpy as np
import pandas as pd
import os
from AutomatedSeriesClassification import cluster_utils
from rdkit import Chem
from rdkit.Chem import AllChem

#-----------------------------------------------
def read_data(filename, seperator, smiles_column):
    
    df = pd.read_csv(filename, sep=seperator);
    
    num_compounds = len(df[smiles_column].to_list());
    #num_compounds= 10000;
    
    print("");
    print("****************************");
    print("******* Reading data *******");
    print("****************************");

    print("Number of compounds: " + repr(num_compounds));

    fp_list = [];
    compound_indices = [];
    smiles_list = [];
    #project_id = [];
    mol_list = [];

    #clean the data and get fingerprints from the compounds
    for tmp_compound in range(num_compounds):

        try:
            if ("*" in df[smiles_column].iloc[tmp_compound]) | ("." in df[smiles_column].iloc[tmp_compound]):
                continue;      
            
            tmp_mol = Chem.MolFromSmiles(df[smiles_column].iloc[tmp_compound]);
            
            if tmp_mol.GetNumHeavyAtoms() < 2:
                continue;
            
            #tmp_mol = Chem.AddHs(tmp_mol);
            tmp_fp = AllChem.GetMorganFingerprintAsBitVect(tmp_mol, 2);
            fp_list.append(tmp_fp);
            compound_indices.append(tmp_compound);
            mol_list.append(tmp_mol);
            smiles_list.append(df['Structure'][tmp_compound]);
                #project_id.append(tmp_pid);
        except:
            alu = 0; 

        num_printout = 10;
        if (tmp_compound % int((num_compounds/num_printout))) == 0:
            progress = 100*tmp_compound/float(num_compounds);
            print('{:.2f}% finished ...'.format(progress));

    num_compounds = len(compound_indices);
    print("Number of compounds after cleaning: " + repr(num_compounds));

    df_filtered = df.iloc[compound_indices, :];

    # set up project database for arthor substructure matching        
    proj_db = cluster_utils.make_project_db(smiles_list);

    df = 0;
    
    return mol_list, proj_db, df_filtered, smiles_list;
