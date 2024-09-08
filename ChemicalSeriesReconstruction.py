# Created by Maximilian Beckers, December 2021, initial Code by Franziska Kruger et al. J. Chem. Inf. Model. 2020, 60, 6, 2888–2902

from AutomatedSeriesClassification import  Clustering, cluster_utils, active_times_detection, chembl_setup
import argparse, sys, os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import pickle

try:
    import arthor;
except ImportError:
    arthor = None;
    
    
class ChemicalSeriesReconstruction:

    #input
    smiles_list = [];
    min_cluster_size = 10;
    flimit = 0.001;
    scaffolds = None;
    use_gpu=False;
    dates = [];
    size_sliding_window = None;
    jaccard_similarity_threshold = None;
    
    #output
    series_data = pd.DataFrame();
    mcs_data = pd.DataFrame();
    
    #generated data
    smiles_list_cleaned = [];
    mol_list = [];
    used_compounds = [];
    
    #databases
    proj_db = 0;
    chembl_db = 0;
    
    
    #********************************************
    #************ Object Constructor ************
    #********************************************
    def __init__(self, smiles_list, min_cluster_size = 10, flimit = 0.001, scaffolds = None, dates = [], size_sliding_window = None, jaccard_similarity_threshold=None, use_gpu=False):
        
        
        self.smiles_list = smiles_list;
        self.min_cluster_size = min_cluster_size;
        self.flimit = flimit;
        self.scaffolds = scaffolds;
        self.dates = dates;
        self.size_sliding_window = size_sliding_window;
        self.jaccard_similarity_threshold = jaccard_similarity_threshold;
        self.use_gpu = use_gpu;
        self.make_mol_list();
        self.proj_db = cluster_utils.make_project_db(self.smiles_list_cleaned);
        self.make_chembl_db();
        self.reconstruct();
            

    
    #********************************************
    #**** function to reconstruct the series ****
    #********************************************
    def reconstruct(self):
    
    
        df = pd.DataFrame();
        df["Structure"] = self.smiles_list_cleaned;
        
        if self.size_sliding_window is not None:
            
            if len(self.dates) != len(self.smiles_list):
                sys.exit('Date list has different lenght than the provided SMILES. Please Check. Exit ...');
            
            tmp_dates = np.array(self.dates)[self.used_compounds];
            df["Registration Date"] = np.array(self.dates)[self.used_compounds];
    
        mcs_list, class_labels, comp_label_list, _, spec_chembl_list, spec_projectdb_list = Clustering.UPGMA(self.mol_list, self.flimit, self.min_cluster_size, self.proj_db, self.chembl_db, self.scaffolds, gpu=self.use_gpu);
        
          
        df = df.iloc[comp_label_list];
        df['Class'] = class_labels;
        
        #add column with mcs and fchembl to dataframe
        df['MCS'] = mcs_list;
        df['Spec. ChEMBL'] = spec_chembl_list;
        df['Spec. Project DB'] = spec_projectdb_list;

        df = cluster_utils.fraction_of_atoms_explained_by_scaffold(df);        
     
        df, df_mcs = cluster_utils.make_mcs_file(df);
    
        if self.jaccard_similarity_threshold is not None:
            df = cluster_utils.merge_scaffolds_according_to_jaccard_similarity(df, self.jaccard_similarity_threshold);
        
        df = cluster_utils.calculate_series_purity(df);
        cluster_utils.calculate_median_tanimoto_similarities_between_series(df, self.min_cluster_size);
    
        #if the series should be split into the active times, do it
        if self.size_sliding_window is not None:
            df = active_times_detection.seperate_series_into_active_phases(df, "Registration Date", size_sliding_window=self.size_sliding_window, min_num_mols_per_window=self.min_cluster_size);
    
        self.series_data = df;
        self.mcs_data = df_mcs;
    
    
    #*********************************************
    #*** function to make the chembl database ****
    #*********************************************
    def make_chembl_db(self):

        #set the chembl database
        curr_path = os.path.dirname(os.path.abspath(__file__));
        outpath  = os.path.join(curr_path, "Data");

        if arthor is not None:

            try:
                chembldb = arthor.SubDb(curr_path + '/Data/chembl_27.atdb');
                chembldb.set_num_processors(16);
                print("Arthor ChEMBl database already exists. Will use the existing one");

            except:

                print("Setting up ChEMBL database for Arthor. This needs to be done only once. This will take ca. 30 minutes ...");

                sdf_path = chembl_setup.download_chembl(outpath);
                chembl_setup.make_chembl_smiles(outpath);

                print("Setting up ChEMBL database for Arthor ...");
                os.system('smi2atdb -j 0 -t {0}{1}.smi {0}{1}.atdb'.format(curr_path, "/Data/chembl_27"));
                os.system('atdb2fp -j 0 {0}{1}.atdb'.format(curr_path, "/Data/chembl_23"));
                os.system('smi2atfp -j 0 -t {0}{1}.smi {0}{1}.atfp'.format(curr_path, "/Data/chembl_27"));

                chembldb = arthor.SubDb(curr_path + '/Data/chembl_27.atdb');
                chembldb.set_num_processors(16);

                os.remove(sdf_path);

            Nchembl = len(chembldb.search('*'));

        else:

            try:
                with open(curr_path + '/Data/chembl27_sssdata.pkl','rb') as file:
                    chembldb = pickle.load(file);
                print("RDKit ChEMBl database already exists. Will use the existing one");

            except:
                print("Setting up ChEMBL database for RDKit. This needs to be done only once. This will take ca. 30 minutes ...");

                sdf_path = chembl_setup.download_chembl(outpath);
                chembl_setup.make_rdkit_substr_lib(outpath);
                with open(curr_path + '/Data/chembl27_sssdata.pkl','rb') as file:
                    chembldb = pickle.load(file);
                os.remove(sdf_path);

            Nchembl = len(chembldb);

        print("Number of compounds in the ChEMBL database: " + repr(Nchembl));
        self.chembl_db = chembldb;
    
    #*******************************************************
    #*** function to make list of mols from the smiles *****
    #*******************************************************
    def make_mol_list(self):
        
        num_compounds = len(self.smiles_list);
        compound_indices = [];
        mol_list = [];
        fp_list = [];
        smiles_list_cleaned = [];
                           
        #clean the data and get fingerprints from the compounds
        for tmp_compound in range(num_compounds):

            try:
                if ("*" in self.smiles_list[tmp_compound]) | ("." in self.smiles_list[tmp_compound]):
                    continue;      
            
                tmp_mol = Chem.MolFromSmiles(self.smiles_list[tmp_compound]);
            
                if tmp_mol.GetNumHeavyAtoms() < 2:
                    continue;
            
                #tmp_mol = Chem.AddHs(tmp_mol);
                tmp_fp = AllChem.GetMorganFingerprintAsBitVect(tmp_mol, 2);
                fp_list.append(tmp_fp);
                compound_indices.append(tmp_compound);
                mol_list.append(tmp_mol);
                smiles_list_cleaned.append(self.smiles_list[tmp_compound]);
            except:
                alu = 0; 

        num_printout = 10;
        if (tmp_compound % int((num_compounds/num_printout))) == 0:
            progress = 100*tmp_compound/float(num_compounds);
            print('{:.2f}% finished ...'.format(progress));

        num_compounds = len(compound_indices);
        print("Number of compounds after cleaning: " + repr(num_compounds));

        self.smiles_list_cleaned = smiles_list_cleaned;
        self.mol_list = mol_list;
        self.used_compounds = compound_indices;
     
     
        
    
