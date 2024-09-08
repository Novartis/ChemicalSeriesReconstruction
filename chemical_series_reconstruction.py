# Created by Maximilian Beckers, December 2021, initial Code by Franziska Kruger et al. J. Chem. Inf. Model. 2020, 60, 6, 2888–2902

import argparse, sys, os
import random
import time
import pickle
from AutomatedSeriesClassification import utilsDrawing, Clustering, DimensionalityReduction, cluster_utils, IO, active_times_detection, chembl_setup
from rdkit import Chem, rdBase
from rdkit.Chem import rdSubstructLibrary
import numpy as np
import pandas as pd
try:
    import arthor
except ImportError:
    arthor = None

# *************************************************************
# ****************** Commandline input ************************
# *************************************************************


cmdl_parser = argparse.ArgumentParser(
    prog=sys.argv[0],
    description='*** Classification of chemical series ***',
    formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=30), add_help=True);

cmdl_parser.add_argument('-data', '--data', metavar="data.csv", type=str, required=True,
                         help='Input of csv file of chemical compounds');
cmdl_parser.add_argument('-min_series_size', '--min_series_size', metavar="10", type=int, required=False,
                         help='Minimal size of chemical series');
cmdl_parser.add_argument('-flimit', '--flimit', metavar="1", type=float, required=False,
                         help='Specificity for ChEMBL search');
cmdl_parser.add_argument('-smiles_column', '--smiles_column', metavar="Structure", type=str, required=False,
                         help='Name of the column that contains the smiles of the compounds. (Default: "Structure").');
cmdl_parser.add_argument('-jaccard_similarity_threshold', '--jaccard_similarity_threshold', metavar="1", type=float, required=False,
                         help='Threshold of the Jaccard similarity for merhing the scaffolds. If none, scaffolds will not be merged.');
cmdl_parser.add_argument('-scaffolds', '--scaffolds', metavar="scaffolds.csv", type=str, required=False,
                         help='Input of csv file of scaffolds for substructure matching. Automated scaffold identification will be skipped if this is done.');
cmdl_parser.add_argument('-size_sliding_window', '--size_sliding_window', metavar="365", type=int, required=False,
                         help='Size of sliding window for active phase determination');
cmdl_parser.add_argument('-sep', '--sep', metavar=",", type=str, required=False,
                         help='Delimiter to use for seperation of columns (Default: ","');
cmdl_parser.add_argument('-date_column', '--date_column', metavar=",", type=str, required=False,
                         help='Name of the column that contains the registration dates of the compounds. This is used for minin active phases of a chemical series (Default: "First Reg Date").');
cmdl_parser.add_argument('-gpu', '--gpu',default=False, action='store_true', required=False,
                         help="Enable GPU support. Default: False");


# ************************************************************
# ********************** main function ***********************
# ************************************************************

def main():

    import warnings
    warnings.filterwarnings("ignore")
    
    start = time.time();
    random.seed(2)

    print('');
    print('');
    print('');
    print('*****************************************************************');
    print('************** Classification of chemical series ****************');
    print('*****************************************************************');
    print('');
    print('');
    print('');
    
    # get command line input
    args = cmdl_parser.parse_args();
    
    filename = args.data;
        
    algorithm = "UPGMA";
        
    if args.flimit is None:
        flimit = 0.002;
    else:
        flimit = args.flimit;

    if args.min_series_size is None:
        min_cluster_size = 10;
    else:
        min_cluster_size = args.min_series_size;
        
    print("Applying a minimal series size of " + repr(min_cluster_size));
    
    if args.sep is None:
        seperator = ",";
    else:
        seperator = args.sep;
        
    if args.date_column is None:
        date_column = "First Reg Date";
    else:
        date_column = args.date_column;
    
    if args.smiles_column is None:
        smiles_column = "Structure";
    else:
        smiles_column = args.smiles_column;
    
    curr_path = os.path.dirname(os.path.abspath(__file__));
    
    #set the chembl database
    chembldb, Nchembl = make_chembl_db();
    """
    if arthor is not None:
        
        print("Setting up ChEMBL database ...");
        os.system('smi2atdb -j 0 -t {0}{1}.smi {0}{1}.atdb'.format(curr_path, "/Data/chembl_23"));
        os.system('atdb2fp -j 0 {0}{1}.atdb'.format(curr_path, "/Data/chembl_23"));
        os.system('smi2atfp -j 0 -t {0}{1}.smi {0}{1}.atfp'.format(curr_path, "/Data/chembl_23"));
    
        chembldb = arthor.SubDb(curr_path + '/Data/chembl_23.atdb');
        chembldb.set_num_processors(16);
        Nchembl = len(chembldb.search('*'));
    
    else:
        print("Setting up ChEMBL database ...");

        with open(curr_path + '/Data/chembl27_sssdata.pkl','rb') as file:
            chembldb = pickle.load(file);

        Nchembl = len(chembldb);"""
                
    #read the data
    mol_list, proj_db, df_filtered, smiles_list = IO.read_data(filename, seperator, smiles_column);
    
    #read the scaffolds, if they are given
    scaffolds = None;
    if args.scaffolds is not None:
    
        df = pd.read_csv(args.scaffolds, sep=",", header=None);
        tmp_scaffolds = df.iloc[:,0].to_list();

        scaffolds = [];
        for tmp_scaffold in tmp_scaffolds:
            try: 
                a = np.isnan(tmp_scaffold);
            except:
                scaffolds.append(tmp_scaffold);
        
    #UPGMA
    mcs_list, class_labels, comp_label_list, pre_clustering_labels, spec_chembl_list, spec_projectdb_list = Clustering.UPGMA(mol_list, flimit, min_cluster_size, proj_db, chembldb, scaffolds, gpu=args.gpu);
        
    df_filtered = df_filtered.iloc[comp_label_list];
        
    #do umap embedding
    #print("Do UMAP embedding ...");
    #fit = umap.UMAP(metric=cluster_utils.tanimoto_distance, low_memory=True);
    #data = fit.fit_transform(fp_array[np.unique(comp_label_list), :]);
            
    #save the embeddin as 2D csv
    #np.savetxt("UMAP_embedding.csv", data, delimiter=';');
        
    #add column with class labels of pre-clustering to dataframe
    #df_filtered['Pre-Class'] = pre_clustering_labels;
        
    #add column with class labels to dataframe
    df_filtered['Class'] = class_labels;
        
    #add column with mcs and fchembl to dataframe

    df_filtered['MCS'] = mcs_list;
    df_filtered['Spec. ChEMBL'] = spec_chembl_list;
    df_filtered['Spec. Project DB'] = spec_projectdb_list;

    df_filtered = cluster_utils.fraction_of_atoms_explained_by_scaffold(df_filtered);        
     
    df_filtered, df_mcs = cluster_utils.make_mcs_file(df_filtered);
    
    df_mcs.to_csv("scaffolds.csv", index=False);
    
    if args.jaccard_similarity_threshold is not None:
        df_filtered = cluster_utils.merge_scaffolds_according_to_jaccard_similarity(df_filtered, args.jaccard_similarity_threshold);
        
    df_filtered = cluster_utils.calculate_series_purity(df_filtered);
    cluster_utils.calculate_median_tanimoto_similarities_between_series(df_filtered, min_cluster_size);
    
    #if the series should be split into the active times, do it
    if args.size_sliding_window is not None:
        df_filtered = active_times_detection.seperate_series_into_active_phases(df_filtered, date_column, size_sliding_window=args.size_sliding_window, min_num_mols_per_window=min_cluster_size);
        
    
        
    # saving the dataframe 
    df_filtered.to_csv('{0}.csv'.format(algorithm), index=False);
        
    end = time.time();
    totalRuntime = end - start;

    print("****** Summary ******");
    print("Runtime: %.2f s" % totalRuntime);
    
    
#****************************************************
#*** Helper function to make the chembl database ****
#****************************************************
def make_chembl_db():
    
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
            print("Number of compounds in the ChEMBL database: " + repr(len(chembldb)));
            
        except:
            print("Setting up ChEMBL database for RDKit. This needs to be done only once. This will take ca. 30 minutes ...");
                
            sdf_path = chembl_setup.download_chembl(outpath);
            chembl_setup.make_rdkit_substr_lib(outpath);
            with open(curr_path + '/Data/chembl27_sssdata.pkl','rb') as file:
                chembldb = pickle.load(file);
            os.remove(sdf_path);
    
        Nchembl = len(chembldb);
        
    return chembldb, Nchembl;
    
#-----------------------------------------------------------
#-----------------------------------------------------------
#-----------------------------------------------------------
#-----------------------------------------------------------
if (__name__ == "__main__"):
    main()
