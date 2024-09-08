# Created by Maximilian Beckers, December 2021, initial Code by Franziska Kruger et al. J. Chem. Inf. Model. 2020, 60, 6, 2888–2902

import numpy as np
import sklearn
from sklearn.cluster import AgglomerativeClustering
from AutomatedSeriesClassification import cluster_utils
try:
    import arthor;
except ImportError:
    arthor = None;

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdSubstructLibrary    
import time
import random
from scipy import sparse, spatial, cluster
import math

#-----------------------------------------------------------------
def UPGMA(mol_list, flimit, MinClusterSize, proj_db, chembldb, scaffolds, gpu=False):
    
    #calculating a numpy array of fingerprints
    fp_array = cluster_utils.calc_fingerprint_matrix(mol_list);
    
    num_fp = fp_array.shape[0];
    compound_indices = np.arange(num_fp);
    
    if arthor is not None:
        Nchembl = len(chembldb.search('*'));
        #Nproj = len(proj_db.search('*'));
    else:
        Nchembl = len(chembldb);
    
    Nproj = len(proj_db);
        
    #if scaffolds are not given, identify scaffolds in the data
    if scaffolds is None:
    
        print("");
        print("****************************");
        print("****** Pre-partioning ******");
        print("****************************");
        print("");

        print("Partition the data ...");
        start = time.time();
        pre_classes = int(math.ceil(num_fp/5000));
        print("Using " + repr(pre_classes) + " classes ...");

        print("Start Farthest-first classification ...");
        start = time.time();

        #filtered_compounds = np.arange(num_fp);
        fp_array_256bits = cluster_utils.calc_fingerprint_matrix(mol_list, num_bits=256);
        pre_clustering_labels = cluster_utils.farthest_first_clustering(fp_array, pre_classes, fp_array_256bits, use_gpu=gpu);

        end = time.time();

        print("Time elapsed for pre-classification in s: " + repr(end-start));
        print("");

        print("Using the following batch sizes: ")
        for tmp_cluster in np.unique(pre_clustering_labels):
            filtered_compounds = compound_indices[pre_clustering_labels==tmp_cluster];
            print("Number of compounds in  batch " + repr(tmp_cluster) + ": " + repr(filtered_compounds.size));


        MCS_list = [];
        fChembl_list = [];


        for tmp_cluster in np.unique(pre_clustering_labels):

            print("")
            print("****************************")
            print("Processing batch " + repr(tmp_cluster));


            filtered_compounds = compound_indices[pre_clustering_labels==tmp_cluster];
            print("Number of compounds in temporary batch: " + repr(filtered_compounds.size));

            start = time.time();
            print("Calculate distance matrix ...");
            distdata_proj_matrix = cluster_utils.get_dist_matrix(fp_array[filtered_compounds], use_gpu=gpu);
            #distdata_proj_vector = cluster_utils.get_dist_vector(fp_array);

            # Apply UPGMA clustering
            print("Start UPGMA clustering ...");

            cluster = AgglomerativeClustering(n_clusters=10, compute_full_tree=True, affinity = 'precomputed', linkage='average');
            cluster.fit(distdata_proj_matrix);
            children = cluster.children_;
            del cluster;

            children = children[:, 0:2];
            children = children.astype(int);

            #free some memory
            del distdata_proj_matrix;

            end = time.time();
            print("Runtime for UPGMA clustering in s: " + repr(end-start));

            # Assign Clusters
            print("Calculate Node assignments ...");
            NumMolList, MolDict = cluster_utils.calc_node_assignments(children, filtered_compounds.size, filtered_compounds);

            start = time.time();
            print("Filter out Nodes and calculate MCS on the respective clusters ...");
            # filter out irrelevant clusters and calculate MCS on selected clusters
            tmp_MCSlist, tmp_fChembl_list = cluster_utils.determine_relevant_mcs(filtered_compounds.size, mol_list, children, MolDict, chembldb, proj_db, flimit, MinClusterSize);

            print("Number of scaffolds identified in temporary batch: " + repr(len(tmp_MCSlist)));

            MCS_list = MCS_list + tmp_MCSlist;
            fChembl_list = fChembl_list + tmp_fChembl_list;

            del children; 

            end = time.time();
            print("Runtime for MCS calculation in s: " + repr(end-start));
   
    else:
        #only estimate specificities for the given scaffolds
        pre_clustering_labels = np.ones(fp_array.size);
        fChembl_list = [];
        MCS_list = [];
        for tmp_scaffold in scaffolds:
            try:
                tmp_fChembl = cluster_utils.get_fchembl(tmp_scaffold, chembldb, Nchembl, Nchembl);
                
                if tmp_fChembl > flimit:
                    continue;
                
                fChembl_list.append(tmp_fChembl);
                MCS_list.append(tmp_scaffold);
            except:
                continue;
        print("Number of scaffolds: " + repr(len(MCS_list)));
        

    print("");
    print("*******************************************");
    print("**** Do structure matchings using MCS *****");    
    print("*******************************************");
    print("");
    
    if scaffolds is None:
        #take the MCS and do substructure match for assigning compounds to clusters    
        MolAssignment, MCS_dict = cluster_utils.assign_series_to_mcs(MCS_list, mol_list, proj_db, chembldb, Nchembl, MinClusterSize, flimit);
    else:
        MolAssignment, MCS_dict = cluster_utils.assign_series_to_mcs_provided_scaffolds(MCS_list, mol_list, proj_db);
    
    #make dictionary for finding the indices of samples
    #if arthor is not None:
    #    res = proj_db.Search('*');
    #    indices = [int(i.decode("utf-8"))  for i in res];
    #    index_dictionary = dict(zip(indices, range(len(indices))));
        #index_dictionary = dict(zip(range(len(indices)), range(len(indices))));
    #else:
    res = proj_db.GetMatches(Chem.MolFromSmarts('*'), maxResults=len(proj_db));
    indices = list(res);
    index_dictionary = dict(zip(range(len(indices)), range(len(indices))));      
        

    class_labels = [];
    mcs_list = [];
    comp_label_list = [];
    spec_chembl_list = [];
    spec_novartis_list = [];

    #mcs_recalc_list = [];
    #spec_chembl_recalc_list = [];
    #spec_novartis_recalc_list = [];
    
    class_label_counter = 0;
    for tmp_cluster in list(MolAssignment.keys()):
        
        tmp_comp_label_list = [];
        
        #get compound labels in temporary cluster
        for x in MolAssignment[tmp_cluster]:            
            tmp_comp_label_list.append(index_dictionary[x]);

        if len(tmp_comp_label_list) < MinClusterSize:
            continue;
            
        tmp_mcs = MCS_dict[tmp_cluster];
        
        #if tmp_mcs == "recalculate":
        if scaffolds is None:
            tmp_mol_list = [mol_list[i] for i in tmp_comp_label_list];
            #max_num_matches = int(math.ceil(flimit*(Nchembl+2) - 1));
            
            spec_chembl_recalc, tmp_mcs_recalc = cluster_utils.mcs_from_mol_list(tmp_mol_list, chembldb, Nchembl, Nchembl);
        
        #calculate specs for ChEMBL and Novartis
        spec_chembl = cluster_utils.get_fchembl(tmp_mcs, chembldb, Nchembl, Nchembl);
        spec_nov = cluster_utils.get_fchembl(tmp_mcs, proj_db, Nproj, Nproj);
        
        if scaffolds is None:
            if spec_chembl != spec_chembl_recalc:
                continue;
        
        #assign everything
        for x in MolAssignment[tmp_cluster]:           
            
            class_labels.append(class_label_counter);

            mcs_list.append(str(tmp_mcs));
            comp_label_list.append(index_dictionary[x]);
            spec_chembl_list.append(spec_chembl);
            spec_novartis_list.append(spec_nov);
            #mcs_recalc_list.append(tmp_mcs_recalc);
            #spec_chembl_recalc_list.append(spec_chembl_recalc);
            #spec_novartis_recalc_list.append(spec_nov_recalc);
            
        class_label_counter = class_label_counter + 1;
    
    #now add all the molecules that were not found by any scaffold to the noise class  
    for tmp_compound in range(num_fp):
        if tmp_compound not in comp_label_list:
            comp_label_list.append(tmp_compound);
            mcs_list.append("");
            spec_chembl_list.append(0);
            spec_novartis_list.append(0);
            #mcs_recalc_list.append("");
            #spec_chembl_recalc_list.append(0);
            #spec_novartis_recalc_list.append(0);
            class_labels.append(-1);

        
    mcs_list = np.array(mcs_list);
    class_labels = np.array(class_labels);
    class_labels = cluster_utils.rename_class_labels(class_labels);
    spec_chembl_list = np.array(spec_chembl_list);
    spec_projectdb_list = np.array(spec_novartis_list);
    #mcs_recalc_list = np.array(mcs_recalc_list);
    #spec_chembl_recalc_list = np.array(spec_chembl_recalc_list);
    #spec_novartis_recalc_list = np.array(spec_novartis_recalc_list);
    
    print("");
    print("*****************************")
    print("Total number of scaffolds found: " + repr(np.unique(class_labels).size - 1));
    print("Total number of molecules in all classes: " + repr(class_labels.size));
    print("");

    pre_clustering_labels = pre_clustering_labels[comp_label_list];
                
    return mcs_list, class_labels, comp_label_list, pre_clustering_labels, spec_chembl_list, spec_projectdb_list;

