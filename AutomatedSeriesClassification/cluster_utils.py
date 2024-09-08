# Created by Maximilian Beckers, December 2021, initial Code by Franziska Kruger et al. J. Chem. Inf. Model. 2020, 60, 6, 2888–2902

from numba import njit, prange, cuda
from AutomatedSeriesClassification import gpu_utils
import sys, os
import pandas as pd
import numpy as np
import time
import scipy
import warnings
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import AllChem
from rdkit.Chem import rdFMCS
from rdkit.Chem import Lipinski
from rdkit.Chem import rdSubstructLibrary
import random
try:
    import arthor;
except ImportError:
    arthor = None;
import math

    
#---------------------------------------------------------------
#numba implementation of Tanimoto distance
@njit(nopython=True)
def tanimoto_distance(v1, v2):
    #Calculates tanimoto distance for two bit vectors
    
    bit_sum = v1 + v2
    bitwise_and = count_loop_equals2(bit_sum);
    bitwise_or = count_loop_bigger0(bit_sum);
    
    jaccard_distance = 1 - (bitwise_and / bitwise_or);
    
    return jaccard_distance;


#--------------------------------------------------------------
@njit()
def count_loop_equals2(a):
    s = 0
    for i in a:
        if i == 2:
            s += 1
    return s

#--------------------------------------------------------------
@njit()
def count_loop_bigger0(a):
    s = 0
    for i in a:
        if i > 0:
            s += 1
    return s


#---------------------------------------------------------------
def filter_compounds_arthor(smiles_list, sim_db, cutoff, lim_num_neighbors):

    num_fp = len(smiles_list);
    filtered_compounds = np.zeros(num_fp);
    
    sim_db.set_num_processors(16);
    
    for tmp_compound in range(num_fp):
        
        tmp_smiles = smiles_list[tmp_compound];
        
        rs = sim_db.search(tmp_smiles, limit=lim_num_neighbors)
        
        df = pd.read_csv(rs, sep='\s+', names=['id', 'similarity'])
        
        tmp_sim = df["similarity"].to_numpy();
        
        if tmp_sim[lim_num_neighbors-1] > cutoff:
            filtered_compounds[tmp_compound] = 1;
        
    indices = np.arange(num_fp);
    filtered_compounds = indices[filtered_compounds == 1];
    
    return filtered_compounds;


#---------------------------------------------------------------
@njit(nopython=True)
def filter_compounds(fp_array, cutoff, lim_num_neighbors):

    num_fp = fp_array.shape[0];
    filtered_compounds = np.zeros(num_fp);
    
    for tmp_fp_1 in range(num_fp):
        
        neighbor_counter = 0;
        fp_1 = fp_array[tmp_fp_1];
        
        for tmp_fp_2 in range(num_fp):
        
            tmp_distance = tanimoto_distance(fp_1, fp_array[tmp_fp_2]);

            if tmp_distance < (1.0-cutoff):
                neighbor_counter = neighbor_counter + 1;
                if neighbor_counter >= lim_num_neighbors:
                    filtered_compounds[tmp_fp_1] = 1;
                    break;
    
    indices = np.arange(num_fp);
    filtered_compounds = indices[filtered_compounds == 1];
    
    return filtered_compounds;

#--------------------------------------------------------------
@njit(nopython=True)
def square_to_condensed(i, j, n):
    if i < j:
        i, j = j, i
    return int(n*j - j*(j+1)//2 + i - 1 - j)

#---------------------------------------------------------------
#numba implementation of distance vector
@njit(nopython=True, parallel=True)
def get_dist_vector(fp_array):
    
    num_fp = fp_array.shape[0];
    distances = np.zeros(int(num_fp * (num_fp-1) / 2), dtype=np.float32);
    
    #calculate distance matrix
    for tmp_fp_1 in prange(num_fp):
        
        tmp_fp_sample_1 = fp_array[tmp_fp_1, :];
        
        for tmp_fp_2 in prange(tmp_fp_1):
                
                #Calculates tanimoto distance for two bit vectors
                bit_sum = tmp_fp_sample_1 + fp_array[tmp_fp_2, :];
                bitwise_and = count_loop_equals2(bit_sum);
                bitwise_or = count_loop_bigger0(bit_sum);
    
                tmp_dist = np.float32(1 - (bitwise_and / float(bitwise_or)));
              
                distances[square_to_condensed(tmp_fp_1, tmp_fp_2, num_fp)] = tmp_dist;
                
    return distances;

#----------------------------------------------------------------
@njit(nopython=True, parallel=True)
def get_dists_between_two_sets(fp_array_1, fp_array_2):
    
    num_fp_1 = fp_array_1.shape[0];
    num_fp_2 = fp_array_2.shape[0];
    
    dists = np.zeros((num_fp_1)*(num_fp_2));
    
    #calculate distances
    for fp_ind_1 in prange(num_fp_1):
        
        tmp_fp_1 = fp_array_1[fp_ind_1, :];
        
        for fp_ind_2 in prange(num_fp_2):
            
            tmp_fp_2 = fp_array_2[fp_ind_2, :];
            
            #Calculates tanimoto distance for two bit vectors
            bit_sum = tmp_fp_1 + tmp_fp_2;
              
            bitwise_and = count_loop_equals2(bit_sum);
            bitwise_or = count_loop_bigger0(bit_sum);
    
            tmp_dist = np.float32(1 - (bitwise_and / float(bitwise_or)));
        
            dists[fp_ind_1*(num_fp_2) + fp_ind_2] = 1-tmp_dist;
                 
    return dists;

#---------------------------------------------------------------
def get_dist_matrix(fp_array, dates=None, use_gpu=False):

    if use_gpu:
                
        nb = fp_array.shape[1];
        gpu_utils.NB = nb;
        num_fp = fp_array.shape[0];
        gpu_utils.NUM_FP = num_fp;
        
        distances = np.zeros((num_fp, num_fp));

        threadsperblock = (32, 32);
        blockspergrid_x = math.ceil(distances.shape[0] / threadsperblock[0])
        blockspergrid_y = math.ceil(distances.shape[1] / threadsperblock[1])
        blockspergrid = (blockspergrid_x, blockspergrid_y)

        gpu_utils.get_dist_matrix_gpu[blockspergrid, threadsperblock](fp_array, distances)

    else:
        distances = get_dist_matrix_cpu(fp_array, dates);

    return distances;

    
#---------------------------------------------------------------
#numba implementation of distance matrix
@njit(nopython=True, parallel=True)
def get_dist_matrix_cpu(fp_array, dates=None):
    
    num_fp = fp_array.shape[0];
    distances = np.zeros((num_fp, num_fp), dtype=np.float32);
    
    #calculate distance matrix
    for tmp_fp_1 in prange(num_fp):
        
        tmp_fp_sample_1 = fp_array[tmp_fp_1, :];
        
        for tmp_fp_2 in prange(tmp_fp_1):
                
                #Calculates tanimoto distance for two bit vectors
                bit_sum = tmp_fp_sample_1 + fp_array[tmp_fp_2, :];
              
                bitwise_and = count_loop_equals2(bit_sum);
                bitwise_or = count_loop_bigger0(bit_sum);
    
                tmp_dist = np.float32(1 - (bitwise_and / float(bitwise_or)));
              
                distances[tmp_fp_1, tmp_fp_2] = tmp_dist;
                distances[tmp_fp_2, tmp_fp_1] = tmp_dist;
                
                if dates is not None:
                    if dates[tmp_fp_1] > dates[tmp_fp_2]:
                         distances[tmp_fp_1, tmp_fp_2] = 0;
                    if dates[tmp_fp_2] > dates[tmp_fp_1]:
                        distances[tmp_fp_2, tmp_fp_1] = 0;
                        
    return distances;

#---------------------------------------------------
def calc_fingerprint_matrix(mol_list, num_bits=1024):

    #get list of fingerprints from smiles
    fp_list = [];
    for tmp_mol in mol_list:
        tmp_fp = AllChem.GetMorganFingerprintAsBitVect(tmp_mol, 2, nBits=num_bits);
        fp_list.append(tmp_fp);
    
    #make array of fingerprints
    num_fp = len(fp_list)

    fp_array = np.zeros((num_fp, num_bits), dtype=np.int8)

    for tmp_fp in range(num_fp):
        tmp_array = np.zeros((0,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp_list[tmp_fp], tmp_array)
        fp_array[tmp_fp, :] = tmp_array
        
    return fp_array
             
#--------------------------------------------------------------
def make_project_db( smiles_array):
         
    print("");
    print("Creating substructure library with RDKit ...");
    mols = rdSubstructLibrary.CachedTrustedSmilesMolHolder();
    fps = rdSubstructLibrary.PatternHolder();
    for smi in smiles_array:
        tmp_mol = Chem.AddHs(Chem.MolFromSmiles(smi));
        mols.AddSmiles(Chem.MolToSmiles(tmp_mol));
        fps.AddFingerprint(Chem.PatternFingerprint(tmp_mol));

    proj_db = rdSubstructLibrary.SubstructLibrary(mols, fps);
        
    return proj_db;

#------------------------------------------------------------
def calc_node_assignments(children, Ndata, filtered_compounds):
    
    # Assigns molecules to the clusters of the UPGMA tree
    NumMolList=[];
    MolDict={};
    
    for i in range(len(children)):
        N=0;
        mols_assigned=[];
        for j in range(len(children[i])):
            if children[i][j] < Ndata:
                N += 1;
                mols_assigned.append(filtered_compounds[children[i][j]]);
            else:
                N += NumMolList[children[i][j]-Ndata];
                mols_assigned += MolDict[children[i][j]];
                
        NumMolList.append(N);
        MolDict[i+Ndata] = mols_assigned;
        
    return NumMolList, MolDict

#-------------------------------------------------------------
def CalcScore(children, distdata, NumMolList):
    
    # Calculates intra-cluster distance scores
    N_init=len(distdata)
    singletons=[1]*N_init
    NumMolList=singletons+NumMolList
    dist_up=np.zeros((N_init*2-1,N_init*2-1))
    dist_up[0:N_init,0:N_init]=distdata
    ScoreDict={}
    for i in range(len(children)):
        c1=children[i][0]
        c2=children[i][1]
        N1=NumMolList[c1]
        N2=NumMolList[c2]
        ScoreDict[i+N_init]=dist_up[c1, c2]
        update_x=np.zeros((1,N_init+i))
        for k in range(N_init+i):
            if k in [c1,c2]:
                dist_k=0
            else: 
                dist_k=(dist_up[c1,k]*N1+dist_up[c2,k]*N2)/(N1+N2)
            update_x[0,k]=dist_k
        dist_up[N_init+i,0:N_init+i]=update_x[0,:]
        dist_up[0:N_init+i,N_init+i]=update_x[0,:]
        
    return ScoreDict


#----------------------------------------------------------------
def determine_relevant_mcs(Ndata, mol_list, children, MolDict, chembldb, proj_db, flimit, MinClusterSize):
    
    # filter out irrelevant clusters and calculate MCS on selected clusters
    #class_labels = np.ones((len(mol_list))) * (-1);
    currlayer=[Ndata*2 - 2];
    MCSlist = [];
    intra_series_similarity = [];
    fChembl_list = [];
        
    if arthor is not None:
        Nchembl=len(chembldb.search('*'));
    else:
        Nchembl = len(chembldb);
        
    max_num_matches = int(math.ceil(flimit*(Nchembl+2) - 1));
    
    while len(currlayer) > 0:
        childlayer=[];
        for c in currlayer:
            if c >= Ndata: 
                if len(MolDict[c]) >= MinClusterSize:
                    
                    tmp_mol_list = [mol_list[i] for i in MolDict[c]];
                    fChembl, Smarts = mcs_from_mol_list(tmp_mol_list, chembldb, Nchembl, max_num_matches);
                              
                    if fChembl>=flimit:
                        childlayer+=children[c-Ndata].tolist();
                    else:
                        MCSlist.append(Smarts);
                        fChembl_list.append(fChembl);

        currlayer = childlayer;
    
    return MCSlist, fChembl_list;

#--------------------------------------------------------------------
def assign_series_to_mcs(MCS_list, mol_list, proj_db, chembldb, Nchembl, min_cluster_size, flimit):
        
    #make dictionary for finding the indices of samples
    res = proj_db.GetMatches(Chem.MolFromSmarts('*'), maxResults=len(proj_db));
    indices = list(res);
    index_dictionary = dict(zip(range(len(indices)), range(len(indices))));   
        
    # assign series to MCS of selected clusters
    #smartslist = [v[2] for v in MCSdict.values()]
    smartslist = MCS_list;
    MolAssign_prel = {}
    MolAssignment = {}
    MCS_dict = {}
    series_size = [];
    merged_series = [];
    
    merging_threshold = 0.75;
              
    print("Number of scaffolds BEFORE merging series with high overlap: " + repr(len(MCS_list)));
        
    #search the database for the smarts and append them
    for s in range(len(smartslist)):
        
        #if arthor is not None: #use arthor utilities
        #    res = proj_db.Search(smartslist[s]);
        #    mols = [int(i.decode("utf-8"))  for i in res];
        #else: #use rdkit
        mols = proj_db.GetMatches(Chem.MolFromSmarts(smartslist[s]), maxResults=len(proj_db));        
              
        MolAssign_prel[s] = list(mols);
        series_size.append(len(mols));

        
    #sort the keys according to size of the series
    sorted_series_ind = np.argsort(series_size).astype(int);
        
    #overlaps = [];
    mergings = [];
    # remove all series that are entirely in another series  
    for tmp_series_1 in range(len(smartslist)):
        key1 = sorted_series_ind[tmp_series_1];
        add = 1;
        
        if len(set(MolAssign_prel[key1])) == 0: #if the scaffold contains no molecules
            add = 0;
            continue;
        
        for tmp_series_2 in range(tmp_series_1, len(smartslist)):
            key2 = sorted_series_ind[tmp_series_2];
            if key2 != key1:                     
                 
                    
                """    
                if len(set(MolAssign_prel[key1]).intersection(set(MolAssign_prel[key2])))/float(len(set(MolAssign_prel[key1]))) > 0.9:
                    add = 0;
                    MCS_list[key2] = MCS_list[key1];
                    break;       
                """
                #overlaps.append(len(set(MolAssign_prel[key1]).intersection(set(MolAssign_prel[key2])))/float(len(set(MolAssign_prel[key1]))));
                

                #are the two series overlapping?
                #len_intersection = len(set(MolAssign_prel[key1]).intersection(set(MolAssign_prel[key2])));
                #len_union = float(len(set(MolAssign_prel[key1]))) + float(len(set(MolAssign_prel[key2]))) - len_intersection;
                if len(set(MolAssign_prel[key1]).intersection(set(MolAssign_prel[key2])))/float(len(set(MolAssign_prel[key1]))) >= merging_threshold:

                    #if overlappping, then merge the series
                    tmpMolAssign = MolAssign_prel[key1] + MolAssign_prel[key2];
                    tmpMolAssign = list(set(tmpMolAssign)); #avoid duplication of molecules
                    
                    #calculate the spec and the mcs of the merged series
                    tmp_comp_label_list = [index_dictionary[x] for x in tmpMolAssign];
                    tmp_mol_list = [mol_list[i] for i in tmp_comp_label_list];
                    
                    if len(tmp_mol_list) <= min_cluster_size:
                        continue;
                    
                    flimit_merging = flimit;
                    max_num_matches = int(math.ceil(flimit_merging*(Nchembl+2) - 1));
                    fChembl, tmp_mcs_recalc = mcs_from_mol_list(tmp_mol_list, chembldb, Nchembl, max_num_matches);
                    
                    #if the mcs of the merged series is a valid scaffold, keep the merging
                    #coverage_of_mcs = fraction_of_atoms_explained(tmp_mol_list, tmp_mcs_recalc);
                    #if coverage_of_mcs > 0.5:
                    #    if intra_series_similarities(tmp_mol_list) > 0.322:
                    if fChembl < flimit_merging:
                            mergings.append([MCS_list[key1], MolAssign_prel[key1], MCS_list[key2], MolAssign_prel[key2]]);
                            MolAssign_prel[key2] = MolAssign_prel[key1] + MolAssign_prel[key2];
                            MolAssign_prel[key2] = list(set(MolAssign_prel[key2])); #avoid duplication of molecules
                            MCS_list[key2] = tmp_mcs_recalc;
                            add = 0;
                        
                
        if (add == 1) and (MolAssign_prel[key1] not in MolAssignment.values()):
            MolAssignment[key1] = MolAssign_prel[key1];
        
    MolAssignment = {k: MolAssignment[k] for k in MolAssignment.keys() if len(MolAssignment[k]) > min_cluster_size};
    #MCSdict = {k:(MCSdict[k][0], len(MolAssignment[k]), MCSdict[k][2], MolAssignment[k]) for k in MolAssignment.keys()}
      
    MCS_dict = {k: MCS_list[k] for k in MolAssignment.keys()};
      
    print("Number of scaffolds AFTER merging series with high overlap: " + repr(len(MolAssignment)));
    
    #write the overlaps to array
    #np.savetxt("series_overlaps.txt", np.array(overlaps), delimiter=";");
    #import pickle as pkl
    #with open('merging_steps.pkl','wb') as f:
    #    pkl.dump(mergings, f);
    
    return MolAssignment, MCS_dict;

#--------------------------------------------------------------------
def assign_series_to_mcs_provided_scaffolds(MCS_list, mol_list, proj_db):  
        
    # assign series to MCS of selected clusters
    #smartslist = [v[2] for v in MCSdict.values()]
    smartslist = MCS_list;
    MolAssignment = {}
    MCS_dict = {}
    series_size = [];
    merged_series = [];
    
              
    print("Number of scaffolds: " + repr(len(MCS_list)));
        
    #search the database for the smarts and append them
    for s in range(len(smartslist)):
        mols = proj_db.GetMatches(Chem.MolFromSmarts(smartslist[s]), maxResults=len(proj_db));        
        
        MolAssignment[s] = list(mols);

    MolAssignment = {k:MolAssignment[k] for k in MolAssignment.keys()};
      
    MCS_dict = {k: MCS_list[k] for k in MolAssignment.keys()};
    
    return MolAssignment, MCS_dict;


#-------------------------------------------------------------
def mcs_from_mol_list(mollist, chembldb, Nchembl, max_num_matches):
    
    MCSSmarts2 = rdFMCS.FindMCS(mollist, atomCompare=rdFMCS.AtomCompare.CompareAny, bondCompare=rdFMCS.BondCompare.CompareOrderExact, ringMatchesRingOnly=False, timeout=1).smartsString
    MCSSmarts = rdFMCS.FindMCS(mollist, atomCompare=rdFMCS.AtomCompare.CompareElements, bondCompare=rdFMCS.BondCompare.CompareOrder, ringMatchesRingOnly=False, timeout=1).smartsString
    
    
    if MCSSmarts2 == '': 
        fChembl2 = 1
    else: 
        fChembl2 = get_fchembl(MCSSmarts2, chembldb, Nchembl, max_num_matches);
    
    if MCSSmarts == '': 
        fChembl = 1
    else:
        fChembl = get_fchembl(MCSSmarts, chembldb, Nchembl, max_num_matches);
        
    if fChembl2 < fChembl:
        fChembl = fChembl2
        MCSSmarts = MCSSmarts2
        
    return fChembl, MCSSmarts;
    
#--------------------------------------------------------------
def get_fchembl(qry, chembldb, Ntot, max_num_matches, qryformat='Smarts'):
    
    if arthor is not None: #use arthor
        if qryformat == 'Smarts':
            try:
                results = chembldb.search(qry, limit = max_num_matches);
            except:
                results=chembldb.GetMatches(Chem.MolFromSmarts(qry), numThreads=16, maxResults = max_num_matches);
                
        elif qryformat == 'MDL':
            with open(qry) as f:
                qryarthor = arthor.Query(f.read(),"Mdl")
            results = chembldb.search(str(qryarthor), limit = max_num_matches);

    else:
        if qryformat=='Smarts':
            qry = Chem.MolFromSmarts(qry);
        elif qryformat=='MDL':
            qry = Chem.MolFromMolFile(qry);
            
        results=chembldb.GetMatches(qry, numThreads=16, maxResults = max_num_matches);
            
    fChembl=(len(results)+1)/(Ntot+2);

    return fChembl;

#--------------------------------------------------------------
def rename_class_labels(class_labels):
    
    class_indices = np.unique(class_labels);
    new_class_labels = np.copy(class_labels);
    num_classes = class_indices.size;
    
    class_counter = 0;
    for tmp_class in class_indices:
    
        if tmp_class != -1:
            new_class_labels[class_labels == tmp_class] = class_counter;
            class_counter = class_counter + 1;
            
    return new_class_labels;
    
#-------------------------------------------------------------
def calculate_mcs_of_clusters(class_labels, mol_list, chembldb, Nchembl):
    
    print("Calculate MCS for each cluster ...");
    num_compounds = class_labels.size;
    indices = np.arange(num_compounds);
    mcs_list = [""]*num_compounds;
    fChembl_list = np.zeros(class_labels.size);
    
    for tmp_class in np.unique(class_labels):
        
        min_f = 10;
        for tmp_iteration in range(10):
            
            tmp_indices = indices[class_labels == tmp_class];
            
            tmp_mol_indices = np.random.choice(tmp_indices, int(0.5*tmp_indices.size), replace=False);
            tmp_mol_list = [mol_list[tmp_mol] for tmp_mol in tmp_mol_indices];
            fChembl, MCSSmarts = mcs_from_mol_list(tmp_mol_list, chembldb, Nchembl);
            
            if fChembl < min_f:
                min_f = fChembl;
                fChembl_list[class_labels == tmp_class] = fChembl;
        
                for tmp_mol in tmp_indices:
                    mcs_list[tmp_mol] = MCSSmarts;

    
    return mcs_list, fChembl_list;

#---------------------------------------------------------------
def fraction_of_atoms_explained(mol_list, smarts):

    mol_smarts = Chem.MolFromSmarts(smarts);
    cov_list = [];
    
    for tmp_mol in mol_list:
        cov = len(tmp_mol.GetSubstructMatch(mol_smarts))/Lipinski.HeavyAtomCount(tmp_mol);
        cov_list.append(cov);
        
    return np.nanmedian(cov_list);
    
#---------------------------------------------------------------
def intra_series_similarities(mol_list, num_bits=1024):
    
    tmp_sims = [];
                
    for ind1 in range(len(mol_list)):
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol_list[ind1], 2, nBits=num_bits);
        for ind2 in np.random.choice(np.arange(len(mol_list)), size=10):
            if ind1==ind2:
                continue;
                
            fp2 = AllChem.GetMorganFingerprintAsBitVect(mol_list[ind2], 2, nBits=num_bits);
            tmp_sims.append(DataStructs.FingerprintSimilarity(fp1, fp2));
        
    median_similarity = np.percentile(tmp_sims, 50);
    
    return median_similarity;
#---------------------------------------------------------------
def calculate_series_purity(df, num_bits=1024):
    
    print("");
    print("Calculating series purity ...");
    print("");
    
    classes = df["Class"].to_numpy();
    num_classes = np.unique(classes).size - 1; #ignore the noise class
    
    percentile_1 = np.zeros(classes.size);
    percentile_10 = np.zeros(classes.size);
    percentile_50 = np.zeros(classes.size);

    for tmp_class in range(num_classes):
    
        tmp_df = df[df["Class"]==tmp_class];
    
        tmp_sims = [];
        tmp_smiles = tmp_df["Structure"].to_list();
                
        for ind1 in range(len(tmp_smiles)):
            fp1 = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(tmp_smiles[ind1]), 2, nBits=num_bits);
            for ind2 in np.random.choice(np.arange(len(tmp_smiles)), size=10):
                if ind1==ind2:
                    continue;
                
                fp2 = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(tmp_smiles[ind2]), 2, nBits=num_bits);
                tmp_sims.append(DataStructs.FingerprintSimilarity(fp1, fp2));
        
        percentile_1[classes == tmp_class] = np.percentile(tmp_sims, 1);
        percentile_10[classes == tmp_class] = np.percentile(tmp_sims, 10);
        percentile_50[classes == tmp_class] = np.percentile(tmp_sims, 50);
        
    df["Intra-class sim. 1% percentile"] = percentile_1;
    df["Intra-class sim. 10% percentile"] = percentile_10;
    df["Intra-class sim. 50% percentile"] = percentile_50;
    
    return df;
    
#-------------------------------------------------------------
def farthest_first_clustering(fp_array, num_clusters, fp_array_256bits, use_gpu=False):
    
    
    outlier_compds = find_outliers(fp_array_256bits,cutoff=0.4, lim_num_neighbors=100, use_gpu=use_gpu);
    centroids = determine_centroids(fp_array, num_clusters, outlier_compds);
    labels = assign_mols_to_centroids(fp_array, centroids);
    
    return labels;
    
#--------------------------------------------------------------
@njit(nopython=True)
def determine_centroids(fp_array, num_clusters, outlier_compds):
    
    num_fp = fp_array.shape[0];
    centroids = np.zeros(num_clusters); 
    
    #set the first centroid
    centroids[0] = 0;
    
    #*************************************************    
    #************ determine the centroids ************
    #*************************************************
    
    for tmp_centroid_assign in range(1, num_clusters):
        
        distances = np.zeros((tmp_centroid_assign, num_fp));

        #calculate the average distance for each mol to the centroid
        for tmp_fp_ind in range(num_fp):

            tmp_fp = fp_array[tmp_fp_ind, :];
        
            for tmp_centroid in range(tmp_centroid_assign):
                        
                tmp_centroid_fp = fp_array[int(centroids[tmp_centroid]), :];
                distances[tmp_centroid, tmp_fp_ind] = tanimoto_distance(tmp_centroid_fp, tmp_fp);
                
        avg_distance = np.zeros(num_fp);
        for tmp_fp_ind in range(num_fp):
            avg_distance[tmp_fp_ind] = np.mean(distances[:, tmp_fp_ind]);
            
            
        #check for the number of neighbours
        sorted_avgs_inds = np.flip(np.argsort(avg_distance));
        for tmp_sort_ind in sorted_avgs_inds:
                
            #check if the compound is already a centroid
            in_centroids = False;
            for i in range(len(centroids)):
                if centroids[i] == tmp_sort_ind:
                    in_centroids = True;
                    break;
                
            if in_centroids:
                continue;

            if outlier_compds[tmp_sort_ind]:
                continue;
            else:
                centroids[tmp_centroid_assign] = tmp_sort_ind;
                break;
    
    return centroids;

#---------------------------------------------------------------
def find_outliers(fp_array, cutoff, lim_num_neighbors, use_gpu=False):
    
    if use_gpu:
        
        cuda.select_device(0);

        print("Using the GPU for outlier cleaning ...");
        cuda.select_device(0);
        print(cuda.detect());
        
        nb = fp_array.shape[1];
        gpu_utils.NB_CLEANING = nb;
        num_fp = fp_array.shape[0];
        gpu_utils.NUM_FP = num_fp;
        
        outlier_compds = np.array([True]*num_fp);
        cutoff = 0.4;
        lim_num_neighbors = 100;

        threadsperblock = 32
        blockspergrid = (outlier_compds.size + (threadsperblock - 1)) // threadsperblock

        gpu_utils.find_outliers_GPU[blockspergrid, threadsperblock](fp_array, outlier_compds, cutoff, lim_num_neighbors);
        

    else:
        outlier_compds = find_outliers_cpu(fp_array, cutoff=0.4, lim_num_neighbors=100);

    return outlier_compds;

#--------------------------------------------------------------
@njit(nopython=True, parallel=True)
def assign_mols_to_centroids(fp_array, centroids):
    
    num_fp = fp_array.shape[0];
    num_clusters = centroids.shape[0];
    class_labels = np.zeros(num_fp);
    
    for tmp_fp_ind in prange(num_fp):
        
        tmp_fp = fp_array[tmp_fp_ind, :];
        
        min_dist = 10;
        for tmp_centroid in range(num_clusters):
            
            tmp_centroid_fp = fp_array[int(centroids[tmp_centroid]), :];
            tmp_dist = tanimoto_distance(tmp_fp, tmp_centroid_fp);
            
            if tmp_dist < min_dist:
                min_dist = tmp_dist;
                tmp_class = tmp_centroid;
                
        class_labels[tmp_fp_ind] = tmp_class;
    
    return class_labels;
    
#--------------------------------------------------------------
@njit(nopython=True)
def find_outliers_cpu(fp_array, cutoff, lim_num_neighbors):

    num_fp = fp_array.shape[0];
        
    outlier_compds = [True]*num_fp;
    for tmp_fp_1 in range(num_fp):
        neighbor_counter = 0;
        for tmp_fp_2 in range(num_fp):

            tmp_distance = tanimoto_distance(fp_array[tmp_fp_1], fp_array[tmp_fp_2]);

            if tmp_distance < (1.0-cutoff):
                neighbor_counter = neighbor_counter + 1;
                if neighbor_counter >= lim_num_neighbors:
                    outlier_compds[tmp_fp_1] = False;
                    break;
     
    return np.array(outlier_compds);

#-------------------------------------------------------------------
def fraction_of_atoms_explained_by_scaffold(df):
    
    print("");
    print("Calculating the fraction of the compounds explained by the scaffolds ...");
    
    smis = df["Structure"].to_numpy();
    mcss = df["MCS"].to_numpy();
    
    frac_atoms_explained = [];
    for tmp_comp in range(len(smis)):
        
        if len(mcss[tmp_comp]) == 0:
            frac_atoms_explained.append(np.nan);
            continue;
        
        tmp_mol = Chem.MolFromSmiles(smis[tmp_comp]);
        tmp_mcs_mol = Chem.MolFromSmarts(mcss[tmp_comp]);
        cov = len(tmp_mol.GetSubstructMatch(tmp_mcs_mol))/Chem.Lipinski.HeavyAtomCount(tmp_mol);
        
        frac_atoms_explained.append(cov);
    
    df["Fraction of atoms explained by MCS"] = frac_atoms_explained;
    
    return df;
    
    
#-----------------------------------------------------------------------------------------
def calculate_median_tanimoto_similarities_between_series(df, min_series_size, num_bits=1024):
    
    print("");
    print("Determining median Tanimoto similarities between series ...");
    
    classes = np.unique(df["Class"].to_numpy());
    num_classes = len(classes[classes>-1]);
    
    #generate mols and fingerprint
    mol_list = [];
    fp_list = [];
    for tmp_smi in df["Structure"].to_list():
        mol = Chem.MolFromSmiles(tmp_smi);
        fp_list.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=num_bits));  
        

    smiles = df["Structure"].to_numpy();
    inds = np.arange(len(df["Class"].to_numpy()));
    class_arr = df["Class"].to_numpy();
    sims_between_classes = np.zeros((num_classes, num_classes));

    for class_1 in range(num_classes):
        
        fps_class_1 = np.random.choice(inds[class_arr==class_1], min_series_size, replace=False);
        smiles_class_1 = smiles[class_arr==class_1];
        
        for class_2 in range(class_1+1, num_classes):   
        
            fps_class_2 = np.random.choice(inds[class_arr==class_2], min_series_size, replace=False);
            smiles_class_2 = smiles[class_arr==class_2];

            #if there is not enough compound overlap, do not merge
            #smiles_intersection = np.intersect1d(smiles_class_1, smiles_class_2);
            #if len(smiles_intersection)/min(len(smiles_class_1), len(smiles_class_2)) > 0.75:
            #    continue;
            
            sims=[];
            for tmp_fp_1 in fps_class_1:

                for tmp_fp_2 in fps_class_2:
                    tmp_sim = DataStructs.FingerprintSimilarity(fp_list[tmp_fp_1], fp_list[tmp_fp_2]);
                    sims.append(tmp_sim);

            sims_between_classes[class_1, class_2] = np.nanmedian(sims);
            sims_between_classes[class_2, class_1] = np.nanmedian(sims);
            
    #adj_matrix = np.copy(sims_between_classes);
    #adj_matrix[adj_matrix < similarity_threshold] = 0.0;
    
    #get the number of connected components
    #num_conn_comp, conn_comps = scipy.sparse.csgraph.connected_components(adj_matrix, directed=False, return_labels=True);
    
    #import pickle as pkl
    #with open('tanimoto_similarities_betwn_series.pkl','wb') as f:
    #    pkl.dump(sims_between_classes, f);
    
#-----------------------------------------------------------------------------------------
def merge_scaffolds_according_to_jaccard_similarity(df, merging_threshold = 0.5):

    print("");
    print("Determining Jaccard similarities between series ...");
    
    #*****************************************************
    #calculate jaccard similarities between series
    #*****************************************************
    
    #get all classes where the compound is contained 
    classes = np.unique(df["Class"].to_numpy());
    classes = classes[classes > -1]; #ignore the noise class
    num_classes = classes.size;
    classes_all_compounds = np.copy(df["Class"].to_numpy());
    
    num_compounds = classes_all_compounds.size;
    structs = df["Structure"].to_numpy();
    _, ID = np.unique(structs, return_inverse=True);

    num_classes_per_compound = [];
    classes_per_compound = [];

    for tmp_compound_ind in range(num_compounds):

        #get all classes where this coumpound is contained 
        tmp_classes = classes_all_compounds[ID == ID[tmp_compound_ind]];
        num_classes_per_compound.append(tmp_classes.size);
        classes_per_compound.append(tmp_classes);

        
    #df["Scaffold IDs"] = classes_per_compound;
    #df["Num. scaffolds per compound"] = num_classes_per_compound;
    classes_per_compound = np.array(classes_per_compound, dtype="object");
       
    #calculate overlap with other series
    overlap_matrix = np.zeros((num_classes, num_classes));
    num_neighbour_classes = np.zeros(num_classes);

    for tmp_class in range(num_classes):
        tmp_classes = np.array(classes_per_compound)[classes_all_compounds==tmp_class];

        for tmp_arr in tmp_classes:

            tmp_arr = tmp_arr.astype(int);
            overlap_matrix[tmp_class, tmp_arr] = overlap_matrix[tmp_class, tmp_arr] + 1;

        tmp_matrix = overlap_matrix[tmp_class, :];
        num_neighbour_classes[tmp_class] = len(tmp_matrix[tmp_matrix>0])-1;

    np.fill_diagonal(overlap_matrix, 0);
    molecule_overlap_matrix = np.copy(overlap_matrix);

    #now get jaccard similarities
    for tmp_class1 in range(num_classes):
        for tmp_class2 in range(num_classes):

            if overlap_matrix[tmp_class1, tmp_class2] == 0:
                continue;

            tmp_df_1 = df[df["Class"] == tmp_class1];
            tmp_df_2 = df[df["Class"] == tmp_class2];

            size_class_1 = len(tmp_df_1["Class"].to_list());
            size_class_2 = len(tmp_df_2["Class"].to_list());
            
            #calculate jaccard similarity
            overlap_matrix[tmp_class1, tmp_class2] = overlap_matrix[tmp_class1, tmp_class2]/ (size_class_1 + size_class_2 - overlap_matrix[tmp_class1, tmp_class2]);
    
    #import pickle as pkl
    #with open('jaccard_similarities_betwn_scaffoldmatches.pkl','wb') as f:
    #    pkl.dump(overlap_matrix, f);
    
    #prune the similarities for edges bigger than the threshold
    overlap_matrix[overlap_matrix<0.5] = 0.0;
    
    #now get the connected components in the graph
    num_conn_comp, conn_comps = scipy.sparse.csgraph.connected_components(overlap_matrix, directed=False, return_labels=True);
    
    print("Number of connected components at a Jaccard similarity threshold of " + repr(merging_threshold) + ": " + repr(num_conn_comp));
    
    #now assign the series to each class
    series_all_compounds = np.copy(classes_all_compounds);
    
    for tmp_class in range(num_classes):
        series_all_compounds[classes_all_compounds == tmp_class] = conn_comps[tmp_class];
    
    #df["Scaffold ID"] = classes_all_compounds;
    df["Class"] = series_all_compounds;
            
    #now remove duplicated molecules in the same merged series and make multiclass labels for the scaffold
    compound_ids = np.arange(len(df["Class"].to_list()));
    ids_to_keep = [];
    multi_scaffold_labels = [];
    
    for tmp_series in np.unique(series_all_compounds):
        
        tmp_ids = ID[series_all_compounds==tmp_series];
        tmp_compound_ids = compound_ids[series_all_compounds==tmp_series];
        tmp_scaffolds = classes_all_compounds[series_all_compounds==tmp_series];
        
        _, unique_comps_in_series = np.unique(tmp_ids, return_index=True);
        
        for tmp_cmpd in unique_comps_in_series:
            ids_to_keep.append(tmp_compound_ids[tmp_cmpd]);
            
            scaffolds_in_series_this_comp = tmp_scaffolds[tmp_ids == tmp_ids[tmp_cmpd]];
            multi_scaffold_labels.append(str(list(scaffolds_in_series_this_comp)));
            
    new_df = df.iloc[ids_to_keep, :].copy();
    #df.loc[:, "Scaffold ID"] = multi_scaffold_labels;
    new_df["Scaffold ID"] = multi_scaffold_labels;
    
    return new_df;
    
#--------------------------------------------------------------------
def make_mcs_file(df):
    
    scaffold_ids = df["Class"].to_numpy();
    _, unids = np.unique(scaffold_ids, return_index=True);
    
    
    out_df = df[["Class", "MCS", "Spec. ChEMBL", "Spec. Project DB", "Fraction of atoms explained by MCS"]];
    out_df = out_df.iloc[unids, :];
    
    out_df.rename(columns={'Class':'Scaffold ID',
                          'MCS':'Scaffold SMARTS',
                          "Fraction of atoms explained by MCS": "Median fraction of atoms of all matching molecules explained by Scaffold"}, 
                 inplace=True);
    
    df = df.drop(columns=["MCS", "Spec. ChEMBL", "Spec. Project DB", "Fraction of atoms explained by MCS"]);
    df_mcs = out_df;
    
    return df, df_mcs;
    
    