# Created by Maximilian Beckers, December 2021

import numba
import numpy as np
import math
from numba import cuda, prange, float32

NUM_FP = 2000;
NB_CLEANING = 256;
NB = 1024;
    
#------------------------------------------------------------
@cuda.jit
def get_dist_matrix_gpu(fp_array, dist_matrix):
    
    x, y = cuda.grid(2);
    
    bit_sum = cuda.local.array(shape=NB, dtype=float32);
    
    if (x < fp_array.shape[0]) and (y < fp_array.shape[0]):
    
        for i in range(bit_sum.shape[0]):
            bit_sum[i] = fp_array[x, i] + fp_array[y, i];
        
    
    if (x < fp_array.shape[0]) and (y < fp_array.shape[0]):
        bitwise_and = 0.0;
        for i in bit_sum:
            if i == 2:
                bitwise_and = bitwise_and + 1.0;

        bitwise_or = 0.0;
        for i in bit_sum:
            if i > 0.0:
                bitwise_or = bitwise_or + 1.0;

        dist_matrix[x,y] = 1.0 - (bitwise_and / bitwise_or);
    
#------------------------------------------------------------
@cuda.jit
def get_dists_to_sample_gpu(fp_sample, fp_array, dist_matrix):
    
    x = cuda.grid(1);
    
    bit_sum = cuda.local.array(shape=NB, dtype=float32);
    
    if x < fp_array.shape[0]:
    
        for i in range(bit_sum.shape[0]):
            bit_sum[i] = fp_sample[i] + fp_array[x, i];
    
        bitwise_and = 0.0;
        for i in bit_sum:
            if i == 2:
                bitwise_and = bitwise_and + 1.0;

        bitwise_or = 0.0;
        for i in bit_sum:
            if i > 0.0:
                bitwise_or = bitwise_or + 1.0;

        dist_matrix[x] = 1.0 - (bitwise_and / bitwise_or);    

#--------------------------------------------------------------
@cuda.jit
def find_outliers_GPU(fp_array, outlier_compds, cutoff, lim_num_neighbors):
        
    x = cuda.grid(1);
        
    dists = cuda.local.array(shape=NUM_FP, dtype=float32);
    bit_sum = cuda.local.array(shape=NB_CLEANING, dtype=float32);
        
    if x < fp_array.shape[0]:

        num_neighbors = 0;
        fp_1 = fp_array[x];
        for tmp_fp_2 in range(fp_array.shape[0]):
            
            for i in range(bit_sum.shape[0]):
                bit_sum[i] = fp_1[i] + fp_array[tmp_fp_2, i];

            bitwise_and = 0.0;
            for i in bit_sum:
                if i == 2:
                    bitwise_and = bitwise_and + 1.0;

            bitwise_or = 0.0;
            for i in bit_sum:
                if i > 0.0:
                    bitwise_or = bitwise_or + 1.0;

            tmp_dist = 1.0 - (bitwise_and / bitwise_or);
            
            if tmp_dist < (1-cutoff):
                num_neighbors = num_neighbors+1;
            
            if num_neighbors >= lim_num_neighbors:
                outlier_compds[x] = False; 
                break;