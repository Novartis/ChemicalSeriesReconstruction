# Created by Maximilian Beckers, December 2021

import sys, os
import pandas as pd
import numpy as np
import time
import warnings
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import rdFMCS
import random
import math


#-----------------------------------------------------
def seperate_series_into_active_phases(df, date_column, size_sliding_window=365, min_num_mols_per_window=10, num_eval_points = 10000):
    
    print("Determining active phases of series and subsetting the compounds in the active phases ...");
    print("");
    
    start = time.time();
    
    df = add_days_column_to_dataframe(df, date_column);
    
    classes = np.unique(df["Class"].to_numpy());
    classes = classes[classes>-1];
    
    df_split = pd.DataFrame();
    
    #do not split the noise class
    tmp_df = df[df["Class"] == -1];
    df_split = tmp_df.copy();
    df_split["Class (Active)"] = [-1] * len(df_split["Class"].to_list());

    new_class_index = 0;
    for tmp_class in classes:
        
        if tmp_class % 100 == 0:
            print("Now analyzing class " + repr(int(tmp_class)) + " ...");
        
        tmp_df = df[df["Class"]==tmp_class];
        
        tmp_times = tmp_df["Days"].to_numpy();
        size_series = tmp_df["Class"].to_numpy().size;
        
        active_compound_indices, interval_indices = find_active_times(tmp_times, size_sliding_window, min_num_mols_per_window, num_eval_points);
        
        #----------------------------------------------------------------
        #append the compounds in inactive regions to the noise class
        inactive_compound_indices = [];
        for tmp_compound_ind in np.arange(size_series):
            if tmp_compound_ind not in active_compound_indices:
                inactive_compound_indices.append(tmp_compound_ind);
                
        tmp_df_inactive = tmp_df.iloc[inactive_compound_indices, :].copy();
        tmp_df_inactive["Class (Active)"] = [-1] * len(inactive_compound_indices);
        df_split = pd.concat([df_split, tmp_df_inactive], ignore_index=False);
        
        #----------------------------------------------------------------
        #append the compounds in active phase to new series
        for tmp_sub_series in np.unique(interval_indices):
            size_sub_series = active_compound_indices[interval_indices==tmp_sub_series].size;
            
            
            if size_sub_series < min_num_mols_per_window:
                tmp_sub_series_df = tmp_df.iloc[active_compound_indices[interval_indices==tmp_sub_series], :].copy();
                tmp_sub_series_df["Class (Active)"] = [-1]*size_sub_series ;
                df_split = pd.concat([df_split, tmp_sub_series_df], ignore_index=False);
            else:
                tmp_sub_series_df = tmp_df.iloc[active_compound_indices[interval_indices==tmp_sub_series], :].copy();

                #set the new class_index
                tmp_sub_series_df["Class (Active)"] = [new_class_index]*size_sub_series ;

                #append the subseries df to the final df
                df_split = pd.concat([df_split, tmp_sub_series_df], ignore_index=False);

                new_class_index = new_class_index + 1;
      
    df_split["Class (Active)"] = df_split["Class (Active)"].to_numpy().astype(int);
            
    print("Number of active periods after splitting into active subseries: " + repr(new_class_index));
    
    end = time.time();
    print("Time elapsed for subsetting active phases in s: " + repr(end-start));
    print("");
    
    return df_split;


#-----------------------------------------------------    
def add_days_column_to_dataframe(df, date_column):
    
    date_list = dt = pd.to_datetime(df[date_column].to_list());

    date = [];
    indices_with_date = [];
    for tmp_date_ind in range(date_list.size):
        tmp_date = str(date_list[tmp_date_ind]);
        try:
            date.append(int(tmp_date[0:4] + tmp_date[5:7] + tmp_date[8:10]));  
            indices_with_date.append(tmp_date_ind);
        except:
            date.append(0);  

    df["Date"] = date;
    df = df.iloc[indices_with_date, :];
    
    #add days columns
    dates = df["Date"];

    years = np.array([int(str(tmp_date)[0:4]) for tmp_date in dates]);
    months = np.array([int(str(tmp_date)[4:6]) for tmp_date in dates]);
    days = np.array([int(str(tmp_date)[6:8]) for tmp_date in dates]);

    total_days = years*365 + months*31 + days;
    days_diff_start = total_days - np.min(total_days);

    df["Reg. Year"] = years;
    df["Days"] = days_diff_start;
    
    return df;

#-----------------------------------------------------
def find_active_times(times, size_sliding_window, min_num_mols_per_window, num_eval_points):
    
    
    num_mols = len(times);
    
    eval_points = np.linspace(np.min(times), np.max(times), num = num_eval_points);
    #kde = stats.gaussian_kde(times);
    #pdf_thresholded = kde.pdf(eval_points);
    #density = np.copy(pdf_thresholded);    

    #generate a list with ones at points in the eval_points where the compounds occur
    tmp_eval_points = eval_points.astype(int);
    indices = np.arange(eval_points.size);
    compounds_in_eval_times = np.zeros(eval_points.size);
    for tmp_compound_time in times:
        compound_index_in_time = np.min(indices[tmp_eval_points==tmp_compound_time]);
        compounds_in_eval_times[compound_index_in_time] = compounds_in_eval_times[compound_index_in_time] + 1 ;
    
    pdf_thresholded = np.zeros(eval_points.size);

    #move sliding window over series
    for i in range(len(times)):
        tmp_num_mols = times[(times >= times[i]) & (times < times[i] + size_sliding_window)].size;
        if tmp_num_mols > min_num_mols_per_window:
            upper_limit = np.max(times[times<(times[i] + size_sliding_window)]);
            pdf_thresholded[(eval_points >= times[i]) & (eval_points <= upper_limit)] = 1;  
            
        
    """    
    #get intervals were the pdf is higher then the filling threshold
    filling_threshold = 10.0/(num_mols*365.0); #this corresponds to 10 molecules per year
    pdf_thresholded[pdf_thresholded > filling_threshold] = 1;
    pdf_thresholded[pdf_thresholded<1] = 0;
    """
    
    #-------------------------------------------------
    #merge all intervals that are closer than given days
    min_time_diff = 365;
    prev_point = -1;
    for tmp_point in range(eval_points.size):
        
        if pdf_thresholded[tmp_point] == 1:
            tmp_time = eval_points[tmp_point];
            
            if prev_point != -1:
                prev_time = eval_points[prev_point];
                if tmp_time-prev_time < min_time_diff:
                    pdf_thresholded[prev_point:tmp_point] = 1;

            prev_point = tmp_point
    
    #--------------------------------------------------    
    #now delete all intervals smaller than 10 compounds
    min_series_size = min_num_mols_per_window;
    prev_point = 0;
    tmp_series_size = 0;
    
    for tmp_point in range(eval_points.size):
        
        if pdf_thresholded[tmp_point] == 0:    
            tmp_series_size = 0;
            prev_point = tmp_point;
        else:
            if compounds_in_eval_times[tmp_point] > 0:
                tmp_series_size = tmp_series_size + compounds_in_eval_times[tmp_point];
            
        if tmp_point == (eval_points.size-1): #we are at the end of the series
            if (tmp_series_size < min_series_size) & (tmp_series_size > 0):
                pdf_thresholded[prev_point : eval_points.size] = 0; 
        else:
            if (pdf_thresholded[tmp_point + 1] == 0) and (tmp_series_size < min_series_size):              
                pdf_thresholded[prev_point : (tmp_point+1)] = 0; 
    
    
    
    #--------------------------------------------------
    #now count the number of intervals
    num_intervals = 0;
    prev_value = 0;
    for tmp_point in range(1, eval_points.size):
        
        if (pdf_thresholded[tmp_point] == 1) and (prev_value==0): 
            num_intervals = num_intervals + 1;
          
        prev_value = pdf_thresholded[tmp_point];
            
        if (pdf_thresholded[tmp_point] == 1):
            pdf_thresholded[tmp_point] = num_intervals;
            

    #return num_intervals, pdf_thresholded;
    
    #-------------------------------------
    #--- get compounds in active times ---
    #-------------------------------------
    
    eval_points = np.linspace(np.min(times), np.max(times), num = num_eval_points).astype(int);
    indices = np.arange(eval_points.size);
    
    active_compound_indices = [];
    interval_indices = [];
    
    for tmp_compound_ind in range(len(times)):
        
        tmp_compound_time = times[tmp_compound_ind];
        compound_index_in_time = np.min(indices[eval_points==tmp_compound_time]);
        
        if pdf_thresholded[compound_index_in_time] != 0:
            active_compound_indices.append(tmp_compound_ind);
            interval_indices.append(pdf_thresholded[compound_index_in_time]);
            
    
    return np.array(active_compound_indices), np.array(interval_indices);