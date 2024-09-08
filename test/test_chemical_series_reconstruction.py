# Created by Maximilian Beckers, December 2021

import unittest
import sys
import os

sys.path.append('..')
from ChemicalSeriesReconstruction import ChemicalSeriesReconstruction
import pandas as pd

#--------------------------------------------------------------------------
class TestChemicalSeriesReconstruction(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        curr_path = os.path.dirname(os.path.abspath(__file__));

        df= pd.read_csv(os.path.join(curr_path, "test_smiles.csv"));
        smiles_list = list(df["Structure"].to_numpy()[:1000]);

        min_cluster_size = 10;
        flimit = 0.001;
        scaffolds = None;
        dates = [];
        size_sliding_window = None;
        jaccard_similarity_threshold = None;
        
        cls.given_series_data = pd.read_csv(os.path.join(curr_path, "series_data.csv"));
        cls.given_mcs_data = pd.read_csv(os.path.join(curr_path, "mcs_data.csv"));
        cls.series = ChemicalSeriesReconstruction(smiles_list, min_cluster_size, flimit, scaffolds, dates, size_sliding_window, jaccard_similarity_threshold);
        
    def test_classes(self):
        self.assertListEqual(self.series.series_data["Class"].to_list(), self.given_series_data["Class"].to_list());
        
        
    def test_mcs(self):
        self.assertListEqual(self.series.mcs_data["Scaffold ID"].to_list(), self.given_mcs_data["Scaffold ID"].to_list());
    
#-----------------------------------------------------------------------------
class TestChemicalSeriesReconstructionJaccard(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        
        curr_path = os.path.dirname(os.path.abspath(__file__));
          
        df= pd.read_csv(os.path.join(curr_path, "test_smiles.csv"));
        smiles_list = list(df["Structure"].to_numpy()[:1000]);

        min_cluster_size = 10;
        flimit = 0.001;
        scaffolds = None;
        dates = [];
        size_sliding_window = None;
        jaccard_similarity_threshold = 0.5;
        
        cls.given_series_data = pd.read_csv(os.path.join(curr_path, "series_data_jaccard.csv"));
        cls.given_mcs_data = pd.read_csv(os.path.join(curr_path, "mcs_data_jaccard.csv"));
        cls.series = ChemicalSeriesReconstruction(smiles_list, min_cluster_size, flimit, scaffolds, dates, size_sliding_window, jaccard_similarity_threshold);
        
    def test_classes(self):
        self.assertListEqual(self.series.series_data["Class"].to_list(), self.given_series_data["Class"].to_list());
        
        
    def test_mcs(self):
        self.assertListEqual(self.series.mcs_data["Scaffold ID"].to_list(), self.given_mcs_data["Scaffold ID"].to_list());
        
#-------------------------------------------------------------------------------
class TestChemicalSeriesReconstructionJaccard(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        
        curr_path = os.path.dirname(os.path.abspath(__file__));
        
        df= pd.read_csv(os.path.join(curr_path, "test_smiles.csv"));
        smiles_list = list(df["Structure"].to_numpy()[:1000]);

        min_cluster_size = 10;
        flimit = 0.001;
        scaffolds = None;
        size_sliding_window = 365;
        jaccard_similarity_threshold = 0.5;
        
        #generate timestamps
        import datetime
        from random import randrange

        dates = [];
        current = datetime.datetime(2013, 9, 20,13, 00);
        for i in range(len(smiles_list)):
            current = current + datetime.timedelta(minutes=100);
            dates.append(current)
        
        
        cls.given_series_data = pd.read_csv(os.path.join(curr_path, "series_data_time.csv"));
        cls.given_mcs_data = pd.read_csv(os.path.join(curr_path, "mcs_data_time.csv"));
        cls.series = ChemicalSeriesReconstruction(smiles_list, min_cluster_size, flimit, scaffolds, dates, size_sliding_window, jaccard_similarity_threshold);
        
    def test_classes(self):
        self.assertListEqual(self.series.series_data["Class"].to_list(), self.given_series_data["Class"].to_list());
        
    def test_active_classes(self):
        self.assertListEqual(self.series.series_data["Class (Active)"].to_list(), self.given_series_data["Class (Active)"].to_list());
            
    def test_dates(self):
        self.assertListEqual(self.series.series_data["Date"].to_list(), self.given_series_data["Date"].to_list());
        
    def test_mcs(self):
        self.assertListEqual(self.series.mcs_data["Scaffold ID"].to_list(), self.given_mcs_data["Scaffold ID"].to_list());