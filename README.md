# ChemicalSeriesReconstruction

This is a command line software tool for automated identification of chemical series in sets of molecules, which is adjusted for dealing with larger amounts of data by means of a farthest-first pre-clustering. Additionally, an option for determining times active phases as well as the possibility to determine multi-scaffold series is implemented here.

This code was used to reconstruct the historical chemical series at Novartis, as published in 

**25 Years of Small-Molecule Optimization at Novartis: A Retrospective Analysis of Chemical Series Evolution**

Maximilian Beckers et al., *Journal of Chemical Information and Modelling (2023)*  https://pubs.acs.org/doi/full/10.1021/acs.jcim.2c00785



**Dependencies**

NumPy, SciPy, scikit-learn, Numba, Pandas and RDKit. Arthor (https://www.nextmovesoftware.com/) will be used when available to speed up the calculations, otherwise the RDKit substructure functionalities will be used.

**Structure of input data**

As input a comma separated txt-file needs to be provided. For the chemical series identification, a column of SMILES strings is required (Default column name is "Structure", but the actual column name can be given with the flag -column_name). For the active phase determination, we require a column with the registration dates of the compounds. As default, the date column needs to be named "First Reg Date", but the name of the date column can be given with the flag -date_column, and the dates have to be specified as "yyyy-mm-dd", e.g. "1998-12-31".

## How to run the commandline interface ##

***Preliminaries***

Before running the actual series reconstruction the first time, the ChEMBL substructure library will be setup. This will happen automatically when running the first time and will take some time to download the ChEMBL and to prepare a local file of the substructure library. This needs only be done once, so if the substructure library is generated, it will be used in all future runs of the series identification.

***Running the series identification***

To run the algorithm, just run *chemical_series_reconstruction.py* with python3. Please make sure, all the required packages are installed. The only required input is the list of compounds, which has to be provided with the flag *-data*. 

The following options are available:

    Required:
 
    -data      A txt-file as described above that contains the SMILES of the compounds and optionally also the registration dates. The name of the column can be given with the flag -smiles_column (see below, Default: "Structure")

    Optional:

    -flimit     The specificity limit for the scaffolds. The smaller, the more specific (Default: 0.002)
    -min_series_size     The minimum number of compounds that is required for defining a chemical series. (Default: 10)
    -smiles_column       Name of the column that contains the smiles of the compounds. (Default: "Structure").
    -jaccard_similarity_threshold        Threshold of the Jaccard similarity for merging the series. If none, scaffolds will not be merged (Default: None)
    -scaffolds     A txt-file containing a single column of SMARTS strings of the scaffolds (Default: None)
    -size_sliding_window     Size of the sliding window, given as the number of days, in which compounds are summed up for determination of the active times of a series, i.e. when the series is actively worked on. If the number of compounds in the sliding window exceeds min-series-size, then we say that the respective days are active times. (If not given, then no active times determination will be done) (Default: None)
    -date_column         Name of the column that contains the registration dates of the compounds. This is used for determining active phases of a chemical series (Default: "First Reg Date").
    -gpu  This flag will enable usage of the GPU. CUDA needs to be available. For setup of CUDA see http://numba.pydata.org/numba-doc/latest/cuda/overview.html#requirements

The following lines of code give an example of a run with default settings:

    $ python -u chemical_series_reconstruction.py -data data.csv

Another example with increased specificity limit and also determination of the active phases can be:

    $ python -u chemical_series_reconstruction.py -data data.csv -flimit 0.00001 -size_sliding_window 365

And last an example with the possibility of multi-scaffold series combined with active phase determination, which will be done on the multi-scaffold series:

    $ python -u chemical_series_reconstruction.py -data data.csv -flimit 0.00001 -size_sliding_window 365 -jaccard_similarity_threshold 0.5

***Output***

Output of the program is a txt-file with the same columns as the input file and has the name "UPGMA.csv". Additionally, the chemical series identifier of each compound is given in the column "Class". If multi-scaffold series are determined using the jaccard-similarity merging, the scaffold IDs are provided in the column "Scaffold ID". The information about the scaffolds is found in the file "scaffolds.csv".
If the active times are calculated as well, then the active series identifiers are given in the column "Class (Active)". 
As a compound can be member of multiple series, a compound will appear multiple times in the output table in such cases. In other words, the output table is in long format.  
    

## How to run the library ##

Just import the module and run the series identification in Python by

    from ChemicalSeriesReconstruction import ChemicalSeriesReconstruction
    series = ChemicalSeriesReconstruction(smiles_list, min_cluster_size, flimit, scaffolds, dates, size_sliding_window, jaccard_similarity_threshold);

and then access the results by

    # This is the pandas dataframe of the series data, same as "UPGMA.csv"
    series.series_data

    # This is the pandas dataframe f the scaffold data, similar to "scaffolds.csv"
    series.mcs_data

The parameters are the same as for the command line tool described above. The compounds are provided as a list of SMILES. E.g. 

    smiles_list = LIST_OF_SMILES;
    min_cluster_size = 10;
    flimit = 0.001;
    scaffolds = None;
    dates = None;
    size_sliding_window = None;
    jaccard_similarity_threshold = None; 
    use_gpu = False;

If active phases should be determined, registration dates should be provided as a list of dates in "dates" and the size of the sliding window has to be specified in "size_sliding_window".

## Testing the code ##

A few test cases have been implemented to check validity of the results. The tests can be run from the main directory of this package using the unittest module by
    
    python -m unittest