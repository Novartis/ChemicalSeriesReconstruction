# Created by Maximilian Beckers, December 2021

import numpy as np
import math
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib import ticker
import umap
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
from scipy.cluster.hierarchy import dendrogram
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
import warnings
import random
import os
import gc
from AutomatedSeriesClassification import cluster_utils

class DimensionalityReduction:

    
    fingerprints = [];
    tsne_embedding = [];
    isomap_embedding = [];
    pca_embedding = [];
    explained_variances = [];
    umap_embedding = [];
    umap_model = [];
    class_labels = [];
    good_classes = [];
    junk_classes = [];
    class_centers = [];
    class_size = [];
    abs_class_size = [];
    

    #**************************************
    def __init__(self, fp_array):
        
        self.fingerprints = fp_array;
        
        num_fp = fp_array.shape[0];
 
        #subsampling of data for PCA and tSNE, otherwise this would take forever
        max_num_samples = 10000;
        if max_num_samples < num_fp:
            print("For dimensionality reduction, data will be downsampled to a sample size of " + repr(max_num_samples))
            self.sample_subset = random.sample(range(num_fp), max_num_samples);
            fp_sample = self.fingerprints[self.sample_subset, :];
        else:
            self.sample_subset = range(num_fp);
            fp_sample = self.fingerprints;

        num_neighbors = int(0.05 * fp_sample.shape[0]);

        #pca
        print("Performing PCA ...");
        pca = PCA();
        pca.fit(fp_sample);
        self.explained_variances = pca.explained_variance_ratio_ * 100;
        pca = PCA(n_components=2);
        self.pca_embedding = pca.fit_transform(fp_sample);
        
        
        #tsne
        print("Calculating tSNE embedding ...");
        self.tsne_embedding = TSNE(n_components=2, perplexity=num_neighbors, init="pca", metric = cluster_utils.tanimoto_distance, random_state=100, n_jobs=-1).fit_transform(fp_sample);
        

        #LLE
        #print("Calculating LLE embedding ...");
        #self.lle_embedding = LocallyLinearEmbedding( n_neighbors=num_neighbors, n_components=2).fit_transform(fp_sample);
        
        #umap
        print("Calculating UMAP embedding ...");
        self.umap_model = umap.UMAP(n_neighbors=num_neighbors, metric = cluster_utils.tanimoto_distance, random_state=100).fit(fp_sample);
        self.umap_embedding = self.umap_model.transform(fp_sample);
        
        
    #***************************************
    def make_plot(self, class_labels = None):
        
        if class_labels is None:
            self.class_labels = np.zeros((self.umap_embedding.shape[0]))
        else:
            self.class_labels = class_labels[self.sample_subset];
        
        #get relative class sizes
        _, self.abs_class_size = np.unique(self.class_labels, return_counts=True);
        self.class_size = self.abs_class_size/float(np.sum(self.abs_class_size));
        
        colors = "jet"
        num_classes = np.unique(self.class_labels).size;
        tmp_label = np.unique(self.class_labels);
        label = [""]*num_classes;
        for class_ind in range(num_classes):
            label[class_ind] = '{}\n({:.0f}%)'.format(tmp_label[class_ind], self.class_size[class_ind]*100);
        
        #if labels are string, make int vector for coloring of plot
        try: 
            a = (self.class_labels[0] < 0); #this will raise an error if not numeric
            class_indices = self.class_labels;
        except:
            class_indices = np.copy(self.class_labels);
            for class_ind in range(num_classes):
                class_indices[self.class_labels == tmp_label[class_ind]] = class_ind;

        class_indices = class_indices.astype(int)
        
        # make diagnostics plot
        plt.rc('xtick', labelsize=8);  # fontsize of the tick labels
        plt.rc('ytick', labelsize=8);  # fontsize of the tick labels
        plt.tight_layout;
        gs = gridspec.GridSpec(2, 2, wspace=0.5, hspace=0.5);

        
        #plot pca
        ax1 = plt.subplot(gs[0, 0]);
        scatter = ax1.scatter(self.pca_embedding[:,0], self.pca_embedding[:, 1], c=class_indices, s=0.5, cmap=colors);
        ax1.set_title('PCA plot');
        ax1.set_xlabel('PC 1');
        ax1.set_ylabel('PC 2');
        #ax1.legend(handles=scatter.legend_elements()[0], labels=label, title='Class\n(rel.size)', fontsize=2, title_fontsize=2, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., markerscale=0.5);
        
        
        #plot explained variances
        ax2 = plt.subplot(gs[0, 1]);
        ax2.plot(range(1, 21, 1), self.explained_variances[0:20], linewidth=2, label="variance");
        ax2.plot(range(1, 21, 1), np.cumsum(self.explained_variances[0:20]), linewidth=2, label="cumulative\nvariance")
        ax2.set_xticks([1,2,3,4,5,10,15,20]);
        ax2.set_ylim(0,100);
        ax2.set_xlabel('Principal Component');
        ax2.set_ylabel('Explained variance [%]');
        ax2.set_title('Explained variances');
        #ax2.legend( title='', fontsize=4, title_fontsize=4, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.);


        #plot LLE
        ax3 = plt.subplot(gs[1, 0]);
        scatter = ax3.scatter(self.tsne_embedding[:,0], self.tsne_embedding[:,1], c=class_indices, s=0.5, cmap=colors);
        ax3.set_title('TSNE plot');
        ax3.set_xlabel('TSNE 1');
        ax3.set_ylabel('TSNE 2');
        #ax3.legend(handles=scatter.legend_elements()[0], labels=label, title='Class\n(rel.size)', fontsize=2, title_fontsize=2, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., markerscale=0.5);

        
        #plot umap
        ax4 = plt.subplot(gs[1, 1]);
        scatter = ax4.scatter(self.umap_embedding[:,0], self.umap_embedding[:,1], c=class_indices, s=0.5, cmap=colors);
        ax4.set_title('UMAP plot');
        ax4.set_xlabel('UMAP 1');
        ax4.set_ylabel('UMAP 2');
        #ax4.legend(handles=scatter.legend_elements()[0], labels=label, title='Class\n(rel.size)', fontsize=2, title_fontsize=2, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., markerscale=0.5);

        
        #plt.colorbar(scatter,ax=ax4);
        
        return plt
        