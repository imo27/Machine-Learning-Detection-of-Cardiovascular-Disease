# Created by Sam Yoon 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import *
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

def pca_variance(data):
    '''
    This function returns a graph of the variance explained by each prinicpal compenent in a give dataset. 
    The data must have its samples on rows and attributes on columns. The function takes in a dataframe as
    an arguement.
    '''
    
    
    cust_palt = ['#111d5e', '#c70039', '#f37121', '#ffbd69', '#ffc93c']
    plt.style.use('ggplot')
    
    scaler=StandardScaler()
    scaler.fit(data)
    scaled_X = scaler.fit_transform(data)
    
    pca = PCA()
    pca.fit(scaled_X)
    pca_samples = pca.transform(scaled_X)
    
    fig, ax = plt.subplots(figsize=(14, 5))
    plt.plot(range(scaled_X.shape[1]), pca.explained_variance_ratio_.cumsum(), linestyle='--', drawstyle='steps-mid', color=cust_palt[0],
         label='Cumulative Explained Variance')
    sns.barplot(np.arange(1,scaled_X.shape[1]+1), pca.explained_variance_ratio_, alpha=0.85, color=cust_palt[1],
            label='Individual Explained Variance')

    plt.ylabel('Explained Variance Ratio', fontsize = 14)
    plt.xlabel('Number of Principal Components', fontsize = 14)
    ax.set_title('Explained Variance', fontsize = 20)
    plt.legend(loc='center right', fontsize = 13)
    
    return