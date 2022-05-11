#Created by Sam Yoon and Ifeanyi Osuchukwu
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import *
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def pca_2d(data,title_g=None, labels=None,target=None,n_comp=2,graph_status = True, print_status=True):
    
    '''
    This function returns a 2D scatter plot and the explained variance of the first two 
    principal components.

    Parameters:
    data : dataframe
    title : title for graph , str
    labels: labels for legened, str/list
    target: dataframe column name of class/target
    n_comp: number of components for pca , int
    graph_status: displays graph
    print_status: print variance amount 
    '''
    
    scaler=StandardScaler()
    scaler.fit(data)
    scaled_X = scaler.fit_transform(data)
    
    pca = PCA(n_comp)
    Y = pca.fit_transform(scaled_X)
    V = pca.explained_variance_ratio_
   
    if graph_status == True: 
        graph = pd.DataFrame({"X": Y[:,0], "Y": Y[:,1], "Category":data[target].values})
        groups = graph.groupby("Category")
        markers = ["o","^"]
        colors = ['r','b']

        fig, ax = plt.subplots(figsize=(14, 5))
        
        for (name, group), mar,col,lab in zip(groups,markers,colors,labels):
            plot(group["X"], group["Y"], marker=mar, color=col ,  linestyle="", label=lab)
        
        # plt.rcParams['figure.figsize'] = [10, 10]
        title(title_g)
        xlabel(f"PC 1 {V[0]*100:.2f}% ")
        ylabel(f"PC 2 {V[1]*100:.2f}% ")
        legend()
    
    total_var = pca.explained_variance_ratio_.sum() * 100
    if print_status == True:
        print(f'Total Explained Variance of 2: {total_var:.2f}%')
    
    if graph_status == False:
        return Y