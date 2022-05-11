#Created by Ifeanyi Osuchukwu
import pandas as pd
import numpy as np 

def get_data(dataset):
    '''
    Accepts a pandas dataframe as an argument and extracts test set, target values, and features of dataset.
    '''
    Test_data = dataset.values[:,:-1] #Data
    Target = dataset.values[:,-1] #Target Data
    features = dataset.columns.tolist()[:-1]
    
    return Test_data,Target,features