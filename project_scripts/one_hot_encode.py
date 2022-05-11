# Created by Amal Alghamdi and Ifeanyi Osuchukwu

import numpy as np 
import pandas as pd

def one_hot_encode (df):
    '''
    This function accepts a pandas dataframe as an object and converts all string data to its 1 hot encode form.
    returns a new df and conversion information for features with binary info. 
    '''
    #convert string data into intergers using 1-hot-encoding 
    conversion = {}
    for col in df.columns:
        if len(np.unique(df[col])) <= 2 and df[col].dtype == object:
            uni_items = np.unique(df[col])
            conversion[col] = {j:i for i,j in enumerate(uni_items)}
            df[col] = np.where(df[col] == uni_items[0], 0, 1)

        elif len(np.unique(df[col])) >=3 and df[col].dtype == object:
            y = pd.get_dummies(df[col],prefix=col)  #create new column with the value 1 if attribute is present 
            df = df.drop([col],axis=1)  #drop col with string data 
            df = pd.concat([df, y], axis=1)
    df = df.astype('float')
    return df , conversion

#test case
'''
import sys,os; sys.path.append(os.environ['BMESAHMETDIR']); import bmes
file = bmes.downloadurl('https://www.cs.drexel.edu/~lht29/tmp/heart.csv','./heart.csv') #the data file is small enough to download into this folder.
df = pd.read_csv(file)

df_onehot, conv=one_hot_encode(df)
#df_onehot (intentionally?) missing class atrribute "HeartDisease", fixed by
df_onehot=pd.concat([df_onehot,df["HeartDisease"]], axis=1) #taget appear last, or
df_onehot=pd.concat([df["HeartDisease"], df_onehot], axis=1) #so that target appear first
#---or---
tmp=df_onehot["HeartDisease"]
df_onehot=df_onehot.drop('HeartDisease',1)
df_onehot=pd.concat([df_onehot,tmp], axis=1)
df_onehot
'''