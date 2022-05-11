#Created by Ifeanyi Osuchukwu

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def bar_graph_age_sex(data,Targets,t_value,target_label ='Heart_Disease'):
    '''
    This function takes in a proccessed pandas dataframe and an array of Targets. It 
    returns a bar graph of positive (instances where the prediction is Heart Disease) Targets by age and gender.
    '''

    sex = [gender for gender,T in zip(data['Sex'].values,Targets) if T == t_value]
    age = [age for age,T in zip(data['Age'].values,Targets) if T == t_value ]
    T = [T for T in Targets if T == t_value ]

    bar_df = pd.DataFrame({"Sex":sex, "Age":age,target_label :T})
    bins_labs = ['31 - 39', '40 - 49', '50 - 59', '60 - 69','70+']
    cut_bins = [31,39,49,59,69,100]
    bar_df['Age_Groups'] = pd.cut(bar_df['Age'], bins = cut_bins,include_lowest=True,labels = bins_labs) # add new column that groups age catergory according to bins variable 
    
    men_hd = [sum(bar_df.loc[(np.where((bar_df["Age_Groups"]==age) & (bar_df["Sex"]==1)))].value_counts()) for age in bins_labs]
    women_hd = [sum(bar_df.loc[(np.where((bar_df["Age_Groups"]==age) & (bar_df["Sex"]==0)))].value_counts()) for age in bins_labs]

    N = len(bins_labs)

    plt.figure(figsize = (10,10))
    ind = np.arange(N) 
    width = 0.35       
    plt.bar(ind, men_hd, width, label="Men",color = 'cornflowerblue')
    plt.bar(ind + width, women_hd, width,color = 'blue',label="Women")

    plt.ylabel(f'{target_label} Count',fontsize=15)
    plt.title(f'{target_label} Distribution by Age and Gender',fontsize=15)

    plt.xticks(ind + width / 2, (bins_labs))
    plt.legend(loc='best')
    plt.show()