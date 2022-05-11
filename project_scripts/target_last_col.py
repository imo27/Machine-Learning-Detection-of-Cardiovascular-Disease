#Created by Ifeanyi Osuchukwu 


def target_last_col (df,name_of_col):
    '''
    Rearrange columns to have the target column appear last, this makes future calculations easier.
    The function accepts a dataframe and the name of target column as the arguements. 
    '''
    temp_cols=df.columns.tolist()
    new_cols=temp_cols[:temp_cols.index(name_of_col)] + temp_cols[temp_cols.index(name_of_col)+1:]
    new_cols.append(temp_cols[temp_cols.index(name_of_col)])
    df=df[new_cols]
    return df