#Created by Ifeanyi Osuchukwu

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def cross_validation(model,X,Y,splits=4):
    '''
    This funciton returns a cross validation score for a classification model. 
    This function accepts a classification model, testing data, target data, and the 
    number of desired splts/folds.
    '''
    clf = model
    skf = StratifiedKFold(n_splits=splits,shuffle=True)
    scores = cross_val_score(clf,X,Y, cv = skf)
    print(f'---{splits}--- fold cross-validation accuracy: {scores.mean()*100:.1f}% (+/- {scores.std()*100:.1f})')
