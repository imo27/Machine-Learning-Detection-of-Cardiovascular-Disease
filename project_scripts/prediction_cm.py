#Created by Ifeanyi Osuchukwu

from random import random
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.metrics import confusion_matrix
import seaborn as sns
from matplotlib import cm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

def prediction_cm(model,X,Y,labels=None,title="Classification Confusion Matrix",bound_info = False, cm = True):
    '''
    This function accepts a classification model,test data, target data, and labels for a confusion matrix.
    This function returns a confusion matrix and predicition statistics. 
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, Y,  test_size=0.20,random_state=15)
    clf = model
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    
    if cm == True:
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10,10))
        ax = plt.subplot()
        sns.heatmap(cm, cmap= "Blues" , annot=True, fmt='g', ax=ax)
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title(title)
        ax.xaxis.set_ticklabels(labels)
        ax.yaxis.set_ticklabels(labels)
        plt.show()
        score = classification_report(y_test, y_pred, target_names=labels,output_dict= True)
        print(classification_report(y_test, y_pred, target_names=labels))
    
    if bound_info == False:
        return y_pred,score
    else:
        return clf.intercept_,clf.coef_