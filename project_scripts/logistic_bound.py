# Created by Ifeanyi Osuchukwu

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import *
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from project_scripts.pca_2d import *
from project_scripts.prediction_cm import *


def logistic_bound(data,y,title):
    transform_x = pca_2d(data,print_status=False,graph_status = False)
    intercept,coef = prediction_cm(LogisticRegression(solver='liblinear'),transform_x,y,bound_info =True,cm=False)
    b = intercept[0]
    w1,w2 = coef.T
    c = -b/w2
    m = -w1/w2
    xmin, xmax = -1, 2
    ymin, ymax = -1, 2.5
    xd = np.array([xmin, xmax])
    yd = m*xd + c
    pca_2d(data,labels = ['Heart Disease', 'No Heart Disease'],title_g = title ,target="HeartDisease",print_status=False,graph_status = True)
    plt.plot(xd, yd, 'k', lw=1, ls='--')