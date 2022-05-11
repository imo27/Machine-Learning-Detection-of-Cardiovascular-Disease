

from sklearn.tree import plot_tree
from sklearn import tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def show_tree(X,Y):
    X_train, X_test, y_train, y_test = train_test_split(X, Y,  test_size=0.20,random_state=15)
    d3 = tree.DecisionTreeClassifier()
    d3 = d3.fit(X_train, y_train)
    plt.figure(figsize=(15,10))
    tree.plot_tree(d3)