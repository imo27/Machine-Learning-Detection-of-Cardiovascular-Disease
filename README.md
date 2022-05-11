# Machine Learning Detection of Cardiovascular Disease

Python: need version 3.0 or later
Make sure all Python Modules could be imported:
sys os sqlite3 pandas numpy matplotlib statsmodels sklearn statistics

Use command prompt to install packagies:
Your python engine may be called "python" or "py". take "py" for example, below command lines to install packagies:

py -m pip install "pandas"

py -m pip install "numpy"

py -m pip install "matplotlib"

py -m pip install "statsmodels"

py -m pip install "sklearn"

py -m pip install "statistics"

py -m pip install "matplotlib.pylot"

py -m pip install "sklearn.metrics"


Files in project: 
heart.csv: comma seperated file , dataset with samples on rows and attributes on the columns

All python script files are in the project_sript folder. 

bar_graph.py: python script that creates double bar graphs for male and female populations. 

cross_validation. py: python script cross validates given classification models. 

get_data.py : python script takes dataframe and produces testing dataset, targets, and features of a given dataset. 

logistic_bound.py: python script that returns a logistic decision boundary. 

one_hot_encode.py: python script turns dataframe columns with string values into 1's and 0's.

pca_2d.py : python script , creates pca plots.

pca_variance.py: python script , returns the graph of principal components and their explained variance. 

prediction_cm.py : python script that cross validates a given model and returns a confusion matrix and a classification report. 

show_tree.py : python script that shows a decision tree plot . 

target_last_col.py : python script places a desired column in a dataframe at the end of dataframe.
