#!/usr/bin/python


import pickle
import numpy as np
import pandas as pd

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA


"""
	Section #1 

	-Data set information, missing data, outliers and cleanning 
	
	First of all lest find out some data about the loaded data set like total number of 
	data point , allocation of target classes (POI vs NON POI), original number of features
	missing values, features containing missing values and outliers detection.
	To accomplish that i will use, as recommended, pandas dataframe.
	Additionaly, i will drop all non-finantial features from the dataset,

"""

### Load the pickle file
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Copy the loaded data to a new structure to manipulate it
my_dataset = data_dict

### Convert the data set to pandas using float64 as dtype and employee names as index
my_dataset = pd.DataFrame.from_dict(my_dataset, orient='index', dtype='float64')

### Obtain total number of data points, poi and non poi
print "Data set information"
print "Total number of datapoints : {}".format(len(my_dataset.index))
print "Allocation of poi / non poi : \n{}".format(my_dataset['poi'].value_counts())
print "Total number of features : {}".format(len(my_dataset.columns))

### Lets obtain the list of original features that came with the dataset
original_features = my_dataset.columns.values

### Lets check for 'Missing data' in the features . Build two list for if is required
### latter 
has_missing_list = []
has_not_missing_list = []

for feat in original_features:
	has_missing = my_dataset[feat].isnull().values
	if has_missing.any() == True:
		has_missing_list.append(feat)
	else:
		has_not_missing_list.append(feat)


### Lets print the lists
print "\n\nData set information"
print "Features with missing values: {}".format(has_missing_list)
print "Features with no missing values: {}".format(has_not_missing_list)

### Drop email related features
mail_features = ['to_messages','shared_receipt_with_poi','from_messages','from_this_person_to_poi','email_address','from_poi_to_this_person']
my_dataset = my_dataset.drop(mail_features, axis=1)
### Print and allocate a list with the remaining features
initial_features = my_dataset.columns.values
print "\n\nFeatures remaining after cleaning mail: {}".format(my_dataset.columns.values)

### Drop 'TOTAL' index and any data point with all 'NaN'
my_dataset = my_dataset.drop(['TOTAL'])
my_dataset = my_dataset.dropna(how='all')

### Identify features with outliers. Will be treated on features engineering
print "\n\nOutliers identification"
print "Features . Distance in max - (2*std)"
print my_dataset.max() - (my_dataset.mean() + (2 * my_dataset.std()))

### Fill NaN with 0
my_dataset = my_dataset.fillna(0)

""" 
	Section #2

	-Feature selection engineering 
	We ll use RandomForestClassifier and feature_importances_ to compare features
	importance in the initial list of features and after the new feaures.
	For the rational behind the new features please look at the doc.
	
"""

### First we will train a model without new features . We'll use the "original_features" list
### which cointains what features remain after cleaning mail features. Lets print the list
print "\n\n This is the list of features for the first train and validation: {}".format(initial_features)

### Extract features and labels from dataset for local testing
labels_series = my_dataset.loc[:, 'poi']
labels = labels_series.values
labels = labels.astype(np.int64)

### Now the features, we should 'index' the slicing with a list of all the features BUT poi
features_columns = []
for col in my_dataset.columns:
    if col != 'poi':
        features_columns.append(col)

### pandas 'as_matrix' helps convert pandas Series to ndarrays based on selected columns
features = my_dataset.as_matrix(columns=features_columns)

### Lets check the results. labels should be a 1D array and features a ND array 
print "labels has this shape: {}".format(labels.shape)
print "features has this shape: {}".format(features.shape)

### Lets build the model for feature exploration
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, stratify=labels, random_state=42)

### Fit the model
forest = RandomForestClassifier(n_estimators=100, random_state=0, max_depth=4)
forest.fit(features_train, labels_train)

### Obtain feature_importances
feat_importance = forest.feature_importances_

### Check which features provide to be th most important
my_dataset_no_poi = my_dataset.drop('poi', axis=1)
feat_names = my_dataset_no_poi.columns.values
test = pd.Series(feat_importance,index=feat_names)
print test[test > 0.10]

### Lets move to build a different feature list creating cash_from_stock and high_exercised_percentile
### labels can be reused, features will be redefined
### Firts copy the dataset 
my_magic_dataset = my_dataset

### Create new feature 'cash_from_stock'
my_magic_dataset['cash_from_stock'] = my_magic_dataset['exercised_stock_options'] + my_magic_dataset['restricted_stock']

### Create new feature 'high_exercisd_percentile'

water_mark = my_magic_dataset['exercised_stock_options'].quantile(.80)
my_magic_dataset['high_exercised_percentile'] = my_magic_dataset['exercised_stock_options'] > water_mark
my_magic_dataset['high_exercised_percentile'] = my_magic_dataset['high_exercised_percentile'].astype(np.int64)

### Align features and labels with the new feature arrangement
my_magic_dataset_no_poi = my_magic_dataset.drop('poi', axis=1)
magic_feat = my_magic_dataset_no_poi.columns.values
my_magic_features = my_magic_dataset_no_poi.as_matrix(columns=magic_feat)

### Check if correct shapes
print "labels has this shape: {}".format(labels.shape)
print "features has this shape: {}".format(my_magic_features.shape)

my_magic_features_train, my_magic_features_test, labels_train, labels_test = train_test_split(my_magic_features, labels, stratify=labels, random_state=42)

### Fit and obtain feature importance
magic_forest = RandomForestClassifier(n_estimators=100, random_state=0, max_depth=4)
magic_forest.fit(my_magic_features_train ,labels_train)
magic_feat_importance = magic_forest.feature_importances_

### Check which features provide to be th most important
magic_feat_names = my_magic_dataset_no_poi.columns.values
magic_test = pd.Series(magic_feat_importance,index=magic_feat_names)
print magic_test[magic_test > 0.10]

"""
	Section #3

	- Model, params tuning , pipe and dump
	First we ll need to build features an labels. Then convert it back to original format.
	Then i will use utility featureFormat and targetFeatureSplit to obtain labels and features
	Then , build the pipe, the GridSearchCV for the LinearSVC
	First i will train and evaluate a RandomForestClassifier with the features i end up
	I will use StratifiedShuffleSplit 
	
"""
my_features_list = ['poi']
for feat in magic_feat:
	my_features_list.append(feat)
# print my_features_list
my_dataset = my_magic_dataset.to_dict(orient='index')
# print my_dataset
# my_features_list = ['poi','cash_from_stock', 'other', 'deferred_income']
data = featureFormat(my_dataset, my_features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Build the param grid for the gridsearch
lsvc_param_grid = {'linearsvc__C': [0.001, 0.01, 0.1, 1, 10, 100], 'linearsvc__tol': [1e-3, 1e-4], 'pca__tol': [0.001,0.01,0.1,1,10,100]}

### Lets build our StratifiedShuffleSplit for the dataset
from sklearn.model_selection import StratifiedShuffleSplit
lsvc_cv = StratifiedShuffleSplit(n_splits=100, test_size=0.5, random_state=42)

### Build the pipe

pipe = make_pipeline(PCA(n_components=6, svd_solver='arpack', random_state=42),
					 Imputer(missing_values=0, strategy='mean'),
					 StandardScaler(),
					 LinearSVC(penalty='l2', class_weight='balanced', dual=False))

### Build the GridSearchCV
pipe_grid = GridSearchCV(pipe, param_grid=lsvc_param_grid, cv=lsvc_cv)

### Fit
pipe_grid.fit(features, labels)

print("Best Linear SVC params: {}".format(pipe_grid.best_params_))
print("Best Linear SVC cv score: {}".format(pipe_grid.best_score_))

### Dump
dump_classifier_and_data(pipe_grid.best_estimator_, my_dataset, my_features_list)