import os
import sys
import subprocess
import requests
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score

def data_analysis (data):
    print ("Shape of the dataset:\n", data.shape) # Give number of rows & columns in the df
    print ('*' * 100)
    print ("Columns:\n", data.columns) # Give the names of the columns
    print ('*' * 100)
    print (data.dtypes.value_counts()) # Give the number of columns of each datatype
    print ('*' * 100)
    print ("Non-object columns:\n", data.columns [data.dtypes != 'object']) # Gives an array of columns of numerical datatype
    print ('*' * 100)
    print ("Object columns:\n", data.columns [data.dtypes == 'object']) # Give an array of columns of categorical datatype
    print ('*' * 100)
    print (data.info())
    print ('*' * 100)
    print (data.describe().transpose()) # Give descriptive statistics of the df
    print ('*' * 100)
    # Count of non-fraudulant (0) & fraudulant (1) transactions
    output = [0, 1]
    output_cent = [data['isFraud'].value_counts()[0]*100/len(data), data['isFraud'].value_counts()[1]*100/len(data)]
    print (output_cent)
    plt.bar (output, output_cent)
    plt.title ('0: No Fraud \n 1: Fraud')
    # Correlation plot
    plt.figure (figsize=(6, 4))
    heatmap = sns.heatmap (data.corr(), vmin=-1, vmax=1, annot=True, cmap='YlGnBu', cbar=True)
    heatmap.set_title ('Correlation Heatmap', pad=10)
    plt.show ()
    print ('*' * 100)
    # Count of non-fraudulant (0) & fraudulant (1) transactions for each type
    print (data.groupby('type')['isFraud'].value_counts())
    zeros = []
    ones = []
    unique_types = data['type'].unique()
    for i in unique_types:
        zeros.append (data[data['type']==i]['isFraud'].value_counts()[0])
        ones.append (data[data['type']==i]['isFraud'].value_counts()[1])
    # Set the width of the bars
    bar_width = 0.35
    # Set the positions of the bars on the x-axis
    x_axis = np.arange (len(unique_types))
    # Plot the first set of bars
    plt.bar (x_axis - bar_width/2, zeros, width=bar_width, label='Not fraud', color='gray')
    # Plot the second set of bars
    plt.bar (x_axis + bar_width/2, ones, width=bar_width, label='Fraud', color='red')
    # Add labels, title, and legend
    plt.xlabel ('Types')
    plt.ylabel ('Count')
    plt.xticks (x_axis, unique_types)
    plt.legend ()
    plt.show ()

def rule_based (data):
    # Mean transaction amount for non-fraudulant (0) & fraudulant (1) transactions
    print ("Average amount for non-fraud transaction (0) & fraud transaction (1):\n", data.groupby('isFraud')['amount'].mean())
    
def data_preprocessing (data):
    data['type'] = data['type'].astype ('category')
    data['isFraud'] = data['isFraud'].astype ('category')
    data['isFlaggedFraud'] = data['isFlaggedFraud'].astype ('category')
    df = data.copy () 
    # Dropping off less important features
    df = df.drop (columns=['step', 'nameOrig', 'nameDest', 'isFlaggedFraud'])
    # Missing data handling
    for c in data.columns:
        if data[c].dtypes != 'category':
            si_numeric = SimpleImputer (missing_values = np.nan, strategy = "mean") #NaN in numeric features are replaced by mean
            si_numeric.fit (df[:, 1:6])
            df[:, 1:6] = si_numeric.transform (df[:, 1:6])
        else:
            si_cat = SimpleImputer (missing_values = np.nan, strategy = "most_frequent") #NaN in categorical features are replaced by mode
            si_cat.fit (df[:, 0])
            df[:, 0] = si_cat.transform (df[:, 0])
    # One hot encoding the 'type' column
    df = pd.get_dummies (df, columns = ['type'])
    return df

def data_splitting (data):
    # Extracting the features (x) & target (y) from the dataframe
    x = df.iloc[:, [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]].values # Matrix of features ('values' is used to convert the sliced df into array so that numpy can work on it)
    y = df.iloc[:, 5].values # 1D array of target
    y = y.reshape ((6362620, 1)) # Reshaping the y-array to make it 2D
    print (y.shape)
    # Splitting the data into training and test dataset (done BEFORE feature scaling)
    x_train, x_test, y_train, y_test = train_test_split (x, y, test_size = 0.2, random_state = 0) #Train:Test :: 8:2

def imbalanced_data (x_train, y_train):
    # Resampling the minority class. The strategy can be changed as required
    sm = SMOTE (random_state=0)
    # Fit the model to generate the data.
    sm_x_train, sm_y_train = sm.fit_resample (x_train, y_train)
    return (sm_x_train, sm_y_train)

def standardisation (sm_x_train, x_test):
    # Feature scaling -- Standardisation & Normalisation -- is performed to get all the features on the same scale, so that 1 feature do not suppress other
    sc = StandardScaler ()
    sm_x_train[:, 0:5] = sc.fit_transform (sm_x_train[:, 0:5])
    x_test[:, 0:5] = sc.transform (x_test[:, 0:5]) # 'fit' function is not used as mean and SD of features used for test dataset is the one that was computed and used for training dataset

def isolationForest (sm_x_train, sm_y_train, x_test, y_test):
    # Implementing Isolation forest model
    iso_for = IsolationForest (n_estimators=100, max_samples=len(sm_x_train), random_state=0)
    iso_for.fit (sm_x_train, sm_y_train)
    y_pred_iso = iso_for.predict (x_test)
    y_pred_iso [y_pred_iso==1] = 0
    y_pred_iso [y_pred_iso==-1] = 1
    print ("Accuracy score:", accuracy_score (y_test, y_pred_iso))
    anomaly_scores = iso_for.decision_function (x_test)
    # Normalize anomaly scores to range [0, 1]
    normalized_scores = (anomaly_scores - np.min(anomaly_scores)) / (np.max(anomaly_scores) - np.min(anomaly_scores))
    # Convert normalized scores to probabilities
    iso_prob = 1 - normalized_scores # Since, lower score means higher probability to being an outlier, so we are subtracting from 1 so that the probaility for being an outlier is higher than that being an inlier
    return iso_prob

def ANN (sm_x_train, sm_y_train, x_test, y_test):
    ann = Sequential ([
        Dense (input_dim=10, units=6, activation='relu'), # Creating the input layer of input_dim=#Features & the 1st hidden layer
        Dense (units=6, activation='relu'), # Creating the 2nd hidden layer
        Dense (units=1, activation='sigmoid'), # Creating the output layer
    ])
    ann.compile (optimizer="adam", loss="binary_crossentropy", metrics=['accuracy']) # Using 'binary crossentropy' to calculate the loss, calculate accuracy & optimising weights using 'adams' (Stochastic gradient descent) to minimise the loss & maximise the accuracy
    ann.fit (sm_x_train, sm_y_train, batch_size=32, epochs = 5)
    ann_prob = ann.predict (x_test)
    y_pred_ann = (ann_prob > 0.5) # If ann_prob>0.5 is True (1), 1 will get stored in y_pred_ann, otherwise 0 will get stored
    ann.save ('ANN.h5')
    print ("Accuracy score:", accuracy_score (y_test, y_pred_ann))
    return ann_prob

def ensemble_model (iso_prob, ann_prob):
    # Combining the results of Isolation forest as well as ANN
    finalpred = (iso_prob + ann_prob)/2
    finalpred = (finalpred > 0.5)
    print ("F1 Score:", f1_score (y_test, finalpred, average="weighted"))
    print ("Accuracy score:", accuracy_score (y_test, finalpred))
    print ("Confusion matrix:", confusion_matrix (y_test, finalpred))
