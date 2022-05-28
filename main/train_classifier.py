#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Code tested on python version 3.7.3
import tqdm
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import xgboost as xgb
import pickle
import glob


if __name__ == '__main__':
    
    trainingData_directory = '../sepsis_data/train/' 
    trainingData_fileList = glob.glob(os.path.join(trainingData_directory + '*.psv'))
    dfList = []
    print("In progress: Reading training data")
    for inputfile in tqdm.tqdm(trainingData_fileList):
        single_df=pd.read_csv(inputfile,sep="|",index_col=None,header=0)
        single_df = single_df.fillna(method='ffill')

        dfList.append(single_df)
    data = pd.concat(dfList, axis=0, ignore_index=True) # Concatenate every patient file into single training dataset
    print("In progress: Pre-processing training data")
    data = data.dropna(axis='rows')
    data = data.drop(["Bilirubin_direct",],axis=1)
    print("In progress: Training model")

    X_train = data.loc[:, data.columns != 'SepsisLabel']
    y_train = data['SepsisLabel']
    
    # model = xgb.XGBClassifier()
    # model.fit(X_train, y_train)

    model = GradientBoostingClassifier(random_state=0)
    model.fit(X_train, y_train)
    print("Accuracy on training set: {:.3f}".format(model.score(X_train, y_train)))
    # print("Accuracy on test set: {:.3f}".format(model.score(X_test, y_test)))
    filename = 'my_model.pkl'
    pickle.dump(model, open(filename, 'wb'))
    f = open('trainingData_colMeans.txt','w')
    data_colMean = data.mean(axis=0)
    for i in range(0, len(data_colMean)):
        f.write(str(data_colMean[i]) + '\n')
    f.close()
    print("Completed: Successfully trained model - my_model.pkl")
