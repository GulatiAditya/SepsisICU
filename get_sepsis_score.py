#!/usr/bin/env python

import numpy as np
import pickle
import pandas as pd
import math
from collections import defaultdict

def missing_value_imputation(data):
    df = pd.DataFrame(data)
    df = df.fillna(method='ffill')
    return df

def get_sepsis_score(data, model):

    imt = missing_value_imputation(data)
    print(imt)
    score = model.predict_proba(imt,validate_features=False)
    label = model.predict(imt,validate_features=False)

    # print(score[0][0])
    # print(label[0])
    if label[0]==1:
        exit()
    # exit()



    # lines = f.readlines()
    # for i in lines:
    #     trainingData_colMean.append(float(i.rstrip()))
    # scores_labels = defaultdict(list)
    # for i in range(0, df.shape[0]):
    #     scores_labels[i] = [model.predict_proba(df.iloc[[i]]).tolist()[0][0],model.predict(df.iloc[[i]]).tolist()[0]]
    
    # score = 0
    # for key, value in scores_labels.items():
    #     if value[1] == 1:
    #         score = 1-value[0]
    #         label = value[1]
    #         break
    #     elif value[1] == 0 and value[0] > score:
    #         score = value[0]
    #         label = value[1]
            
    return score[0][0], label[0]


def load_sepsis_model():
    # read in saved model pickle file here and return the model pickle variable
    
    trainedModel_filename = '../my_model_xgb.pkl'
    model = pickle.load(open(trainedModel_filename, 'rb'))

    # for root, dirs, files in os.walk('./'):
    #     if trainedModel_filename in files:
    #         trainedModel_filepath = os.path.join(root, trainedModel_filename)
    try:
        model = pickle.load(open(trainedModel_filename, 'rb'))
    except Exception as e:
        print(str(e) + '\n')
        print("ERROR: Model pickle not loaded. Verify file location, filename, filesize")
    return model