import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.model_selection import train_test_split
import numpy as np
import argparse
import pickle
def func():
    parser = argparse.ArgumentParser('Spam Classification')
    path_settings = parser.add_argument_group('path settings')
    path_settings.add_argument('--test_file', default='./data_test_hw4_problem1.csv',
                               help='Path of pre-trained input data')
    args=parser.parse_args()
    testfilePath=args.test_file
    data=pd.read_csv(testfilePath)
    X_test=data["text"]
    cv = CountVectorizer()
    X_test=cv.fit_transform(X_test)
    filename = './finalized_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    Y=loaded_model.predict(X_test)
    df2=pd.DataFrame()
    df2["spam"]=pd.Series(data=list(Y),dtype=str)
    df2.to_csv("./predictions.csv",index=False)


if __name__ == '__main__':
    func()
