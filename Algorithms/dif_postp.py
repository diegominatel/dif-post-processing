# -*- coding: utf-8 -*-

''' General packages '''
import pandas as pd
import numpy as np
from datetime import datetime

from dif import Dif
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_curve
from classification_validation import _StratifiedBy


''' Experiment parameterization '''
report_columns = ['method', 'model', 'valid_size', 'threshold']
n_items = [16, 32]
per = [0.05, 0.10, 0.20, 0.40, 0.45]

def is_correct(true, predict):
    if true == predict:
        return 1
    else:
        return 0    

def readjust_predict(predict_proba, positive_value):
    if predict_proba >= positive_value:
        return 1
    else:
        return 0    
       
is_correct = np.vectorize(is_correct)
readjust_predict = np.vectorize(readjust_predict)
    
def repeat_columns(data):
    drop_columns = []
    columns = data.columns
    i = 0
    while i < len(data.columns) - 1:
        if columns[i] not in drop_columns:
            j = i + 1
            while j < len(data.columns):
                if pd.DataFrame.equals(data[columns[i]], data[columns[j]]) == True:
                    drop_columns.append(columns[j])
                j += 1
        i += 1
    return drop_columns

def only_one_response(data):
    drop_columns = []
    for group in list(data['group'].unique()):
        aux_data = data.loc[data['group'] == group]
        for col in aux_data.columns:
            if col != 'group':
                if len(aux_data[col].unique()) == 1:
                    if col not in drop_columns:
                        drop_columns.append(col)
    
    return drop_columns

class DIF_PostP(BaseEstimator, ClassifierMixin):
    
    def __init__(self, clf, focal, valid_size, DIF_methods, random_state):
        ''' '''
        self.clf = clf
        self.focal = focal
        self.valid_size = valid_size
        self.focal = focal
        self.DIF_methods = DIF_methods
        self.thresholds = None
        self.ppa_counter = 0
        self.random_state = random_state
        
    def get_params(self, model):
        return {'group' : 'group', 'focal' : self.focal, 'model' : model}
    
    def _initialize_report(self):
        self.thresholds = pd.DataFrame(columns=report_columns)
    
    def update_except_thresholds(self, method_name, model, threshold):
        self.thresholds.loc[self.ppa_counter] = [method_name, model, self.valid_size, threshold]
        self.ppa_counter += 1
            
    def update_thresholds(self, method_name, model):
        threshold = float(self.validation_values['DIF'].idxmin())
        self.thresholds.loc[self.ppa_counter] = [method_name, model, self.valid_size, threshold]
        self.ppa_counter += 1
        
    def generate_item_modeling(self, x_test_valid, y_test_valid, threshold_items):
        item_modeling =  pd.DataFrame(columns=threshold_items)
        for i, value in enumerate(threshold_items):
            predict = readjust_predict(self.proba_positive, value)
            item_modeling[value] = is_correct(y_test_valid, predict)
        r_columns = repeat_columns(item_modeling)
        item_modeling = item_modeling.drop(r_columns, axis=1)
        item_modeling['group'] = list(x_test_valid.index)
        self.validation_values = self.validation_values.drop(r_columns, axis=0)
        
        drop = only_one_response(item_modeling)
        if len(drop) > 0:
            item_modeling = item_modeling.drop(drop, axis=1)
            self.validation_values = self.validation_values.drop(drop, axis=0)
            
        return item_modeling
    
    
    def fit(self, x, y):
        self._initialize_report()
        by = getattr(_StratifiedBy, 'group_target')(x, y)
        x_train_valid, x_test_valid, y_train_valid, y_test_valid = train_test_split(x, y, stratify=by, 
                                                                                    test_size=self.valid_size,
                                                                                    random_state=self.random_state)
        print('-------------------------------------------------------------')
        print('[%s] - Train the classifier for validation' % datetime.now())
        self.clf.fit(x_train_valid, y_train_valid)
        proba = self.clf.predict_proba(x_test_valid)
        self.proba_positive = np.transpose(proba)[1]  
        _, _, all_thresholds = roc_curve(y_test_valid, self.proba_positive)

        for p in per:
            for N in n_items:
                select_thresholds = all_thresholds[round(len(all_thresholds)*p):-(round(len(all_thresholds)*p))]
                threshold_items = np.sort(select_thresholds)
                if len(threshold_items) > N:
                    threshold_items = [threshold_items[i] for i in np.linspace(0.5,len(threshold_items)-0.5, N, dtype=int)]

                self.validation_values = pd.DataFrame(index=threshold_items, columns=report_columns)
                item_modeling = self.generate_item_modeling(x_test_valid, y_test_valid, threshold_items)

                print('[%s] - Generates the item modeling (%s)' % (datetime.now(), str(item_modeling.shape)))
                for method_name, model in self.DIF_methods: 
                    params = self.get_params(model)
                    ''' Run DIF '''
                    print('[%s] - Run DIF (%s)' % (datetime.now(), method_name))
                    methodDif = Dif(params)
                    
                    try:
                        if item_modeling.shape[1] < 5:
                            raise ValueError('Item modeling: insufficient number of columns')
                        if len(item_modeling.loc[item_modeling['group'] == self.focal]) < 5:
                            raise ValueError('Item modeling: insufficient number of privileged group instances')
                        if len(item_modeling.loc[item_modeling['group'] != self.focal]) < 5:
                            raise ValueError('Item modeling: insufficient number of unprivileged group instances')
                        methodDif.calculate(item_modeling, model)
                        self.validation_values['DIF'] = methodDif.dif['DIF'].astype(float)
                        self.update_thresholds(model + '_N-' + str(N) + '_P-' + str(p), method_name)
                    except:
                        percents = pd.DataFrame()
                        percents['Group'] = x.index
                        percents['Target'] = y
                        percents = percents.value_counts()
                        middle = (percents['Privileged', 0] + percents['Unprivileged', 0])/len(x)
                        self.update_except_thresholds(model + '_N-' + str(N) + '_P-' + str(p), method_name, middle)
                ''' Retrain the classifier '''
                print('[%s] - Train the classifier' % (datetime.now()))
                self.clf.fit(x, y)
                print('[%s] - End of this processing' % (datetime.now()))
        
    def predict(self, x):
        proba = self.clf.predict_proba(x)
        proba_positive = np.transpose(proba)[1]
        predicts = pd.DataFrame(columns=list(self.thresholds.index))
        for idx in self.thresholds.index:
            predicts[idx] = readjust_predict(proba_positive, self.thresholds.loc[idx, 'threshold'])
            
        return predicts