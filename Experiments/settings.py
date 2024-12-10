import math
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from aif360.sklearn.inprocessing import AdversarialDebiasing
from sklearn.linear_model import LogisticRegression

def set_configs(n_columns):
    
    adversarial_debiasing = {
        'AD' : [AdversarialDebiasing,
                {'prot_attr' : ['Group'],
                 'num_epochs' : list(range(50, 210, 10)),
                 'random_state' : [42]}]
    }
    
    ada = {
        'ADA' : [AdaBoostClassifier,
                {'n_estimators' : list(range(50, 450, 25)), 
                 'random_state' : [42]}]
    }
    
    lr = {
        'LR' : [LogisticRegression,
               {'C' : list(np.arange(0.65, 1.45, 0.05)), 'n_jobs' : [-1], 'random_state' : [42], 'n_jobs' : [-1]}]
    
    }
    
    mlp = {
        'MLP' : [MLPClassifier,
                 {'hidden_layer_sizes' : list(range(5, 36, 2)),
                  'random_state' : [42]}]
    }
    
    random_forest = {
        'RF' : [RandomForestClassifier,
                {'n_estimators' : list(range(50, 450, 25)),
                 'min_samples_split' : [math.floor(abs(math.sqrt(n_columns - 1)))], 
                 'random_state' : [42], 'n_jobs' : [-1]}]
    }
    
    xgb = {
        'XGB' : [XGBClassifier,
                 {'n_estimators' : list(range(50, 450, 25)),
                  'random_state' : [42]}]
    }    
    
    all_configs = {
        'ada'   : ada,
        'ad'    : adversarial_debiasing,
        'lr'    : lr,
        'mlp'   : mlp,
        'rf'    : random_forest,
        'xgb'   : xgb,
    }
    
    return all_configs