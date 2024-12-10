import pandas as pd
import numpy as np
import math
import sys
from IPython.display import display
from itertools import combinations
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, friedmanchisquare, rankdata
from Orange.evaluation import compute_CD, graph_ranks

measures_multicriteria = ['ratio_selection_rate', 'ratio_recall', 'ratio_odds']
datasets = ['adult', 'amphet', 'arrhythmia', 'bank', 'compasmen', 'compaswomen', 'crack',
            'contraceptive', 'ecstasy', 'german', 'heart', 'heroin', 'nicotine', 'student', 'titanic']

def multicriteria(measurement_values):
    pairs = list(combinations(measurement_values, 2))
    area = 0
    for a, b in pairs:
        area += (a*b*math.sin((2*math.pi)/3)/2)    
    return area

def multicriteria_validation(validation):
    matrix = validation[measures_multicriteria].to_numpy()
    validation['multicriteria'] = [multicriteria(row) for row in matrix]
    return validation

def measure_report(measure_selection, measure_view):

    results_valid = pd.DataFrame(columns = ['Dataset', 'Config', 'Clf', 'SPP', 'DIF-PP', 'CEOP', 'ROBC'])
    count = 0

    for dataset in datasets:
        for config in ['ad', 'ada', 'lr', 'mlp', 'rf', 'xgb']:
            validation = pd.read_csv('../Experiments/' + dataset + '_' + config + '_validation.csv', 
                                      sep=';', index_col=0)
            validation = validation.reset_index()
            ''' Calculate ratio odds '''
            validation['ratio_odds'] = (validation['ratio_fpr'] + validation['ratio_recall'])/2
            ''' Calculate multicriteria '''
            validation = multicriteria_validation(validation)
            ''' Inicialização '''
            
            
            for name in validation['clf_name'].unique():
                
                values = []
                values.append(dataset)
                values.append(config)
                values.append(name)
            
                for postp in ['without', 'dif-postp', 'ceo', 'roc']:

                    aux_val = validation.loc[(validation['clf_name'] == name) & (validation['postp'] == postp)]
                    idx = aux_val[measure_selection].idxmax()
                    values.append(aux_val.loc[idx, measure_view])      
                
                results_valid.loc[count] = values
                count += 1
                
    return results_valid

def raw_report():

    results = pd.DataFrame()
    count = 0

    for dataset in datasets:
        for config in ['ad', 'ada', 'lr', 'mlp', 'rf', 'xgb']:
            validation = pd.read_csv('../Experiments/' + dataset + '_' + config + '_validation.csv', 
                                      sep=';', index_col=0)
            validation = validation.reset_index()
            ''' Calculate ratio odds '''
            validation['ratio_odds'] = (validation['ratio_fpr'] + validation['ratio_recall'])/2
            ''' Calculate multicriteria '''
            validation = multicriteria_validation(validation)
            ''' Inicialização '''
            
            validation['Dataset'] = [dataset]*len(validation)
            results = pd.concat([results, validation])
                
    return results
    
def cd_diagram(df, name):
    df = df[['SPP', 'DIF-PP', 'CEOP', 'ROBC']]
    algorithms_names = ['WPP', 'DIF-PP', 'CEOP', 'ROBC']
    results_array = df.values
    ranks_test = np.array([rankdata(-p) for p in results_array])
    average_ranks = np.mean(ranks_test, axis=0)
    print('\n'.join('({}) Rank average: {}'.format(a, r) for a, r in zip(algorithms_names, average_ranks)))  
    # This method computes the critical difference for Nemenyi test
    cd = compute_CD(average_ranks, n=len(df), alpha='0.05', test='nemenyi')
    print('CD = ', cd)
    # This method generates the plot.
    graph_ranks(average_ranks, names=algorithms_names, cd=cd, width=4, textspace=0.75, reverse=False, 
                filename = name + '.pdf')
    
def report_plot():
    df = pd.DataFrame(columns = ['Dataset', 'Config', 'Postp', 'MCPM', 'F1'])
    df_multicriteria = measure_report('multicriteria', 'multicriteria')
    df_f1 = measure_report('multicriteria', 'f1_macro')
    i = 0

    for aux_mult, aux_f1 in zip(df_multicriteria.iterrows(), df_f1.iterrows()):
        for criterion in ['SPP', 'DIF-PP', 'CEOP', 'ROBC']:
            df.loc[i] = [aux_f1[1]['Dataset'], aux_f1[1]['Config'], criterion, aux_mult[1][criterion], aux_f1[1][criterion]]
            i += 1
    return df