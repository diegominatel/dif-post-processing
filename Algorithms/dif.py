# -*- coding: utf-8 -*-

''' General packages '''
import math
import numpy as np
import pandas as pd

''' class Mirt from mirt.py '''
from mirt import Mirt


def _value_readjustment(x, value):
    ''' Readjust values 
    '''
    if x < 0 and x < -(value):
        return -(value)
    if x > 0 and x > value:
        return value
    else: 
        return x    

def _difference_logistic_curves(theta, ref_a, ref_b, foc_a, foc_b):
    ''' Calculate the Area DIF
    Args: 
        theta: (list) range of abilities values  
        ref_a: (float) estimated value of the parameter a of the reference group
        ref_b: (float) estimated value of the parameter a of the reference group
        foc_a: (float) estimated value of the parameter b of the focus group
        foc_b: (float) estimated value of the parameter b of the focus group
    '''
    ''' Readjust values to acceptable range ''' 
    ref_a = _value_readjustment(ref_a, 50)
    foc_a = _value_readjustment(foc_a, 50)
    ref_b = _value_readjustment(ref_b, 10)
    foc_b = _value_readjustment(foc_b, 10)
    ''' Calculate the area between logistic curves '''
    ref_v = 1/(1 + (math.exp(-ref_a*(theta-ref_b)))) 
    foc_v = 1/(1 + (math.exp(-foc_a*(theta-foc_b)))) 
    return abs(ref_v-foc_v)

class Dif:
    '''
    Class to calculate the DIF of classifiers
    '''
    
    def __init__(self, params):
        '''
        Args:
            dif_model_name: (str) nome do métod de detecção de DIF
            params: (dict) com o nome e valor dos atributos do método de DIF
        '''
        self.params = params
        self.group = params['group']
        self.focal = params['focal']
        self.dif = None
        self.irt_pars = None
    
    def calculate(self, item_modeling, itemtype):
        ''' Executa o método de detecção de DIF
        Args:
            item_modeling: (pd.DataFrame) where rows indicate individuals and columns indicate items, each value pair i,j 
                           represents whether individual i correctly answered item j. 
        '''
        ''' Copy the groups '''
        group = list(item_modeling['group'])
        ''' Insert the groups to examples correct and incorrects (adaption to always run) '''
        group.append('Privileged')
        group.append('Privileged')
        group.append('Unprivileged')
        group.append('Unprivileged')
        item_modeling = item_modeling.loc[:, item_modeling.columns != 'group']
        item_modeling.loc[item_modeling.shape[0]] = [0]*item_modeling.shape[1]
        item_modeling.loc[item_modeling.shape[0] + 1] = [1]*item_modeling.shape[1]
        item_modeling.loc[item_modeling.shape[0] + 2] = [0]*item_modeling.shape[1]
        item_modeling.loc[item_modeling.shape[0] + 3] = [1]*item_modeling.shape[1]
        ''' Run Mirt '''   
        mirt_model = Mirt()
        mirt_model.calculate(item_modeling, itemtype, group)
        ''' estimated item parameters for reference group (Privileged) '''
        ref_irt = mirt_model.params_privileged
        ''' estimated item parameters for reference group (Privileged) '''
        focal_irt = mirt_model.params_unprivileged
        ''' calculate the area DIF '''
        mult = 0.001 # use as parameter?
        abilities = np.arange(-4, 4, mult)
        self.dif = pd.DataFrame(index=ref_irt.index, columns=['DIF'])
        difference_areas = np.vectorize(_difference_logistic_curves)    
        for idx in ref_irt.index:
            dif = sum(difference_areas(abilities, ref_irt.loc[idx, 'a'], ref_irt.loc[idx, 'b'], focal_irt.loc[idx, 'a'], 
                                       focal_irt.loc[idx, 'b']))*mult
            self.dif.loc[idx, 'DIF'] = dif

        
