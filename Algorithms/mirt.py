# -*- coding: utf-8 -*-

''' General packages '''
import pandas as pd

''' Load the methods to use R '''
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

''' Load the 'mirt' package in R '''
mirt_package = importr('mirt')
mirt_package.mirtCluster()


''' Hardcode in R of the coef_function that calls mirt's coef method (Required: only way this method (coef) works) '''
ro.r('''
        # create a function `f`
        coefPrivileged <- function(model) {
            return(coef(model, IRTpars=TRUE, simplify=TRUE)$Privileged$items)
        }
        ''')


ro.r('''
        # create a function `f`
        coefUnprivileged <- function(model) {
            return(coef(model, IRTpars=TRUE, simplify=TRUE)$Unprivileged$items)
        }
        ''')


class Mirt:
    '''
    Class that runs the mirt package in R
    '''
    
    def __init__(self):
        self.__model = None
        self.params_privileged = None
        self.params_unprivileged = None
        
    def mirt_config(self, itemtype, group):
        '''
        Returns the parameters used in mirt
        '''
        technical = ro.r('list(NCYCLES = 500, message=FALSE, warn=FALSE)') 
        params = dict() 
        params['technical'] = technical
        params['itemtype'] = itemtype
        params['group'] = group
        params['verbose'] = True
        
        return params
        
    def calculate(self, item_modeling, itemtype, group):
        '''
        Args:
                item_modeling:  (pd.DataFrame) where rows indicate individuals and columns indicate items, each value pair i,j 
                                represents whether individual i correctly answered item j.
                group:   (list) indicates the group ('Privileged' or 'Unprivileged') to which the individual belongs.
        '''
        item_names = list(item_modeling.columns)
        ''' Convert DataFrame to R Frame '''
        with localconverter(ro.default_converter + pandas2ri.converter):
            item_modeling = ro.conversion.py2rpy(item_modeling)
        ''' Convert List to R Vector '''    
        group = ro.vectors.StrVector(group)
        ''' Run mirt '''
        self.__model = mirt_package.multipleGroup(item_modeling, 1, **self.mirt_config(itemtype, group))
        ''' Estimated item parameters for the privileged group. '''
        coefP = ro.r['coefPrivileged']
        params_privileged = coefP(self.__model)
        ''' Estimated item parameters for the unprivileged group. '''
        coefU = ro.r['coefUnprivileged']
        params_unprivileged = coefU(self.__model)
        ''' Set the attributes of the class (Convert R Frame to Pandas) '''
        with localconverter(ro.default_converter + pandas2ri.converter):
            self.params_privileged = pd.DataFrame(ro.conversion.rpy2py(params_privileged), index = item_names,
                                    columns = ['a', 'b', 'c', 'u'])
            self.params_unprivileged = pd.DataFrame(ro.conversion.rpy2py(params_unprivileged), index = item_names,
                                    columns = ['a', 'b', 'c', 'u'])
        