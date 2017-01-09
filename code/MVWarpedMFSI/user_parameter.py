# -*- coding: utf-8 -*-
"""
Created on Sat Mar 07 10:23:02 2015

@author: Rukawa
"""


class user_parameter(object):      
    def __init__(self,n_ranks=3,I_list=[3],func_type_list=['tanh'],R=[10,10],alpha=0.0001,beta=0.0001,max_iters=1000,convergence_threshold=0.0001,n_processes=1,random_state=0,n_folds=5,filter_threshold=10):
        self.random_state = random_state
        self.n_folds = n_folds
        self.filter_threshold = filter_threshold

        self.R = R       # Number of ranks for svd 
        self.K = n_ranks # Number of factors

        self.alpha = alpha
        self.beta = beta
        
        self.I_list = I_list
        self.func_type_list = func_type_list
        self.max_iters = max_iters
        self.convergence_threshold = convergence_threshold
        self.n_processes = n_processes

        

