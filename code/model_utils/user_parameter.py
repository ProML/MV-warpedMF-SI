# -*- coding: utf-8 -*-
"""
Created on Sat Mar 07 10:23:02 2015

@author: Rukawa
"""


class user_parameter(object):      
    def __init__(self,n_ranks=3,n_folds=5,R=[100,100],max_iters=1000,convergence_threshold=0.00001,n_processes=1,random_state=0):
        self.random_state = random_state
        self.n_folds = n_folds        
        self.K = n_ranks
        self.max_iters = max_iters
        self.convergence_threshold = convergence_threshold
        self.n_processes = n_processes
        self.R = R

        

