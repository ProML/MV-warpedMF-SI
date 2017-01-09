# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 16:53:06 2015

@author: Rukawa
"""
import pickle
import numpy

from model_utils.user_parameter import user_parameter
from model_utils.corpus import corpus

from run_MVWarpedMF import run_MVWarpedMF
from run_MVWarpedMFSI import run_MVWarpedMFSI


def load_pickle_variable(filename,variable_name):
    pickle_file = open(filename, 'rb')
    unpickled = pickle.load(pickle_file)
    pickle_file.close()
    return unpickled[variable_name]

toy_data_filename = r'../Data/toy_data.pickle'
Data = load_pickle_variable(toy_data_filename,'Data')
Sigma_u = load_pickle_variable(toy_data_filename,'Sigma_u')
Sigma_v = load_pickle_variable(toy_data_filename,'Sigma_v')
mat_mask_train = load_pickle_variable(toy_data_filename,'mat_mask_train')
mat_mask_test = load_pickle_variable(toy_data_filename,'mat_mask_test')
method_name_list = ['MVWarpedMF','MVWarpedMFSI']
cp = corpus(numpy.copy(Data),Sigma_u=Sigma_u,Sigma_v=Sigma_v,description = 'Toy example')
user_param = user_parameter(n_ranks = 5,n_folds=5,R=[10,10],max_iters=50)
#==============================================================================
# Train the models 
#==============================================================================
model = numpy.full((len(method_name_list)),None,dtype=object)

for m in xrange(len(method_name_list)):

    if method_name_list[m] == 'MVWarpedMF':
        model = run_MVWarpedMF('multi',cp,user_param,mat_mask_train,mat_mask_test)

    if method_name_list[m] == 'MVWarpedMFSI':
        model = run_MVWarpedMFSI('multi',cp,user_param,mat_mask_train,mat_mask_test)

