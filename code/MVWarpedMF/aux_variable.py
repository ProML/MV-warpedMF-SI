# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 16:39:02 2015

@author: Rukawa
"""


import numpy

class aux_variable(object):
    def __init__(self,obs):
        self.R = [[]] * obs.N
        for n in xrange(obs.N):
            self.R[n] = numpy.full([obs.D,obs.G],numpy.nan)

def get_F(model_params,n):
    return (model_params.a[n])*model_params.M_u.dot(model_params.M_v.transpose()) + model_params.M_r[n]+ model_params.M_c[n]

        
def get_Zeta(model_params):
    return (model_params.M_u**2).dot(model_params.S_v.transpose()) + model_params.S_u.dot((model_params.M_v**2).transpose()) + (model_params.S_u).dot(model_params.S_v.transpose())

def get_Xi(model_params,aux_params,n):
    return aux_params.R[n]**2 + (model_params.a[n]**2) * get_Zeta(model_params) + model_params.S_r[n] + model_params.S_c[n]
