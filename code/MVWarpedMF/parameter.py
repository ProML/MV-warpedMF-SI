# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 16:01:20 2015

@author: Rukawa
"""

import numpy

class parameter(object):
    def __init__(self,obs,user_param):
        self.Z = obs.S.copy()

        self.M_u = numpy.full([obs.D,user_param.K],numpy.nan)
        self.S_u = numpy.full([obs.D,user_param.K],numpy.nan)
        self.M_v = numpy.full([obs.G,user_param.K],numpy.nan)
        self.S_v = numpy.full([obs.G,user_param.K],numpy.nan)
                

        # ARD parameters
        self.alpha = numpy.nan
        self.beta = numpy.full(user_param.K,numpy.nan)
        
        self.rho = numpy.full(user_param.K,numpy.nan)

        # Dataset-dependent parameters
        self.tau = numpy.full([obs.N],numpy.nan)
        # Linear transformation parameters
        self.a = numpy.full(obs.N,numpy.nan)
        self.M_r = numpy.full([obs.N,obs.D,obs.G],numpy.nan)
        self.M_c = numpy.full([obs.N,obs.D,obs.G],numpy.nan)
        self.S_r = numpy.full([obs.N,obs.D,obs.G],numpy.nan)
        self.S_c = numpy.full([obs.N,obs.D,obs.G],numpy.nan)
        self.Tau_r = numpy.full([obs.N,obs.D,obs.G],numpy.nan)
        self.Tau_c = numpy.full([obs.N,obs.D,obs.G],numpy.nan)
    
    def rand_init(self,obs,user_param):
        

        self.alpha = 0.0001
        self.beta[:] = 0.0001
        self.rho[:] = 1.0

        self.M_u = numpy.random.normal(0, 1, (obs.D,user_param.K))
        self.S_u[:,:] = 1.0
        self.M_v = numpy.random.normal(0, 1, (obs.G,user_param.K))
        self.S_v[:,:] = 1.0
        

        self.tau[:] = 1.0
        self.a[:] = 1.0
        self.M_r[:] = 0.0
        self.M_c[:] = 0.0
        self.S_r[:] = 1.0
        self.S_c[:] = 1.0
        self.Tau_r[:] = 1.0
        self.Tau_c[:] = 1.0
            

  