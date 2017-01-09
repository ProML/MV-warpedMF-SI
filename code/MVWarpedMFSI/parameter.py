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
        self.C_u = numpy.full([user_param.K,obs.D,obs.D],numpy.nan)
        self.M_v = numpy.full([obs.G,user_param.K],numpy.nan)
        self.C_v = numpy.full([user_param.K,obs.G,obs.G],numpy.nan)
                


        # Dataset-dependent parameters
        self.tau = numpy.full([obs.N],numpy.nan)
        # Linear transformation parameters
        self.a = numpy.full(obs.N,numpy.nan)
        self.M_r = [[]] * obs.N
        self.M_c = [[]] * obs.N
        self.S_r = [[]] * obs.N
        self.S_c = [[]] * obs.N
        self.Tau_r = [[]] * obs.N        
        self.Tau_c = [[]] * obs.N        
        for n in xrange(obs.N):
            self.M_r[n] = numpy.full([obs.D,obs.G],numpy.nan)
            self.M_c[n] = numpy.full([obs.D,obs.G],numpy.nan)
            self.S_r[n] = numpy.full([obs.D,obs.G],numpy.nan)
            self.S_c[n] = numpy.full([obs.D,obs.G],numpy.nan)
            self.Tau_r[n] = numpy.full([obs.D,obs.G],numpy.nan)
            self.Tau_c[n] = numpy.full([obs.D,obs.G],numpy.nan)

        self.delta_u = numpy.nan
        self.delta_v = numpy.nan

    def rand_init(self,obs,user_param):

#        A = numpy.random.rand(obs.D,obs.D)
#        B = numpy.random.rand(obs.G,obs.G)
        for k in xrange(user_param.K):
#            self.C_u[k] = numpy.dot(A,A.transpose())
#            self.C_v[k] = numpy.dot(B,B.transpose())
#            self.C_u[k] = numpy.copy(obs.Sigma_u.dot(obs.Sigma_u.T))
#            self.C_v[k] = numpy.copy(obs.Sigma_v.dot(obs.Sigma_v.T))
            self.C_u[k] = numpy.identity(obs.D)
            self.C_v[k] = numpy.identity(obs.G)
            
        for k in xrange(user_param.K):
            self.M_u[:,k] = numpy.random.multivariate_normal([0]*obs.D,self.C_u[k])
            self.M_v[:,k] = numpy.random.multivariate_normal([0]*obs.G,self.C_v[k])
                
        self.tau[:] = 1.0
        self.a[:] = 1.0
        self.delta_u = 1.0
        self.delta_v = 1.0
        for n in xrange(obs.N):
            self.M_r[n][:] = 0.0
            self.M_c[n][:] = 0.0
            self.S_r[n][:] = 1.0
            self.S_c[n][:] = 1.0
            self.Tau_r[n][:] = 1.0
            self.Tau_c[n][:] = 1.0
            
