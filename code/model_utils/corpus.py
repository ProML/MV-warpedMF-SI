# -*- coding: utf-8 -*-
"""
Created on Fri May  8 18:53:46 2015

@author: Rukawa
"""

import numpy

class corpus():
    def __init__(self,S,Sigma_u=None,Sigma_v=None,R=[None,None],P_u=None,Q_u=None,Q_u_inv=None,P_v=None,Q_v=None,Q_v_inv=None,description='MF',transforming_var = None):
        self.S = S
        self.N = len(self.S)
        self.D,self.G = numpy.shape(self.S[0])
        self.W = [[]]*self.N
        self.density = [[]]*self.N
        self.mean_matrix = [[]]*self.N
        self.mean_row = [[]]*self.N
        self.mean_col = [[]]*self.N
        self.mask_basis = [[]]*self.N
        self.description = description
        self.transforming_var = transforming_var

        for n in xrange(self.N):
            W_n = numpy.reshape(~numpy.isnan(self.S[n]),(self.D,self.G))
            self.S[n][~W_n] = 0.0

        
        self.decomposed_u = False
        self.decomposed_v = False
        if P_u is not None:
            self.decomposed_u = True  
            self.P_u = P_u
            self.Q_u = Q_u
            self.Q_u_inv = Q_u_inv      
        if P_v is not None:
            self.decomposed_v = True   
            self.P_v = P_v
            self.Q_v = Q_v
            self.Q_v_inv = Q_v_inv      
			     
        if Sigma_u is None:
            self.Sigma_u = numpy.identity(self.D)
        else:
            self.Sigma_u = Sigma_u

        if Sigma_v is None:
            self.Sigma_v = numpy.identity(self.G)
        else:
            self.Sigma_v = Sigma_v
