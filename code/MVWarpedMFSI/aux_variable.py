# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 16:39:02 2015

@author: Rukawa
"""


import numpy
import scipy

class aux_variable(object):
    def __init__(self,obs,user_param):
        print 'Aux initialising ...'
        self.R = [[]] * obs.N
        for n in xrange(obs.N):
            self.R[n] = numpy.full([obs.D,obs.G],numpy.nan)
        self.S_u = numpy.full([obs.D,user_param.K],numpy.nan)
        self.S_v = numpy.full([obs.G,user_param.K],numpy.nan)
        self.logdet_Sigma_u = numpy.nan
        self.Sigma_u = numpy.full([obs.D,obs.D],numpy.nan)
        self.Sigma_v = numpy.full([obs.G,obs.G],numpy.nan)
        self.inv_Sigma_u = numpy.full([obs.D,obs.D],numpy.nan)
        self.inv_Sigma_v = numpy.full([obs.G,obs.G],numpy.nan)
        self.P_u = numpy.full([obs.D,obs.D],numpy.nan) 
        self.P_v = numpy.full([obs.G,obs.G],numpy.nan) 
        self.Q_u = numpy.full([obs.D,obs.D],numpy.nan) 
        self.Q_v = numpy.full([obs.G,obs.G],numpy.nan) 
        self.Q_u_inv = numpy.full([obs.D,obs.D],numpy.nan) 
        self.Q_v_inv = numpy.full([obs.G,obs.G],numpy.nan) 
        self.PQinvPt_u = numpy.full([obs.D,obs.D],numpy.nan)
        self.PQinvPt_v = numpy.full([obs.G,obs.G],numpy.nan)


def get_F(model_params,n):
    return (model_params.a[n])*model_params.M_u.dot(model_params.M_v.transpose()) + model_params.M_r[n]+ model_params.M_c[n]

        
def get_Zeta(model_params,aux_params):
    return (model_params.M_u**2).dot(aux_params.S_v.transpose()) + aux_params.S_u.dot((model_params.M_v**2).transpose()) + (aux_params.S_u).dot(aux_params.S_v.transpose())

def get_Xi(model_params,aux_params,n):
    return aux_params.R[n]**2 + (model_params.a[n]**2) * get_Zeta(model_params,aux_params) + model_params.S_r[n] + model_params.S_c[n]
    

def svd(Sigma,R=100):
    print 'SVD computing ...'
    mu_inv = 0.1
    if Sigma.shape[0] == Sigma.shape[1]:
        P, s, V = numpy.linalg.svd(Sigma, full_matrices=True)
        Q = numpy.diag(s)
        Q_inv = numpy.diag(1./s)
        if R < len(P):
            mu_inv = numpy.mean((1./s)[:R])
#            return P[:,-R:],Q[-R:,-R:],Q_inv[-R:,-R:]
            return P[:,-R:],Q[-R:,-R:],Q_inv[-R:,-R:],mu_inv
    else:
        P, s, V = numpy.linalg.svd(Sigma, full_matrices=False)
        Q = numpy.diag(s**2)
        Q_inv = numpy.diag(1./(s**2))
    return P,Q,Q_inv,mu_inv

def matrix_inv(A_diag,P,Q):
    A_diag_inv = 1./A_diag
#    if numpy.sum(A_diag==0) > 0:
#        print 'WARNING: zero division'
    A_inv = numpy.diag(A_diag_inv)
#    if not numpy.allclose(numpy.identity(len(A_inv)),numpy.diag(A_diag).dot(A_inv)):
#        print 'WARNING: wrong inversion in A'
    Q_inv = numpy.copy(Q)
    numpy.fill_diagonal(Q_inv,1./numpy.diag(Q))
    B = Q_inv + P.T.dot(A_inv).dot(P)
    B_inv = scipy.linalg.solve(B,numpy.identity(len(B)))
#    if not numpy.allclose(B.dot(B_inv),numpy.identity(len(B))):
#        print 'WARNING: wrong inversion in B'
    res = A_inv - A_inv.dot(P).dot(B_inv).dot(P.T).dot(A_inv)
#    if not numpy.allclose(numpy.identity(len(res)),res.dot(numpy.diag(A_diag)+P.dot(Q).dot(P.T))):
#        print 'WARNING: wrong inversion in matrix inv'
#    print res.dot(numpy.diag(A_diag)+P.dot(Q).dot(P.T))

    return res

        
    
