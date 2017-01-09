# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 09:54:50 2015

@author: Rukawa
@version: v07_0
"""

import numpy
import scipy.special

import evaluation

from parameter import parameter
import aux_variable
import warping_functions

from const import const


class inference(object):
    
    def __init__(self,obs,user_param,mat_mask):  
        self.obs = obs
        self.hist_ELBO = []
        self.hist_sum_rmse = []
        self.hist_convergence = []
        self.hist_rmse = [[]] * self.obs.N
        self.converged = False
        self.user_param = user_param
        self.mat_mask = mat_mask    
        self.isProgramError = False

#==============================================================================
# Initialise parameters          
#==============================================================================
        self.param = parameter(self.obs,user_param)
        self.param.rand_init(self.obs,self.user_param)

#==============================================================================
# Initialise auxillary variables          
#==============================================================================
        print 'Aux initialising ...'
        self.aux = aux_variable.aux_variable(self.obs,self.user_param)
        if self.obs.decomposed_u:
            self.aux.P_u = self.obs.P_u.copy()
            self.aux.Q_u = self.obs.Q_u.copy()
            self.aux.Q_u_inv = self.obs.Q_u_inv.copy()
        else:
            if self.obs.Sigma_u.shape[0] == self.obs.Sigma_u.shape[1]:
                self.aux.P_u,self.aux.Q_u,self.aux.Q_u_inv,self.param.delta_u = aux_variable.svd(self.obs.Sigma_u,self.user_param.R[0])
            else:
                self.aux.P_u,self.aux.Q_u_inv,self.aux.Q_u,self.param.delta_u = aux_variable.svd(numpy.linalg.pinv(self.obs.Sigma_u.T))
        self.aux.PQinvPt_u = self.aux.P_u.dot(self.aux.Q_u_inv).dot(self.aux.P_u.T)
        self.aux.inv_Sigma_u = self.aux.PQinvPt_u + numpy.diag(numpy.full((self.obs.D),self.param.delta_u))
        self.aux.Sigma_u = aux_variable.matrix_inv(numpy.full((self.obs.D),self.param.delta_u),self.aux.P_u,self.aux.Q_u_inv)
        self.aux.logdet_Sigma_u = numpy.linalg.slogdet(self.aux.Sigma_u)[1]
            
        if self.obs.decomposed_v:
            self.aux.P_v = self.obs.P_v.copy()
            self.aux.Q_v = self.obs.Q_v.copy()
            self.aux.Q_v_inv = self.obs.Q_v_inv.copy()
        else:
            if self.obs.Sigma_v.shape[0] == self.obs.Sigma_v.shape[1]:
                self.aux.P_v,self.aux.Q_v,self.aux.Q_v_inv,self.param.delta_v = aux_variable.svd(self.obs.Sigma_v.T,self.user_param.R[1])
            else:
                self.aux.P_v,self.aux.Q_v_inv,self.aux.Q_v,self.param.delta_v = aux_variable.svd(numpy.linalg.pinv(self.obs.Sigma_v.T))
        self.aux.PQinvPt_v = self.aux.P_v.dot(self.aux.Q_v_inv).dot(self.aux.P_v.T)
        self.aux.inv_Sigma_v = self.aux.PQinvPt_v + numpy.diag(numpy.full((self.obs.G),self.param.delta_v))
        self.aux.Sigma_v = aux_variable.matrix_inv(numpy.full((self.obs.G),self.param.delta_v),self.aux.P_v,self.aux.Q_v_inv)
        self.aux.logdet_Sigma_v = numpy.linalg.slogdet(self.aux.Sigma_v)[1]
            
        for n in xrange(self.obs.N):
            self.aux.R[n] = (self.param.Z[n] - aux_variable.get_F(self.param,n)) * self.mat_mask[n] # Signed zeros
        for k in xrange(self.user_param.K):
            self.aux.S_u[:,k] = numpy.diag(self.param.C_u[k])
            self.aux.S_v[:,k] = numpy.diag(self.param.C_v[k])
            
#==============================================================================
# Initialise warping fucntion object          
#==============================================================================
        self.warping = [[]] * self.obs.N
        for n in xrange(self.obs.N):
            self.warping[n] = warping_functions.composite_warping(self.user_param.I_list,self.user_param.func_type_list)        


#==============================================================================
# Compute ELBO          
#==============================================================================

    def compute_ELBO(self):
        # $\mathbb{E}_q[\log p(\bm{Z|U},\bm{V})]$
        l_bound = 0
        for n in xrange(self.obs.N):
            l_bound += -0.5*numpy.sum(self.mat_mask[n])*(const.log2pi-numpy.log(self.param.tau[n]))
            l_bound += -0.5*numpy.sum(self.param.tau[n]*self.mat_mask[n]*aux_variable.get_Xi(self.param,self.aux,n))
        print l_bound
    
        # $\mathbb{E}_q[\log p(\bm{U}|\bm{\Sigma}^u]$
        l_bound -= 0.5*self.obs.D * self.user_param.K*const.log2pi
        l_bound -= 0.5* self.user_param.K * self.aux.logdet_Sigma_u
        for k in xrange(self.user_param.K):
            l_bound -= 0.5*numpy.trace(self.aux.inv_Sigma_u.dot(self.param.C_u[k]))
            l_bound -= 0.5*(self.param.M_u[:,k].dot(self.aux.inv_Sigma_u).dot(self.param.M_u[:,k]))  
        print l_bound
    
        # $\mathbb{E}_q[\log p(\bm{V}|\bm{\Sigma}^v]$
        l_bound -= 0.5*self.obs.G * self.user_param.K*const.log2pi
        l_bound -= 0.5* self.user_param.K * self.aux.logdet_Sigma_v
        for k in xrange(self.user_param.K):
            l_bound -= 0.5*numpy.trace(self.aux.inv_Sigma_v.dot(self.param.C_v[k]))
            l_bound -= 0.5*(self.param.M_v[:,k].dot(self.aux.inv_Sigma_v).dot(self.param.M_v[:,k]))  
        print l_bound

        # $-\mathbb{E}_q[\log q(\bm{U})]$
        l_bound += 0.5*self.obs.D * self.user_param.K*const.log2pi        
        for k in xrange(self.user_param.K):
            l_bound += 0.5*(numpy.linalg.slogdet(self.param.C_u[k])[1])
        l_bound += 0.5*self.obs.D * self.user_param.K
        print l_bound

        # $-\mathbb{E}_q[\log q(\bm{V})]$
        l_bound += 0.5*self.obs.G * self.user_param.K*const.log2pi        
        for k in xrange(self.user_param.K):
            l_bound += 0.5*(numpy.linalg.slogdet(self.param.C_v[k])[1])
        l_bound += 0.5*self.obs.G * self.user_param.K
        print l_bound

        # \sum\limits_{n,d} \mathcal{L}^{b^r}
        for n in xrange(self.obs.N):
            l_bound += 0.5* numpy.sum(numpy.log(self.param.Tau_r[n][:,0]) + numpy.log(self.param.S_r[n][:,0]))
            l_bound -= 0.5* numpy.sum(self.param.Tau_r[n][:,0] * (self.param.M_r[n][:,0]**2+self.param.S_r[n][:,0]))
            l_bound += 0.5* self.obs.D
        print l_bound

        # \sum\limits_{n,g} \mathcal{L}^{b^c}
        for n in xrange(self.obs.N):
            l_bound += 0.5* numpy.sum(numpy.log(self.param.Tau_c[n][0,:]) + numpy.log(self.param.S_c[n][0,:]))
            l_bound -= 0.5* numpy.sum(self.param.Tau_c[n][0,:] * (self.param.M_c[n][0,:]**2+self.param.S_c[n][0,:]))
            l_bound += 0.5* self.obs.G
        print l_bound
        return l_bound
#==============================================================================
# Update variational parameters          
#==============================================================================
       
    def update_q_u(self,k):
        m_u_k_old = self.param.M_u[:,k].copy()
        c_u_k_partial = numpy.zeros(self.obs.D)
        m_u_partial = numpy.zeros(self.obs.D)
        which_observed = set()
        for n in xrange(self.obs.N):
            for g in xrange(self.obs.G):
                if numpy.sum(self.mat_mask[n][:,g])>0:
                    which_observed.add(n)
                    c_u_k_partial += self.mat_mask[n][:,g] * self.param.tau[n]* (self.param.a[n]**2) * (self.param.M_v[g,k]**2 + self.param.C_v[k,g,g])
                    m_u_partial   += self.param.tau[n]*self.param.a[n]*self.param.M_v[g,k]* self.mat_mask[n][:,g]*(self.aux.R[n][:,g]+self.param.a[n]*self.param.M_v[g,k]*self.param.M_u[:,k])
       
        if len(which_observed) > 0:
#            self.param.C_u[k] = numpy.linalg.solve((numpy.diag(c_u_k_partial) + self.aux.inv_Sigma_u ),numpy.identity(self.obs.D))
            self.param.C_u[k] = aux_variable.matrix_inv(c_u_k_partial+self.param.delta_u,self.aux.P_u,self.aux.Q_u_inv)
            self.aux.S_u[:,k] = numpy.diag(self.param.C_u[k])
            self.param.M_u[:,k] = self.param.C_u[k].dot(m_u_partial)      
            
            # Update aux
            for n in which_observed:
#                for g in xrange(self.obs.G):
#                    self.aux.R[n][:,g] = self.aux.R[n][:,g]-self.param.a[n]*(self.param.M_u[:,k]-m_u_k_old)*(self.param.M_v[g,k])
                self.aux.R[n] = self.aux.R[n]-self.param.a[n]*numpy.outer((self.param.M_u[:,k]-m_u_k_old),self.param.M_v[:,k])

    def update_q_v(self,k):
        m_v_k_old = self.param.M_v[:,k].copy()
        c_v_k_partial = numpy.zeros(self.obs.G)
        m_v_partial = numpy.zeros(self.obs.G)
        which_observed = set()
        for n in xrange(self.obs.N):
            for d in xrange(self.obs.D):
                if numpy.sum(self.mat_mask[n][d,:])>0:
                    which_observed.add(n)
                    c_v_k_partial += self.mat_mask[n][d,:] * self.param.tau[n]* (self.param.a[n]**2) * (self.param.M_u[d,k]**2 + self.param.C_u[k,d,d])
                    m_v_partial   += self.param.tau[n]*self.param.a[n]*self.param.M_u[d,k]* self.mat_mask[n][d,:]*(self.aux.R[n][d,:]+self.param.a[n]*self.param.M_u[d,k]*self.param.M_v[:,k])
       
        if len(which_observed) > 0:
#            self.param.C_v[k] = numpy.linalg.solve((numpy.diag(c_v_k_partial) + self.aux.inv_Sigma_v ),numpy.identity(self.obs.G))
            self.param.C_v[k] = aux_variable.matrix_inv(c_v_k_partial+self.param.delta_v,self.aux.P_v,self.aux.Q_v_inv)
        
            self.aux.S_v[:,k] = numpy.diag(self.param.C_v[k])
            self.param.M_v[:,k] = self.param.C_v[k].dot(m_v_partial)      
            
            # Update aux
            for n in which_observed:
#                for d in xrange(self.obs.D):
#                    self.aux.R[n][d,:] = self.aux.R[n][d,:]-self.param.a[n]*(self.param.M_v[:,k]-m_v_k_old)*(self.param.M_u[d,k])
                self.aux.R[n] = self.aux.R[n]-self.param.a[n]*numpy.outer(self.param.M_u[:,k],(self.param.M_v[:,k]-m_v_k_old))

            
    def update_q_b_r(self):
        for n in xrange(self.obs.N):
            for i in xrange(self.obs.D):
                m_r_n_i_old = self.param.M_r[n][i,0]
                j = numpy.where(self.mat_mask[n][i,:]==1)[0]
                if len(j)>0:                    
                    self.param.M_r[n][i,:] = numpy.sum(self.aux.R[n][i,j] + m_r_n_i_old)*self.param.tau[n]/(len(j)*self.param.tau[n]+self.param.Tau_r[n][i,0])
                    # Update R
                    self.aux.R[n][i,j] = self.aux.R[n][i,j] - (self.param.M_r[n][i,0] - m_r_n_i_old)
                    # Update S_r
                    self.param.S_r[n][i,:] = 1.0/(self.param.tau[n] * len(j) + self.param.Tau_r[n][i,0])

    def update_q_b_c(self):
        for n in xrange(self.obs.N):
            for j in xrange(self.obs.G):
                m_c_n_j_old = self.param.M_c[n][0,j]
                i = numpy.where(self.mat_mask[n][:,j]==1)[0]
                if len(i)>0:                    
                    self.param.M_c[n][:,j] = numpy.sum(self.aux.R[n][i,j] + m_c_n_j_old)*self.param.tau[n]/(len(i)*self.param.tau[n]+self.param.Tau_c[n][0,j])
                    # Update R
                    self.aux.R[n][i,j] = self.aux.R[n][i,j] - (self.param.M_c[n][0,j] - m_c_n_j_old) 
                    # Update S_c
                    self.param.S_c[n][:,j] = 1.0/(self.param.tau[n] * len(i) + self.param.Tau_c[n][0,j])
                
#==============================================================================
# Update warping parameters          
#==============================================================================
    def update_warping_parameters(self):
        for n in xrange(self.obs.N):
            F_n = aux_variable.get_F(self.param,n)
            self.warping[n].update_parameters(self.obs.S[n],F_n,self.param.tau[n],self.mat_mask[n]) 
        # Update Z
        for n in xrange(self.obs.N):
            z_n_old = self.param.Z[n].copy()
            self.param.Z[n] = self.warping[n].f(self.obs.S[n],i=-1) 
            # Update aux
            self.aux.R[n] = (self.aux.R[n] + (self.param.Z[n] - z_n_old) ) * self.mat_mask[n] # Signed zeros

#==============================================================================
# Update hyper-parameters          
#==============================================================================
    def update_tau(self):
        for n in xrange(self.obs.N):
            self.param.tau[n] = numpy.sum(self.mat_mask[n])/numpy.sum(self.mat_mask[n]*aux_variable.get_Xi(self.param,self.aux,n))
     
    def update_a(self):
        mat_latent = self.param.M_u.dot(self.param.M_v.transpose())
        for n in xrange(self.obs.N):
            a_n_old = self.param.a[n]
            opt_a = numpy.sum(self.mat_mask[n]*mat_latent*(self.param.Z[n]-self.param.M_r[n]-self.param.M_c[n]))/numpy.sum( self.mat_mask[n]* ((mat_latent**2) + aux_variable.get_Zeta(self.param,self.aux)) )
#            self.param.a[n] = max(1,opt_a)
            self.param.a[n] = opt_a
            # Update aux
            self.aux.R[n] = self.mat_mask[n]*(self.aux.R[n] - (self.param.a[n] - a_n_old)*mat_latent)
    def update_tau_r(self):
        for n in xrange(self.obs.N):
            self.param.Tau_r[n] = 1.0/(self.param.M_r[n]**2+self.param.S_r[n])
        
    def update_tau_c(self):
        for n in xrange(self.obs.N):
            self.param.Tau_c[n] = 1.0/(self.param.M_c[n]**2+self.param.S_c[n])

    def update_delta_u(self):
        options = dict()
        options['maxiter'] = 200
        res = scipy.optimize.minimize(_f_delta, jac=False, x0=numpy.random.random(), method='L-BFGS-B',args=(self.aux.P_u,self.aux.Q_u,self.param.M_u,self.param.C_u),bounds=[(0.001,None)],options=options)
        if res.success and res.nit > 0: # and res.jac < 1E-4
            self.param.delta_u = res.x
            self.aux.inv_Sigma_v = self.aux.PQinvPt_u + numpy.diag(numpy.full((self.obs.D),self.param.delta_u))
            self.aux.Sigma_v = aux_variable.matrix_inv(numpy.full((self.obs.D),self.param.delta_u),self.aux.P_u,self.aux.Q_u_inv)
            self.aux.logdet_Sigma_u = numpy.linalg.slogdet(self.aux.Sigma_u)[1]
        print res
        print self.param.delta_u

    def update_delta_v(self):
        options = dict()
        options['maxiter'] = 50
        res = scipy.optimize.minimize(_f_delta, jac=False, x0=numpy.random.random(), method='L-BFGS-B',args=(self.aux.P_v,self.aux.Q_v,self.param.M_v,self.param.C_v),bounds=[(0.001,None)] ,options=options)
        if res.success and res.nit > 0: # and res.jac < 1E-4
            print 'update delta'
            self.param.delta_v = res.x
            self.aux.inv_Sigma_v = self.aux.PQinvPt_v + numpy.diag(numpy.full((self.obs.G),self.param.delta_v))
            self.aux.Sigma_v = aux_variable.matrix_inv(numpy.full((self.obs.G),self.param.delta_v),self.aux.P_v,self.aux.Q_v_inv)
            self.aux.logdet_Sigma_v = numpy.linalg.slogdet(self.aux.Sigma_v)[1]


#==============================================================================
# Fit          
#==============================================================================
    def fit(self):  
        
        # Start iterating the coordinate descent
#        l_bound = self.compute_ELBO()
        self.hist_ELBO = []
        self.converged = False
        iteration = 0
        checked = True
        while not self.converged and iteration < self.user_param.max_iters:
            iteration += 1
 
            for k in xrange(self.user_param.K):
                self.update_q_u(k)
#            checked = l_bound <= self.compute_ELBO()
            print 'q(U) ' + str(checked)
#            stop_error(checked)

            for k in xrange(self.user_param.K):
                self.update_q_v(k)
#            checked = l_bound <= self.compute_ELBO()
            print 'q(V) ' + str(checked)
#            stop_error(checked)

                
            
            self.update_tau()
#            checked = l_bound <= self.compute_ELBO()
            print 'tau ' + str(checked)
#            stop_error(checked)

            self.update_a()
#            checked = l_bound <= self.compute_ELBO()
            print 'a ' + str(checked)
#            stop_error(checked)

            self.update_q_b_r()
#            checked = l_bound <= self.compute_ELBO()
            print 'q(b_r) ' + str(checked)
#            stop_error(checked)


            self.update_q_b_c()
#            checked = l_bound <= self.compute_ELBO()
            print 'q(b_c) ' + str(checked)
#            stop_error(checked)

            self.update_tau_r()
#            checked = l_bound <= self.compute_ELBO()
            print 'tau_r ' + str(checked)
#            stop_error(checked)

            self.update_tau_c()
#            checked = l_bound <= self.compute_ELBO()
            print 'tau_c ' + str(checked)
#            stop_error(checked)

            self.update_delta_u()
#            checked = l_bound <= self.compute_ELBO()
            print 'delta_u ' + str(checked)
#            stop_error(checked)

#            self.update_delta_v()
#            checked = l_bound <= self.compute_ELBO()
#            print 'delta_v ' + str(checked)
#            stop_error(checked)

            self.update_warping_parameters()
#            checked = l_bound <= self.compute_ELBO()
            print 'Warping params ' + str(checked)
#            stop_error(checked)
            
            # Check convergence
#            l_bound_old = l_bound
#            l_bound = self.compute_ELBO()
#            self.hist_ELBO.append(l_bound)
             
#            sum_rmse = 0 
#            for n in xrange(self.obs.N):
#                pred = aux_variable.get_F(self.param,n)
#                y_hat = self.param.Z[n][self.mat_mask[n]]
#                y = pred[self.mat_mask[n]]
#                rmse_n = evaluation.get_rmse(y_hat,y)
#                sum_rmse += rmse_n
#                self.hist_rmse[n].append(rmse_n)
#            self.hist_sum_rmse.append(sum_rmse)

#            if (l_bound - l_bound_old) < 0:
#                self.isProgramError = True
#            conv = numpy.abs(l_bound - l_bound_old)/numpy.abs(l_bound_old)
#            self.hist_convergence.append(conv)
#            self.converged = conv < self.user_param.convergence_threshold
#            print 'iter: ' + str(iteration) + ' ... ELBO: ' + str(l_bound) + ' ... Convergence: ' + str(conv)
            print 'iter: ' + str(iteration) 
            print 'a = '+ str(self.param.a)
            print 'tau = '+ str(self.param.tau)

def stop_error(corrected):
    if not corrected:
#        print corrected
        input("The program is wrong !")

def _f_delta(delta,P,Q,M,C):
    K = len(C)
    J = len(Q)
    Q_diag = numpy.zeros(len(M))
    Q_diag[:J] = Q.diagonal()
    Q_inv = numpy.diag(1.0/(Q.diagonal() + delta))
    Sigma_inv = P.dot(Q_inv).dot(P.T)
    f = 0.0
    for k in xrange(K):
        f += 0.5*numpy.trace(Sigma_inv.dot(C[k]))
#        print f
        f += 0.5*M[:,k].dot(Sigma_inv).dot(M[:,k])
#        print f
    f += 0.5*K*numpy.linalg.slogdet(P.dot(Q + delta).dot(P.T))[1]
    return f