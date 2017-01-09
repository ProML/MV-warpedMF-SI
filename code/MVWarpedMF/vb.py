# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 09:54:50 2015

@author: Rukawa
@version: v06_0
"""

import numpy
from joblib import Parallel, delayed
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
        self.hist_convergence = []
        self.hist_sum_rmse = []
        self.hist_rmse = [[]] * self.obs.N
        self.hist_r2 = [[]] * self.obs.N
        self.hist_rp = [[]] * self.obs.N
        self.converged = False
        self.user_param = user_param
        self.mat_mask = mat_mask    
        self.isProgramError = False

        self.param = parameter(self.obs,user_param)
        self.param.rand_init(self.obs,self.user_param)

#==============================================================================
# Initialise auxillary variables          
#==============================================================================
         
        self.aux = aux_variable.aux_variable(self.obs)
        for n in xrange(self.obs.N):
            self.aux.R[n] = (self.param.Z[n] - aux_variable.get_F(self.param,n)) * self.mat_mask[n] # Signed zeros

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

        l_bound = 0
        
        # Jacobian term
        for n in xrange(self.obs.N):
            l_bound += numpy.nansum(self.mat_mask[n]* (numpy.log(self.warping[n].f_prime(self.obs.S[n],i=-1) ) ) )
        
        # \sum\limits_n \sum\limits_{(i,j)}\mathcal{L}_{ij} 
        for n in xrange(self.obs.N):
            l_bound += -0.5*numpy.sum(self.mat_mask[n])*(const.log2pi-numpy.log(self.param.tau[n]))
            l_bound += -0.5*numpy.sum(self.param.tau[n]*self.mat_mask[n]*aux_variable.get_Xi(self.param,self.aux,n))
    
        # \sum\limits_{(i,k)}\mathcal{L}^u_{ij} 
        l_bound += 0.5*numpy.sum(numpy.log(self.param.rho) + numpy.log(self.param.S_u))
        l_bound -= 0.5*numpy.sum(self.param.rho * (self.param.M_u**2+self.param.S_u))
        l_bound += 0.5*self.obs.D*self.user_param.K
    
        # \sum\limits_{(i,k)}\mathcal{L}^v_{ik} 
        l_bound += 0.5*numpy.sum(scipy.special.psi(self.param.alpha) - numpy.log(self.param.beta) + numpy.log(self.param.S_v))
        l_bound -= 0.5*numpy.sum(self.param.alpha/self.param.beta * (self.param.M_v**2+self.param.S_v))
        l_bound += 0.5*self.obs.G*self.user_param.K
        
        # \sum\limits_k \mathcal{L}^\gamma_k 
        l_bound += (self.user_param.alpha-self.param.alpha)*numpy.sum(scipy.special.psi(self.param.alpha)-numpy.log(self.param.beta))
        l_bound += numpy.sum( self.user_param.alpha*numpy.log(self.user_param.beta) - scipy.special.gammaln(self.user_param.alpha) -self.param.alpha*numpy.log(self.param.beta) + scipy.special.gammaln(self.param.alpha))
        l_bound += self.param.alpha * numpy.sum((1.0 - self.user_param.beta/self.param.beta))

        # \sum\limits_{n,d} \mathcal{L}^{b^r}
        for n in xrange(self.obs.N):
            l_bound += 0.5* numpy.sum(numpy.log(self.param.Tau_r[n][:,0]) + numpy.log(self.param.S_r[n][:,0]))
            l_bound -= 0.5* numpy.sum(self.param.Tau_r[n][:,0] * (self.param.M_r[n][:,0]**2+self.param.S_r[n][:,0]))
            l_bound += 0.5* self.obs.D

        # \sum\limits_{n,g} \mathcal{L}^{b^c}
        for n in xrange(self.obs.N):
            l_bound += 0.5* numpy.sum(numpy.log(self.param.Tau_c[n][0,:]) + numpy.log(self.param.S_c[n][0,:]))
            l_bound -= 0.5* numpy.sum(self.param.Tau_c[n][0,:] * (self.param.M_c[n][0,:]**2+self.param.S_c[n][0,:]))
            l_bound += 0.5* self.obs.G

        return l_bound
        
#==============================================================================
# Update variational parameters          
#==============================================================================
    def update_q_u(self,k):
        u_bar_k_old = self.param.M_u[:,k].copy()

        s_u_partial = self.param.rho[k]
        u_bar_partial = 0.0
        for n in xrange(self.obs.N):
            s_u_partial += self.param.tau[n]* (self.param.a[n]**2) * self.mat_mask[n].dot(self.param.M_v[:,k]**2 + self.param.S_v[:,k])           
            u_bar_partial += self.param.tau[n] * self.param.a[n] * ((self.aux.R[n]*self.mat_mask[n]).dot(self.param.M_v[:,k]) + self.param.a[n]*numpy.sum(self.mat_mask[n]*numpy.outer(self.param.M_u[:,k],(self.param.M_v[:,k]**2)),axis=1))
                    
        i_obs = numpy.sum(self.mat_mask,axis=(0,2))!=0
        self.param.S_u[i_obs,k] = 1.0/s_u_partial[i_obs]           
        self.param.M_u[i_obs,k] = self.param.S_u[i_obs,k]*u_bar_partial[i_obs]
        # Update R
        for n in xrange(self.obs.N):
            self.aux.R[n]= self.aux.R[n] - self.mat_mask[n]*self.param.a[n]*numpy.outer((self.param.M_u[:,k]-u_bar_k_old),self.param.M_v[:,k])
        
#        u_bar_ik_old = self.param.M_u[i,k]
#    
#        s_u_partial = self.param.rho[k]
#        u_bar_partial = 0
#        which_observed = []
#        for n in xrange(self.obs.N):
#            j = numpy.where(self.mat_mask[n][i,:]==1)[0]
#            if len(j)>0:
#                which_observed.append(n)
#                s_u_partial += self.param.tau[n]* (self.param.a[n]**2) * numpy.sum(self.param.M_v[j,k]**2 + self.param.S_v[j,k])           
#                u_bar_partial += self.param.tau[n] * self.param.a[n] * numpy.sum((self.aux.R[n][i,j]+self.param.a[n]*self.param.M_u[i,k]*self.param.M_v[j,k])*self.param.M_v[j,k])            
#            
#        if len(which_observed)>0:
#            self.param.S_u[i,k] = 1.0/s_u_partial            
#            self.param.M_u[i,k] = self.param.S_u[i,k]*u_bar_partial
#            # Update aux
#            for n in which_observed:
#                j = numpy.where(self.mat_mask[n][i,:]==1)[0]
#                self.aux.R[n][i,j] = self.aux.R[n][i,j]-self.param.a[n]*(self.param.M_u[i,k]-u_bar_ik_old)*(self.param.M_v[j,k])

    def update_q_v(self,k):
        v_bar_k_old = self.param.M_v[:,k].copy()

        s_v_partial = self.param.alpha/self.param.beta[k]
        v_bar_partial = 0.0
        for n in xrange(self.obs.N):
            s_v_partial += self.param.tau[n]* (self.param.a[n]**2) * self.mat_mask[n].T.dot(self.param.M_u[:,k]**2 + self.param.S_u[:,k])           
            v_bar_partial += self.param.tau[n] * self.param.a[n] * ((self.aux.R[n]*self.mat_mask[n]).T.dot(self.param.M_u[:,k]) + self.param.a[n]*numpy.sum(self.mat_mask[n]*numpy.outer((self.param.M_u[:,k]**2),self.param.M_v[:,k]),axis=0))
                    
        j_obs = numpy.sum(self.mat_mask,axis=(0,1))!=0
        self.param.S_v[j_obs,k] = 1.0/s_v_partial[j_obs]           
        self.param.M_v[j_obs,k] = self.param.S_v[j_obs,k]*v_bar_partial[j_obs]
        # Update R
        for n in xrange(self.obs.N):
            self.aux.R[n]= self.aux.R[n] - self.mat_mask[n]*self.param.a[n]*numpy.outer((self.param.M_v[:,k]-v_bar_k_old),self.param.M_u[:,k]).T
#        v_bar_jk_old = self.param.M_v[j,k]
#        
#        s_v_partial = self.param.alpha/self.param.beta[k]
#        v_bar_partial = 0
#        which_observed = []
#        for n in xrange(self.obs.N):
#            i = numpy.where(self.mat_mask[n][:,j]==1)[0]
#            if len(i)>0:
#                which_observed.append(n)
#                s_v_partial += self.param.tau[n]* (self.param.a[n]**2) * numpy.sum(self.param.M_u[i,k]**2 + self.param.S_u[i,k])
#                v_bar_partial += self.param.tau[n] * self.param.a[n] * numpy.sum((self.aux.R[n][i,j]+self.param.a[n]*self.param.M_v[j,k]*self.param.M_u[i,k])*self.param.M_u[i,k])
#        if len(which_observed) > 0:
#            self.param.S_v[j,k] = 1.0/s_v_partial            
#            self.param.M_v[j,k] = self.param.S_v[j,k]*v_bar_partial   
#            # Update aux             
#            for n in which_observed:
#                i = numpy.where(self.mat_mask[n][:,j]==1)[0]
#                self.aux.R[n][i,j] = self.aux.R[n][i,j]-self.param.a[n]*(self.param.M_v[j,k]-v_bar_jk_old)*(self.param.M_u[i,k])
                
    
    def update_q_gamma(self):
        self.param.alpha = self.user_param.alpha + 0.5*(self.obs.G) 
        self.param.beta = self.user_param.beta + 0.5*numpy.sum((self.param.M_v**2+self.param.S_v),axis=0)


    def update_q_b_r(self):
        for n in xrange(self.obs.N):
            for i in xrange(self.obs.D):
                m_r_n_i_old = self.param.M_r[n][i,0]
                j = numpy.where(self.mat_mask[n][i,:]==1)[0]
                if len(j)>0:                    
                    self.param.M_r[n][i,:] = numpy.sum(self.aux.R[n][i,j] + m_r_n_i_old)*self.param.tau[n]/(len(j)*self.param.tau[n]+self.param.Tau_r[n][i,0])
                    # Update aux
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
                    # Update aux
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

        for n in xrange(self.obs.N): 
            # Update Z
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
            opt_a = numpy.sum(self.mat_mask[n]*mat_latent*(self.param.Z[n]-self.param.M_r[n]-self.param.M_c[n]))/numpy.sum( self.mat_mask[n]* ((mat_latent**2) + aux_variable.get_Zeta(self.param)) )
            self.param.a[n] = max(1,opt_a)
            # Update R
            self.aux.R[n] = self.mat_mask[n]*(self.aux.R[n] - (self.param.a[n] - a_n_old)*mat_latent)
            
                
    def update_tau_r(self):
        for n in xrange(self.obs.N):
            self.param.Tau_r[n] = 1.0/(self.param.M_r[n]**2+self.param.S_r[n])
        
    def update_tau_c(self):
        for n in xrange(self.obs.N):
            self.param.Tau_c[n] = 1.0/(self.param.M_c[n]**2+self.param.S_c[n])

    def update_rho(self):
        self.param.rho = self.obs.D/numpy.sum((self.param.M_u**2+self.param.S_u),axis=0)
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
#            checked = l_bound < self.compute_ELBO()
            print 'q(U) ' + str(checked)
#            stop_error(checked)
            
            for k in xrange(self.user_param.K):
                self.update_q_v(k)
#            checked = l_bound < self.compute_ELBO()
            print 'q(V) ' + str(checked)
#            stop_error(checked)
            
            self.update_q_gamma()
#            checked = l_bound < self.compute_ELBO()
            print 'q(gamma) ' + str(checked)
#            stop_error(checked)
            
            # Update tau: point estimate
            self.update_tau()
#            checked = l_bound < self.compute_ELBO()
            print 'tau ' + str(checked)
#            stop_error(checked)

            # Update a: point estimate
#            self.update_a()
#            checked = l_bound < self.compute_ELBO()
#            print 'a ' + str(checked)
#            stop_error(checked)

            self.update_q_b_r()
#            checked = l_bound < self.compute_ELBO()
            print 'q(b_r) ' + str(checked)
#            stop_error(checked)


            self.update_q_b_c()
#            checked = l_bound < self.compute_ELBO()
            print 'q(b_c) ' + str(checked)
#            stop_error(checked)

            self.update_tau_r()
#            checked = l_bound < self.compute_ELBO()
            print 'tau_r ' + str(checked)
#            stop_error(checked)

            self.update_tau_c()
#            checked = l_bound < self.compute_ELBO()
            print 'tau_c ' + str(checked)
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
#                rp_n = scipy.stats.pearsonr(y_hat,y)[0]
#                r2_n = evaluation.get_r2(y_hat,y)
#                sum_rmse += rmse_n
#                self.hist_rmse[n].append(rmse_n)
#                self.hist_r2[n].append(r2_n)
#                self.hist_rp[n].append(rp_n)
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
        input("The program is wrong !")
