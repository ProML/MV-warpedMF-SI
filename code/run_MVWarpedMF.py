# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 12:22:54 2015

@author: Rukawa
"""
import numpy
import scipy.stats


from MVWarpedMF.user_parameter import user_parameter
from MVWarpedMF.vb import inference
from MVWarpedMF import evaluation
from MVWarpedMF import aux_variable

def run_MVWarpedMF(obs,up,mat_mask_train,mat_mask_test,is_latent_space=False):
    
    user_param = user_parameter(n_ranks = up.K,n_folds=up.n_folds,max_iters=up.max_iters,convergence_threshold=up.convergence_threshold,n_processes=1,learn_bias_terms=up.learn_bias_terms)

    mf = inference(obs,user_param,mat_mask_train)
    mf.fit()
            
    return mf

def evaluate_model(obs,mf,mat_mask_train,mat_mask_test,is_latent_space=False):
    rmse = [[]] * obs.N
    r2 = [[]] * obs.N
    rs = [[]] * obs.N    
    for n in xrange(obs.N):
        mask = mat_mask_test[n].copy()
        mask[:,numpy.where(numpy.sum(mat_mask_train[n],axis=0)==0)[0]] = False
        mask[numpy.where(numpy.sum(mat_mask_train[n],axis=1)==0)[0],:] = False
        if is_latent_space:
            """
            Performance in the latent space
            """
            pred_n = aux_variable.get_F(mf.param,n)
            y_hat = pred_n.flatten()[mask.flatten()]
            y = mf.warping[n].f(obs.S[n].flatten()[mask.flatten()],i=-1)
        else:
            """
            Performance in the observation space
            """            
            pred_n = mf.warping[n].f_inv_gauss_hermite(aux_variable.get_F(mf.param,n),mf.param.tau[n])
            y_hat = pred_n.flatten()[mask.flatten()]
            y = obs.S[n].flatten()[mask.flatten()]
            
        rmse[n] = evaluation.get_rmse(y_hat,y)
        r2[n] = evaluation.get_r2(y_hat,y)
        rs[n] = scipy.stats.spearmanr(y_hat,y)[0]
            
    results = {'rmse':rmse,'r2':r2,'rs':rs}

    return results

    
