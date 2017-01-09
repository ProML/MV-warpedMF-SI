# -*- coding: utf-8 -*-
"""
Created on Tue May 12 10:52:37 2015

@author: Rukawa
"""
import numpy

def get_r2(y_hat,y):
    ssr = numpy.sum((y-y_hat)**2)
    tss = numpy.sum((y-numpy.mean(y))**2)
    r2 = 1.0-ssr/tss
    return r2
    
def get_rmse(y_hat,y):
    return numpy.sqrt(numpy.mean((y - y_hat)**2))


def get_each_nrmse(pred_list,obs_list,mask_list,N):
    each_nrmse = numpy.full([N],numpy.nan)
    for n in xrange(N):    
        norm = numpy.max(obs_list[n]*mask_list[n]) - numpy.min(obs_list[n]*mask_list[n])
        each_nrmse[n] = numpy.sqrt(numpy.mean(((obs_list[n] - pred_list[n])*mask_list[n])**2))/norm
    return each_nrmse
    
def get_learning_mean(mat_mask,y,ori_mat_mask):
    new_mat_mask = mat_mask*ori_mat_mask
    y_hat = numpy.copy(y)
    y_hat[~new_mat_mask] = numpy.nan
    col_mean = numpy.nanmean(y_hat,axis=0)
    col_mean[numpy.isnan(col_mean)] = 0.0
    row_mean = numpy.nanmean(y_hat,axis=1)
    matrix_mean = numpy.nanmean(y_hat)
    row_mean[numpy.isnan(row_mean)] = 0.0    
    return matrix_mean,row_mean,col_mean