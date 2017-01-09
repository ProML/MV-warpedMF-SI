# -*- coding: utf-8 -*-
"""
Created on Wed May 13 13:53:35 2015

@author: Rukawa
"""
import numpy

class vis_settings(object):

    train_color = 'green'
    test_color = 'red'
    alpha = 0.5
    marker_size = 3
    marker = 'o'
    bin_size = 0.005
    train_label = 'Train'
    test_label = 'Test'
    pred_label = 'Prediced'
    lw_horizontal = 2
    
    @staticmethod
    def get_bin_list(min_edge,max_edge,bin_size):
        N = (max_edge-min_edge)/bin_size
        Nplus1 = N + 1
        return numpy.linspace(min_edge, max_edge, Nplus1)
