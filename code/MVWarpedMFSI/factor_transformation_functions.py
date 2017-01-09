# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 11:41:30 2015

@author: Rukawa
"""

import numpy

_exp_lim_val = numpy.finfo(numpy.float64).max
_lim_val = 36.0
epsilon = numpy.finfo(numpy.float64).resolution

class linear(object):
    def __init__(self,x):
        self.x = x
        
    def f(self):
        return self.x
    def f_prime(self):
        return 1.0
    def __str__(self):
        return ''

class log_exp(object):
    def __init__(self,x):
        self.x = x
    def f(self):
        return numpy.log(1. + numpy.exp(self.x))
    def f_prime(self):
        return 1. - numpy.exp(-self.x)
    def __str__(self):
        return '+ve'

class exp(object):
    def __init__(self,x):
        self.x = x
    def f(self):
        return numpy.exp(self.x)
    def f_prime(self):
        return numpy.exp(self.x)
    def __str__(self):
        return '+ve'
