# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 14:41:10 2015

@author: Rukawa
"""


import numpy
import scipy.optimize
from matplotlib import pyplot as plt
import factor_transformation_functions
from const import const

class warping_functions(object):
    
    """
    Abstract methods for the warping functions
    z = f(x)
    """

    def __init__(self):
        raise NotImplementedError

    def f(self,x):
        """
        Function transformation
        x = observed data
        """
        raise NotImplementedError

    def f_prime(self,x):
        """
        Gradient of f w.r.t. to x
        """
        raise NotImplementedError

    def update_parameters(self):
        """
        Update parameters
        """
        raise NotImplementedError

    def pdf(self,d,g,mu_z,var_z,space):
#        sd_z = numpy.sqrt(1.00/tau_z)
        sd_z = numpy.sqrt(var_z)
        z = numpy.linspace(mu_z-3*sd_z,mu_z+3*sd_z,1000)
        pdf = scipy.stats.norm.pdf(z,loc=mu_z,scale=sd_z)
        if space == 'observation':
            x = self.f_inv(z)
            pdf *= self.f_prime(x,i=-1) 
            return x,pdf
        else:
            return z,pdf


    def f_inv(self,z,x=None,rate = 0.1, max_iters=50):
        """
        Inverse function transformation
        """
        if x is None:
            x = numpy.zeros_like(z).astype(float) # How to choose starting points !
        update = numpy.inf
        t = 0
        while t == 0 or (numpy.abs(update).sum() > (1e-5) and t < max_iters):
            update = rate*(self.f(x) - z) / self.f_prime(x)
            x -= update
            t += 1 
        if t == max_iters:
            print "WARNING: maximum number of iterations reached in f_inv "
        return x

    def f_inv_gauss_hermite(self,mu_z,tau_z,deg=10):
        """
        Mean of the inverse function transformation under a Gaussian density
        """
        x,w = numpy.polynomial.hermite.hermgauss(deg)
        sqrt_2_over_tau_z = numpy.sqrt(2.0/tau_z)
        mu_x = numpy.zeros_like(mu_z).astype(float)
        for t in xrange(deg):
            new_arg = x[t]*sqrt_2_over_tau_z + mu_z 
            mu_x += w[t]*self.f_inv(new_arg)
        return (1.0/numpy.sqrt(numpy.pi)) * mu_x


    def plot(self, xmin, xmax):
        """
        Plot the warping function
        """
        y = numpy.arange(xmin, xmax, 0.01)
        f_y = self.f(y)
        plt.figure()
        plt.plot(y, f_y)
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Warping function')
        
            
#==============================================================================
# Class composite_warping
#==============================================================================
class composite_warping(warping_functions):
    def __init__(self,I_list=[3,3],func_type_list=['tanh','linear_curvature']):
        self.I_list = I_list
        self.func_type_list = func_type_list
        self.n_functions = len(self.func_type_list)
        self.warping_f_list = numpy.empty(self.n_functions,dtype=object)
        for nf in xrange(self.n_functions):
            if self.func_type_list[nf] == 'tanh':
                self.warping_f_list[nf] = tanh_warping(self.I_list[nf])
            elif self.func_type_list[nf] == 'linear_curvature':
                self.warping_f_list[nf] = linear_curvature_warping(self.I_list[nf])
            elif self.func_type_list[nf] == 'logistic':
                self.warping_f_list[nf] = logistic_warping(self.I_list[nf])
            elif self.func_type_list[nf] == 'exp':
                self.warping_f_list[nf] = exp_warping(self.I_list[nf])
    
    def f(self,x,nf=-1,i=-1):
        """
        Function transformation
        x = observed data
        """        
        res = 0.0
        for l in xrange(self.n_functions):
            if l != nf:
                res += self.warping_f_list[l].f(x,i=-1)        
            else:
                res += self.warping_f_list[l].f(x,i)        
        return x + res

    def f_prime(self,x,nf=-1,i=-1):
        """
        Gradient of f w.r.t. to x
        """
        res = 0.0
        for l in xrange(self.n_functions):
            if l != nf:
                res += self.warping_f_list[l].f_prime(x,i=-1)        
            else:
                res += self.warping_f_list[l].f_prime(x,i)        
        return 1.0 + res
        
    def update_parameters(self,X,F,tau,mat_mask):
        """
        Update parameters
        """
        for nf in xrange(self.n_functions):
            for i in xrange(self.I_list[nf]):
                partial_f_val = self.f(X,nf,i)
                partial_f_prime_val = self.f_prime(X,nf,i)
                self.warping_f_list[nf].update_parameters(i,X,F,tau,mat_mask,partial_f_val,partial_f_prime_val)        

#==============================================================================
# Class tanh_warping
#==============================================================================

class tanh_warping(warping_functions):
    
    def __init__(self,I=3):
        self.I = I
        self.a_bounds = [(None,None)] * self.I
        self.b_bounds = [(None,None)] * self.I
        self.c_bounds = [(None,None)] * self.I
        self.options = dict()
        self.options['maxiter'] = 50
        self.factor_ft_type = dict()
        self.factor_ft_type['a'] = factor_transformation_functions.log_exp
        self.factor_ft_type['b'] = factor_transformation_functions.log_exp
        self.factor_ft_type['c'] = factor_transformation_functions.linear
        self.a = [[]] * self.I
        self.b = [[]] * self.I
        self.c = [[]] * self.I
        for i in xrange(self.I):
            self.a[i] = self.factor_ft_type['a'](numpy.random.random())
            self.b[i] = self.factor_ft_type['b'](numpy.random.random())
            self.c[i] = self.factor_ft_type['c'](numpy.random.random())
        self.method = 'L-BFGS-B'
#        self.method = 'TNC'
        self.univariate_opt = True


    def f(self,x,i=-1):
        """
        Transform x using all warping terms except the i^th term
        If i is -1, all warping terms are used. 
        """
        res = 0.0
        for l in xrange(self.I):
            if l!=i: 
                res += _tanh(x,self.a[l],self.b[l],self.c[l])
        return res
    

    def f_prime(self,x,i=-1):
        """
        Gradient of f w.r.t. to x of all warping terms except the i^th term
        """
        res = 0.0
        for l in xrange(self.I):
            if l!=i: 
                res += _tanh_prime(x,self.a[l],self.b[l],self.c[l])
        return res



    def update_parameters(self,i,X,F,tau,mat_mask,partial_f_val,partial_f_prime_val):
#        print 'scalar'
#        res = scipy.optimize.minimize_scalar(fun=self._f_ELBO, bounds=self.a_bounds[i], args=('a',i,X,F,tau,mat_mask,partial_f_val,partial_f_prime_val), method='bounded', tol=None, options=self.options)
        res = scipy.optimize.minimize(self._f_ELBO, jac=self._f_ELBO_d_param, x0=numpy.random.random(), method=self.method,args=('a',i,X,F,tau,mat_mask,partial_f_val,partial_f_prime_val),bounds=[self.a_bounds[i]],options=self.options)
        if res.success and res.nit > 0 and res.jac < 1E-4:
            self.a[i].x = res.x
        print 'a = ' + str(self.a[i].x)

        res = scipy.optimize.minimize(self._f_ELBO, jac=self._f_ELBO_d_param, x0=numpy.random.random(), method=self.method,args=('b',i,X,F,tau,mat_mask,partial_f_val,partial_f_prime_val),bounds=[self.b_bounds[i]],options=self.options)
        if res.success and res.nit > 0 and res.jac < 1E-4:
            self.b[i].x = res.x            
        print 'b = ' + str(self.b[i].x)
            
        res = scipy.optimize.minimize(self._f_ELBO, jac=self._f_ELBO_d_param, x0=numpy.random.random(), method=self.method,args=('c',i,X,F,tau,mat_mask,partial_f_val,partial_f_prime_val),bounds=[self.c_bounds[i]],options=self.options)
        if res.success and res.nit > 0 and res.jac < 1E-4:
            self.c[i].x = res.x            
        print 'c = ' + str(self.c[i].x)

    def _f_ELBO(self,param,param_opt,i,X,F,tau,mat_mask,partial_f_val,partial_f_prime_val):

        if param_opt == 'a':
            a_i = self.factor_ft_type['a'](param)
        else:
            a_i = self.a[i]
        if param_opt == 'b':
            b_i = self.factor_ft_type['b'](param)
        else:
            b_i = self.b[i]
        if param_opt == 'c':
            c_i = self.factor_ft_type['c'](param)
        else:
            c_i = self.c[i]

        Z = partial_f_val + _tanh(X,a_i,b_i,c_i)
        ans = numpy.log(partial_f_prime_val + _tanh_prime(X,a_i,b_i,c_i)) 
        ans -= 0.5*tau * ((Z - F)**2)
        func = numpy.sum( mat_mask * ans)
        return -func

        
    def _f_ELBO_d_param(self,param,param_opt,i,X,F,tau,mat_mask,partial_f_val,partial_f_prime_val):

        if param_opt == 'a':
            a_i = self.factor_ft_type['a'](param)
        else:
            a_i = self.a[i]
        if param_opt == 'b':
            b_i = self.factor_ft_type['b'](param)
        else:
            b_i = self.b[i]
        if param_opt == 'c':
            c_i = self.factor_ft_type['c'](param)
        else:
            c_i = self.c[i]

        inv_f_prime_val = 1.0/(partial_f_prime_val + _tanh_prime(X,a_i,b_i,c_i))        
        Z = partial_f_val + _tanh(X,a_i,b_i,c_i)

        tau_Z_F = tau * (Z-F)
        tanh_val = numpy.tanh(b_i.f()*(X + c_i.f()))
        d_tanh_val = 1 - tanh_val**2

    
        if param_opt == 'a':
            grad = inv_f_prime_val * b_i.f() * d_tanh_val 
            grad -= tau_Z_F * tanh_val
            grad *= a_i.f_prime()
           
        if param_opt == 'b':
            grad = inv_f_prime_val * ( a_i.f() * d_tanh_val - a_i.f() * b_i.f() * (2 * tanh_val) * d_tanh_val * (X + c_i.f()) )
            grad -= tau_Z_F * a_i.f() * d_tanh_val * (X + c_i.f())
            grad *= b_i.f_prime()

        if param_opt == 'c':
            grad = (-1.0) * inv_f_prime_val * a_i.f() * (b_i.f() ** 2) * (2 * tanh_val) * d_tanh_val
            grad -= tau_Z_F * a_i.f() * b_i.f() * d_tanh_val
            grad *= c_i.f_prime()
        return numpy.array([(-1.0) * numpy.sum(mat_mask*grad)])


#==============================================================================
# Class box_cox_warping
#==============================================================================

#==============================================================================
# Class linear_curvature_warping
#==============================================================================
class linear_curvature_warping(warping_functions):
    def __init__(self,I=3):
        self.I = I
        self.a_bounds = [(None,None)] * self.I
        self.b_bounds = [(None,None)] * self.I
        self.c_bounds = [(None,None)] * self.I
        self.d_bounds = [(None,None)] * self.I
        self.options = dict()
        self.options['maxiter'] = 200
        self.factor_ft_type = dict()
        self.factor_ft_type['a'] = factor_transformation_functions.linear
        self.factor_ft_type['b'] = factor_transformation_functions.log_exp
        self.factor_ft_type['c'] = factor_transformation_functions.log_exp
        self.factor_ft_type['d'] = factor_transformation_functions.linear
        self.a = [[]] * self.I
        self.b = [[]] * self.I
        self.c = [[]] * self.I
        self.d = [[]] * self.I
        for i in xrange(self.I):
            self.a[i] = self.factor_ft_type['a'](numpy.random.random())
            self.b[i] = self.factor_ft_type['b'](numpy.random.random())
            self.c[i] = self.factor_ft_type['c'](numpy.random.random())
            self.d[i] = self.factor_ft_type['d'](numpy.random.random())
        self.method = 'L-BFGS-B'
        self.univariate_opt = True

    def f(self,x,i=-1):
        """
        Transform x using all warping terms except the i^th term
        If i is -1, all warping terms are used. 
        """
        res = 0.0
        for l in xrange(self.I):
            if l!=i: 
                res += _linear_curvature(x,self.a[l],self.b[l],self.c[l],self.d[l])
        return res
    

    def f_prime(self,x,i=-1):
        """
        Gradient of f w.r.t. to x of all warping terms except the i^th term
        """
        res = 0.0
        for l in xrange(self.I):
            if l!=i: 
                res += _linear_curvature_prime(x,self.a[l],self.b[l],self.c[l],self.d[l])
        return res


    def update_parameters(self,i,X,F,tau,mat_mask,partial_f_val,partial_f_prime_val):
        res = scipy.optimize.minimize(self._f_ELBO, jac=False, x0=numpy.random.random(), method=self.method,args=('a',i,X,F,tau,mat_mask,partial_f_val,partial_f_prime_val),bounds=[self.a_bounds[i]],options=self.options)
        if res.success and res.nit > 0 and res.jac < 1E-4:
            self.a[i].x = res.x
        res = scipy.optimize.minimize(self._f_ELBO, jac=False, x0=numpy.random.random(), method=self.method,args=('b',i,X,F,tau,mat_mask,partial_f_val,partial_f_prime_val),bounds=[self.b_bounds[i]],options=self.options)
        if res.success and res.nit > 0 and res.jac < 1E-4:
            self.b[i].x = res.x            
            
        res = scipy.optimize.minimize(self._f_ELBO, jac=False, x0=numpy.random.random(), method=self.method,args=('c',i,X,F,tau,mat_mask,partial_f_val,partial_f_prime_val),bounds=[self.c_bounds[i]],options=self.options)
        if res.success and res.nit > 0 and res.jac < 1E-4:
            self.c[i].x = res.x

        res = scipy.optimize.minimize(self._f_ELBO, jac=False, x0=numpy.random.random(), method=self.method,args=('d',i,X,F,tau,mat_mask,partial_f_val,partial_f_prime_val),bounds=[self.d_bounds[i]],options=self.options)
        if res.success and res.nit > 0 and res.jac < 1E-4:
            self.d[i].x = res.x
        
    def _f_ELBO(self,param,param_opt,i,X,F,tau,mat_mask,partial_f_val,partial_f_prime_val):

        if param_opt == 'a':
            a_i = self.factor_ft_type['a'](param)
        else:
            a_i = self.a[i]
        if param_opt == 'b':
            b_i = self.factor_ft_type['b'](param)
        else:
            b_i = self.b[i]
        if param_opt == 'c':
            c_i = self.factor_ft_type['c'](param)
        else:
            c_i = self.c[i]
        if param_opt == 'd':
            d_i = self.factor_ft_type['d'](param)
        else:
            d_i = self.d[i]

        Z = partial_f_val + _linear_curvature(X,a_i,b_i,c_i,d_i)
        ans = numpy.log(partial_f_prime_val + _linear_curvature_prime(X,a_i,b_i,c_i,d_i)) 
        ans -= 0.5*tau * ((Z - F)**2)
        func = numpy.sum( mat_mask * ans)
        return -func

        

#==============================================================================
# Class logistic_warping
#==============================================================================

class logistic_warping(warping_functions): 
    
    def __init__(self,I=3):
        self.I = I
        self.a_bounds = [(None,None)] * self.I
        self.b_bounds = [(None,None)] * self.I
        self.c_bounds = [(None,None)] * self.I
        self.options = dict()
        self.options['maxiter'] = 200
        self.factor_ft_type = dict()
        self.factor_ft_type['a'] = factor_transformation_functions.log_exp
        self.factor_ft_type['b'] = factor_transformation_functions.log_exp
        self.factor_ft_type['c'] = factor_transformation_functions.linear
        self.a = [[]] * self.I
        self.b = [[]] * self.I
        self.c = [[]] * self.I
        for i in xrange(self.I):
            self.a[i] = self.factor_ft_type['a'](numpy.random.random())
            self.b[i] = self.factor_ft_type['b'](numpy.random.random())
            self.c[i] = self.factor_ft_type['c'](numpy.random.random())
        self.method = 'L-BFGS-B'
        self.univariate_opt = True


    def f(self,x,i=-1):
        """
        Transform x using all warping terms except the i^th term
        If i is -1, all warping terms are used. 
        """
        res = 0.0
        for l in xrange(self.I):
            if l!=i: 
                res += _logistic(x,self.a[l],self.b[l],self.c[l])
        return res
    

    def f_prime(self,x,i=-1):
        """
        Gradient of f w.r.t. to x of all warping terms except the i^th term
        """
        res = 0.0
        for l in xrange(self.I):
            if l!=i: 
                res += _logistic_prime(x,self.a[l],self.b[l],self.c[l])
        return res

    def update_parameters(self,i,X,F,tau,mat_mask,partial_f_val,partial_f_prime_val):

        res = scipy.optimize.minimize(self._f_ELBO, jac=False, x0=numpy.random.random(), method=self.method,args=('a',i,X,F,tau,mat_mask,partial_f_val,partial_f_prime_val),bounds=[self.a_bounds[i]],options=self.options)
        print res
        if res.success and res.nit > 0 and res.jac < 1E-4:
            self.a[i].x = res.x
        res = scipy.optimize.minimize(self._f_ELBO, jac=False, x0=numpy.random.random(), method=self.method,args=('b',i,X,F,tau,mat_mask,partial_f_val,partial_f_prime_val),bounds=[self.b_bounds[i]],options=self.options)
        print res
        if res.success and res.nit > 0 and res.jac < 1E-4:
            self.b[i].x = res.x            
            
        res = scipy.optimize.minimize(self._f_ELBO, jac=False, x0=numpy.random.random(), method=self.method,args=('c',i,X,F,tau,mat_mask,partial_f_val,partial_f_prime_val),bounds=[self.c_bounds[i]],options=self.options)
        print res
        if res.success and res.nit > 0 and res.jac < 1E-4:
            self.c[i].x = res.x

    def _f_ELBO(self,param,param_opt,i,X,F,tau,mat_mask,partial_f_val,partial_f_prime_val):

        if param_opt == 'a':
            a_i = self.factor_ft_type['a'](param)
        else:
            a_i = self.a[i]
        if param_opt == 'b':
            b_i = self.factor_ft_type['b'](param)
        else:
            b_i = self.b[i]
        if param_opt == 'c':
            c_i = self.factor_ft_type['c'](param)
        else:
            c_i = self.c[i]

        Z = partial_f_val + _logistic(X,a_i,b_i,c_i)
        ans = numpy.log(partial_f_prime_val + _logistic_prime(X,a_i,b_i,c_i)) 
        ans -= 0.5*tau * ((Z - F)**2)
        func = numpy.sum( mat_mask * ans)
        return -func

        
#==============================================================================
# Class exp_warping 
#==============================================================================
class exp_warping(warping_functions):  # not ready
    
    def __init__(self,I=3):
        self.I = I
        self.a_bounds = [(None,None)] * self.I
        self.options = dict()
        self.options['maxiter'] = 200
        self.factor_ft_type = dict()
        self.factor_ft_type['a'] = factor_transformation_functions.exp
        self.a = [self.factor_ft_type['a'](numpy.random.random())] * self.I
        self.method = 'L-BFGS-B'
        self.univariate_opt = True


    def f(self,x,i=-1):
        """
        Transform x using all warping terms except the i^th term
        If i is -1, all warping terms are used. 
        """
        res = 0.0
        for l in xrange(self.I):
            if l!=i: 
                res += _gauss2exp(x,self.a[l])
        return res
    

    def f_prime(self,x,i=-1):
        """
        Gradient of f w.r.t. to x of all warping terms except the i^th term
        """
        res = 0.0
        for l in xrange(self.I):
            if l!=i: 
                res += _gauss2exp_prime(x,self.a[l])
        return res



    def update_parameters(self,i,X,F,tau,mat_mask,partial_f_val,partial_f_prime_val):

        res = scipy.optimize.minimize(self._f_ELBO, jac=False, x0=numpy.random.random(), method=self.method,args=('a',i,X,F,tau,mat_mask,partial_f_val,partial_f_prime_val),bounds=[self.a_bounds[i]],options=self.options)
        print res
        if res.success and res.nit > 0 and res.jac < 1E-4:
            self.a[i].x = res.x
                


    def _f_ELBO(self,param,param_opt,i,X,F,tau,mat_mask,partial_f_val,partial_f_prime_val):
        if param_opt == 'a':
            a_i = self.factor_ft_type['a'](param)
        else:
            a_i = self.a[i]

        Z = partial_f_val + _gauss2exp(X,a_i)
        
        ans = numpy.log(partial_f_prime_val + _gauss2exp_prime(X,a_i)) 
        ans -= 0.5*tau * ((Z - F)**2)
        func = numpy.sum( mat_mask * ans)
        return -func

        
            
#==============================================================================
# Warping unit functions
#==============================================================================
def _tanh(x,a,b,c):
    return a.f() * numpy.tanh( b.f()*(x + c.f()) )

def _tanh_prime(x,a,b,c):
    """    
    Derivative of _tanh w.r.t. x 
    """
    return a.f() * b.f() * (1.0-numpy.tanh(b.f()*(x + c.f()))**2)


def _linear_curvature(x,a,b,c,d):
    F = numpy.exp(a.f()*b.f()*(x-d.f()))
    G = numpy.exp(a.f()*c.f()*(x-d.f()))
    if a.f()==0:
        return 0.0
    else:
        return (1.0/a.f()) * numpy.log(F+G)

def _linear_curvature_prime(x,a,b,c,d):
    """    
    Derivative of _linear_curvature w.r.t. x 
    """
    F = numpy.exp(a.f()*b.f()*(x-d.f()))
    G = numpy.exp(a.f()*c.f()*(x-d.f()))
    return (b.f()*F + c.f()*G)/(F+G)

def _logistic(x,a,b,c):
    return a.f()*1.0/(1.0+numpy.exp(-b.f()*(x-c.f())))

def _logistic_prime(x,a,b,c):
    return a.f() * b.f() * _logistic(x,a,b,c) * (1.0-_logistic(x,a,b,c))


def _gauss2exp(x,a): 
    return - (1./ a.f()) * numpy.log(0.5 - 0.5*scipy.special.erf(x/const.sqrt2))

def _gauss2exp_prime(x,a): # check this !
#    return (1.0/ (const.sqrt2pi * a.f())) * numpy.exp(a.f() * _gauss2exp(x,a) - 0.5*(x**2))
    return (1.0/ (const.sqrt2pi * a.f())) * (1.0/(0.5 - 0.5*scipy.special.erf(x/const.sqrt2))) * numpy.exp(0.5*(x**2))
           
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
    