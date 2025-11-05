#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os,sys,jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.scipy as jsc
import numpy as np
from jax import jacrev, vmap, jit, hessian

from abc import ABC, abstractmethod


import gwfast.population.POPutils as utils
import gwfast.population.Globals as glob


import h5py
from scipy.optimize import root


def _ensure_derivative_shape(raw):
    arr = np.asarray(raw)
    if arr.ndim == 1:
        return arr[:, None]
    if arr.ndim == 2:
        if arr.shape[1] == 2 and arr.shape[0] != 2:
            return arr.T
        elif arr.shape[0] == 2:
            return arr
        else:
            return arr.T
    raise ValueError("Derivative has unexpected ndim: %d" % arr.ndim)

def _ensure_hessian_shape(raw):
    arr = np.asarray(raw)
    if arr.ndim == 2:
        return arr[:, :, None]
    if arr.ndim == 3:
        # if shape is (Nevents,2,2) -> transpose
        if arr.shape[1] == 2 and arr.shape[2] == 2 and arr.shape[0] != 2:
            return np.transpose(arr, (1,2,0))
        return arr
    raise ValueError("Hessian has unexpected ndim: %d" % arr.ndim)


class DeltaDistribution(ABC):
    '''
    Abstract class to compute deltaPN distributions.

    :param dict, optional hyperparameters: Dictionary containing the hyperparameters of the redshift model as keys and their fiducial value as entry.
    :param dict, optional priorlims_parameters: Dictionary containing the parameters of the redshift model as keys and their prior limits value as entry, given as a tuple :math:`(l, h)`.
    
    '''

    def __init__(self,):
        
        self.par_list = ['deltaPN']
        self.hyperpar_dict = {}
        self.priorlims_dict = {}
    
    def set_hyperparameters(self, hyperparameters):
        '''
        Setter method for the hyperparameters of the delta PN model.

        :param dict new_hyperparameters: Dictionary containing the hyperparameters of the delta PN model as keys and their new value as entry.
        '''
        self.hyperpar_dict = hyperparameters

    def update_hyperparameters(self, new_hyperparameters):
        '''
        Method to update the hyperparameters of the delta PN model.

        :param dict new_hyperparameters: Dictionary containing the new hyperparameters of the delta PN model as keys and their new value as entry.
        '''
        for key in new_hyperparameters.keys():
            if key in self.hyperpar_dict.keys():
                self.hyperpar_dict[key] = new_hyperparameters[key]
            else:
                raise ValueError('The hyperparameter '+key+' is not present in the hyperparameter dictionary.')
    def set_priorlimits(self, limits):
        '''
        Setter method for the prior limits on the parameters of the delta PN model.

        :param dict limits: Dictionary containing the parameters of the delta PN model as keys and their prior limits value as entry, given as a tuple :math:`(l, h)`.
        '''
        self.priorlims_dict = limits
        
    def update_priorlimits(self, new_limits):
        '''
        Method to update the prior limits on the parameters of the delta PN model.
        
        :param dict new_limits: Dictionary containing the new prior limits on the parameters of the delta PN model as keys and their new value as entry, given as a tuple :math:`(l, h)`.
        '''
        for key in new_limits.keys():
            if key in self.priorlims_dict.keys():
                self.priorlims_dict[key] = new_limits[key]
            else:
                raise ValueError('The parameter '+key+' is not present in the prior limits dictionary.')

    def _isin_prior_range(self, par,  val):
        """
        Function to check if a value is in the prior range of a parameter.

        :param str par: Parameter name.
        :param float val: Parameter value.
        :return: Boolean value.
        :rtype: bool
        """

        return (val >= self.priorlims_dict[par][0]) & (val <= self.priorlims_dict[par][1])
        
    @abstractmethod
    def delta_function(self,):
        pass

    @abstractmethod
    def sample_population(self, size):
        pass
    
    @abstractmethod
    def delta_function_derivative(self,):
        pass
    
class Gauss_DeltaDistribution(DeltaDistribution):
    '''
    Gaussian delta PN distribution.
    
    Parameters:
        * mu_PN: loc of the normal distribution
        * sigma_PN: scale of the normal ditribution ì
    
    :param dict, optional hyperparameters: Dictionary containing the hyperparameters of the delta PN model as keys and their fiducial value as entry.
    :param dict, optional priorlims_parameters: Dictionary containing the parameters of the delta PN model as keys and their prior limits value as entry, given as a tuple :math:`(l, h)`.
    
    '''
    
    def __init__(self, hyperparameters=None, priorlims_parameters=None):
        
        self.expected_hyperpars = ['mu_PN', 'sigma_PN']
        super().__init__()
        
        if hyperparameters is not None:
            self.set_hyperparameters(hyperparameters)
        else:
            print('Expected hyperparameters: ', self.expected_hyperpars)
            basevalues = {'mu_PN': 0.0, 'sigma_PN': 0.05}
            print('Base values are: ', list(basevalues.items()))
            self.set_hyperparameters(basevalues)
        
        if priorlims_parameters is not None:
            self.set_priorlimits(priorlims_parameters)
        else:
            self.set_priorlimits({'deltaPN': (-0.5, 0.5)})
            
        self.derivative_par_nums = {'mu_PN':0, 'sigma_PN':1}
    

    def delta_function(self, deltaPN, mu_PN=None, sigma_PN=None, uselog=False):
        '''
        Delta PN distribution function.
        
        :param array deltaPN: delta PN.
        :param float, optional mu_PN, sigma_PN: loc and scale of the distribution of delta PN.
        :param bool, optional uselog: Boolean specifying whether to return the probability or log-probability, defaults to False.
        
        :return: Delta PN distribution value at the input deltas.
        :rtype: array
        '''
        
        mu_c = self.hyperpar_dict['mu_PN'] if mu_PN is None else mu_PN
        sigma_c = self.hyperpar_dict['sigma_PN'] if sigma_PN is None else sigma_PN
        
        goodsamples = self._isin_prior_range('deltaPN', deltaPN)
        deltagrid = jnp.geomspace(self.priorlims_dict['deltaPN'][0], self.priorlims_dict['deltaPN'][1], 1000)

        #distr = jsc.stats.norm.pdf(deltaPN, loc = mu_c, scale = sigma_c)
        distr = utils.gaussian_norm(deltaPN, mu_c, sigma_c)
        logdistr = jsc.stats.norm.logpdf(deltaPN, loc = mu_c, scale = sigma_c)

        if not uselog:
            return jnp.where(goodsamples, distr, 0.)
        else:
            return jnp.where(goodsamples, logdistr, -jnp.inf)#np.NINF
    
    def sample_population(self, size):
        '''
        Function to sample the delta PN model.
        
        :param int size: Size of the deltaPN sample.

        :return: Sampled deltas.
        :rtype: dict(array)
        '''
        
        deltas = utils.inverse_cdf_sampling(self.delta_function, size, self.priorlims_dict['deltaPN'])

        
        return {'deltaPN':deltas}
        
    def delta_function_derivative(self, deltaPN, mu_PN=None, sigma_PN=None, uselog=False):
        '''
        First derivative with respect to the hyperparameters of the deltaPN function.
        
        :param array deltaPN: delta PN.
        :param float, optional mu_PN, sigma_PN: loc and sigma of the delta distribution.
        :param bool uselog: Boolean specifying whether to use the probability or log-probability, defaults to False.
        
        :return: Array containing the derivatives of the deltaPN_PN distribution.
        :rtype: array
        '''
    
        mu_c = self.hyperpar_dict['mu_PN'] if mu_PN is None else mu_PN
        sigma_c = self.hyperpar_dict['sigma_PN'] if sigma_PN is None else sigma_PN

        funder = lambda deltaPN, mu_c, sigma_c: self.delta_function(deltaPN, mu_c, sigma_c, uselog=uselog)

        raw = np.squeeze(np.asarray(jacrev(funder, argnums=(1,2))(deltaPN, jnp.array([mu_c]), jnp.array([sigma_c]))))

        derivs_all = _ensure_derivative_shape(np.squeeze(np.asarray(raw)))
        return derivs_all 
    
    def delta_function_hessian(self, deltaPN, mu_PN=None, sigma_PN=None, uselog=False):
        '''
        Hessian with respect to the hyperparameters of the deltaPN function.
        
        :param array deltaPN1: deltaPN.
        :param float, optional mu_PN, sigma_PN: loc and scale of the deltaPN distribution.
        :param bool uselog: Boolean specifying whether to use the probability or log-probability, defaults to False.
        
        :return: Array containing the Hessians of the deltaPN distribution.
        :rtype: array
        '''
    
        mu_c = self.hyperpar_dict['mu_PN'] if mu_PN is None else mu_PN
        sigma_c = self.hyperpar_dict['sigma_PN'] if sigma_PN is None else sigma_PN
        
        funder = lambda deltaPN, mu_c, sigma_c: self.delta_function(deltaPN, mu_c, sigma_c, uselog=uselog)

        rawH = np.squeeze(np.asarray(hessian(funder, argnums=(1,2))(deltaPN, jnp.array([mu_c]), jnp.array([sigma_c]))))

        hess_all = _ensure_hessian_shape(np.squeeze(np.asarray(rawH)))
        return hess_all 
