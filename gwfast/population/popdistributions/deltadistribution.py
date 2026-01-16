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


class PhiPNDistribution(ABC):
    '''
    Abstract class to compute Phi_PN distributions.

    :param dict, optional hyperparameters: Dictionary containing the hyperparameters of the redshift model as keys and their fiducial value as entry.
    :param dict, optional priorlims_parameters: Dictionary containing the parameters of the redshift model as keys and their prior limits value as entry, given as a tuple :math:`(l, h)`.
    
    '''

    def __init__(self,):
        
        self.par_list = []
        self.hyperpar_dict = {}
        self.priorlims_dict = {}

    def set_parameters(self, parameters):
        '''
        Setter method for the parameters of the mass model.

        :param list parameters: List containing the parameters of the mass model.
        '''
        self.par_list = parameters
    
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
    def PhiPN_function(self,):
        pass

    @abstractmethod
    def sample_population(self, size):
        pass
    
    @abstractmethod
    def PhiPN_function_derivative(self,):
        pass

    
class Gauss_PhiPNDistribution(PhiPNDistribution):
    '''
    Gaussian Phi_PN distribution for a single PN order.
    Questa versione è stata riscritta per usare una lambda esplicita con jacrev,
    mantenendo la coerenza di stile con la classe N-dimensionale.
    
    :param list, optional orderPN: List containing the single PN order to model, e.g., [0]. Defaults to [0].
    '''
    
    def __init__(self, hyperparameters=None, priorlims_parameters=None, orderPN=None):
        
        if orderPN is None:
            orderPN = [0]
        
        if len(orderPN) != 1:
            raise ValueError("Gauss_PhiPNDistribution only supports a single PN order. For multiple orders, use Gauss2D_PhiPNDistribution or other distributions.")

        self.orderPN = orderPN
        self.order_idx = self.orderPN[0]

        self.phi_name = f'Phi_{self.order_idx}'
        self.mu_name = f'mu_PN{self.order_idx}'
        self.sigma_name = f'sigma_{self.order_idx}{self.order_idx}'

        self.expected_hyperpars = [self.mu_name, self.sigma_name]
        super().__init__()
        
        self.set_parameters([self.phi_name])
        
        if hyperparameters is not None:
            self.set_hyperparameters(hyperparameters)
        else:
            print('Expected hyperparameters: ', self.expected_hyperpars)
            basevalues = {self.mu_name: 0.0, self.sigma_name: 0.05}
            print('Base values are: ', list(basevalues.items()))
            self.set_hyperparameters(basevalues)
        
        if priorlims_parameters is not None:
            self.set_priorlimits(priorlims_parameters)
        else:
            self.set_priorlimits({self.phi_name: (-0.5, 0.5)})
            
        self.derivative_par_nums = {name: i for i, name in enumerate(self.expected_hyperpars)}
    

    def PhiPN_function(self, phi_val, uselog=False, **hyperparams):
        '''
        Delta PN distribution function for a single order.
        La firma ora accetta il valore di Phi come primo argomento posizionale.
        '''
        mu_c = hyperparams.get(self.mu_name, self.hyperpar_dict[self.mu_name])
        sigma_c = hyperparams.get(self.sigma_name, self.hyperpar_dict[self.sigma_name])
        
        goodsamples = self._isin_prior_range(self.phi_name, phi_val)

        #distr = utils.gaussian_norm(phi_val, mu_c, sigma_c)
        distr = jsc.stats.norm.pdf(phi_val, loc = mu_c, scale = sigma_c)
        logdistr = jsc.stats.norm.logpdf(phi_val, loc=mu_c, scale=sigma_c)

        if not uselog:
            return jnp.where(goodsamples, distr, jnp.zeros_like(distr))
        else:
            return jnp.where(goodsamples, logdistr, jnp.full_like(logdistr, -jnp.inf))
    
    def sample_population(self, size, **hyperparams):
        '''
        Function to sample the delta PN model.
        '''
        mu_c = hyperparams.get(self.mu_name, self.hyperpar_dict[self.mu_name])
        sigma_c = hyperparams.get(self.sigma_name, self.hyperpar_dict[self.sigma_name])

        def func_to_sample(phi_val):
            return self.PhiPN_function(phi_val, uselog=False, **{self.mu_name: mu_c, self.sigma_name: sigma_c})

        deltas = utils.inverse_cdf_sampling(func_to_sample, size, self.priorlims_dict[self.phi_name])
        
        return {self.phi_name: deltas}

    def PhiPN_function_derivative(self, phi_val, uselog=False, **hyperparams):
        '''
        First derivative, riscritta con la logica della lambda posizionale.
        '''
        mu_c = hyperparams.get(self.mu_name, self.hyperpar_dict[self.mu_name])
        sigma_c = hyperparams.get(self.sigma_name, self.hyperpar_dict[self.sigma_name])

        funder = lambda p, m, s: self.PhiPN_function(p, uselog=uselog, **{self.mu_name: m, self.sigma_name: s})

        argnums_to_diff = (1, 2)
        
        raw = jacrev(funder, argnums=argnums_to_diff)(phi_val, jnp.array(mu_c), jnp.array(sigma_c))

        derivs_all = jnp.stack(raw, axis=-1)
        return np.asarray(derivs_all).T
    
    def PhiPN_function_hessian(self, phi_val, uselog=False, **hyperparams):
        '''
        Hessian, riscritta con la logica della lambda posizionale.
        '''
        mu_c = hyperparams.get(self.mu_name, self.hyperpar_dict[self.mu_name])
        sigma_c = hyperparams.get(self.sigma_name, self.hyperpar_dict[self.sigma_name])
        
        funder = lambda p, m, s: self.PhiPN_function(p, uselog=uselog, **{self.mu_name: m, self.sigma_name: s})

        argnums_to_diff = (1, 2)

        rawH = hessian(funder, argnums=argnums_to_diff)(phi_val, jnp.array(mu_c), jnp.array(sigma_c))

        num_hyper = len(self.expected_hyperpars)
        hess_matrix = jnp.stack([jnp.stack([rawH[i][j] for j in range(num_hyper)], axis=0)for i in range(num_hyper)],axis=0)
        
        return np.asarray(hess_matrix)


class Gauss2D_PhiPNDistribution(PhiPNDistribution):
    '''
    N-dimensional Gaussian Phi_PN distribution for a generic list of PN orders.
    This class generalizes the logic to any number of dimensions based on the 'orderPN' parameter.
    
    :param list, optional orderPN: List of PN orders to model, e.g., [0, 1]. Defaults to [0, 1].
    :param dict, optional hyperparameters: Dictionary of hyperparameters.
    :param dict, optional priorlims_parameters: Dictionary of prior limits for the Phi parameters.
    '''
    
    def __init__(self, hyperparameters=None, priorlims_parameters=None, orderPN=None):

        self.key = jax.random.PRNGKey(42)
        
        if orderPN is None:
            orderPN = [0, 1]
        self.orderPN = sorted(orderPN)
        self.N_dims = len(self.orderPN)
        
        self.phi_names = [f'Phi_{i}' for i in self.orderPN]
        self.mu_names = [f'mu_PN{i}' for i in self.orderPN]
        self.sigma_names = []
        
        for i in range(self.N_dims):
            for j in range(i, self.N_dims):
                idx1, idx2 = sorted((self.orderPN[i], self.orderPN[j]))
                sigma_name = f'sigma_{idx1}{idx2}'
                if sigma_name not in self.sigma_names:
                    self.sigma_names.append(sigma_name)
        
        self.expected_hyperpars = self.mu_names + self.sigma_names
        
        super().__init__()
        
        self.set_parameters(self.phi_names)
        
        if hyperparameters is not None:
            self.set_hyperparameters(hyperparameters)
        else:
            print('Expected hyperparameters: ', self.expected_hyperpars)
            basevalues = {}
            for mu_name in self.mu_names:
                basevalues[mu_name] = 0.0
            for sigma_name in self.sigma_names:
                indices = sigma_name.split('_')[1]
                if indices[0] == indices[-1]:
                    basevalues[sigma_name] = 0.05 
                else:
                    basevalues[sigma_name] = 0.00
            print('Base values are: ', list(basevalues.items()))
            self.set_hyperparameters(basevalues)
        
        if priorlims_parameters is not None:
            self.set_priorlimits(priorlims_parameters)
        else:
            prior_limits = {name: (-0.5, 0.5) for name in self.phi_names}
            self.set_priorlimits(prior_limits)
            
        self.derivative_par_nums = {name: i for i, name in enumerate(self.expected_hyperpars)}

    def _build_mean_and_cov(self, **hyperparams):
        """Helper function to dynamically build mean vector and covariance matrix."""
        mean_vector = jnp.array([
            hyperparams.get(name, self.hyperpar_dict[name]) for name in self.mu_names
        ])

        cov_matrix = jnp.zeros((self.N_dims, self.N_dims))
        for i in range(self.N_dims):
            for j in range(i, self.N_dims):
                idx1, idx2 = sorted((self.orderPN[i], self.orderPN[j]))
                sigma_name = f'sigma_{idx1}{idx2}'
                sigma_val = hyperparams.get(sigma_name, self.hyperpar_dict[sigma_name])
                
                if i == j:
                    cov_matrix = cov_matrix.at[i, i].set(sigma_val**2)
                else:
                    cov_matrix = cov_matrix.at[i, j].set(sigma_val)
                    cov_matrix = cov_matrix.at[j, i].set(sigma_val) #Matrice simmetrica

        for i in range(self.N_dims):
            for j in range(i, self.N_dims):
                if i != j:
                    if abs(cov_matrix[i,j]) >= jnp.sqrt(cov_matrix[i,i] * cov_matrix[j,j]):
                            raise ValueError(f"Invalid covariance: |cov_ij|={abs(sigma_val)} must be < {cov_matrix[i,i]*cov_matrix[j,j]}")
        
        return mean_vector, cov_matrix

    def PhiPN_function(self, *phis, uselog=False, **hyperparams):
        if len(phis) != self.N_dims:
            raise ValueError(f"Expected {self.N_dims} Phi arguments, but got {len(phis)}")

        mean_vector, cov_matrix = self._build_mean_and_cov(**hyperparams)
        
        samples = jnp.stack(phis, axis=-1)
        
        good_samples = jnp.full(phis[0].shape, True)
        for i, phi_val in enumerate(phis):
            good_samples &= self._isin_prior_range(self.phi_names[i], phi_val)

        distr = jsc.stats.multivariate_normal.pdf(samples, mean=mean_vector, cov=cov_matrix)
        logdistr = jsc.stats.multivariate_normal.logpdf(samples, mean=mean_vector, cov=cov_matrix)

        if not uselog:
            return jnp.where(good_samples, distr, jnp.zeros_like(distr))
        else:
            return jnp.where(good_samples, logdistr, jnp.full_like(logdistr, -jnp.inf))

    def sample_population(self, size, **hyperparams):
        
        mean_vector, cov_matrix = self._build_mean_and_cov(**hyperparams)
        
        self.key, subkey = jax.random.split(self.key)
        sampled_points = jax.random.multivariate_normal(subkey, mean=mean_vector, cov=cov_matrix, shape=(size,))
        
        return {self.phi_names[i]: np.asarray(sampled_points[:, i]) for i in range(self.N_dims)}

    def PhiPN_function_derivative(self, *phis, uselog=False, **hyperparams):
        if len(phis) != self.N_dims:
            raise ValueError(f"Expected {self.N_dims} Phi arguments, but got {len(phis)}")

        hyperparam_values = [hyperparams.get(name, self.hyperpar_dict[name]) for name in self.expected_hyperpars]

        def funder(*all_args):
            phis_in = all_args[:self.N_dims]
            hyperparams_in = all_args[self.N_dims:]
            hyperparams_dict = {name: val for name, val in zip(self.expected_hyperpars, hyperparams_in)}
            
            return self.PhiPN_function(*phis_in, uselog=uselog, **hyperparams_dict)

        argnums_to_diff = tuple(range(self.N_dims, self.N_dims + len(self.expected_hyperpars)))
    
        all_args = list(phis) + hyperparam_values
        raw = jacrev(funder, argnums=argnums_to_diff)(*all_args)

        derivs_all = jnp.stack(raw, axis=-1)
        return np.asarray(derivs_all).T

    def PhiPN_function_hessian(self, *phis, uselog=False, **hyperparams):
        if len(phis) != self.N_dims:
            raise ValueError(f"Expected {self.N_dims} Phi arguments, but got {len(phis)}")

        hyperparam_values = [hyperparams.get(name, self.hyperpar_dict[name]) for name in self.expected_hyperpars]

        def funder(*all_args):
            phis_in = all_args[:self.N_dims]
            hyperparams_in = all_args[self.N_dims:]
            hyperparams_dict = {name: val for name, val in zip(self.expected_hyperpars, hyperparams_in)}
            return self.PhiPN_function(*phis_in, uselog=uselog, **hyperparams_dict)
        
        argnums_to_diff = tuple(range(self.N_dims, self.N_dims + len(self.expected_hyperpars)))
        
        all_args = list(phis) + hyperparam_values
        rawH = hessian(funder, argnums=argnums_to_diff)(*all_args)

        num_hyper = len(self.expected_hyperpars)
        hess_matrix = jnp.stack([jnp.stack([rawH[i][j] for j in range(num_hyper)], axis=0)for i in range(num_hyper)],axis=0)
        
        return np.asarray(hess_matrix)


class Gauss3D_PhiPNDistribution(PhiPNDistribution):
    '''
    N-dimensional Gaussian Phi_PN distribution for a generic list of PN orders.
    This class generalizes the logic to any number of dimensions based on the 'orderPN' parameter.

    To create a 3D distribution, initialize with: orderPN=[0, 1, 2]
    To create a 5D distribution, initialize with: orderPN=[0, 1, 2, 3, 4]
    
    :param list, optional orderPN: List of PN orders to model, e.g., [0, 1, 2]. Defaults to [0, 1, 2].
    :param dict, optional hyperparameters: Dictionary of hyperparameters.
    :param dict, optional priorlims_parameters: Dictionary of prior limits for the Phi parameters.
    '''
    
    def __init__(self, hyperparameters=None, priorlims_parameters=None, orderPN=None):

        self.key = jax.random.PRNGKey(42)
        
        if orderPN is None:
            orderPN = [0, 1, 2]
        self.orderPN = sorted(orderPN)
        self.N_dims = len(self.orderPN)
        
        self.phi_names = [f'Phi_{i}' for i in self.orderPN]
        self.mu_names = [f'mu_PN{i}' for i in self.orderPN]
        self.sigma_names = []
        
        for i in range(self.N_dims):
            for j in range(i, self.N_dims):
                idx1, idx2 = sorted((self.orderPN[i], self.orderPN[j]))
                sigma_name = f'sigma_{idx1}{idx2}'
                if sigma_name not in self.sigma_names:
                    self.sigma_names.append(sigma_name)
        
        self.expected_hyperpars = self.mu_names + self.sigma_names
        
        super().__init__()
        
        self.set_parameters(self.phi_names)
        
        if hyperparameters is not None:
            self.set_hyperparameters(hyperparameters)
        else:
            print('Expected hyperparameters: ', self.expected_hyperpars)
            basevalues = {}
            for mu_name in self.mu_names:
                basevalues[mu_name] = 0.0
            for sigma_name in self.sigma_names:
                indices = sigma_name.split('_')[1]
                if indices[0] == indices[-1]:
                    basevalues[sigma_name] = 0.05 
                else:
                    basevalues[sigma_name] = 0.0
            print('Base values are: ', list(basevalues.items()))
            self.set_hyperparameters(basevalues)
        
        if priorlims_parameters is not None:
            self.set_priorlimits(priorlims_parameters)
        else:
            prior_limits = {name: (-0.5, 0.5) for name in self.phi_names}
            self.set_priorlimits(prior_limits)
            
        self.derivative_par_nums = {name: i for i, name in enumerate(self.expected_hyperpars)}

    def _build_mean_and_cov(self, **hyperparams):
        """Helper function to dynamically build mean vector and covariance matrix."""
        mean_vector = jnp.array([hyperparams.get(name, self.hyperpar_dict[name]) for name in self.mu_names])

        cov_matrix = jnp.zeros((self.N_dims, self.N_dims))
        for i in range(self.N_dims):
            for j in range(i, self.N_dims):
                idx1, idx2 = sorted((self.orderPN[i], self.orderPN[j]))
                sigma_name = f'sigma_{idx1}{idx2}'
                sigma_val = hyperparams.get(sigma_name, self.hyperpar_dict[sigma_name])
                
                if i == j:
                    cov_matrix = cov_matrix.at[i, i].set(sigma_val**2)
                else:
                    cov_matrix = cov_matrix.at[i, j].set(sigma_val)
                    cov_matrix = cov_matrix.at[j, i].set(sigma_val)

        for i in range(self.N_dims):
            for j in range(i, self.N_dims):
                if i != j:
                    if abs(cov_matrix[i,j]) >= jnp.sqrt(cov_matrix[i,i] * cov_matrix[j,j]):
                            raise ValueError(f"Invalid covariance: |cov_ij|={abs(sigma_val)} must be < {cov_matrix[i,i]*cov_matrix[j,j]}")
        
        return mean_vector, cov_matrix

    def PhiPN_function(self, *phis, uselog=False, **hyperparams):
        if len(phis) != self.N_dims:
            raise ValueError(f"Expected {self.N_dims} Phi arguments, but got {len(phis)}")

        mean_vector, cov_matrix = self._build_mean_and_cov(**hyperparams)
        
        samples = jnp.stack(phis, axis=-1)
        
        good_samples = jnp.full(jnp.shape(phis[0]), True)
        for i, phi_val in enumerate(phis):
            good_samples &= self._isin_prior_range(self.phi_names[i], phi_val)

        distr = jsc.stats.multivariate_normal.pdf(samples, mean=mean_vector, cov=cov_matrix)
        logdistr = jsc.stats.multivariate_normal.logpdf(samples, mean=mean_vector, cov=cov_matrix)

        if not uselog:
            return jnp.where(good_samples, distr, jnp.zeros_like(distr))
        else:
            return jnp.where(good_samples, logdistr, jnp.full_like(logdistr, -jnp.inf))

    def sample_population(self, size, **hyperparams):
        
        mean_vector, cov_matrix = self._build_mean_and_cov(**hyperparams)
        
        self.key, subkey = jax.random.split(self.key)
        sampled_points = jax.random.multivariate_normal(subkey, mean=mean_vector, cov=cov_matrix, shape=(size,))
        
        return {self.phi_names[i]: np.asarray(sampled_points[:, i]) for i in range(self.N_dims)}

    def PhiPN_function_derivative(self, *phis, uselog=False, **hyperparams):
        if len(phis) != self.N_dims:
            raise ValueError(f"Expected {self.N_dims} Phi arguments, but got {len(phis)}")

        hyperparam_values = [hyperparams.get(name, self.hyperpar_dict[name]) for name in self.expected_hyperpars]

        def funder(*all_args):
            phis_in = all_args[:self.N_dims]
            hyperparams_in = all_args[self.N_dims:]
            hyperparams_dict = {name: val for name, val in zip(self.expected_hyperpars, hyperparams_in)}
            
            return self.PhiPN_function(*phis_in, uselog=uselog, **hyperparams_dict)

        argnums_to_diff = tuple(range(self.N_dims, self.N_dims + len(self.expected_hyperpars)))
    
        all_args = list(phis) + hyperparam_values
        raw = jacrev(funder, argnums=argnums_to_diff)(*all_args)

        derivs_all = jnp.stack(raw, axis=-1)
        return np.asarray(derivs_all).T

    def PhiPN_function_hessian(self, *phis, uselog=False, **hyperparams):
        if len(phis) != self.N_dims:
            raise ValueError(f"Expected {self.N_dims} Phi arguments, but got {len(phis)}")

        hyperparam_values = [hyperparams.get(name, self.hyperpar_dict[name]) for name in self.expected_hyperpars]

        def funder(*all_args):
            phis_in = all_args[:self.N_dims]
            hyperparams_in = all_args[self.N_dims:]
            hyperparams_dict = {name: val for name, val in zip(self.expected_hyperpars, hyperparams_in)}
            return self.PhiPN_function(*phis_in, uselog=uselog, **hyperparams_dict)
        
        argnums_to_diff = tuple(range(self.N_dims, self.N_dims + len(self.expected_hyperpars)))
        
        all_args = list(phis) + hyperparam_values
        rawH = hessian(funder, argnums=argnums_to_diff)(*all_args)

        num_hyper = len(self.expected_hyperpars)
        hess_matrix = jnp.stack([jnp.stack([rawH[i][j] for j in range(num_hyper)], axis=0)for i in range(num_hyper)],axis=0)
        
        return np.asarray(hess_matrix)



class Gauss5D_PhiPNDistribution(PhiPNDistribution):
    '''
    N-dimensional Gaussian Phi_PN distribution for a generic list of PN orders.
    This class generalizes the logic to any number of dimensions based on the 'orderPN' parameter.

    The default is a 5D distribution for orders [0, 1, 2, 3, 4].
    
    :param list, optional orderPN: List of PN orders to model, e.g., [0, 1, 2, 3, 4]. Defaults to [0, 1, 2, 3, 4].
    :param dict, optional hyperparameters: Dictionary of hyperparameters.
    :param dict, optional priorlims_parameters: Dictionary of prior limits for the Phi parameters.
    '''
    
    def __init__(self, hyperparameters=None, priorlims_parameters=None, orderPN=None):

        self.key = jax.random.PRNGKey(42)
        
        if orderPN is None:
            orderPN = [0, 1, 2, 3, 4]  #Default to 5D case
        self.orderPN = sorted(orderPN)
        self.N_dims = len(self.orderPN)
        
        self.phi_names = [f'Phi_{i}' for i in self.orderPN]
        self.mu_names = [f'mu_PN{i}' for i in self.orderPN]
        self.sigma_names = []
        
        for i in range(self.N_dims):
            for j in range(i, self.N_dims):
                idx1, idx2 = sorted((self.orderPN[i], self.orderPN[j]))
                sigma_name = f'sigma_{idx1}{idx2}'
                if sigma_name not in self.sigma_names:
                    self.sigma_names.append(sigma_name)
        
        self.expected_hyperpars = self.mu_names + self.sigma_names
        
        super().__init__()
        
        self.set_parameters(self.phi_names)
        
        if hyperparameters is not None:
            self.set_hyperparameters(hyperparameters)
        else:
            print('Expected hyperparameters: ', self.expected_hyperpars)
            basevalues = {}
            for mu_name in self.mu_names:
                basevalues[mu_name] = 0.0
            for sigma_name in self.sigma_names:
                indices = sigma_name.split('_')[1]
                if indices[0] == indices[-1]:
                    basevalues[sigma_name] = 0.05
                else:
                    basevalues[sigma_name] = 0.0 #024 
            print('Base values are: ', list(basevalues.items()))
            self.set_hyperparameters(basevalues)
        
        if priorlims_parameters is not None:
            self.set_priorlimits(priorlims_parameters)
        else:
            prior_limits = {name: (-0.5, 0.5) for name in self.phi_names}
            self.set_priorlimits(prior_limits)
            
        self.derivative_par_nums = {name: i for i, name in enumerate(self.expected_hyperpars)}

    def _build_mean_and_cov(self, **hyperparams):
        """Helper function to dynamically build mean vector and covariance matrix."""
        mean_vector = jnp.array([hyperparams.get(name, self.hyperpar_dict[name]) for name in self.mu_names])

        cov_matrix = jnp.zeros((self.N_dims, self.N_dims))
        for i in range(self.N_dims):
            for j in range(i, self.N_dims):
                idx1, idx2 = sorted((self.orderPN[i], self.orderPN[j]))
                sigma_name = f'sigma_{idx1}{idx2}'
                sigma_val = hyperparams.get(sigma_name, self.hyperpar_dict[sigma_name])
                
                if i == j:
                    cov_matrix = cov_matrix.at[i, i].set(sigma_val**2)
                else:
                    cov_matrix = cov_matrix.at[i, j].set(sigma_val)
                    cov_matrix = cov_matrix.at[j, i].set(sigma_val)

        for i in range(self.N_dims):
            for j in range(i, self.N_dims):
                if i != j:
                    limit = jnp.sqrt(cov_matrix[i,i] * cov_matrix[j,j])
                    if abs(cov_matrix[i,j]) > limit:
                        raise ValueError(f"Invalid covariance: |cov_ij|={abs(sigma_val)} must be < {limit}")
        
        return mean_vector, cov_matrix

    def PhiPN_function(self, *phis, uselog=False, **hyperparams):
        if len(phis) != self.N_dims:
            raise ValueError(f"Expected {self.N_dims} Phi arguments, but got {len(phis)}")

        mean_vector, cov_matrix = self._build_mean_and_cov(**hyperparams)
        
        samples = jnp.stack(phis, axis=-1)
        
        good_samples = jnp.full(jnp.shape(phis[0]), True)
        for i, phi_val in enumerate(phis):
            good_samples &= self._isin_prior_range(self.phi_names[i], phi_val)

        distr = jsc.stats.multivariate_normal.pdf(samples, mean=mean_vector, cov=cov_matrix)
        logdistr = jsc.stats.multivariate_normal.logpdf(samples, mean=mean_vector, cov=cov_matrix)

        if not uselog:
            return jnp.where(good_samples, distr, jnp.zeros_like(distr))
        else:
            return jnp.where(good_samples, logdistr, jnp.full_like(logdistr, -jnp.inf))

    def sample_population(self, size, **hyperparams):
        
        mean_vector, cov_matrix = self._build_mean_and_cov(**hyperparams)
        
        self.key, subkey = jax.random.split(self.key)
        sampled_points = jax.random.multivariate_normal(subkey, mean=mean_vector, cov=cov_matrix, shape=(size,))
        
        return {self.phi_names[i]: np.asarray(sampled_points[:, i]) for i in range(self.N_dims)}

    def PhiPN_function_derivative(self, *phis, uselog=False, **hyperparams):
        if len(phis) != self.N_dims:
            raise ValueError(f"Expected {self.N_dims} Phi arguments, but got {len(phis)}")

        hyperparam_values = [hyperparams.get(name, self.hyperpar_dict[name]) for name in self.expected_hyperpars]

        def funder(*all_args):
            phis_in = all_args[:self.N_dims]
            hyperparams_in = all_args[self.N_dims:]
            hyperparams_dict = {name: val for name, val in zip(self.expected_hyperpars, hyperparams_in)}
            
            return self.PhiPN_function(*phis_in, uselog=uselog, **hyperparams_dict)

        argnums_to_diff = tuple(range(self.N_dims, self.N_dims + len(self.expected_hyperpars)))
    
        all_args = list(phis) + hyperparam_values
        raw = jacrev(funder, argnums=argnums_to_diff)(*all_args)

        derivs_all = jnp.stack(raw, axis=-1)
        return np.asarray(derivs_all).T

    def PhiPN_function_hessian(self, *phis, uselog=False, **hyperparams):
        if len(phis) != self.N_dims:
            raise ValueError(f"Expected {self.N_dims} Phi arguments, but got {len(phis)}")

        hyperparam_values = [hyperparams.get(name, self.hyperpar_dict[name]) for name in self.expected_hyperpars]

        def funder(*all_args):
            phis_in = all_args[:self.N_dims]
            hyperparams_in = all_args[self.N_dims:]
            hyperparams_dict = {name: val for name, val in zip(self.expected_hyperpars, hyperparams_in)}
            return self.PhiPN_function(*phis_in, uselog=uselog, **hyperparams_dict)
        
        argnums_to_diff = tuple(range(self.N_dims, self.N_dims + len(self.expected_hyperpars)))
        
        all_args = list(phis) + hyperparam_values
        rawH = hessian(funder, argnums=argnums_to_diff)(*all_args)

        num_hyper = len(self.expected_hyperpars)
        hess_matrix = jnp.stack([jnp.stack([rawH[i][j] for j in range(num_hyper)], axis=0)for i in range(num_hyper)],axis=0)
        
        return np.asarray(hess_matrix)


class Gauss7D_PhiPNDistribution(PhiPNDistribution):
    """
    N-dimensional Gaussian Phi_PN distribution for a generic list of PN orders.
    Default is 7D distribution for orders [0,1,2,3,4,5,6].

    :param list orderPN: list of PN orders to model, defaults to [0..6]
    :param dict hyperparameters: optional dict with hyperparameter values
    :param dict priorlims_parameters: optional prior limits for Phi params
    """

    def __init__(self, hyperparameters=None, priorlims_parameters=None, orderPN=None):
        self.key = jax.random.PRNGKey(42)

        if orderPN is None:
            orderPN = [0, 1, 2, 3, 4, 5, 6]  #Default 7D
        self.orderPN = sorted(orderPN)
        self.N_dims = len(self.orderPN)

        self.phi_names = [f'Phi_{i}' for i in self.orderPN]
        self.mu_names = [f'mu_PN{i}' for i in self.orderPN]

        self.sigma_names = []
        for i in range(self.N_dims):
            for j in range(i, self.N_dims):
                idx1, idx2 = sorted((self.orderPN[i], self.orderPN[j]))
                sigma_name = f'sigma_{idx1}{idx2}'
                if sigma_name not in self.sigma_names:
                    self.sigma_names.append(sigma_name)

        self.expected_hyperpars = self.mu_names + self.sigma_names

        super().__init__()

        self.set_parameters(self.phi_names)
        if hyperparameters is not None:
            self.set_hyperparameters(hyperparameters)
        else:
            #default: mu = 0.0, diagonal sigma small (0.05), off-diagonal 0.0
            basevalues = {}
            for mu_name in self.mu_names:
                basevalues[mu_name] = 0.0
            for sigma_name in self.sigma_names:
                indices = sigma_name.split('_')[1]
                # diagonal entries like sigma_00, sigma_11 -> set small positive std
                if indices[0] == indices[-1]:
                    basevalues[sigma_name] = 0.05
                else:
                    basevalues[sigma_name] = 0.0
            self.set_hyperparameters(basevalues)

        if priorlims_parameters is not None:
            self.set_priorlimits(priorlims_parameters)
        else:
            prior_limits = {name: (-0.5, 0.5) for name in self.phi_names}
            self.set_priorlimits(prior_limits)

        self.derivative_par_nums = {name: i for i, name in enumerate(self.expected_hyperpars)}

    def _build_mean_and_cov(self, **hyperparams):
        mean_vector = jnp.array([hyperparams.get(name, self.hyperpar_dict[name]) for name in self.mu_names])

        cov_matrix = jnp.zeros((self.N_dims, self.N_dims))
        for i in range(self.N_dims):
            for j in range(i, self.N_dims):
                idx1, idx2 = sorted((self.orderPN[i], self.orderPN[j]))
                sigma_name = f'sigma_{idx1}{idx2}'
                sigma_val = hyperparams.get(sigma_name, self.hyperpar_dict[sigma_name])

                if i == j:
                    cov_matrix = cov_matrix.at[i, i].set(sigma_val**2)
                else:
                    cov_matrix = cov_matrix.at[i, j].set(sigma_val)
                    cov_matrix = cov_matrix.at[j, i].set(sigma_val)
                    if abs(sigma_val) >= jnp.sqrt(cov_matrix[i,i] * cov_matrix[j,j]):
                        raise ValueError("Invalid covariance: |cov_ij| must be < sigma_i * sigma_j. Try a smaller value.")

        return mean_vector, cov_matrix

    def PhiPN_function(self, *phis, uselog=False, **hyperparams):
        if len(phis) != self.N_dims:
            raise ValueError(f"Expected {self.N_dims} Phi arguments, but got {len(phis)}")

        mean_vector, cov_matrix = self._build_mean_and_cov(**hyperparams)

        samples = jnp.stack(phis, axis=-1)

        good_samples = jnp.full(jnp.shape(phis[0]), True)
        for i, phi_val in enumerate(phis):
            good_samples &= self._isin_prior_range(self.phi_names[i], phi_val)

        distr = jsc.stats.multivariate_normal.pdf(samples, mean=mean_vector, cov=cov_matrix)
        logdistr = jsc.stats.multivariate_normal.logpdf(samples, mean=mean_vector, cov=cov_matrix)

        if not uselog:
            return jnp.where(good_samples, distr, jnp.zeros_like(distr))
        else:
            return jnp.where(good_samples, logdistr, jnp.full_like(logdistr, -jnp.inf))

    def sample_population(self, size, **hyperparams):
        mean_vector, cov_matrix = self._build_mean_and_cov(**hyperparams)

        self.key, subkey = jax.random.split(self.key)
        sampled_points = jax.random.multivariate_normal(subkey, mean=mean_vector, cov=cov_matrix, shape=(size,))

        return {self.phi_names[i]: np.asarray(sampled_points[:, i]) for i in range(self.N_dims)}

    def PhiPN_function_derivative(self, *phis, uselog=False, **hyperparams):
        if len(phis) != self.N_dims:
            raise ValueError(f"Expected {self.N_dims} Phi arguments, but got {len(phis)}")

        hyperparam_values = [hyperparams.get(name, self.hyperpar_dict[name]) for name in self.expected_hyperpars]

        def funder(*all_args):
            phis_in = all_args[:self.N_dims]
            hyperparams_in = all_args[self.N_dims:]
            hyperparams_dict = {name: val for name, val in zip(self.expected_hyperpars, hyperparams_in)}
            return self.PhiPN_function(*phis_in, uselog=uselog, **hyperparams_dict)

        argnums_to_diff = tuple(range(self.N_dims, self.N_dims + len(self.expected_hyperpars)))

        all_args = list(phis) + hyperparam_values
        raw = jacrev(funder, argnums=argnums_to_diff)(*all_args)

        derivs_all = jnp.stack(raw, axis=-1)
        return np.asarray(derivs_all)

    def PhiPN_function_hessian(self, *phis, uselog=False, **hyperparams):
        if len(phis) != self.N_dims:
            raise ValueError(f"Expected {self.N_dims} Phi arguments, but got {len(phis)}")

        hyperparam_values = [hyperparams.get(name, self.hyperpar_dict[name]) for name in self.expected_hyperpars]

        def funder(*all_args):
            phis_in = all_args[:self.N_dims]
            hyperparams_in = all_args[self.N_dims:]
            hyperparams_dict = {name: val for name, val in zip(self.expected_hyperpars, hyperparams_in)}
            return self.PhiPN_function(*phis_in, uselog=uselog, **hyperparams_dict)

        argnums_to_diff = tuple(range(self.N_dims, self.N_dims + len(self.expected_hyperpars)))

        all_args = list(phis) + hyperparam_values
        rawH = hessian(funder, argnums=argnums_to_diff)(*all_args)

        num_hyper = len(self.expected_hyperpars)
        #assemble block Hessian matrix (num_hyper x num_hyper)
        hess_matrix = jnp.block([[jnp.array(rawH[i][j]) for j in range(num_hyper)] for i in range(num_hyper)])

        return np.asarray(hess_matrix)