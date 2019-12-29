import pymc3 as pm
import pandas as pd
import sys
from theano.printing import Print
import theano.tensor as tt
import numpy as np
from src.modeling import base_model


def Print(x):  # Comment out to turn on printing
    return lambda y: y


class ExponentialDuration(base_model.BaseModel):
    def __init__(self):
        self.params = ['lam']
        self.priors = {'lam_mu': 1, 'lam_sd': 100} #keeping it vague for now
        self.flat_priors = {'lam_mu': 1, 'lam_sd': 100}
        self.name = 'exponential_duration'

    @staticmethod
    def sample_rand_params():
        mu=1
        sd=10
        alpha=(mu^2)/(sd^2)
        beta=mu/(sd^2)
        return {'lam': np.random.gamma(shape=alpha, scale=beta)}

    def init_model(self, data, flat=False):
        with pm.Model() as model:
            lam = pm.Gamma('lam',
                            mu=self.flat_priors['lam_mu'] if flat else self.priors['lam_mu'],
                            sd=self.flat_priors['lam_sd'] if flat else self.priors['lam_sd'])
            obs = pm.Exponential('duration', lam=lam, observed=data['duration'])
        return model

    @staticmethod
    def init_prior_model(fixed_params):
        with pm.Model() as prior_model:
            obs = pm.Exponential('duration', lam=fixed_params['lam'])
        return prior_model


# [WIP]
# class FirstOrderSleepEpochsWithDurationModel(base_model.BaseModel):
#     def __init__(self):
#         self.params = ['tp1']
#         self.nstages = 6
#         self.priors = {'1back_baserate': np.tile([0.05, 0.05, 0.5, 0.2, 0.2, 0.001],(self.nstages,1))}
#         self.flat_priors = {'1back_baserate': np.ones((self.nstages, self.nstages))}
#         self.name = 'first_order_sleep_stage'
#
#     @staticmethod
#     def sample_rand_params():
#         return {
#             'tp0': param_sample_tools.rand_trans_0back(6),
#             'tp1': param_sample_tools.rand_trans_1back(6)
#         }
#
#     def init_model(self, data, flat=False):
#         with pm.Model() as model:
#             tp1 = pm.Dirichlet('tp1', a=self.flat_priors['1back_baserate']
#                                             if flat else self.priors['1back_baserate'],
#                                shape=(self.nstages, self.nstages),
#                                testval=np.ones((self.nstages, self.nstages))/self.nstages)
#             tp1_epoched = tp1[data['current_epoch'], :]
#             obs = pm.Categorical('obs', p=tp1_epoched, observed=data['next_epoch'])
#         return model
#
#     @staticmethod
#     def init_prior_model(fixed_params):
#         with pm.Model() as prior_model:
#             current_epoch = pm.Categorical('current_epoch', p=fixed_params['tp0'])
#             tp1 = tt.as_tensor_variable(fixed_params['tp1'])
#             obs = pm.Categorical('next_epoch', p=tp1[current_epoch,:])
#         return prior_model