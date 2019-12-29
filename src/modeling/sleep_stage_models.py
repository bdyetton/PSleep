import pymc3 as pm
import pandas as pd
import sys
import theano.tensor as tt
from theano import shared
import numpy as np
from src.modeling import param_sample_tools, base_model
from pymc3.distributions.distribution import draw_values, generate_samples

from theano.printing import Print
def pm_print(x, attrs=None, on=True):
    if on:
        if attrs is not None:
            return Print(x.name, attrs=attrs)(x)
        else:
            return Print(x.name)(x)
    else:
        return x

def pm_shape(x, on=True):
    if on:
        return Print(x.name, attrs=['shape'])(x)
    else:
        return x


class ZerothOrderSleepEpochsModel(base_model.BaseModel):
    def __init__(self):
        super().__init__()
        self.params = ['tp0']
        self.priors = {'stage_baserate': np.array([0.05, 0.05, 0.5, 0.2, 0.2, 0.001])}
        self.nstages = 6
        self.flat_priors = {'stage_baserate': np.ones((self.nstages, ))}
        self.name = 'zeroth_order_sleep_stage'

    @staticmethod
    def sample_rand_params():
        return {'tp0': param_sample_tools.rand_trans_0back(6)}

    def init_model(self, data, flat=False):
        self.output_vars.append('next_epoch')
        with pm.Model() as model:
            tp0 = pm.Dirichlet('tp0', a=self.get_prior('stage_baserate', flat))
            obs = pm.Categorical('next_epoch', p=tp0, observed=data['next_epoch'])
        return model

    @staticmethod
    def init_prior_model(fixed_params):
        with pm.Model() as prior_model:
            obs = pm.Categorical('next_epoch', p=fixed_params['tp0'])
        return prior_model


class FirstOrderSleepEpochsModel(base_model.BaseModel):
    def __init__(self):
        super().__init__()
        self.params = ['tp1']
        self.nstages = 6
        self.priors = {'1back_baserate': np.tile([0.05, 0.05, 0.5, 0.2, 0.2, 0.001],(self.nstages,1))}
        self.flat_priors = {'1back_baserate': np.ones((self.nstages, self.nstages))}
        self.name = 'first_order_sleep_stage'

    @staticmethod
    def sample_rand_params():
        return {
            'tp0': param_sample_tools.rand_trans_0back(6),
            'tp1': param_sample_tools.rand_trans_1back(6)
        }

    def init_model(self, data, flat=False):
        self.input_vars['current_epoch'] = shared(data['current_epoch'].values)
        self.output_vars.append('next_epoch')
        with pm.Model() as model:
            tp1 = pm.Dirichlet('tp1', a=self.get_prior('1back_baserate',flat),
                               shape=(self.nstages, self.nstages),
                               testval=np.ones((self.nstages, self.nstages))/self.nstages)
            tp1_epoched = tp1[self.input_vars['current_epoch'], :]
            obs = pm.Categorical('next_epoch', p=tp1_epoched, observed=data['next_epoch'])
        return model

    @staticmethod
    def init_prior_model(fixed_params):
        with pm.Model() as prior_model:
            current_epoch = pm.Categorical('current_epoch', p=fixed_params['tp0'])
            tp1 = tt.as_tensor_variable(fixed_params['tp1'])
            obs = pm.Categorical('next_epoch', p=tp1[current_epoch,:])
        return prior_model


class ParetoHarzardRateModel(base_model.BaseModel):
    def __init__(self):
        super().__init__()
        self.params = ['tp1']
        self.nstages = 5
        alpha_prior = np.ones((self.nstages, self.nstages)) * 0.01
        trans_prior = np.ones((self.nstages, self.nstages)) * 0.01
        for s in range(self.nstages):
            alpha_prior[s, s] = 0.75
            trans_prior[s, s] = 0.5
        self.priors = {'alpha': alpha_prior,
                       'trans_baserate': trans_prior}
        self.flat_priors = {'alpha': 0.5,
                            'trans_baserate': 0.5}
        self.name = 'pareto_hazard_sleep_stage'

    @staticmethod
    def sample_rand_params():
        return {
            'tp0': param_sample_tools.rand_trans_0back(6),
            'tp1': param_sample_tools.rand_trans_1back(6)
        }

    def init_model(self, data, flat=False):
        self.input_vars['current_epoch'] = shared(data['current_epoch'].values)
        self.input_vars['tau'] = shared(data['tau'].values)
        self.output_vars.append('next_epoch')
        with pm.Model() as model:
            BoundedNormal= pm.Bound(pm.Normal, lower=0, upper=1)
            alpha = BoundedNormal('alpha',
                                 mu=self.get_prior('alpha',flat),
                                 sd=1e5,
                                 shape=(self.nstages, self.nstages))
            trans_baserate = BoundedNormal('trans_baserate',
                                           mu=self.get_prior('trans_baserate',flat),
                                           sd=1e5,
                                           shape=(self.nstages, self.nstages))

            # current_idx = pm.Deterministic('current_epoch', var=tt.cast(self.input_vars['current_epoch'], 'int16'))
            # tau = pm.Deterministic('tau', var=self.input_vars['tau'])

            current_idx = tt.cast(self.input_vars['current_epoch'], 'int16')
            tau = self.input_vars['tau']

            alpha_t = alpha[current_idx, :] / tau[:, np.newaxis]
            trans_baserate_t = trans_baserate[current_idx, :]

            trans_p = pm.Deterministic('trans_p', self.norm_trans_p(alpha_t + trans_baserate_t))

            obs = pm.Categorical('next_epoch', p=trans_p, observed=data['next_epoch'])

        return model


class FullHarzardRateModel(base_model.BaseModel):
    def __init__(self):
        super().__init__()
        self.params = ['tp1']
        self.nstages = 5
        self.power_stages = [0,2,3] #['n2', 'n3', 'waso']
        alpha_prior = np.ones((self.nstages, self.nstages)) * 0.01
        trans_prior = np.ones((self.nstages, self.nstages)) * 0.01
        for s in range(self.nstages):
            trans_prior[s, s] = 0.5
            alpha_prior[s, s] = 0.75
        self.priors = {'alpha': alpha_prior,
                       'trans_baserate': trans_prior}
        self.flat_priors = {'alpha': 0.5,
                            'trans_baserate': 0.5}
        self.name = 'full_hazard_sleep_stage'

    @staticmethod
    def sample_rand_params():
        return {
            'tp0': param_sample_tools.rand_trans_0back(6),
            'tp1': param_sample_tools.rand_trans_1back(6)
        }

    def init_model(self, data, flat=False):
        tau_shared = shared(data['tau'].values)
        current_epoch_shared = shared(data['current_epoch'].values)
        self.input_vars['current_epoch'] = current_epoch_shared
        self.input_vars['tau'] = tau_shared
        self.output_vars.append('next_epoch')
        #exp_stages = ['n1', 'rem']
        with pm.Model() as model:
            BoundedNormal = pm.Bound(pm.Normal, lower=0, upper=1)
            alpha = BoundedNormal('alpha', mu=self.get_prior('alpha', flat),
                                 sd=1e5, shape=(self.nstages, self.nstages))

            trans_baserate = BoundedNormal('trans_baserate', mu=self.get_prior('trans_baserate', flat),
                                 sd=1e5, shape=(self.nstages, self.nstages))

            current_idx = pm.Deterministic('current_epoch', var=tt.cast(self.input_vars['current_epoch'], 'int16'))
            tau = pm.Deterministic('tau', var=self.input_vars['tau'])

            is_n1 = tt.eq(current_idx, 1)
            is_rem = tt.eq(current_idx, 4)
            is_exp = tt.or_(is_n1, is_rem)
            hazard_rate = is_exp[:, np.newaxis] * (alpha[current_idx, :] / tau[:, np.newaxis]) + trans_baserate[current_idx, :]
            trans_p = pm.Deterministic('trans_p', self.norm_trans_p(hazard_rate))
            next_epoch = pm.Categorical('next_epoch', p=trans_p, observed=data['next_epoch'])

        return model

