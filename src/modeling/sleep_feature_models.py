import pymc3 as pm
import pandas as pd
import sys
import theano.tensor as tt
from theano import shared
import numpy as np
from src.modeling import param_sample_tools, base_model
from pymc3.distributions.distribution import draw_values, generate_samples

from theano.printing import Print
# def Print(x, attrs=None):  # Comment out to turn on printing
#     return lambda y: y

class SpindlesModel(base_model.BaseModel):
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
        self.output_vars.append('spindle_density')

        with pm.Model() as model:
            BoundedLower = pm.Bound(pm.Normal, lower=0)
            mu_baseline = BoundedLower('spindle_baseline_mu', mu=self.get_prior('spindle_baseline_mu', flat),
                                 sd=1e5, shape=self.nstages)

            BoundedLower = pm.Bound(pm.Normal, lower=0)
            mu_time = BoundedLower('spindle_time_mu', mu=self.get_prior('spindle_time_mu', flat),
                                 sd=1e5, shape=self.nstages)

            current_idx = tt.cast(data['current_epoch'], 'int16')
            mu_inst = mu_baseline[current_idx] + mu_time[current_idx]
            spindle_density = pm.Poisson('spindle_density', mu=mu_inst, observed=data['spindle_density'])

        return model


#TODO: include effects of time
#TODO: age, sex, ethnicity effects
#TODO: Spindle model - start separately?

