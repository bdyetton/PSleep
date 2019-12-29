import pymc3 as pm
import pandas as pd
import sys
import theano.tensor as tt
from theano import shared
import numpy as np
from src.modeling import param_sample_tools, base_model

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



class MetaModel(base_model.BaseModel):
    def __init__(self, name, terms, maps):
        super().__init__()
        self.params = ['alpha', 'trans_baserate', 'clocktime_rate']
        self.nstages = 5
        self.terms = terms
        alpha_prior = np.ones((self.nstages, self.nstages)) * 0.01
        trans_prior = np.ones((self.nstages, self.nstages)) * 0.01
        for s in range(self.nstages):
            alpha_prior[s, s] = 0.75
            trans_prior[s, s] = 0.5
        self.priors = {'alpha_p': alpha_prior,
                       'trans_baserate_p': trans_prior,
                       'clocktime_rate_p': [0,],
                       'sex_p': 0,
                       'age_p': 0,
                       'eth_p': 0,
                       'age_sex_p':0,
                       'clocktime_p': 0,
                       'clocktime_p^2':0}
        self.flat_priors = {'alpha_p': 0.5,
                            'trans_baserate_p': 0.5,
                            'clocktime_baserate_p':0.01,
                            'age_p':0,
                            'sex_p':0}
        self.name = name
        self.maps = maps
        self.nstudies = len(maps['studyid'])
        self.det_vars = {}

    def init_model(self, data, outcome='next_epoch', flat=False):
        assert np.all(data['current_epoch'].isin(np.arange(0, self.nstages)))
        if ~isinstance(outcome, list):
            outcome = [outcome]
        cols = [col for col in list(self.terms.keys()) + ['current_epoch', 'studyid'] + outcome if col in data.columns]
        dat_before = data.shape[0]
        data = data.loc[:, cols].dropna()
        print('Fitting on', data.shape[0],'records (dropped', dat_before-data.shape[0],')')

        trans_p_terms = []

        self.input_vars['current_epoch'] = shared(data['current_epoch'].values)
        self.input_vars['studyid'] = shared(data['studyid'].values)
        for term in self.terms:
            if term in data.columns:
                self.input_vars[term] = shared(data[term].values)
        self.output_vars += outcome

        with pm.Model() as model:

            current_idx = pm.Deterministic('current_epoch', var=tt.cast(self.input_vars['current_epoch'], 'int16'))
            self.det_vars['current_epoch'] = current_idx
            studyid = pm.Deterministic('studyid', var=tt.cast(self.input_vars['studyid'], 'int16'))
            self.det_vars['studyid'] = current_idx
            if 'previous_bout' in self.terms:
                previous_bout = pm.Deterministic('previous_bout', var=tt.cast(self.input_vars['previous_bout'], 'int16'))
                self.det_vars['previous_bout'] = previous_bout
            for term in self.terms:
                if term in data.columns:
                    self.det_vars[term] = pm.Deterministic(term, var=self.input_vars[term])

            assert 'baseline' in self.terms
            if self.terms['baseline']['type'] == 'nested':
                baseline_effect_t = self.add_baseline_nested(studyid, current_idx)
            elif self.terms['baseline']['type'] == 'fixed':
                baseline_effect_t = self.add_baseline_fixed(current_idx)
            elif self.terms['baseline']['type'] == 'restricted':
                baseline_effect_t = self.add_baseline_restricted(current_idx)
            else:
                baseline_effect_t = self.add_baseline(current_idx)
            trans_p_terms.append(baseline_effect_t)

            if 'previous_bout' in self.terms:
                previous_bout_t = self.add_previous_bout(previous_bout, current_idx, **self.terms['previous_bout'])
                trans_p_terms.append(previous_bout_t)

            if 'clocktime' in self.terms and self.terms['clocktime'] is not None:
                if self.terms['clocktime']['type'] == 'linear' or self.terms['clocktime']['type'] == 'quadratic':
                    clocktime_effect_t = self.add_linear_effect('clocktime_p', self.det_vars['clocktime'], current_idx, **self.terms['clocktime'])
                    trans_p_terms.append(clocktime_effect_t)
                if self.terms['clocktime']['type'] == 'quadratic':
                    clocktime_effect_t = self.add_quadratic_effect('clocktime_p', self.det_vars['clocktime'], current_idx, **self.terms['clocktime'])
                    trans_p_terms.append(clocktime_effect_t)
                if self.terms['clocktime']['type'] == 'sine':
                    clocktime_effect_t = self.add_clocktime_effects(current_idx, self.det_vars['clocktime'])
                    trans_p_terms.append(clocktime_effect_t)

            if 'age_sex' in self.terms:
                age_sex_t = self.add_interaction_effect('age_p','sex_p',
                                                        self.det_vars['age'], self.det_vars['sex'],
                                                        current_idx,
                                                        **self.terms['age_sex'])
                trans_p_terms.append(age_sex_t)

            if 'tau' in self.terms:
                if 'type' not in self.terms['tau'] or self.terms['tau']['type'] == 'pareto':
                    tau_t = self.add_pareto_tau_effect(current_idx, self.det_vars['tau'], **self.terms['tau'])
                    trans_p_terms.append(tau_t)
                elif self.terms['tau']['type'] == 'linear' or self.terms['tau']['type']=='quadratic':
                    tau_t = self.add_linear_effect('tau_lin_p', current_idx, self.det_vars['tau'], **self.terms['tau'])
                    trans_p_terms.append(tau_t)
                elif self.terms['tau']['type'] == 'quadratic':
                    tau_t = self.add_quadratic_effect('tau_lin_p', current_idx, self.det_vars['tau'], **self.terms['tau'])
                    trans_p_terms.append(tau_t)

            if 'age' in self.terms:
                age_t = self.add_linear_effect('age_p', self.det_vars['age'], current_idx, **self.terms['age'])
                trans_p_terms.append(age_t)

            if 'sex' in self.terms:
                sex_t = self.add_linear_effect('sex_p', self.det_vars['sex'], current_idx, **self.terms['sex'])
                trans_p_terms.append(sex_t)

            if 'ethnicity' in self.terms:
                sex_t = self.add_linear_effect('ethnicity_p', self.det_vars['ethnicity'], current_idx, **self.terms['ethnicity'])
                trans_p_terms.append(sex_t)

            if outcome == ['next_epoch']:
                trans_p = pm.Deterministic('trans_p', self.norm_trans_p(sum(trans_p_terms)))
                obs = pm.Categorical('next_epoch', p=trans_p, observed=data['next_epoch'])
            elif outcome in ['spindle_count','slow_osc_count','rem_event_count']: #FIXME set overlap
                obs = pm.Poisson(outcome, mu=sum(trans_p_terms), observed=data[outcome])
            else:
                errors = pm.HalfNormal('band_errors', mu=0, sd=10*3, shape=len(outcome))
                obs = pm.Normal(outcome, mu=sum(trans_p_terms), sd=errors, observed=data[outcome]) #https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4234023/

        return model

    def add_previous_bout(self, previous_bout, current_idx, flat=False, **kwargs):
        BoundedNormal = pm.Bound(pm.Normal, lower=0, upper=1)
        trans_prev_bout = BoundedNormal('previous_bout_p',
                                       mu=self.get_prior('previous_bout_p', flat),
                                       sd=1e2,
                                       shape=(self.nstages, self.nstages, self.nstages))
        #previous_bout != current_idx by definition
        return trans_prev_bout[previous_bout, current_idx, :]

    def add_clocktime_effects(self, current_idx, clocktime, **kwargs):
        clocktime_phase = pm.VonMises('clocktime_phase_p',
                                      mu=0, kappa=0.01, shape=self.nstages)
        clocktime_effect =  self.add_parameter('clocktime_rate_p', current_idx, **kwargs)*(
                    pm.math.sin(2 * np.pi * clocktime / 24 + clocktime_phase[current_idx]) + 0.5)[:, np.newaxis]
        return clocktime_effect

    def add_baseline(self, current_idx, flat=False):
        BoundedNormal = pm.Bound(pm.Normal, lower=0, upper=1)
        trans_baserate = BoundedNormal('trans_baserate_p',
                                       mu=self.get_prior('trans_baserate_p', flat),
                                       sd=1e2,
                                       shape=(self.nstages, self.nstages))
        return trans_baserate[current_idx, :]


    def add_baseline_fixed(self, current_idx, flat=False):
        trans_baserate = pm.Normal('trans_baserate_p',
                                   mu=self.get_prior('trans_baserate_p', flat),
                                   sd=1e2,
                                   shape=(self.nstages, self.nstages))
        for i in range(self.nstages):
            trans_baserate = tt.set_subtensor(trans_baserate[i,i], 1)
        return trans_baserate[current_idx, :]

    def add_baseline_restricted(self, current_idx, flat=False):
        sd = np.ones((self.nstages, self.nstages))*1e2
        for i in range(self.nstages):
            sd[i,i] = 1e-5
        trans_baserate = pm.Normal('trans_baserate_p',
                                   mu=self.get_prior('trans_baserate_p', flat),
                                   sd=sd,
                                   shape=(self.nstages, self.nstages))
        for i in range(self.nstages):
            trans_baserate = tt.set_subtensor(trans_baserate[i,i], 1)
        return trans_baserate[current_idx, :]


    def add_baseline_nested(self, studyid, current_idx, flat=False): #TODO may need to fix one stage at a specific rate
        BoundedNormal = pm.Bound(pm.Normal, lower=0, upper=1)
        trans_baserate_global = BoundedNormal('trans_baserate_global_p',
                                       mu=self.get_prior('trans_baserate_p', flat),
                                       sd=1e2,
                                       shape=(self.nstages, self.nstages))
        spread_in_baserates = pm.HalfNormal('trans_baserate_sd', sd=1e2)
        trans_baserate = BoundedNormal('trans_baserate_p',
                                       mu=trans_baserate_global[:,:,np.newaxis],
                                       sd=spread_in_baserates,
                                       shape=(self.nstages, self.nstages, self.nstudies))

        return trans_baserate[current_idx, :, studyid]

    def add_linear_effect(self, name, variable, current_idx, **kwargs):
        return variable[:, np.newaxis] * self.add_parameter(name, current_idx, **kwargs)


    def add_quadratic_effect(self, name, variable, current_idx, **kwargs):
        return (variable[:, np.newaxis]**2) * self.add_parameter(name+'^2', current_idx, **kwargs)


    def add_interaction_effect(self, name1, name2, variable1, variable2, current_idx, **kwargs):
        return (variable1*variable2)[:, np.newaxis] * self.add_parameter(name1+'X'+name2, current_idx, **kwargs)


    def add_parameter(self, name, current_idx, stages='all', self_trans_only=True, flat=False, nested=False, studyid=None, **kwargs):
        BoundedNormal = pm.Bound(pm.Normal, lower=0, upper=1)
        n_params = (self.nstages, self.nstages) if not self_trans_only else self.nstages
        if nested and studyid is not None:
            name = name+'_global'
        effect_base = BoundedNormal(name,
                               mu=self.get_prior(name, flat),
                               sd=1e2,
                               shape=n_params)

        if nested and studyid is not None: #TODO test me
            effect_base = BoundedNormal(name.replace('_global',''),
                               mu=effect_base[:, :, np.newaxis],
                               sd=1e2,
                               shape=n_params)
            effect_base = effect_base[:,:,studyid]

        effect = tt.zeros((self.nstages, self.nstages))
        for i in range(self.nstages):
            if stages != 'all':
                idx_mask = [self.maps['current_epoch'][stage] for stage in stages]
                if i not in idx_mask:
                    continue
            if self_trans_only:
                effect = tt.set_subtensor(effect[i, i], effect_base[i])
            else:
                effect = tt.set_subtensor(effect[i, :], effect_base[i, :])
            #TODO set for off_diag only


        return effect[current_idx, :]


    def add_pareto_tau_effect(self, current_idx, tau, stages='all', flat=False):
        BoundedNormal = pm.Bound(pm.Normal, lower=0, upper=1)
        alpha = BoundedNormal('alpha_p',
                              mu=self.get_prior('alpha_p', flat),
                              sd=1e2,
                              shape=(self.nstages, self.nstages))

        if stages == 'all':
            return alpha[current_idx, :] / tau[:, np.newaxis]
        else:
            idx_mask = [self.maps['current_epoch'][stage] for stage in stages]
            mask = tt.zeros((self.nstages, ))
            tt.set_subtensor(mask[idx_mask], 1)
            return mask[current_idx][:, np.newaxis] * (alpha[current_idx, :] / tau[:, np.newaxis])


