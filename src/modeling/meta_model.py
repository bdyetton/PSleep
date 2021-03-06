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
    def __init__(self, name, terms, maps, priors=None, sd_priors=None, outcome='next_epoch'):
        super().__init__()
        self.params = ['alpha', 'baserate', 'clocktime_rate']
        self.nstages = len(maps['current_epoch'])
        self.npreviousstages = len(maps['previous_bout'])
        self.noutcomes = len(maps['next_epoch']) if outcome==['next_epoch'] else len(outcome)
        self.terms = terms
        self.outcome = outcome
        tau_prior = np.ones((self.nstages, self.noutcomes)) * 0.01
        trans_prior = np.ones((self.nstages, self.noutcomes)) * 0.01
        if outcome=='next_epoch':
            for s in range(self.nstages):
                tau_prior[s, s] = 0.75
                trans_prior[s, s] = 0.9
        self.priors = {'tau_p': 0,
                       'baserate_p': np.ones((1,5))/self.noutcomes,
                       'clocktime_rate_p': [0,],
                       'timeslept_rate_p': [0,],
                       'default':0}
        if priors:
            self.priors.update(priors)
        self.sd_priors = {
            'previous_bout_p': 0.5,
            'baserate_p': 1,
            'default':0.5,
            'default^2':0.5,
            'current_epoch_p':0.5,
        }
        if sd_priors:
            self.sd_priors.update(sd_priors)
        self.flat_priors = {}
        self.name = name
        self.maps = maps
        self.nstudies = len(maps['studyid'])
        self.det_vars = {}
        self.nu_ticker = 0

    def init_model(self, data, flat=False):
        assert np.all(data['current_epoch'].isin(np.arange(0, self.nstages)))
        if not isinstance(self.outcome, list):
            self.outcome = [self.outcome]
        cols = set()
        for col in data.columns:
            if any([col in c for c in list(self.terms.keys()) + ['current_epoch', 'studyid']+self.outcome]):
                cols.add(col)
        dat_before = data.shape[0]
        data = data.loc[:, cols].dropna()
        print('Fitting on', data.shape[0],'records (dropped', dat_before-data.shape[0],')')


        self.input_vars['current_epoch'] = shared(data['current_epoch'].values)
        self.input_vars['studyid'] = shared(data['studyid'].values)
        for term in self.terms:
            if self.terms[term]['type'] == 'interaction':
                for term_ in term.split('X'):
                    if (term_ not in self.input_vars) and (term_ in data.columns):
                            self.input_vars[term_] = shared(data[term_].values)
            elif term in data.columns:
                self.input_vars[term] = shared(data[term].values)
        self.output_vars += self.outcome
        self.input_vars['consider_previous_bout'] = shared(1)
        self.input_vars['consider_current_epoch'] = shared(1)

        with pm.Model() as model:
            predictor_terms = []  # FIXME

            #%% Set up all determistics so we can sample these values from the trace later
            current_idx = pm.Deterministic('current_epoch', var=self.input_vars['current_epoch'])
            self.det_vars['current_epoch'] = current_idx

            add_subid = False
            for term in self.terms:
                term = term.replace('^2','')
                if term == 'previous_bout':
                    previous_bout = pm.Deterministic('previous_bout',
                                                     var=self.input_vars['previous_bout'])
                    self.det_vars['previous_bout'] = previous_bout
                    continue
                if 'nested' in self.terms[term] and self.terms[term]['nested']:
                    add_subid = True
                if self.terms[term]['type'] == 'interaction':
                    for term_ in term.split('X'):
                        if term_ not in self.det_vars:
                            self.det_vars[term_] = pm.Deterministic(term_, var=self.input_vars[term_])
                if term in data.columns and term not in self.det_vars:
                    self.det_vars[term] = pm.Deterministic(term, var=self.input_vars[term])

            if add_subid:
                studyid = pm.Deterministic('studyid', var=self.input_vars['studyid'])
                self.det_vars['studyid'] = studyid
            else:
                studyid = 0

            # #%% Add in terms
            for term, term_kwargs in self.terms.items():
                if term == 'baserate':
                    baserate_effect_t = self.add_baserate(studyid, **self.terms['baserate'])
                    predictor_terms.append(baserate_effect_t)
                elif term == 'previous_bout':
                    previous_bout_t = self.add_previous_bout(self.input_vars['consider_previous_bout'], previous_bout, current_idx, **self.terms['previous_bout'])
                    predictor_terms.append(previous_bout_t)
                elif term == 'current_epoch':
                    current_epoch_t = self.add_current_epoch(self.input_vars['consider_current_epoch'], studyid, current_idx, **self.terms['current_epoch']) #FIXME
                    predictor_terms.append(current_epoch_t)
                elif 'type' in term_kwargs:
                    if term_kwargs['type'] == 'linear':
                        effect_t = self.add_linear_effect(term, self.det_vars[term], studyid, current_idx, **term_kwargs)
                        predictor_terms.append(effect_t)
                    elif term_kwargs['type'] == 'quadratic':
                        term = term.replace('^2','')
                        effect_t = self.add_quadratic_effect(term, self.det_vars[term], studyid, current_idx, **term_kwargs)
                        predictor_terms.append(effect_t)
                    elif term_kwargs['type'] == 'pareto':
                        effect_t = self.add_pareto_tau_effect(term, self.det_vars[term], studyid, current_idx, **term_kwargs)
                        predictor_terms.append(effect_t)
                    elif term_kwargs['type'] == 'sine':
                        effect_t = self.add_sine_effects(term, self.det_vars[term], studyid, current_idx, **term_kwargs)
                        predictor_terms.append(effect_t)
                    elif term_kwargs['type'] == 'exp':
                        effect_t = self.add_exp_effect(term, self.det_vars[term], studyid, current_idx, **term_kwargs)
                        predictor_terms.append(effect_t)
                    elif term_kwargs['type'] == 'interaction':
                        term_chunks = term.split('X')
                        effect_t = self.add_interaction_effect(term, self.det_vars[term_chunks[0]],
                                                               self.det_vars[term_chunks[1]], studyid,
                                                               current_idx, **term_kwargs)
                        predictor_terms.append(effect_t)
                else:
                    effect_t = self.add_linear_effect(term+'_p', self.det_vars[term], studyid, current_idx, **term_kwargs)
                    predictor_terms.append(effect_t)

            #%% Deal with outcome
            if self.outcome == ['next_epoch']:
                print('Running next epoch model')
                trans_p = pm.Deterministic('trans_p', tt.nnet.softmax(sum(predictor_terms)))
                obs = pm.Categorical('next_epoch', p=trans_p, observed=data['next_epoch'])

            elif any([v in ['spindle_count','slow_osc_count','rem_event_count'] for v in self.outcome]): #FIXME set overlap
                if 'correl' not in self.terms['baserate']:
                    print('Running sleep feature model (no correl)')
                    feature_rate = pm.Deterministic('feature_rate_p', tt.nnet.softplus(sum(predictor_terms)))
                else:
                    print('Running sleep feature model (with correl)')
                    sd_dist = pm.HalfNormal.dist(sd=1e2, shape=self.noutcomes)
                    chol_packed = pm.LKJCholeskyCov('feature_chol',
                                                    n=self.noutcomes, eta=5, sd_dist=sd_dist)
                    chol = pm.expand_packed_triangular(self.noutcomes, chol_packed)
                    covar = pm.Deterministic('feature_covar', tt.dot(chol,tt.transpose(chol)))
                    diag_covar = tt.sqrt(tt.diag(covar))
                    per = pm.Deterministic('feature_person', tt.dot(diag_covar,tt.dot(covar, diag_covar)))
                    feature_rate = pm.Deterministic('feature_rate_p', tt.nnet.softplus(tt.dot(chol, sum(predictor_terms).T).T))
                if self.outcome == ['rem_event_count']:
                    obs = pm.Exponential('_'.join(self.outcome), lam=1/feature_rate, observed=data[self.outcome])
                else:
                    obs = pm.Poisson('_'.join(self.outcome), mu=feature_rate, observed=data[self.outcome])

            else:
                print('Running band model')
                if 'correl' not in self.terms['baserate']:
                    sd_dist = pm.HalfNormal('band_power_sd', sd=1e2, shape=self.noutcomes)
                    band_power = pm.Deterministic('band_power_p', sum(predictor_terms))
                    obs = pm.Normal('_'.join(self.outcome), mu=band_power, sd=sd_dist, observed=data[self.outcome]) #https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4234023/
                else:
                    sd_dist = pm.HalfNormal.dist(sd=1e4, shape=self.noutcomes)
                    chol_packed = pm.LKJCholeskyCov('band_chol',
                                                    n=self.noutcomes, eta=5, sd_dist=sd_dist)
                    chol = pm.expand_packed_triangular(self.noutcomes, chol_packed)
                    covar = pm.Deterministic('feature_covar', tt.dot(chol, tt.transpose(chol)))
                    diag_covar = tt.sqrt(tt.diag(covar))
                    per = pm.Deterministic('feature_person', tt.dot(diag_covar, tt.dot(covar, diag_covar)))
                    band_power = pm.Deterministic('feature_rate_p', tt.dot(chol, sum(predictor_terms).T).T)
                    sd = pm.HalfNormal('feature_rate_shared_sd', sd=1e4)
                    obs = pm.Normal('_'.join(self.outcome), mu=band_power, sd=sd, observed=data[self.outcome]) #https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4234023/
                    # band_power = pm.Deterministic('band_power_p', sum(predictor_terms))
                    #obs = pm.MvNormal('_'.join(self.outcome), mu=band_power, chol=chol, observed=data[self.outcome]) #https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4234023/
        return model

    def dist_by_name(self, prior_dist='normal'):
        if prior_dist == 'normal':
            return pm.Normal
        elif prior_dist == 'lower_normal':
            return pm.Bound(pm.Normal, lower=0)
        elif prior_dist == 'gamma':
            return pm.Gamma
        elif prior_dist == 't':
            nu = pm.Exponential('nu'+str(self.nu_ticker), 1. / 10, testval=5.)
            self.nu_ticker += 1
            return lambda name, mu, sd, shape: pm.StudentT(name, mu=mu, sd=sd, nu=nu, shape=shape)

    def add_parameter(self, name,
                      stages='all',
                      self_trans_only=False,
                      baserate=False,
                      flat=False,
                      nested=False,
                      studyid=0,
                      stage_idx=0,
                      prior_dist='normal',
                      **kwargs):

        studyid = studyid if nested else 0
        stage_idx = 0 if baserate else stage_idx

        nstudies = self.nstudies
        nstages = self.nstages
        nglobal_stages = self.nstages
        if self_trans_only:
            nstages = self.nstages
            nglobal_stages = self.nstages
        if baserate:
            nstages = 1
            nglobal_stages = 1
        if not nested or nested is None:
            nstudies = 1

        n_global_params = (1, nglobal_stages, self.noutcomes)
        n_params = (nstudies, nstages, self.noutcomes)

        dist_name = name+'_global_p' if nested else name + '_p'

        effect_base = self.dist_by_name(prior_dist)(dist_name,
                                                    mu=self.get_prior(name+'_p', flat),
                                                    sd=self.get_sd_prior(name+'_p'),
                                                    shape=n_global_params)
        if nested:
            spread_in_baserates = pm.HalfNormal(name+'_spread_p', 5, shape=n_global_params)
            study_offsets = pm.Normal(name+'_study_offsets_p', mu=0, sd=1, shape=n_params)
            effect_base = pm.Deterministic(name+'_p', effect_base + spread_in_baserates * study_offsets)
            effect = tt.zeros(n_params)
        else:
            effect = tt.zeros(n_params)

        if stages == 'all':
            ravel_3_to_2_index(effect_base, studyid, stage_idx)

        idx_map = {v:k for k,v in self.maps['current_epoch'].items()}
        for i in range(self.nstages):
            if stages != 'all':
                idx_mask = [self.maps['current_epoch'][stage] for stage in stages]
                if i not in idx_mask:
                    print(name, idx_map[i], 'has no effect')
                    continue
            if self_trans_only and self.outcome==['next_epoch']:
                assert not nested, "self_trans + nesting not implemented"
                print(name,idx_map[i],'is self trans only')
                effect = tt.set_subtensor(effect[0, i, i], effect_base[0, 0, i])
            else:
                if baserate:
                    effect = tt.set_subtensor(effect[:, :, i], effect_base[:, :, i])
                else:
                    effect = tt.set_subtensor(effect[:, i], effect_base[:, i])
        return ravel_3_to_2_index(effect, studyid, stage_idx)

    def add_previous_bout(self, consider_previous_bout, previous_bout, current_index, flat=False, **kwargs):
        trans_prev_bout = pm.Normal('previous_bout_p',
                                    mu=self.get_prior('previous_bout_p', flat),
                                    sd=self.get_sd_prior('previous_bout_p'),
                                    shape=(self.npreviousstages, self.nstages, self.noutcomes))
        return consider_previous_bout*ravel_3_to_2_index(trans_prev_bout, previous_bout, current_index)  #previous_bout != current_idx by definition

    def add_sine_effects(self, name, variable, studyid, current_index, flat=False, **kwargs):
        sine_phase = pm.Normal(name+'_phase_p',
                               transform=pm.distributions.transforms.Circular(),
                               mu=self.get_prior(name+'_phase_p', flat), sd=self.get_sd_prior(name+'_kappa_p'), shape=(1, self.noutcomes)) #TODO test noutcomes vs nstages
        sine_rate = self.add_parameter(name +'_rate', studyid=studyid, stage_idx=current_index, **kwargs)
        sine_effect =  sine_rate*pm.math.sin(2 * np.pi * variable[:, np.newaxis] / 24 + sine_phase)
        return sine_effect

    def add_current_epoch(self, consider_current_epoch, studyid, current_index, **kwargs):
        return consider_current_epoch*self.add_parameter('current_epoch', self_trans_only=False, studyid=studyid, stage_idx=current_index, **kwargs)

    def add_baserate(self, studyid, **kwargs):
        return self.add_parameter('baserate', baserate=True, studyid=studyid, **kwargs)

    def add_linear_effect(self, name, variable, studyid, current_index, **kwargs):
        return variable[:, np.newaxis] * self.add_parameter(name, studyid=studyid, stage_idx=current_index, **kwargs)

    def add_exp_effect(self, name, variable, studyid, current_index, **kwargs):
        alpha = self.add_parameter(name + '_alpha_exp', studyid=studyid, stage_idx=current_index, **kwargs)
        lam = self.add_parameter(name + '_lam_exp', studyid=studyid, stage_idx=current_index, **kwargs)
        return alpha * pm.math.exp(variable[:, np.newaxis] * lam)

    def add_quadratic_effect(self, name, variable, studyid, current_index, **kwargs):
        return (variable[:, np.newaxis]**2) * self.add_parameter(name +'^2', studyid=studyid, stage_idx=current_index, **kwargs)

    def add_interaction_effect(self, name, variable1, variable2, studyid, current_index, **kwargs):
        return (variable1*variable2)[:, np.newaxis] * self.add_parameter(name, studyid=studyid, stage_idx=current_index, **kwargs)

    def add_pareto_tau_effect(self, name, variable, studyid, current_index, **kwargs):
        alpha = self.add_parameter(name +'_par', studyid=studyid, stage_idx=current_index, **kwargs)
        return alpha / variable[:, np.newaxis]

def ravel_3_to_2_index(a,idx1,idx2):
    a_shape = tt.shape(a)
    a_reshaped = tt.reshape(a, (a_shape[0]*a_shape[1],a_shape[2]))
    return a_reshaped[idx1*a_shape[1]+idx2]

# def add_min(a):
#     return a - tt.min(a, axis=1, keepdims=True)
#
# def add_min_norm(a):
#     non_zero = a - tt.min(a, axis=1, keepdims=True)
#     return non_zero/tt.sum(non_zero, axis=1, keepdims=True)



