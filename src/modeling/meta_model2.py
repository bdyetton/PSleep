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
    def __init__(self, name, terms, maps, outcome='next_epoch'):
        super().__init__()
        self.params = ['alpha', 'baserate', 'clocktime_rate']
        self.nstages = len(maps['current_epoch'])
        self.npreviousstages = len(maps['previous_bout'])
        self.noutcomes = len(maps['next_epoch']) if outcome==['next_epoch'] else len(outcome)
        self.force_all_trans = outcome!=['next_epoch']
        self.terms = terms
        self.outcome = outcome
        tau_prior = np.ones((self.nstages, self.noutcomes)) * 0.01
        trans_prior = np.ones((self.nstages, self.noutcomes)) * 0.01
        if outcome=='next_epoch':
            for s in range(self.nstages):
                tau_prior[s, s] = 0.75
                trans_prior[s, s] = 0.9
        self.priors = {'tau_p': tau_prior,
                       'tau^2_p': 0.1,
                       'tau_par_p': 0.1,
                       'tau_alpha_exp_p':0.1,
                       'tau_lam_exp_p': 0.1,
                       'previous_bout_p': 0.1,
                       'baserate_p': np.ones((1,5))/self.noutcomes,
                       'current_epoch_p': 0,
                       'clocktime_rate_p': [0.1,],
                       'timeslept_rate_p': [0.1,],
                       'timeslept_alpha_exp_p':0.1,
                       'clocktime_alpha_exp_p':0.1,
                       'timeslept_lam_exp_p': 0.1,
                       'clocktime_lam_exp_p': 0.1,
                       'sex_p': 0.1,
                       'age_p': 0.1,
                       'age^2_p': 0.1,
                       'eth_p': 0.1,
                       'ageXsex_p':0.1,
                       'age^2Xsex_p':0.1,
                       'timeslept_p':0.1,
                       'timeslept^2_p':0.1,
                       'clocktime_p': 0.1,
                       'clocktime^2_p':0}
        self.sd_priors = {
            'previous_bout_p': 1,
            'default':1e2,
            'default^2':1,
            'current_epoch_p':0.5
        }
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

        with pm.Model() as model:
            predictor_terms = []  # FIXME

            #%% Set up all determistics so we can sample these values from the trace later
            current_idx = pm.Deterministic('current_epoch', var=self.input_vars['current_epoch'])
            # self.det_vars['current_epoch'] = current_idx

            # add_subid = False
            # for term in self.terms:
            #     term = term.replace('^2','')
            #     if term == 'previous_bout':
            #         previous_bout = pm.Deterministic('previous_bout',
            #                                          var=self.input_vars['previous_bout'])
            #         self.det_vars['previous_bout'] = previous_bout
            #         continue
            #     if 'nested' in self.terms[term] and self.terms[term]['nested']:
            #         add_subid = True
            #     if self.terms[term]['type'] == 'interaction':
            #         for term_ in term.split('X'):
            #             if term_ not in self.det_vars:
            #                 self.det_vars[term_] = pm.Deterministic(term_, var=self.input_vars[term_])
            #     if term in data.columns and term not in self.det_vars:
            #         self.det_vars[term] = pm.Deterministic(term, var=self.input_vars[term])

            # if add_subid:
            #     studyid = pm.Deterministic('studyid', var=self.input_vars['studyid'])
            #     self.det_vars['studyid'] = studyid
            # else:
            studyid = None

            # #%% Add in terms
            for term, term_kwargs in self.terms.items():
                if term == 'baserate':
                    baserate_effect_t = self.add_baserate(studyid, **self.terms['baserate'])
                    predictor_terms.append(baserate_effect_t)
                elif term == 'previous_bout':
                    previous_bout_t = self.add_previous_bout(previous_bout, self.input_vars['consider_previous_bout'], current_idx, **self.terms['previous_bout'])
                    predictor_terms.append(previous_bout_t)
                elif term == 'current_epoch':
                    current_epoch_t = self.add_current_epoch(data['current_epoch'].values, studyid, **self.terms['current_epoch']) #FIXME
                    predictor_terms.append(current_epoch_t)
                elif 'type' in term_kwargs:
                    if term_kwargs['type'] == 'linear':
                        effect_t = self.add_linear_effect(term, self.det_vars[term], current_idx, **term_kwargs)
                        predictor_terms.append(effect_t)
                    elif term_kwargs['type'] == 'quadratic':
                        term = term.replace('^2','')
                        effect_t = self.add_quadratic_effect(term, self.det_vars[term], current_idx, **term_kwargs)
                        predictor_terms.append(effect_t)
                    elif term_kwargs['type'] == 'pareto':
                        effect_t = self.add_pareto_tau_effect(term, self.det_vars[term], current_idx, **term_kwargs)
                        predictor_terms.append(effect_t)
                    elif term_kwargs['type'] == 'sine':
                        effect_t = self.add_sine_effects(term, self.det_vars[term], current_idx, **term_kwargs)
                        predictor_terms.append(effect_t)
                    elif term_kwargs['type'] == 'exp':
                        effect_t = self.add_exp_effect(term, self.det_vars[term], current_idx, **term_kwargs)
                        predictor_terms.append(effect_t)
                    elif term_kwargs['type'] == 'interaction':
                        term_chunks = term.split('X')
                        effect_t = self.add_interaction_effect(term, self.det_vars[term_chunks[0]],
                                                               self.det_vars[term_chunks[1]],
                                                               current_idx, **term_kwargs)
                        predictor_terms.append(effect_t)
                else:
                    effect_t = self.add_linear_effect(term+'_p', self.det_vars[term], current_idx, **term_kwargs)
                    predictor_terms.append(effect_t)

            #%% Deal with outcome
            if self.outcome == ['next_epoch']:
                print('Running next epoch model')
                # baserate_t = pm.Normal('baserate_p',
                #                             mu=1 / 5 * np.ones((1, 5)),
                #                             sd=1,
                #                             shape=(1, 5))

                trans_p = pm.Deterministic('trans_p', tt.nnet.softmax(sum(predictor_terms))) #FIXME
                obs = pm.Categorical('next_epoch', p=trans_p[current_idx], observed=data['next_epoch']) #FIXME

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
                      self_trans_only=True,
                      flat=False,
                      nested=False,
                      prior_dist='normal',
                      **kwargs):
        self_trans_only = False if self.force_all_trans else self_trans_only
        n_params = (self.nstages, self.noutcomes) if not self_trans_only else self.nstages

        if nested:
            dist_name = name+'_global_p'
            n_params = (1, self.nstages, self.noutcomes)
        else:
            dist_name = name+'_p'
        effect_base = self.dist_by_name(prior_dist)(dist_name,
                                              mu=self.get_prior(name+'_p', flat),
                                              sd=self.get_sd_prior(name+'_p'),
                                              shape=n_params)
        # return effect
        # if self.outcome==['next_epoch'] and name == 'baserate':
        #     for i in range(self.nstages):
        #         if nested:
        #             effect_base = tt.set_subtensor(effect_base[0, i, i], 1) #0 for broadcastable study dim
        #         else:
        #             effect_base = tt.set_subtensor(effect_base[i, i], 1)

        if nested:
            print(name, 'is nested with non-centered parameterization')
            spread_in_baserates = pm.HalfCauchy(name+'_spread_p', 5, shape=(1, self.nstages, self.noutcomes))
            study_offsets = pm.Normal(name+'_study_offsets_p', mu=0, sd=0.1, shape=(self.nstudies, self.nstages, self.noutcomes))
            effect_base = pm.Deterministic(name+'_p', effect_base + spread_in_baserates * study_offsets)
            effect = tt.zeros((self.nstudies, self.nstages, self.noutcomes))
        else:
            effect = tt.zeros((self.nstages, self.noutcomes))


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
                effect = tt.set_subtensor(effect[i, i], effect_base[i])
            else:
                if not nested:
                    effect = tt.set_subtensor(effect[i], effect_base[i])
                else:
                    effect = tt.set_subtensor(effect[:, i], effect_base[:, i])
        #     #TODO set for off_diag only?
        return effect

    def add_previous_bout(self, previous_bout, consider_previous_bout, current_idx, flat=False, **kwargs):
        trans_prev_bout = pm.Normal('previous_bout_p',
                                    mu=self.get_prior('previous_bout_p', flat),
                                    sd=self.get_sd_prior('previous_bout_p'),
                                    shape=(self.npreviousstages, self.nstages, self.noutcomes))
        # for i in range(self.nstages):
        #     for j in range(self.noutcomes):
        #         trans_prev_bout = tt.set_subtensor(trans_prev_bout[:, i, j], 0)
        return consider_previous_bout*ravel_3_to_2_index(trans_prev_bout, previous_bout, current_idx)  #previous_bout != current_idx by definition

    def add_sine_effects(self, name, variable, current_idx, **kwargs):
        sine_phase = pm.VonMises(name+'_phase_p',
                                      mu=0, kappa=0.01, shape=self.nstages)
        sine_effect =  self.add_parameter(name+'_rate', **kwargs)[current_idx]*\
                    pm.math.sin(2 * np.pi * variable / 24 + sine_phase[current_idx])[:, np.newaxis]
        return sine_effect

    # def add_baserate(self, current_idx, studyid=None, nested=False, decomposed=False, **kwargs):
    #
    #     baserate = self.add_parameter('baserate', stages='all', nested=nested, self_trans_only=False, studyid=studyid, **kwargs)
    #
    #     if not nested:
    #         if self.outcome == ['next_epoch']:
    #             for i in range(self.nstages):
    #                 baserate = tt.set_subtensor(baserate[i,i], 0.5)
    #         return baserate[current_idx] #TODO could try a softmax here?
    #     else:
    #         if self.outcome == ['next_epoch']:
    #             for i in range(self.nstages):
    #                 baserate = tt.set_subtensor(baserate[:,i,i], 0.5)
    #         return ravel_3_to_2_index(baserate, studyid, current_idx)

    def add_current_epoch(self, current_idx, studyid=None, nested=False, **kwargs):
        return self.add_parameter('current_epoch', nested=nested, self_trans_only=False, **kwargs)
        #retur
        if nested:
            return ravel_3_to_2_index(baserate, studyid, current_idx)
        else:
            return baserate
            #return baserate[current_idx]


    def add_baserate(self, studyid=None, nested=False, **kwargs):
        baserate_global = pm.Normal('baserate_global_p' if nested else 'baserate_p',
                                    mu=(1 / self.noutcomes)*np.ones((1, self.noutcomes)),
                                    sd=1,
                                    shape=(1, self.noutcomes))
        if nested:
            spread_in_baserates = pm.HalfCauchy('baserate_spread_p', 5, shape=(1, self.noutcomes))
            study_offsets = pm.Normal('baserate_study_offsets_p', mu=0, sd=0.1, shape=(self.nstudies, self.noutcomes))
            baserate_p = pm.Deterministic('baserate_p', baserate_global + spread_in_baserates * study_offsets)
            return baserate_p[studyid]
        else:
            return baserate_global

    def add_linear_effect(self, name, variable, current_idx, **kwargs):
        return variable[:, np.newaxis] * self.add_parameter(name, **kwargs)[current_idx,:]

    def add_exp_effect(self, name, variable, current_idx, **kwargs):
        alpha = self.add_parameter(name + '_alpha_exp', **kwargs)
        lam = self.add_parameter(name + '_lam_exp', **kwargs)
        return alpha[current_idx] * pm.math.exp(variable[:, np.newaxis] * lam[current_idx])

    def add_quadratic_effect(self, name, variable, current_idx, **kwargs):
        return (variable[:, np.newaxis]**2) * self.add_parameter(name+'^2', **kwargs)[current_idx]


    def add_interaction_effect(self, name, variable1, variable2, current_idx, **kwargs):
        return (variable1*variable2)[:, np.newaxis] * self.add_parameter(name,  **kwargs)[current_idx]


    def add_pareto_tau_effect(self, name, variable, current_idx, stages='all', **kwargs):
        alpha = self.add_parameter(name+'_par', stages=stages, **kwargs)
        return alpha[current_idx] / variable[:, np.newaxis]

def ravel_3_to_2_index(a,idx1,idx2):
    a_shape = tt.shape(a)
    a_reshaped = tt.reshape(a, (a_shape[0]*a_shape[1],a_shape[2]))
    return a_reshaped[idx1*a_shape[1]+idx2]

def add_min(a):
    return a - tt.min(a, axis=1, keepdims=True)

def add_min_norm(a):
    non_zero = a - tt.min(a, axis=1, keepdims=True)
    return non_zero/tt.sum(non_zero, axis=1, keepdims=True)



