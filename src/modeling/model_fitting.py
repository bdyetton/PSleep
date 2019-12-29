import pymc3 as pm
import pandas as pd
pd.options.display.max_colwidth = 150
import numpy as np
import matplotlib.pyplot as plt
from src.modeling import base_model, meta_model
import pickle
import glob
import os
import warnings


class ModelFitter:
    def __init__(self, outcome, data, maps, priors=None, sd_priors=None, inital_set=[{'baserate': {'type': 'fixed'}}]):
        #self.name = name
        self.data = data
        self.maps = maps
        self.outcome = outcome
        self.sd_priors = sd_priors
        self.priors = priors
        self.term_sets = []
        self.models_run = []
        if inital_set is not None:
            self.term_sets.append(inital_set)

    def add_term_set(self, term_set):
        self.term_sets.append(term_set)

    def add_regular_set(self, replace=None, remove_pareto=False):
        if remove_pareto:
            tau = [{'tau': {'type': 'linear'}}, {'tau': {'type': 'linear'}, 'tau^2': {'type': 'quadratic'}},
             {'tau': {'type': 'exp'}}]
        else:
            tau = [{'tau': {'type': 'linear'}}, {'tau': {'type': 'linear'}, 'tau^2': {'type': 'quadratic'}},
             {'tau': {'type': 'pareto'}}, {'tau': {'type': 'exp'}}]
        reg_set = [
            [{'previous_bout':{'type':''}}],
            tau,
            [{'clocktime':{'type':'linear'}}, {'clocktime':{'type':'linear'}, 'clocktime^2':{'type':'quadratic'}}, {'clocktime':{'type':'sine'}}, {'clocktime':{'type':'exp'}}],
            #[{'timeslept':{'type':'linear'}}, {'timeslept':{'type':'linear'}, 'timeslept^2':{'type':'quadratic'}}, {'timeslept':{'type':'sine'}}, {'timeslept':{'type':'exp'}}],
            [{'age':{'type':'linear'}}, {'age':{'type':'linear'}, 'age^2':{'type':'quadratic'}}],
            [{'sex':{'type':'linear'}}],
            [{'ageXsex':{'type':'interaction'},'age':{'type':'linear'}, 'sex':{'type':'linear'}},
             {'ageXsex': {'type': 'interaction'},
              'age': {'type': 'linear'}, 'age^2': {'type': 'quadratic'} ,'sex': {'type': 'linear'}},
             ],
                  ]
        if replace is not None:
            for i, terms in replace.items():
                reg_set[i] = terms
        self.term_sets += reg_set

    def make_name_from_terms(self, terms):
        name = '-'.join(self.outcome) + '_'
        for term_name, term_kwargs in terms.items():
            if term_kwargs['type'] not in ['fixed','']:
                type_str = '-'+ term_kwargs['type'][0]
            else:
                type_str = ''
            if len(term_kwargs) > 1:
                other_map = {'self_trans_only':'sto'}
                others = '-'+'-'.join([other_map[k] if k in other_map else k for k,v in term_kwargs.items() if k != 'type' and v])
            else:
                others = ''
            if others=='-':
                others = ''
            name += term_name + type_str + others + '_'
        return name

    def run_fitter(self, force_refresh=False):
        terms_to_try = {}
        last_best_name = None
        for i, term_set in enumerate(self.term_sets):
            for terms in term_set:
                new_terms_to_try = terms_to_try[last_best_name].copy() if i!=0 else {}
                new_terms_to_try.update(terms)
                last_added_name = self.make_name_from_terms(new_terms_to_try)
                terms_to_try[last_added_name] = new_terms_to_try
                self.models_run.append(last_added_name)
            mods_to_compare = self.run_meta_models(terms_to_try, force_refresh)
            mod_df = compare_models(mods_to_compare)
            last_best_name = mod_df.index[0]
            for mod_name in mod_df.index[1:]:
                terms_to_try.pop(mod_name) #pop everything but the best
            print('best model this set is', last_best_name)
        print('best model overall is', last_best_name)
        return self.models_run

    def compare_all(self, return_best=False):
        mods = load_models(self.data.name, self.models_run)
        best_mod = compare_models(mods, return_best=return_best)
        if return_best:
            return load_models(self.data.name, best_mod)
        else:
            return best_mod

    def get_models(self, model_name=None):
        if model_name is not None:
            return load_models(self.data.name, mod_names=model_name)[0]
        else: #return all run models
            return load_models(self.data.name, mod_names=self.models_run)

    def run_meta_models(self, terms_to_try, force_refit=False):
        models_to_compare = []
        for name, terms in terms_to_try.items():
            refit = force_refit
            if not refit:
                try:
                    mod = base_model.load_model(self.data.name, name)
                    if hasattr(mod, 'data_name') and self.data.name != mod.data_name:
                        raise FileNotFoundError
                    print('Loaded', name)
                except FileNotFoundError:
                    refit = True
            if refit:
                print('Fitting', name)
                mod = meta_model.MetaModel(name=name, terms=terms, priors=self.priors, sd_priors=self.sd_priors, maps=self.maps, outcome=self.outcome)
                mod.fit(self.data)
                #mod.traceplot()
                mod.summary()
                #plt.show()
                print(pm.loo(mod.trace, mod.model))
                mod.save_model(data_name=self.data.name, model_name=name)
            models_to_compare.append(mod)
        return models_to_compare

    def load_from_folder(self, data_name):
        mod_names = glob.glob('../../data/models/' + data_name + '/*.pkl')
        self.models_run = [os.path.split(name)[1] for name in mod_names if 'data_n' not in os.path.split(name)[1] and 'sampled_' not in os.path.split(name)[1]]

def compare_models(models, printing=True, return_best=False):
    print('Comparing',len(models),'models')
    model_comp_dict = {m.model: m.trace for m in models}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        compare_df = pm.compare(model_comp_dict, ic='LOO')  # ic=WAIC
    compare_df.index = [idx.name for idx in compare_df.index]
    if printing:
        print_compare_df = compare_df.copy()
        idxs = list(print_compare_df.index)
        common_stuff =''
        break_parent = False
        for i, s in enumerate(idxs[0]):
            for other in idxs[1:]:
                if i>=len(other) or s!=other[i]:
                    break_parent=True
                    break
            if break_parent:
                break
            common_stuff += s
        print('--------------', common_stuff, '---------------')
        idxs = [idx.replace(common_stuff,'') for idx in idxs]
        print_compare_df.index = idxs
        print(print_compare_df.drop(['rank','p_loo','d_loo','se','dse','loo_scale','warning'], axis=1))
    if return_best:
        return compare_df.index[0]
    else:
        return compare_df

def load_models(data_name, mod_names=None):
    models_to_compare = []
    if mod_names is None:
        mod_names = glob.glob('../../data/models/' + data_name + '/*.pkl')
        mod_names = [name for name in mod_names if 'data_n' not in os.path.split(name)[1]]
        for mod_name in mod_names:
            loaded_mod = pickle.load(open(mod_name, 'rb'))
            if isinstance(loaded_mod, meta_model.MetaModel):
                if loaded_mod.name != mod_name:
                    loaded_mod.name = os.path.split(mod_name)[1].replace('.pkl','')
                    loaded_mod.model.name = os.path.split(mod_name)[1].replace('.pkl','')
                models_to_compare.append(loaded_mod)
    else:
        if isinstance(mod_names, str):
            mod_names = [mod_names]
        for name in mod_names:
            print('loading', name)
            mod = base_model.load_model(data_name=data_name, model_name=name)
            if mod.name != name:
                mod.name = name
            #if mod.data_name != data_name:
            #    print('Model loaded does not match data! Skipping load.')
            #    continue
            models_to_compare.append(mod)
    if len(models_to_compare) == 1:
        return models_to_compare[0]
    return models_to_compare