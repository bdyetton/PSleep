import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pymc3 as pm
import pandas as pd
import sys
import numpy as np
from src.modeling import param_sample_tools
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_palette(sns.color_palette("Set1", n_colors=8, desat=.5))
import dill as pickle
import os


class BaseModel():
    def __init__(self):
        """Initalize basemodel - these are all placeholder variables"""
        self.params = None
        self.priors = None
        self.sd_priors = None
        self.flat_priors = None
        self.prior_model = None
        self.model = None
        self.name = None
        self.trace = None
        self.input_vars = {}
        self.output_vars = []
        self.det_vars = {}
        self.data_name = ''

    def simulate_and_recover(self, true_params: dict, nsamples: int=1000):
        """
        Simulate some data based on some randomly selected parameters, then the same model back on these parameters
        to see if we can recover our parameters back
        :param nsamples: how many samples to take
        :return: a dataframe with a col for each param, its recovery probability, and number of samples in that run
        """

        model = self.init_prior_model(true_params) #initalize model with these parameters
        with model:
            trace_sim = pm.sample_prior_predictive(samples=nsamples) #simulate some data
        model = self.init_model(trace_sim, flat=True) #initalize our real model with this simulated data
        with model: #fit the real model
            trace_fit = pm.sample()
        if any([v > 1.05 for k,v in pm.diagnostics.gelman_rubin(trace_fit).items()]):
            return None # did not converge
        params_to_validate = {}
        for param in self.params: #get a param:true_value map for each param we care about
            params_to_validate[param] = true_params[param]
        param_recovery = self.validate_params(params_to_validate, trace_fit) #get an estimate of parameter recovery
        param_recovery['number of epochs'] = nsamples #add the number of samples to df samples
        return param_recovery

    def plot_param_recovery_curve(self):
        """Plots a curve of parameter recovery probability vs samples.
        This gives a rough estimate of the sample size required to fit this model or if there are issues
        with recovering parameters due to a possible underspecified (not enough data) or overspecified (multiple correct choices) model"""
        param_cont = []
        for run in range(4):  # sample  random param values for each sample size
            true_params = self.sample_rand_params()  #get some random parameter values
            for nsamples in np.logspace(2, 5, num=4): #logspace for number of samples req to get a general feel
                print('Simulate and recover for', int(nsamples),'samples. Run (of 5):',run+1)
                param_recovery = self.simulate_and_recover(true_params, int(nsamples))
                if param_recovery is None:
                    continue
                param_cont.append(param_recovery)

        param_recovery_df = pd.concat(param_cont, axis=0)
        for param in self.params: #do actual plotting
            trans_recover_df = param_recovery_df.loc[param_recovery_df['parameter'] == param, :]
            grid = sns.lineplot(data=trans_recover_df, x='number of epochs', y='distance')
            grid.set(xscale="log")
            plt.title('Distance between true and recovered parameter '+param)
            plt.show()

    @staticmethod
    def validate_params(true_params, trace):
        true_param_cont = []
        for true_param, true_param_value in true_params.items():
            recovery_cont = []
            for idx, val in np.ndenumerate(true_param_value): #iterate through possibly multi-dimentional parameter
                idx_multi = [slice(None, None)] + list(idx) #create index to get all samples for that parameter (all of trace dim)
                samples = trace[true_param][idx_multi]

                sns.distplot(samples)
                y_lims = plt.ylim()
                plt.axvline(val, y_lims[0], y_lims[1])
                plt.axvline(np.mean(samples),y_lims[0], y_lims[1], color='r')
                plt.show()
                d_stat = abs(val-np.mean(samples))
                # This is broken right now, but it worth considering for the future
                # Do kernel density estimation
                # bkde = gaussian_kde(samples.flatten())
                # # Get probability for range of values
                # start = val - p_width  # Start of the range. FIXME these values should be more theory driven...
                # end = val + p_width  # End of the range
                # if end > max(samples):
                #     end = max(samples)
                # if start < min(samples):
                #     start=min(samples) #bounded distributions cause issues where most points will eval to 0 prob
                # N = 200  # Number of evaluation points
                # print(start,end)
                # print(min(samples), max(samples))
                # step = (end - start) / (N - 1)  # Step size
                # x = np.linspace(start, end, N)[:, np.newaxis]  # Generate values in the range
                # kd_vals = np.array([kde.evaluate(x_) for x_ in x])  # Get PDF values for each x
                # probability = np.sum(kd_vals * step)  # Approximate the integral of the PDF
                recovery_cont.append(d_stat)
            true_param_cont.append(pd.DataFrame({'parameter': [true_param] * len(recovery_cont),
                                                 'recovery': recovery_cont}))
        return pd.concat(true_param_cont, axis=0)

    def fit(self, data):
        self.data_name = data.name
        self.model = self.init_model(data)
        self.model.name = self.name
        with self.model:
            #try:
            #    self.trace = pm.sample(init='ADVI', target_accept=0.90, cores=2, max_treedepth=10)
            #except FloatingPointError:
            self.trace = pm.sample(target_accept=0.90, cores=2, max_treedepth=10, init='adapt_diag')
        return self.model, self.trace

    def get_prior(self, name, flat):
        return self.flat_priors[name] if flat else self.priors[name]

    def get_sd_prior(self, name):
        return self.sd_priors[name]

    def save_model(self, data_name, model_name=None, save_dets=False):
        model_name = self.name if model_name is None else model_name
        save_path = '../../data/models/'+data_name+'/' + model_name + '.pkl'
        if not save_dets:
            for var in self.det_vars:
                self.trace.remove_values(var)
        pickle.dump(self, open(save_path, 'wb'))
        print('Model "' + model_name + '" saved with filesize:', round(os.stat(save_path).st_size / (1024.0 ** 3), 3), 'GiB')
        return save_path

    def init_model(self, data, flat=False):
        pass

    @staticmethod
    def init_prior_model(fixed_params):
        pass

    @staticmethod
    def sample_rand_params():
        pass

    @staticmethod
    def trace_to_data():
        pass

    def sample_posterior_predictive(self, num_draws_from_params=100, vars=None, out_of_sample=False):
        if out_of_sample:
            for v in self.model.named_vars:
                if 'study_offset' in v:
                    self.trace.remove_values(v)
                    assert v not in self.trace
        if vars is None:
            return pm.sample_posterior_predictive(model=self.model,
                                                  trace = self.trace,
                                                  samples=num_draws_from_params)
        else:
            return pm.sample_posterior_predictive(model=self.model,
                                                  trace = self.trace,
                                                  samples=num_draws_from_params,
                                                  vars=[self.model.named_vars[v] for v in vars])


    def run_ppcs(self, data):
        # for var_name in self.input_vars:
        #     self.input_vars[var_name].set_value(data[var_name])
        pps = self.sample_posterior_predictive()
        for var_i, var in enumerate(self.output_vars):
            plt.figure()
            if isinstance(data[var].iloc[0], (int, np.integer)):
                fit_counts = pd.concat([pd.Series(draw).value_counts() for draw in pps[var]], axis=1).T
                real_counts = data[var].value_counts()
                for i, c in enumerate(real_counts):
                    plt.subplot(1, len(real_counts), i+1)
                    plt.hist(fit_counts.iloc[:,i])
                    plt.axvline(real_counts[i], color='r')
            else:
                if len(self.output_vars)>1:
                    output_means = np.array([draw.mean(axis=0) for draw in pps['_'.join(self.output_vars)]])
                    plt.hist(output_means[:,var_i], alpha=0.5)
                else:
                    plt.hist([draw.mean() for draw in pps[var]], alpha=0.5) #Deal with multi output
                plt.axvline(data[var].mean(), color='r')
            plt.gca().set(title='Posterior predictive of '+var+' mean',
                          xlabel='mean('+var+')',
                          ylabel='Frequency')

    def traceplot(self, combine_chains=False):
        to_rep = ['_interval__', '_lowerbound__','_upperbound__', '_log__', '_cholesky-cov-packed__','_circular__']
        var_names = []
        for rv in self.model.free_RVs:
            name = rv.name
            for r in to_rep:
                name = name.replace(r,'')
            var_names.append(name)
        pm.traceplot(self.trace, var_names=var_names, combined=combine_chains)

    def summary(self, print_sum=True):
        to_rep = ['_interval__', '_lowerbound__', '_upperbound__', '_log__', '_cholesky-cov-packed__', '_circular__']
        var_names = []
        for rv in self.model.free_RVs:
            name = rv.name
            for r in to_rep:
                name = name.replace(r, '')
            var_names.append(name)
        summary_df = pm.summary(self.trace, var_names = var_names)
        if print_sum:
            print(summary_df.round(2))
        return summary_df


    def profile(self):
        func = self.model.logp_dlogp_function(profile=True)
        func.set_extra_values({})
        x = np.random.randn(func.size)
        for _ in range(1000):
            func(x)
        print(func.profile.summary())


def load_model(data_name, model_name: str) -> BaseModel:
    mod = pickle.load(open('../../data/models/' + data_name +'/' + model_name.replace('.pkl','') + '.pkl', 'rb'))
    if mod.name != model_name:
        mod.name = model_name
        mod.model.name = model_name
        pickle.dump(mod, open('../../data/models/' + data_name + '/' + model_name.replace('.pkl', '') + '.pkl', 'wb'))
    return mod
