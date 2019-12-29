import pymc3 as pm
import theano.tensor as tt
import pandas as pd
import sys
import numpy as np
from src.modeling import param_sample_tools
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from src.preprocessing.download_dataset import parse_epochstages_to_series
from src.modeling.run_models import format_data_for_modeling
from src.modeling.visualization import extract_mean_as_array
from mednickdb_pysleep import sleep_dynamics
from theano.printing import Print


def Print(x, attrs=None):  # Comment out to turn on printing
    return lambda y: y

def PrintShape(x):
    return Print(x,attrs=['shape'])


stages_to_consider = ['waso','n1','n2']
epoch_len = 0.5
real_alphas = [0.9, 1.1, 1.3]

# Pareto
if True:
    cont = {}
    with pm.Model() as par_model:
        alphas = real_alphas
        m = epoch_len
        for alpha, stage in zip(alphas,stages_to_consider):
            cont[stage] = pm.Pareto(stage, alpha=alpha, m=m)

    epochstage_df_cont = []
    epochstage_dict_cont = []
    for sub in range(1):
        trace = pm.sample_prior_predictive(model=par_model, samples=20*1000)
        for stage in stages_to_consider:
            print('Av duration in',stage,'=',np.mean(trace[stage]))

        cont = []

        epochstages_cont = []
        tup_stages = np.array([trace[s] for s in stages_to_consider]).T
        for stage_len in tup_stages:
            for idx, stage_idx in enumerate(stages_to_consider):
                num_epochs = int(round(stage_len[idx] / epoch_len))
                if num_epochs < 1:
                    continue
                epochstages_cont += [stage_idx] * num_epochs
        epochstage_dict_cont += [{'sleep_scoring.epochstages':epochstages_cont, 'subjectid':sub}] #hacky to make it work with parse function
    data_pre = pd.concat([parse_epochstages_to_series(row) for row in epochstage_dict_cont])

    data_pre = data_pre.loc[(data_pre['next_epoch']!='wase') & (data_pre['previous_bout']!='wbso'),:]

    zero_pre, first_pre, second_pre = sleep_dynamics.transition_counts(data_pre['current_epoch'], normalize=True, count_self_trans=True, stages_to_consider=stages_to_consider)
    print(first_pre)

    data, maps = format_data_for_modeling(data_pre, stages_to_consider=stages_to_consider)

    zero, first, second = sleep_dynamics.transition_counts(data['current_epoch'], normalize=True,
                                                                       count_self_trans=True,
                                                                       stages_to_consider=(0,1,2))

    assert np.all(first_pre==first)
    #print(data)
    episilon = 0.000000001

    alpha_prior = np.ones((len(stages_to_consider),len(stages_to_consider)))*0.01
    alpha_prior[0,0] = 0.75
    alpha_prior[1,1] = 0.75
    alpha_prior[2,2] = 0.75

    with pm.Model() as par_model_fit:
        #alpha = pm.Normal('alpha', mu=alpha_prior, sd=1e4, shape=(len(stages_to_consider),len(stages_to_consider)))
        BoundedLower = pm.Bound(pm.Normal, lower=0, upper=1)
        alpha = BoundedLower('alpha', mu=0.5, sd=1e5, shape=(len(stages_to_consider),len(stages_to_consider)))
        BoundedNormal = pm.Bound(pm.Normal, lower=0, upper=1)
        alpha_baseline = BoundedNormal('alpha_baseline', mu=0.5, sd=1e5, shape=(len(stages_to_consider),len(stages_to_consider)))
        m=epoch_len

        def logp(current_epoch, next_epoch, tau):
            current_idx = tt.cast(current_epoch, 'int16')
            next_idx = tt.cast(next_epoch, 'int16')
            tau = tt.as_tensor_variable(tau[:, np.newaxis])
            alpha_t = alpha[current_idx, :] / tau
            alpha_baseline_t = alpha_baseline[current_idx, :]
            cat = pm.Categorical.dist(p=alpha_t+alpha_baseline_t)
            return cat.logp(next_idx)

        survival = pm.DensityDist('survival', logp,
                                  observed={'current_epoch':data['current_epoch'],
                                            'tau': data['tau'],
                                            'next_epoch':data['next_epoch']})
        trace = pm.sample(nuts_kwargs={'target_accept':0.9, 'max_treedepth':15})

    pickle.dump(trace, open('../../data/scrap/survival_rate_trace.pkl','wb'))
    summary_df = pm.summary(trace)
    print(summary_df)
    pm.traceplot(trace)
    summary_df.to_pickle('../../data/scrap/survival_rate.pkl')

trace = pickle.load(open('../../data/scrap/survival_rate_trace.pkl','rb'))

n_samples = 200
pd_cont = []
for sample_idx, sample in enumerate(np.random.randint(0,len(trace), n_samples)):
    for r, current_stage in enumerate(stages_to_consider):
        tau = np.arange(0, 60) * epoch_len
        alpha_row = trace['alpha'][sample,r,:]/tau[:, np.newaxis] + trace['alpha_baseline'][sample_idx,r,:]
        alpha_row = alpha_row / alpha_row.sum(axis=1)[:, np.newaxis]
        pd_dict = {'tau':tau,'current_stage': current_stage, 'sample':sample_idx}
        sample_df = pd.DataFrame(pd_dict)
        for c, next_stage in enumerate(stages_to_consider):
            sample_df['next_stage'] = next_stage
            sample_df['trans_prob'] = alpha_row[:,c]
            pd_cont.append(sample_df.copy())

simulated_trans_prob = pd.concat(pd_cont)
g = sns.FacetGrid(simulated_trans_prob, col='current_stage')
g.map_dataframe(sns.lineplot, x='tau', y='trans_prob', units='sample', hue='next_stage', estimator=None, alpha=0.05)
plt.show()

sys.exit()
pm.traceplot(trace)
plt.show()
alpha_fit = extract_mean_as_array(trace, var='alpha', astype='xarray').values
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
print('real', first,'\n')
print('fit1', alpha_fit/alpha_fit.sum(axis=1)[:,np.newaxis])
trans_baseline_fit = extract_mean_as_array(trace, var='alpha_baseline', astype='xarray').values
print('fit2', trans_baseline_fit/trans_baseline_fit.sum(axis=1)[:,np.newaxis])
print('fit3', (trans_baseline_fit+alpha_fit)/(trans_baseline_fit+alpha_fit).sum(axis=1)[:,np.newaxis])




# sample = 1
# for r, current_stage in enumerate(stages_to_consider):
#     print(trace['alpha'][sample,:,:])
#     alpha_row = trace['alpha'][sample,r,:]
#     alpha_sm = np.exp(alpha_row) / np.sum(np.exp(alpha_row))
#     for c, next_stage in enumerate(stages_to_consider):
#         alpha_ns = alpha_sm[c]
#         print(current_stage, next_stage, alpha_ns)
#         t = np.arange(1,101)*epoch_len
#         y = (alpha_ns*(m**alpha_ns))/(t**(alpha_ns+1))
#         plt.plot(t,y)
#     plt.legend(stages_to_consider)
#     plt.title(current_stage)
#     plt.show()
# print('Done')




