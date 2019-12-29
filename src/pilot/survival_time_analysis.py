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

# Exponential
# with pm.Model() as ex_model:
#     wake_beta = 3
#     sleep_beta = wake_beta + 4
#     wake_durs = pm.Exponential('wake', lam=1/wake_beta)
#     sleep_durs = pm.Exponential('sleep', lam=1/sleep_beta)

# Pareto
with pm.Model() as par_model:
    wake_alpha = 0.9
    sleep_alpha = 1.1
    m = 0.5
    wake_durs = pm.Pareto('wake', alpha=wake_alpha, m=m)
    sleep_durs = pm.Pareto('sleep', alpha=sleep_alpha, m=m)



trace = pm.sample_prior_predictive(model=par_model, samples=5000)
print('wake', np.mean(trace['wake']))
print('sleep', np.mean(trace['sleep']))
tau_step = 0.1
cont = []
for stage_idx in ('wake', 'sleep'):
    for stage_len in trace[stage_idx]:
        epoch_len = int(round(stage_len/tau_step))
        if epoch_len < 1:
            continue
        epochstages = [stage_idx]*epoch_len
        time = np.arange(1,epoch_len+1)*tau_step
        failure = np.zeros_like(time).astype(int)
        failure[-1] = 1
        cont.append(pd.DataFrame({'time':time,'failure':failure, 'stage':epochstages}))

data = pd.concat(cont, axis=0).reset_index(drop=True)
data['stage'] = data['stage'].map({'wake':0,'sleep':1}).astype(int)
print(data)


# with pm.Model() as ex_model_fit:
#     beta = pm.Normal('beta', mu=0.0, sd=1e5, shape=2)
#
#     def logp(sleep, failure, time):
#         lam = pm.math.exp(pm.math.switch(sleep, beta[1], beta[0]))
#         return (failure * (pm.math.log(lam) - lam * time)).sum()
#
#     survival = pm.DensityDist('survival', logp,
#                               observed={'sleep':data['stage'], 'failure': data['failure'], 'time': data['time']})
#     trace = pm.sample()
#
# summary_df = pm.summary(trace)
# print(summary_df)
# print('wake', np.exp(-summary_df.loc['beta__0','mean']))
# print('sleep', np.exp(-summary_df.loc['beta__1','mean']))
# pm.traceplot(trace)
# plt.show()


with pm.Model() as par_model_fit:
    BoundedNormal = pm.Bound(pm.Normal, lower=0)
    alpha = BoundedNormal('alpha', mu=1.0, sd=1e5, shape=2)
    m=0.5

    def logp(sleep, failure, time):
        alpha_i = pm.math.switch(sleep, alpha[1], alpha[0])
        like = pm.math.log(alpha_i) - (alpha_i+1)*pm.math.log(time) + alpha_i*pm.math.log(m)
        return (failure * like).sum()

    survival = pm.DensityDist('survival', logp,
                              observed={'sleep':data['stage'], 'failure': data['failure'], 'time': data['time']})
    trace = pm.sample()


summary_df = pm.summary(trace)
print(summary_df)
print('wake', summary_df.loc['alpha__0','mean'])
print('sleep', summary_df.loc['alpha__1','mean'])
pm.traceplot(trace)
plt.show()


