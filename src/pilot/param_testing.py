import pymc3 as pm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import theano

tau_data = np.linspace(0.5,60)
rem_data = 2+np.linspace(0,60*0.005)

priors = {
    'baserate_p': 2,  # doubel check rem properties
    'clocktime_p': 0.05,  # assuming hours
    'timeslept_p': -0.03,
    'clocktime_phase': 0,
    'clocktime_rate': 0,
    'tau_p': -0.005,
    'tau_p^2': -0.0005,
    'age_p': -0.05,
    'age_p^2': 0,
    'sex_p': -0.5,
    'ageXsex_p': 0.05,
    'tau_alpha_exp_p': 0.5,
    'tau_lam_exp_p': -0.01,
}

sd_priors = {
    'clocktime_p': 0.5,  # assuming hours
    'timeslept_p': 0.5,
    'clocktime_rate': 3,
    'previous_bout_p': 0.5,
    'tau_p': 0.01,
    'tau_p^2': 0.001,
    'age_p': 0.1,
    'age_p^2': 0.001,
    'sex_p': 3,
    'ageXsex_p': 0.1,
    'tau_alpha_exp_p': 5,
    'tau_lam_exp_p': 2,
}

with pm.Model() as mod:
    tau_t1 = pm.Normal('1',mu=priors['tau_p'], sd=sd_priors['tau_p'])
    tau_t0 = pm.Normal('2', mu=priors['baserate_p'], sd=0.5)
    err = pm.HalfNormal('err', sd=1)
    rem = pm.Normal('rem',  tau_t0 + tau_t1*tau_data, sd=err, observed=rem_data)
    data = pm.sample_prior_predictive()
#print(data['rem'].shape)
#print(tau_data[np.newaxis,:].repeat(500,0).shape)
#plt.scatter(tau_data[np.newaxis,:].repeat(500,0).flatten(), data['rem'].flatten())
#plt.show()

tau_new = tau_data[np.newaxis,:].repeat(500,0).flatten()
rem_new = data['rem'].flatten()
with pm.Model() as mod:
    #alpha * pm.math.exp(variable[:, np.newaxis] * lam)
    offset = pm.Normal('offset',mu=2, sd=0.2)
    alpha = pm.Normal('alpha',mu=priors['tau_alpha_exp_p'], sd=sd_priors['tau_alpha_exp_p'])
    lamb = pm.Normal('lam', mu=priors['tau_lam_exp_p'], sd=sd_priors['tau_lam_exp_p'])
    err = pm.HalfNormal('err', sd=2)
    rem = pm.Normal('rem',  offset + alpha * pm.math.exp(tau_new * lamb), sd=err, observed=rem_new)
    trace = pm.sample(init='adapt_diag')
pm.traceplot(trace)
plt.show()

