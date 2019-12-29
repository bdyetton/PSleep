from mednickdb_pyapi.mednickdb_pyapi import MednickAPI
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import HMM
import pymc3 as pm
import theano as tt
from mednickdb_pysleep import sleep_dynamics
from scipy.stats import gaussian_kde


def sample_from_rand_priors_hmm(nsamples, nback, nstages, hirach=True):
    with pm.Model() as model:
        if not hirach and nback == 1:
            tps = {'p_t0': rand_trans_0back(nstages), 'p_trans': rand_trans_1back(nstages)}
            HMM.Markov1stOrder('markov', p_t0=tps['p_t0'], p_trans=tps['p_trans'], observed=np.ones((nsamples, )))

        elif not hirach and nback == 2:
            tps = {'p_t0': rand_trans_0back(nstages),
                   'p_t1': rand_trans_1back(nstages),
                   'p_trans': rand_trans_2back(nstages)}
            HMM.Markov2ndOrder('markov', p_t0=tps['p_t0'], p_t1=tps['p_t1'], p_trans=tps['p_trans'], observed=np.ones((nsamples, )))

        elif hirach and nback == 1:
            p_trans, b0, b1 = rand_trans_1back_tdep(nsamples, nstages)
            tps = {'p_t0': rand_trans_0back(nstages),
                   'p_trans': p_trans}
            HMM.FunctionalMarkov1stOrder('markov', p_t0=tps['p_t0'], p_trans=tps['p_trans'], observed=np.ones((nsamples, )))


    trace = pm.sample_prior_predictive(model=model, samples=1)
    return trace['markov '][0], tps


def fit_hmm(data, nback, nstages):
    with pm.Model() as model:
        if nback == 1:
            param_names = ['p_t0', 'p_trans']
            tp_dist0 = pm.Dirichlet(param_names[0], a=np.ones((nstages,)))
            tp_dist1 = pm.Dirichlet(param_names[1], a=np.ones((nstages, nstages)),
                                    shape=(nstages, nstages),
                                    testval=np.ones((nstages, nstages)))
            HMM.Markov1stOrder('markov', p_t0=tp_dist0, p_trans=tp_dist1, observed=data)

        elif nback == 2:
            param_names = ['p_t0', 'p_t1', 'p_trans']
            tp_dist0 = pm.Dirichlet(param_names[0], a=np.ones((nstages,)))
            tp_dist1 = pm.Dirichlet(param_names[1], a=np.ones((nstages, nstages)),
                                    shape=(nstages, nstages),
                                    testval=np.ones((nstages, nstages)))
            tp_dist2 = pm.Dirichlet(param_names[2], a=np.ones((nstages, nstages, nstages)),
                                    shape=(nstages, nstages, nstages),
                                    testval=np.ones((nstages, nstages, nstages)))
            HMM.Markov2ndOrder('markov', p_t0=tp_dist0, p_t1=tp_dist1, p_trans=tp_dist2, observed=data)

    return pm.sample(model=model)


def validate_params(true_params, trace):
    true_param_cont = []
    for true_param, true_param_value in true_params.items():
        recovery_probability_cont = []
        for idx, val in np.ndenumerate(true_param_value):
            idx_multi = [slice(None, None)]+list(idx)
            samples = trace[true_param][idx_multi]
            # Do kernel density estimation
            kde = gaussian_kde(samples.flatten())
            # Get probability for range of values
            start = val-0.03  # Start of the range
            end = val+0.03  # End of the range
            N = 100  # Number of evaluation points
            step = (end - start) / (N - 1)  # Step size
            x = np.linspace(start, end, N)[:, np.newaxis]  # Generate values in the range
            kd_vals = np.array([kde.evaluate(x_) for x_ in x])  # Get PDF values for each x
            probability = np.sum(kd_vals * step)  # Approximate the integral of the PDF
            recovery_probability_cont.append(probability)
        true_param_cont.append(pd.DataFrame({'parameter': [true_param]*len(recovery_probability_cont),
                                             'recovery probability': recovery_probability_cont}))
    return pd.concat(true_param_cont, axis=0)


def rand_trans_0back(nshape):
    tp = np.random.dirichlet([1]*nshape)
    assert np.sum(tp) - 1 < 0.00000001
    return tp


def rand_trans_1back(nshape):
    tp = np.random.dirichlet([1]*nshape, nshape)
    for row in range(nshape):
        assert np.sum(tp[row, :]) - 1 < 0.00000001
    return tp


def rand_trans_1back_tdep(nsamples, nshape):
    b0 = np.abs(np.random.normal(0.5, 0.3, size=(1, nshape, nshape)))
    b0 += np.identity(nshape)
    b0 = b0/np.sum(b0, axis=1)
    b1 = np.random.normal(0.01, 0.01, size=(1, nshape, nshape))  # FIXME not sure about this...
    t = np.moveaxis(np.arange(nsamples-1)[np.newaxis][np.newaxis], 2, 0)
    tps = b0+b1*t
    tps = tps/np.expand_dims(np.sum(tps, axis=2),-1)
    return tps, b0, b1


def rand_trans_2back(nshape):
    stacker = []
    for _ in range(nshape):
        stacker.append(np.random.dirichlet([1]*nshape, nshape))
    tp = np.stack(stacker, axis=0)
    for row in range(nshape):
        for col in range(nshape):
            assert np.sum(tp[row, col, :]) - 1 < 0.00000001
    return tp

if __name__ == "__main__":
    nsamples_range = list(range(100, 14101, 2000))
    param_cont = []
    n_back = 1
    for nsamples in [10000,100000]:
        samples, true_params = sample_from_rand_priors_hmm(nsamples=nsamples, nback=n_back, nstages=5)
        # [_, tp1, tp2] = sleep_dynamics.transition_counts(samples,
            #                                              stages_to_consider=list(range(0, 5)),
            #                                              count_self_trans=True,
            #                                              normalize=True)
        trace = fit_hmm(samples, nback=n_back, nstages=5)
        param_recovery = validate_params(true_params, trace)
        param_recovery['number of epochs'] = nsamples
        param_cont.append(param_recovery)
    param_recovery_df = pd.concat(param_cont, axis=0)
    trans_recover_df = param_recovery_df.loc[param_recovery_df['parameter']=='p_trans',:]
    sns.lineplot(data=trans_recover_df, x='number of epochs', y='recovery probability')
    plt.title('Probability of true parameter under dist of fit parameter')
    plt.show()




