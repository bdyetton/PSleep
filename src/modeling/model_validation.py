import pymc3 as pm
from src.modeling.param_sample_tools import sample_priors
import numpy as np


def simulate_data_from_model(model_func, priors, samples_template):
    model = model_func(samples_template, priors)
    trace = pm.sample_prior_predictive(model=model, samples=len(samples_template))
    return trace['obs'][0]


def run_simulation_check(model_func, num_runs=5):
    for run in enumerate(num_runs):
        sim_data = simulate_data_from_model(model_func, params=sample_priors(), samples_template=np.ones((1000,1)))





