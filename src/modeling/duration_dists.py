from abc import ABC
from typing import Tuple
import pymc3 as pm
import pandas as pd
import numpy as np

class BaseDurationDist(ABC):
    def __init__(self, data: pd.DataFrame):
        """
        Base class for duration distributions. This is an abstract baseclass (ABC)
        :param data: dataframe of data we want to fit a model too, should have at least "duration" col
        """
        self.data_len = len(data['duration'])
        self.trace = None
        self.model = None
        self.name = "This is an ABC, don't use directly"

    def fit(self) -> Tuple[pm.Model, pm.backends.base.MultiTrace]:
        """Fit parameters with MCMC"""
        self.trace = pm.sample(model=self.model, nchains=4, draws=2000)
        return self.model, self.trace

    def sample_posterior_predictive(self, num_draws_from_params: int=20) -> np.ndarray:
        """
        Sample from the posterior predictive, i.e. simulate data from the fit model
        :param num_draws_from_params: how many times we should sample the parameters (and then sample data)
        :return: simulated duration samples
        """
        samples = pm.sample_posterior_predictive(self.trace, model=self.model,
                                                      samples=num_draws_from_params)
        duration_samples = samples['duration'].flatten()
        duration_samples = np.round(duration_samples * 2) / 2 #our data only comes in 0.5 bin intervals, so round
        np.random.shuffle(duration_samples)
        return duration_samples[0:self.data_len]


class Exponential(BaseDurationDist):
    """Initializes a model where duration of a sleep stage is exponentially distributed
    We limit the duration to something reasonable because exponential has a heavy tail."""
    def __init__(self, data):
        super().__init__(data)
        self.name='Exponential'
        with pm.Model() as self.model:
            lam = pm.Gamma('lam', mu=3, sd=10**5)
            obs_data = pm.Exponential('duration', lam=lam, observed=data['duration'])
            pm.Potential("bound", pm.math.switch(obs_data <= max(data['duration']) * 1.5, 0, -np.inf))
        self.model.name = 'Exponential'


class Pareto(BaseDurationDist):
    def __init__(self, data):
        """Initializes a model where duration of a sleep stage is pareto distributed.
        We limit the duration to something reasonable because pareto has a heavy tail."""
        super().__init__(data)
        self.name = 'Pareto'
        with pm.Model() as self.model:
            alpha = pm.Gamma('alpha', mu=5, sd=10**5)
            m = min(data['duration'])
            obs_data = pm.Pareto('duration', alpha=alpha, m=m, observed=data['duration'])
            pm.Potential("bound", pm.math.switch(obs_data <= max(data['duration'])*1.5, 0, -np.inf))
        self.model.name = 'Pareto'


class Weibull(BaseDurationDist):
    def __init__(self, data):
        """Initializes a model where duration of a sleep stage is pareto distributed.
        We limit the duration to something reasonable because pareto has a heavy tail."""
        super().__init__(data)
        self.name = 'Weibull'
        with pm.Model() as self.model:
            alpha = pm.InverseGamma('alpha', mu=1, sd=10**5)
            beta = pm.InverseGamma('beta', mu=1, sd=10**5)
            obs_data = pm.Weibull('duration', alpha=alpha, beta=beta, observed=data['duration'])
        self.model.name = 'Weibull'


class Gamma(BaseDurationDist):
    def __init__(self, data):
        """Initializes a model where duration of a sleep stage is gamma distributed"""
        super().__init__(data)
        self.name = 'Gamma'
        with pm.Model() as self.model:
            alpha = pm.Gamma('alpha', mu=5, sd=10**5)
            beta = pm.Gamma('beta', mu=0.5, sd=10**5)
            obs_data = pm.Gamma('duration', alpha=alpha, beta=beta, observed=data['duration'])
        self.model.name = 'Gamma'


class GammaAge(BaseDurationDist):
    def __init__(self, data):
        """Initializes a model where duration of a sleep stage is gamma distributed"""
        super().__init__(data)
        self.name = 'Gamma'
        with pm.Model() as self.model:
            LowerBoundedNormal = pm.Bound(pm.Normal, lower=0)
            alpha = LowerBoundedNormal('alpha', mu=5, sd=10**2, shape=2)
            beta = pm.Gamma('beta', mu=0.5, sd=10**2)
            obs_data = pm.Gamma('duration', alpha=alpha[0]+alpha[1]*data['age'], beta=beta, observed=data['duration'])
        self.model.name = 'GammaAge'
