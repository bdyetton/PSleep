# Code originally from a gist by Fred Callaway: https://gist.github.com/fredcallaway/1903a395f9118323de8a1a02e40492f4
import numpy as np
import pymc3 as pm
import theano

class StochasticMatrix(pm.Continuous):
    """A stochastic matrix has rows that sum to 1."""
    def __init__(self, theta, *args, **kwargs):
        shape = (theta.shape[-1], theta.shape[-1])
        kwargs.setdefault('shape', shape)
        super(StochasticMatrix, self).__init__(*args, **kwargs)
        self.theta = theta
        self.row_distribution = pm.Dirichlet.dist(a=self.theta, transform=None)

    def logp(self, value):
        results, updates = theano.scan(self.row_distribution.logp, value)
        return results.sum()

    def random(self):
        return pm.Dirichlet('rand', self.theta).random(size=len(self.theta))


if __name__ == '__main__':
    with pm.Model() as model:
        transition = StochasticMatrix('transition', theta=np.array([0.5, 0.5]), testval=np.array([[0.9, 0.1], [0.9, 0.1]]))
        trace = pm.sample(5)
        print(trace)

    #     mean
    #     sd...n_eff
    #     Rhat
    # alpha__0
    # 9.997001
    # 0.002087...
    # 197.971928
    # 1.004594
    # alpha__1
    # 11.499420
    # 0.000396...
    # 242.654012
    # 1.002539
    #
    # [2 rows x 7 columns]
    # waso
    # 9.997000887264914
    # n1
    # 11.49941981395759