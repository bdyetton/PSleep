import pymc3 as pm
import numpy as np
import theano.tensor as tt


class Markov1stOrder(pm.Categorical):
    """
    Hidden Markov Model States
    Adapted from https://github.com/hstrey/Hidden-Markov-Models-pymc3/blob/master/Multi-State%20HMM.ipynb

    Parameters
    ----------
    p_trans : tensor
        transition probability
        shape = (N_states,N_states)

    p_t0 : tensor
         starting probability
         shape = (N_states)

    """

    def __init__(self, p_t0=None, p_trans=None, *args, **kwargs):
        super(pm.Categorical, self).__init__(*args, **kwargs)
        self.p_trans = tt.as_tensor_variable(p_trans)  # Tran probabilities (1st order trans probabilities)
        self.p_t0 = tt.as_tensor_variable(p_t0)  # Initial probability for t=0 (0th order trans probabilities)
        self.mean = 0.
        self.mode = tt.cast(0, dtype='int64')

    def logp(self, value):
        """How to sample the distribution given observed points (x)"""
        p_trans = self.p_trans
        p_t0 = self.p_t0

        # We need a the probability of the next state given the current state P(x_t | x_t-1)
        # Index into trans matrix to generate categorical probabilities for the next point which is based on the
        # previous point (except the last)
        p_t = p_trans[value[:-1], :]

        # value_i is what we are trying to predict
        value_i = value[1:]  # The first point is not transitioned too, and not sampled from trans distribution
        likelihood_t = pm.Categorical.dist(p_t).logp(value_i)  # Likelihood of transitioned to points (dynamic process_datasets)
        return pm.Categorical.dist(p_t0).logp(value[0]) + tt.sum(likelihood_t)  # Static first point + dynamic next points

    def random(self, point=None, size=None):
        p_trans = self.p_trans  # Tran probabilities (1st order trans probabilities)
        p_t0 = self.p_t0
        p_t0_, p_trans_ = pm.distributions.multivariate.draw_values([p_t0, p_trans], point=point, size=size)

        def single_dim_gen_sample(p_t0, p_trans, size, hmm_len):
            x_i = pm.distributions.dist_math.random_choice(p=p_t0, size=size)  # first sample of the chain
            x = [x_i]

            for _ in range(hmm_len-1):
                x_i = pm.distributions.dist_math.random_choice(p=p_trans[x_i, :], size=size)  # all others
                x.append(x_i)

            return np.stack(x,-1)

        return pm.distributions.multivariate.generate_samples(single_dim_gen_sample,
                                                              p_t0=p_t0_, p_trans=p_trans_,
                                                              size=size,
                                                              broadcast_shape=(size,),
                                                              hmm_len=self.shape)


class Markov2ndOrder(pm.Categorical):
    """
    Hidden Markov Model States
    Adapted from https://github.com/hstrey/Hidden-Markov-Models-pymc3/blob/master/Multi-State%20HMM.ipynb

    Parameters
    ----------
    p_trans : tensor
        transition probability
        shape = (N_states,N_states,N_states)

    p_t0 : tensor
         starting probability
         shape = (N_states)

    p_t1 : tensor
         Trans probability for t0->t1 probability
         shape = (N_states, N_states)

    """

    def __init__(self, p_t0=None, p_t1=None, p_trans=None, *args, **kwargs):
        super(pm.Categorical, self).__init__(*args, **kwargs)
        self.p_trans = tt.as_tensor_variable(p_trans)  # Tran probabilities (2nd order trans probabilities)
        self.p_t1 = tt.as_tensor_variable(p_t1)  # Initial probability for t=1 | t=0 (1st order trans probabilities)
        self.p_t0 = tt.as_tensor_variable(p_t0)  # Initial probability for t=0 (0th order trans probabilities)
        self.mean = 0.
        self.mode = tt.cast(0, dtype='int64')

    def logp(self, value):
        """How to sample the distribution given observed points (x)"""
        p_trans = self.p_trans
        p_t0 = self.p_t0  # Probability for first point (unconditional)
        p_t1 = self.p_t1  # Probability for second point (conditional on first)

        # Probability of the next points (conditional on previous 2 points) - P(x_t | x_t-1, x_t-1)
        p_t = p_trans[value[:-2], value[1:-1], :]  # Index into trans matrix to generate categorical probabilities for each point,
        # where p_trans is indexed by furthest back point first i.e. p_trans[x_t-2, x_t-1]

        x_ti = value[2:]  # The first and second point is not based on 2 back, this is what we are trying to predict
        likelihood_ti = pm.Categorical.dist(p_t).logp(x_ti)  # likelihood for dynamic process_datasets
        # Full likelihood is static first point + static second point + dynamic next points
        return pm.Categorical.dist(p_t0).logp(value[0]) + pm.Categorical.dist(p_t1[value[0], :]).logp(value[1]) + tt.sum(likelihood_ti)

    def random(self, point=None, size=None):
        p_trans = self.p_trans  # Tran probabilities (1st order trans probabilities)
        p_t1 = self.p_t1
        p_t0 = self.p_t0
        p_t0_, p_t1_, p_trans_ = pm.distributions.multivariate.draw_values([p_t0, p_t1, p_trans], point=point, size=size)

        def single_dim_gen_sample(p_t0, p_t1, p_trans, size, hmm_len):
            x_i = pm.distributions.dist_math.random_choice(p=p_t0, size=size)  # first sample of the chain
            x = [x_i]
            #second sample
            x_ii = pm.distributions.dist_math.random_choice(p=p_t1[x_i, :], size=size)  # all others
            x.append(x_ii)

            for _ in range(hmm_len-2): #all others
                x_ = pm.distributions.dist_math.random_choice(p=p_trans[x_ii, x_i, :], size=size)  # all others
                x.append(x_)
                x_ii = x_i
                x_i = x_

            return np.stack(x, -1)

        return pm.distributions.multivariate.generate_samples(single_dim_gen_sample,
                                                              p_t0=p_t0_, p_t1=p_t1_, p_trans=p_trans_,
                                                              size=size,
                                                              broadcast_shape=(size,),
                                                              hmm_len=self.shape)


class FunctionalMarkov1stOrder(pm.Categorical):
    """
    Hidden Markov Model States
    Adapted from https://github.com/hstrey/Hidden-Markov-Models-pymc3/blob/master/Multi-State%20HMM.ipynb

    Parameters
    ----------
    p_trans : tensor
        transition probability
        shape = (N_states,N_states)

    p_t0 : tensor
         starting probability
         shape = (N_states)

    """

    def __init__(self, p_t0=None, p_trans=None, *args, **kwargs):
        super(pm.Categorical, self).__init__(*args, **kwargs)
        self.p_trans = tt.as_tensor_variable(p_trans)  # Tran probabilities (1st order trans probabilities)
        self.p_t0 = tt.as_tensor_variable(p_t0)  # Initial probability for t=0 (0th order trans probabilities)
        self.mean = 0.
        self.mode = tt.cast(0, dtype='int64')

    def logp(self, value):
        """How to sample the distribution given observed points (x)"""
        p_trans = self.p_trans
        p_t0 = self.p_t0

        # We need a the probability of the next state given the current state P(x_t | x_t-1)
        # Index into trans matrix to generate categorical probabilities for the next point which is based on the
        # previous point (except the last)
        p_t = p_trans[tt.arange(p_trans.shape[0]), value[:-1], :]

        # value_i is what we are trying to predict
        value_i = value[1:]  # The first point is not transitioned too, and not sampled from trans distribution
        likelihood_t = pm.Categorical.dist(p_t).logp(value_i)  # Likelihood of transitioned to points (dynamic process_datasets)
        return pm.Categorical.dist(p_t0).logp(value[0]) + tt.sum(likelihood_t)  # Static first point + dynamic next points

    def random(self, point=None, size=None):
        p_trans = self.p_trans  # Tran probabilities (1st order trans probabilities)
        p_t0 = self.p_t0
        p_t0_, p_trans_ = pm.distributions.multivariate.draw_values([p_t0, p_trans], point=point, size=size)

        def single_dim_gen_sample(p_t0, p_trans, size, hmm_len):
            x_i = pm.distributions.dist_math.random_choice(p=p_t0, size=size)  # first sample of the chain
            x = [x_i]

            for i in range(hmm_len-1):
                x_i = pm.distributions.dist_math.random_choice(p=p_trans[i, x_i, :], size=size)  # all others
                x.append(x_i)

            return np.stack(x, -1)

        return pm.distributions.multivariate.generate_samples(single_dim_gen_sample,
                                                              p_t0=p_t0_, p_trans=p_trans_,
                                                              size=size,
                                                              broadcast_shape=(size,),
                                                              hmm_len=self.shape)

# class FunctionalMarkov1stOrder():
#     def __init__(self, p_t0=None, p_t1=None, p_trans=None, *args, **kwargs):
#         super(pm.Categorical, self).__init__(*args, **kwargs)
#         self.p_trans = tt.as_tensor_variable(p_trans)  # Tran probabilities (2nd order trans probabilities)
#         self.p_t1 = tt.as_tensor_variable(p_t1)  # Initial probability for t=1 | t=0 (1st order trans probabilities)
#         self.p_t0 = tt.as_tensor_variable(p_t0)  # Initial probability for t=0 (0th order trans probabilities)
#         self.mean = 0.
#         self.mode = tt.cast(0, dtype='int64')
#
#     def logp(self, value):
#         """How to sample the distribution given observed points (x)"""
#         p_trans = self.p_trans
#         p_t0 = self.p_t0  # Probability for first point (unconditional)
#         p_t1 = self.p_t1  # Probability for second point (conditional on first)
#
#         # Probability of the next points (conditional on previous 2 points) - P(x_t | x_t-1, x_t-1)
#         p_t = p_trans[:, value[:-2], value[1:-1], :]  # Index into trans matrix to generate categorical probabilities for each point,
#         # where p_trans is indexed by furthest back point first i.e. p_trans[x_t-2, x_t-1]
#
#         x_ti = value[2:]  # The first and second point is not based on 2 back, this is what we are trying to predict
#         likelihood_ti = pm.Categorical.dist(p_t).logp(x_ti)  # likelihood for dynamic process_datasets
#         # Full likelihood is static first point + static second point + dynamic next points
#         # TODO some kind of scan here... or array indexes
#         return pm.Categorical.dist(p_t0).logp(value[0]) + pm.Categorical.dist(p_t1[value[0], :]).logp(value[1]) + tt.sum(likelihood_ti)
#
#     def random(self, point=None, size=None):
#         """TODO start here"""
#         p_trans = self.p_trans  # Tran probabilities (1st order trans probabilities)
#         p_t1 = self.p_t1
#         p_t0 = self.p_t0
#         p_t0_, p_t1_, p_trans_ = pm.distributions.multivariate.draw_values([p_t0, p_t1, p_trans], point=point, size=size)
#
#         def single_dim_gen_sample(p_t0, p_t1, p_trans, size, hmm_len):
#             x_i = pm.distributions.dist_math.random_choice(p=p_t0, size=size)  # first sample of the chain
#             x = [x_i]
#             #second sample
#             x_ii = pm.distributions.dist_math.random_choice(p=p_t1[x_i, :], size=size)  # all others
#             x.append(x_ii)
#
#             for _ in range(hmm_len-2): #all others
#                 x_ = pm.distributions.dist_math.random_choice(p=p_trans[:, x_ii, x_i, :], size=size)  # all others
#                 x.append(x_)
#                 x_ii = x_i
#                 x_i = x_
#
#             return np.stack(x, -1)
#
#         return pm.distributions.multivariate.generate_samples(single_dim_gen_sample,
#                                                               p_t0=p_t0_, p_t1=p_t1_, p_trans=p_trans_,
#                                                               size=size,
#                                                               broadcast_shape=(size,),
#                                                               hmm_len=self.shape)

class Markov3rdOrder(pm.Categorical):
    """
    Hidden Markov Model States
    Adapted from https://github.com/hstrey/Hidden-Markov-Models-pymc3/blob/master/Multi-State%20HMM.ipynb

    Parameters
    ----------
    p_trans : tensor
        transition probability
        shape = (N_states,N_states, N_states, N_states)


    p_t1 : tensor
         Trans probability for t0->t1 probability
         shape = (N_states, N_states)


    p_t2 : tensor
         Trans probability for t1->t2 probability
         shape = (N_states, N_states, N_states)

    """

    def __init__(self, p_t0=None, p_t1=None, p_t2=None, p_trans=None, *args, **kwargs):
        super(pm.Categorical, self).__init__(*args, **kwargs)
        self.p_trans = p_trans  # Tran probabilities (2nd order trans probabilities)
        self.p_t0 = p_t0  # Initial probability for t=0 (0th order trans probabilities)
        self.p_t1 = p_t1  # Initial probability for t=1 | t=0 (1st order trans probabilities)
        self.p_t2 = p_t2  # Initial probability for t=2 | t=0, t=1 (2nd order trans probabilities)
        self.mean = 0.
        self.mode = tt.cast(0, dtype='int64')

    def logp(self, x):
        """How to sample the distribution given observed points (x)"""
        p_trans = self.p_trans
        p_t0 = self.p_t0  # Probability for first point (unconditional)
        p_t1 = self.p_t1  # Probability for second point (conditional on first)
        p_t2 = self.p_t2  # Probability for third point (conditional on first and second)

        # Probability of the next points (conditional on previous 3 points) - P(x_t | x_t-1, x_t-1)
        p_t = p_trans[x[:-3], x[1:-2], x[2:-1]]  # Index into trans matrix to generate categorical probabilities for each point,
        # where p_trans is indexed by furthest back point first i.e. p_trans[x_t-2, x_t-1]

        x_ti = x[3:]  # The first and second and third point is not based on 2 back, this is what we are trying to predict
        likelihood_ti = pm.Categorical.dist(p_t).logp(x_ti)  # likelihood for dynamic process_datasets
        # Full likelihood is static first point + static second point + dynamic next points
        return pm.Categorical.dist(p_t0).logp(x[0]) + pm.Categorical.dist(p_t1[x[0]]).logp(x[1]) \
               + pm.Categorical.dist(p_t2[x[0],x[1]]).logp(x[2]) + tt.sum(likelihood_ti)
