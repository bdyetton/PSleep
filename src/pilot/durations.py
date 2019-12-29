import pymc3 as pm
from mednickdb_pysleep import sleep_dynamics
from mednickdb_pyapi.mednickdb_pyapi import MednickAPI
from mednickdb_pysleep.defaults import stages_to_consider
from mednickdb_pysleep.scorefiles import extract_epochstages_from_scorefile
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


class BaseFitter():
    def __init__(self, data):
        self.data_len = len(data)
        self.trace = None
        self.model = None
        self.rand_durs = None
        self.samples = None

    def fit(self):
        self.trace = pm.sample(model=self.model, nchains=4)

    def sample(self, num_draws_from_params=20):
        self.samples = pm.sample_posterior_predictive(self.trace, model=self.model,
                                                      samples=int(self.data_len / num_draws_from_params),
                                                      size=num_draws_from_params)
        self.rand_durs = self.samples['duration'].flatten()
        return self.resample()  # FIXME why does it get soooo many more samples???

    def resample(self):
        np.random.shuffle(self.rand_durs)
        return self.rand_durs[0:self.data_len]


class Exponential(BaseFitter):
    def __init__(self, data):
        super().__init__(data)
        with pm.Model() as self.model:
            lam = pm.Normal('lam', mu=3, sd=10**10)
            obs_data = pm.Exponential('duration', lam=lam, observed=data)


class Pareto(BaseFitter):
    def __init__(self, data):
        super().__init__(data)
        with pm.Model() as self.model:
            alpha = pm.Normal('alpha', mu=5, sd=10**10)
            m = min(data)
            obs_data = pm.Pareto('duration', alpha=alpha, m=m, observed=data)


def find_and_sample_best_dist(data, dists=(Pareto, Exponential)):
    best_loo = np.inf
    best_dist = None
    for Dist in dists:
        dist = Dist(data)
        dist.fit()
        loo = pm.loo(trace=dist.trace, model=dist.model)
        if loo.LOO < best_loo:
            best_loo = loo.LOO
            best_dist = dist
    return best_dist.sample(), best_dist.__class__.__name__



# def fit_power(data):
#     with pm.Model() as model:
#         alpha = pm.Normal('alpha', mu=5, sd=10**10)
#         m = min(data)
#         obs_data = pm.Pareto('duration', alpha=alpha, m=m, observed=data)
#     trace = pm.sample(model=model, nchains=4)
#
#     return model, trace

def extract_dur_dist_data(data):
    data = data.loc[:, ['sleep_scoring.epochstage', 'subjectid']]
    data = data.dropna()
    all_dur_dists_dict = {s: [] for s in stages_to_consider}
    for idx, row in data.iterrows():
        epochstages = row['sleep_scoring.epochstage']
        dur_dists = sleep_dynamics.bout_durations(epochstages, stages_to_consider=stages_to_consider)
        for stage, dur_dist_data in dur_dists.items():
            all_dur_dists_dict[stage].extend(dur_dist_data)
    return all_dur_dists_dict


def investigate_and_fit_duration_dists(data):
    all_dur_dists_dict = extract_dur_dist_data(data)
    dur_data_cont = []
    for stage, dur_dist_data in all_dur_dists_dict.items():
        fit_dur_data, best_model_name = find_and_sample_best_dist(dur_dist_data)
        print(stage,'best dist =',best_model_name,'\n\n')
        dur_data_cont.append(pd.DataFrame({'duration': dur_dist_data,
                                           'stage': [stage]*len(dur_dist_data),
                                           'best_fit_name':[best_model_name]*len(dur_dist_data),
                                           'fit_duration': fit_dur_data}))
    dur_data = pd.concat(dur_data_cont, axis=0)

    # Initialize the FacetGrid object
    pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
    g = sns.FacetGrid(dur_data, col="stage", hue="stage", aspect=1, height=2, palette=pal, sharex=False, sharey=False)

    # Draw the densities in a few steps
    g.map(sns.kdeplot, "duration", clip_on=False, shade=True, alpha=1, lw=1.5, bw=.2)
    g.map(sns.kdeplot, "fit_duration", clip_on=False, color="k", lw=2, bw=.2)
    # g.map(plt.axhline, y=0, lw=2, clip_on=False)

    # Define and use a simple function to label the plot in axes coordinates
    # def label(x, color, label):
    #     ax = plt.gca()
    #     ax.text(0, .2, label, fontweight="bold", color=color,
    #             ha="left", va="center", transform=ax.transAxes)
    #
    # g.map(label, "duration")

    # Set the subplots to overlap
    #g.fig.subplots_adjust(hspace=-1)

    # Remove axes details that don't play well with overlap
    #g.set_titles("")
    g.set(yticks=[])
    g.despine(bottom=True, left=True)
    plt.show()


def parse_sleep_scoring_to_durations(scorefile):
    epochstages = extract_epochstages_from_scorefile(scorefile)



if __name__ == '__main__':

    scorefiles = glob.iglob('../../')

    # #connect to DB
    # med_api = MednickAPI(username=os.environ['mednickapi_username'], password=os.environ['mednickapi_password'])
    #
    # #Get the data, so easy :)
    # data = med_api.get_data('studyid=NSRR_CFS', format_as='dataframe_single_index')
    # print('Got', data.shape[0], 'records')




    #for
    investigate_and_fit_duration_dists(data)

    #Steps:
    # Show getting the data
    # Show munging the data into