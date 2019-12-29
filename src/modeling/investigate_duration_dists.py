# imports and defs
import pymc3 as pm
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import yaml
import pickle
from scipy.stats import ks_2samp
from typing import List, Tuple
from mednickdb_pysleep.scorefiles import extract_epochstages_from_scorefile
from mednickdb_pysleep.sleep_dynamics import bout_durations
from mednickdb_pyapi.upload_helper import extract_key_values_from_filepath
from src.modeling.duration_dists import Pareto, Gamma, Exponential, BaseDurationDist, GammaAge, Weibull
from src.modeling.run_models import format_data_for_modeling
import os
from mednickdb_pyapi.pyapi import MednickAPI


def find_best_dist(data: pd.DataFrame, dists: list=(Pareto, Gamma, Exponential)) -> BaseDurationDist:
    """
    Fit the given distributions to the duration data, and return the best fitting dist (based on LOO esimate)
    :param data: duration data
    :param dists: list of distributions to fit
    :return: the best fitting distribution
    """
    models_to_compare = {}
    models_fit = {}
    for Dist in dists: #fit each dist to data
        dist = Dist(data)
        model, trace = dist.fit()
        models_fit[dist.name] = dist
        models_to_compare[model] = trace

    #Rank using LOO to find the best fitting model
    compare_df = pm.compare(models_to_compare, ic='LOO') #LOO CV is an estimate of the out-of-sample predictive fit
    print('\n',compare_df,'\n')
    return models_fit[compare_df.index[0]] #return best


def parse_sleep_scoring_to_durations(scorefile_path: str, stage_map: dict, stages_to_consider=('n2','n3','rem')) -> pd.DataFrame:
    """
    Parse a subjects scorefile into duration data for modeling
    :param scorefile_path: path to scorefile to parse
    :param stage_map: map from stages in scorefile to standard mednickdb stages
    :return: pd.Dataframe with columns of stage, duration, and specifiers (subjectid, etc) and a row per bout
    """
    # parse raw scorefile to durations
    # e.g. ['n1','n1','n1','n2','n2','n1','n2','rem','rem'] -> {n1: [90, 30], n2:[60,30], 'rem':[60]}
    epochstages, _, _ = extract_epochstages_from_scorefile(scorefile_path, stage_map) #parse xml file
    durations_per_stage = bout_durations(epochstages, stages_to_consider=stages_to_consider) #convert to list of durations

    # make a dataframe for easy modeling
    rows_to_add_cont = []
    for stage, bouts_for_stage in durations_per_stage.items():
        rows_to_add_cont.append(pd.DataFrame({'stage': [stage] * len(bouts_for_stage),
                                              'duration': bouts_for_stage}))
    subjects_bouts = pd.concat(rows_to_add_cont)

    #get subject information from filename and add to dataframe
    specifiers = extract_key_values_from_filepath(scorefile_path, '*cfs-visit*-{subjectid}-nsrr.xml')
    for spec, spec_val in specifiers.items():
        subjects_bouts[spec] = spec_val

    return subjects_bouts

def parse_epochstages_to_durations(epochstages: str, specifiers:dict, stages_to_consider=('n2','n3','rem')) -> pd.DataFrame:
    """
    Parse a subjects scorefile into duration data for modeling
    :param scorefile_path: path to scorefile to parse
    :param stage_map: map from stages in scorefile to standard mednickdb stages
    :return: pd.Dataframe with columns of stage, duration, and specifiers (subjectid, etc) and a row per bout
    """
    # parse raw scorefile to durations
    # e.g. ['n1','n1','n1','n2','n2','n1','n2','rem','rem'] -> {n1: [90, 30], n2:[60,30], 'rem':[60]}
    durations_per_stage = bout_durations(epochstages, stages_to_consider=stages_to_consider) #convert to list of durations

    # make a dataframe for easy modeling
    rows_to_add_cont = []
    for stage, bouts_for_stage in durations_per_stage.items():
        rows_to_add_cont.append(pd.DataFrame({'stage': [stage] * len(bouts_for_stage),
                                              'duration': bouts_for_stage}))
    subjects_bouts = pd.concat(rows_to_add_cont)

    #get subject information from filename and add to dataframe
    for spec, spec_val in specifiers.items():
        subjects_bouts[spec] = spec_val

    return subjects_bouts


if __name__ == '__main__':
    #% Pre-process_datasets
    # #grab all the scorefiles from our data dir
    # scorefiles = glob.iglob('../../data/nsrr/cfs/polysomnography/annotations-events-nsrr/*.xml')
    #
    # #load the study settings that provide stage conversion map
    # study_settings = yaml.safe_load(open('../../data/nsrr/cfs/mednickdb/nsrr_cfs_study_settings.yaml','r+'))

    med_api = MednickAPI(username=os.environ['mednickapi_username'], password=os.environ['mednickapi_password'])

    #Get the data, so easy :)
    data = med_api.get_data(query='studyid=cfs and filetype=sleep_scoring', format_as='dataframe_single_index')
    print('Got', data.shape[0], 'records')

    specifiers = ['subjectid','visitid']

    data_rows = data.loc[:, specifiers+['sleep_scoring.epochstages']]
    all_rows = data_rows.to_dict(orient='records')

    # parse data into something model-able
    stages_to_consider = ('n2',)
    data = pd.concat([parse_epochstages_to_durations(row.pop('sleep_scoring.epochstages'),
                                                       row, stages_to_consider)
                        for row in all_rows])


    #% Initial look at the data
    # Lets check out the distribution of these bad boys
    # grid = sns.FacetGrid(data, col='stage', sharey=False, sharex=False)
    # grid.map(sns.distplot,'duration', kde=False, bins=np.arange(0,50,2))
    # plt.show()

    #% fit a single model to make sure we are on the right track:
    # exp_dist = Pareto(data) #Modeling duration of sleep bouts as Pareto distributed (1 lambda parameter)
    # model, trace = exp_dist.fit()
    #run MCMC fit:
    # - We want posterior P(theta | x)
    # - Initialize our best guess for how the data was generated (likelihood - P(x | theta)), and what the generating parameters were (prior - P(theta)).
    # - Change those parameters by jumping in a random direction
    # - Evaluate the likelihood of the data under the new parameter values, accept with some probability (e.g. target 50%)
    # pm.traceplot(trace)
    # plt.show()

    #% fit some models, find the best one
    # We will try Gamma, Pareto and Exponential for each stage:
    best_dist_per_stage = {}
    for idx, (stage, stage_data) in enumerate(data.groupby('stage')):
        best_dist = find_best_dist(stage_data, dists=[Gamma, Pareto, Exponential, Weibull])
        best_dist_per_stage[stage] = best_dist
        print('best dist for',stage,'is',best_dist.name)
        axes = pm.traceplot(best_dist.trace)
        plt.show()

    #Save out for later debug
    pickle.dump(best_dist_per_stage,open('../../data/models/best_dist_per_stage.pkl','wb'))

    #% Validate our models
    # Lets sample from our fit models and visually compare this simulated data to the real data
    stage_df_cont = []
    for stage, best_dist in best_dist_per_stage.items():
        ppc_samples = best_dist.sample_posterior_predictive(num_draws_from_params=70)
        stage_df_cont.append(pd.DataFrame({'stage':[stage]*len(ppc_samples),
                                           'duration':ppc_samples,
                                           'data_type':['simulated']*len(ppc_samples)
                                           }))

    data['data_type'] = 'real'
    data_samples_and_real = pd.concat(stage_df_cont+[data], axis=0)

    # % save this out for debug
    data_samples_and_real.to_pickle('../../data/scrap/real_and_sim.pkl')
    data_samples_and_real = pd.read_pickle('../../data/scrap/real_and_sim.pkl')

    #% plotting to compare sim and real:
    grid = sns.FacetGrid(data_samples_and_real, col='stage', row='data_type', sharey=False, sharex=False)
    grid.map(sns.distplot,'duration', kde=False)
    plt.show()

    #% test if distributions are different using frequentist stats
    for stage, stage_data in data_samples_and_real.groupby('stage'):
        data_by_datatype = stage_data.groupby('data_type')
        print(stage,ks_2samp(data_by_datatype.get_group('simulated')['duration'],
                                        data_by_datatype.get_group('real')['duration']))


    #% Assuming we have the distributions right, lets add some effects of age:
    # demographics = pd.read_csv('../../data/nsrr/cfs/datasets/demographics.csv')
    # data = pd.merge(data, demographics, on=['subjectid'], how='inner')
    # data_adults = data.loc[data['age']>=18,:] # look at adults only
    # data_adults['age'] -= 18 #center on 18, bc thats our youngest subject
    #
    # # Lets make sure that is true:
    # sns.regplot('age','duration', data_adults)
    # plt.show()
    #
    # # For example, lets take REM, and fit a model that includes age:
    # data_rem = data_adults.loc[data_adults['stage']=='rem',:]
    # gamma_age_dist = GammaAge(data)
    # rem_age_model, rem_age_trace = gamma_age_dist.fit()
    # pm.traceplot(rem_age_trace)
    # plt.show()
    #
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print(pm.summary(rem_age_trace))
    #
    # gamma_dist = Gamma(data)
    # rem_model, rem_trace = gamma_dist.fit()
    #
    # #Lets tripled check that with LOO:
    # model_rank_dict = {rem_model.model:rem_trace,
    #               rem_age_model:rem_age_trace}
    # print(pm.compare(model_rank_dict, ic='LOO'))



    # #connect to DB
    # med_api = MednickAPI(username=os.environ['mednickapi_username'], password=os.environ['mednickapi_password'])
    #
    # #Get the data, so easy :)
    # data = med_api.get_data('studyid=NSRR_CFS', format_as='dataframe_single_index')
    # print('Got', data.shape[0], 'records')
