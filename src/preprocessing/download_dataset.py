"""A script to create the data expected for modeling work"""
from mednickdb_pyapi.pyapi import MednickAPI
import os
import warnings
file_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(file_dir, '../../data/')
import pandas as pd
import numpy as np
from io import BytesIO
import datetime
from mednickdb_pysleep import pysleep_utils

specifiers = ['studyid', 'versionid', 'subjectid', 'visitid', 'sessionid']
epoch_len = 0.5
studyids_to_download = ('ucddb','wamsley_ken','wamsley_r21','wamsley_future',
                        'mass_ss1','mass_ss2','mass_ss3','mass_ss4','mass_ss5',
                        'cfs','homepap','mros','sof','shhs')

stages_to_consider = ('n1','n2','n3','rem','waso')


def remove_outliers(data, idxs=None):
    if idxs is not None:
        data_idxs = data.loc[:, idxs]
        data_ = data.drop(idxs, axis=1)
    else:
        data_ = data
    mean, std = data_.agg(np.nanmean), data_.agg(np.nanstd)
    data_naned = data_.where(((mean - 3 * std) < data_) & (data_ < (mean + 3 * std)), other=np.nan)
    if idxs is not None:
        return pd.concat([data_idxs, data_naned], axis=1)
    else:
        return data_naned


def download_and_filter_data():
    med_api = MednickAPI(username=os.environ['mednickapi_username'], password=os.environ['mednickapi_password'])

    # Get the data, so easy :)
    query = 'studyid in [' + ','.join(studyids_to_download) + ']'
    sleep_scoring_only = ' and filetype=sleep_scoring'
    healthy_only = ' and health.healthy=true'
    raw_data = med_api.get_data(query + sleep_scoring_only + healthy_only,
                                format_as='dataframe_single_index')

    # plt.figure()
    # sns.regplot(data=raw_data, x='demographics.age', y='sleep_scoring.sleep_efficiency')
    # plt.title('Before')

    raw_data['had_unknowns'] = raw_data['sleep_scoring.epochstages'].apply(lambda x: 'unknown' in x)
    print('There are',sum(raw_data['had_unknowns']),'records with unknowns')
    raw_data['sleep_scoring.epochstages'] = raw_data['sleep_scoring.epochstages'].apply(pysleep_utils.fill_unknown_stages)

    print('Got', raw_data.shape[0], 'records')
    raw_data['eff_cutoff'] = -0.15/70*(raw_data['demographics.age']-20) + 0.85
    raw_data = raw_data.loc[raw_data['eff_cutoff'] < raw_data['sleep_scoring.sleep_efficiency'],:]
    print('Got', raw_data.shape[0], 'after eff cutoff')

    raw_data = raw_data.loc[raw_data['demographics.age'] < 90,:]
    raw_data = raw_data.loc[raw_data['demographics.age'] > 18,:]
    print('Got', raw_data.shape[0], 'after age cutoff')

    def fix_datetimes(row):
        start_datetime = row['sleep_scoring.start_datetime'].replace(day=1, month=1, year=2000)
        start_datetime = start_datetime + datetime.timedelta(seconds=row['sleep_scoring.lights_off_secs'])
        start_datetime = start_datetime \
            if start_datetime > datetime.datetime(year=2000, month=1, day=1, hour=14) \
            else start_datetime + datetime.timedelta(seconds=60 * 60 * 24)
        return start_datetime

    raw_data['lights_off_datetime'] = raw_data.apply(fix_datetimes, axis=1)
    raw_data['lights_on_datetime'] = raw_data.apply(lambda row:
                                                    row['lights_off_datetime']
                                                    +datetime.timedelta(seconds=row['sleep_scoring.lights_on_secs']),
                                                    axis=1)
    latency = raw_data['sleep_scoring.sleep_latency_mins'].apply(lambda x: datetime.timedelta(seconds=x*60))
    raw_data = raw_data.loc[raw_data['lights_off_datetime'] + latency > datetime.datetime(year=2000, month=1, day=1, hour=22),:]
    raw_data = raw_data.loc[raw_data['lights_off_datetime'] + latency < datetime.datetime(year=2000, month=1, day=2, hour=2),:]
    raw_data = raw_data.loc[raw_data['lights_on_datetime'] < datetime.datetime(year=2000, month=1, day=2, hour=9),:]
    print('Got', raw_data.shape[0], 'after timing cutoff')

    age_bins = [0, 35, 50, 65, 80, 95]
    raw_data['age_cat'] = pd.cut(raw_data['demographics.age'], bins=age_bins, labels=range(len(age_bins)-1))

    def balanced_sample(df, grouper='studyid', total=190):
        if df.shape[0] < total:
            return df
        studyids = df[grouper].unique()
        if grouper=='studyid':
            studyids = sorted(studyids, key=lambda x: studyids_to_download.index(x))
        cont = []
        studys_rem = len(studyids)
        for study, study_data in df.groupby(grouper):
            n_per_study = int(total / studys_rem)
            if study_data.shape[0] < n_per_study:
                total -= study_data.shape[0]
                cont.append(study_data)
            else:
                total -= n_per_study
                cont.append(study_data.sample(n=n_per_study))
            studys_rem -= 1
        df_out = pd.concat(cont, axis=0)
        return df_out

    raw_data = raw_data.groupby(['demographics.sex','age_cat']).apply(balanced_sample).reset_index(drop=True)

    print('Got', raw_data.shape[0], 'after balancing cutoff')

    # plt.figure()
    # sns.regplot(data=raw_data, x='demographics.age', y='sleep_scoring.sleep_efficiency')
    # plt.title('After')
    # plt.show()

    #print(raw_data.groupby(['demographics.sex', 'age_cat', 'studyid']).size())
    print(raw_data.groupby(['age_cat','demographics.sex']).size())

    #subsample mros:
    #raw_data = raw_data.loc[~((raw_data['studyid']=='sof') & (np.mod(raw_data['subjectid'],3)==0)),:]
    #raw_data = raw_data.loc[~((raw_data['studyid']=='shhs') & (raw_data['demographics.sex']=='F') & (np.mod(raw_data['subjectid'],2)==0)),:]

    return raw_data

def download_and_format_all_data():
    raw_data = download_and_filter_data()
    stage_data = parse_stage_data(raw_data)
    stage_data.to_pickle(data_dir + 'processed/epoch_x_epoch_scoring_only.pkl')
    all_data = get_sleep_features(stage_data)
    outlier_cols = ['delta','theta','SWA','alpha','beta', 'sigma']
    idx_cols = [col for col in all_data.columns if col not in outlier_cols]
    all_data = all_data.groupby(['studyid','subjectid','current_epoch']).apply(lambda x: remove_outliers(x, idx_cols)).reset_index(drop=True)
    all_data = all_data.groupby(['current_epoch']).apply(lambda x: remove_outliers(x, idx_cols+['studyid','subjectid'])).reset_index(drop=True)
    all_data = all_data.loc[:, ~all_data.columns.duplicated()]
    all_data.to_pickle(data_dir + 'processed/epoch_x_epoch.pkl')

def parse_stage_data(raw_data) -> pd.DataFrame:
    """
    Download stage data (epochstages) for all subjects in the given studyids and parse to the epoch by epoch
    format suitable for modeling. TODO slice out only healthy subjects when the true/false comparision on mednickdb_pyapi works.
    :param studyids: which studyids to download and parse for.
    :return: a dataframe where each row is an epoch for a subject in a study.
    Variables (columns) are: current_epoch, next_epoch, previous_bout, demographics, time, tau (time in stage)
    """
    data = raw_data

    #drop unnessary cols
    to_keep = ['demographics.'+a for a in ['age','sex','ethnicity']] + \
        ['sleep_scoring.epochstages', 'std_sleep_eeg.epochs_with_artifacts', 'lights_off_datetime'] + \
        ['visitid','studyid','subjectid']
    to_drop = [col for col in data.columns if col not in to_keep] #dont need any of the scoring stuff for modeling, we'll build our own
    data = data.drop(to_drop, axis=1)
    data = data.dropna(axis=1, how='all')
    data['std_sleep_eeg.epochs_with_artifacts'] = data['std_sleep_eeg.epochs_with_artifacts'].apply(lambda x: [] if isinstance(x,float) else x)
    data['visitid'] = data['visitid'].fillna(-1)
    data['demographics.ethnicity'] = data['demographics.ethnicity'].fillna('unknown')
    cols = [col for col in data.columns if col not in ['subjectid','visitid','studyid',
                                                       'std_sleep_eeg.epochs_with_artifacts',
                                                       'demographics.ethnicity',
                                                       'demographics.age',
                                                       'demographics.sex']]
    data = data.dropna(subset=cols)
    data = data.rename({'demographics.ethnicity':'ethnicity', 'demographics.age':'age', 'demographics.sex':'sex'}, axis=1)
    print('Records after dropna is:', data.shape[0])
    #fix datatype of cols
    data_specifiers = [k for k in specifiers if k in data.columns] # convert types

    # parse to desired format
    epoch_data_all = pd.concat([parse_epochstages_to_series(row) for idx, row in data.iterrows()])
    epoch_data_all = epoch_data_all.reset_index(drop=True).dropna()

    data = data.drop('sleep_scoring.epochstages', axis=1)

    # merge all the subject data with the epoch data we just created (i.e. attached sex, age, etc to each epoch)
    data_all = pd.merge(data, epoch_data_all, on=data_specifiers, how='inner')

    data_all = data_all.drop('std_sleep_eeg.epochs_with_artifacts', axis=1)

    return data_all


def parse_epochstages_to_series(row):
    """
    Parse epochstages i.e. ['waso','waso','1','1','2', etc]
    :param row: a subjects row, with sleep_scoring.epochstages and specifiers
    :return:
    """
    base_specifiers = {k: row[k] for k in specifiers if k in row}
    epoch_stages = row['sleep_scoring.epochstages']
    bad_epochs = row['std_sleep_eeg.epochs_with_artifacts']
    bad_epochs_df = pd.DataFrame({'stage_idx':bad_epochs})
    bad_epochs_df['bad_epoch'] = True

    if not isinstance(epoch_stages, list):
        base_specifiers.update(
            {'current_epoch': None, 'next_epoch': None, 'previous_bout': None, 't': None, 'tau': None})
        return pd.DataFrame(base_specifiers, index=[0])
    num_stages = len(epoch_stages)
    base_specifiers = {k: [v] * num_stages for k, v in base_specifiers.items()}
    next_epoch = epoch_stages[1:] + ['wase']
    transitions = np.insert(np.where(np.array(epoch_stages) != np.array(next_epoch)), 0, -1)
    if transitions[-1] != len(epoch_stages):
        transitions = np.append(transitions, (len(epoch_stages) - 1))
    previous_bouts = np.empty_like(epoch_stages)
    tau = np.empty_like(epoch_stages)
    bout_idx = np.empty_like(epoch_stages)
    for bout, (s, e) in enumerate(zip(transitions, transitions[1:])):
        previous_bouts[s + 1:e + 1] = 'wbso' if s == -1 else epoch_stages[s]
        len_bout = e - s
        tau[s + 1:e + 1] = np.arange(1, len_bout+1)
        bout_idx[s + 1:e + 1] = bout
    # previous_bouts[-1] = epoch_stages[-2]
    start_datetime = row['lights_off_datetime'] - datetime.datetime(year=2000, month=1, day=1, hour=12)
    start_datetime_hrs = start_datetime.total_seconds()/60/60
    clocktime = [start_datetime_hrs + x for x in np.arange(0, len(epoch_stages))*epoch_len/60]
    total_time_slept = 0
    timeslept = []
    for epoch in epoch_stages:
        if epoch in stages_to_consider and epoch!='waso':
            total_time_slept+=epoch_len/60
        timeslept.append(total_time_slept)
    datetime_ = [row['lights_off_datetime'] + datetime.timedelta(seconds=epoch_len*60*x) for x in np.arange(0, len(epoch_stages))]
    epoch_data = {
        'current_epoch': epoch_stages,
        'next_epoch': next_epoch,
        'previous_bout': previous_bouts,
        'clocktime': clocktime,
        'timeslept': timeslept,
        'datetime': datetime_,
        'stage_idx': np.arange(0, len(epoch_stages)),
        'tau': tau.astype(int)*epoch_len,
        'tau_idx':tau.astype(int),
        'bout_idx':bout_idx
    }
    epoch_data.update(base_specifiers)
    epoch_data = pd.DataFrame(epoch_data)
    epoch_data = pd.merge(epoch_data, bad_epochs_df, on='stage_idx', how='left')
    epoch_data.loc[pd.isna(epoch_data['bad_epoch']),'bad_epoch'] = False

    return epoch_data


def get_sleep_features(stage_data):
    med_api = MednickAPI(username=os.environ['mednickapi_username'], password=os.environ['mednickapi_password'])
    sub_and_vis = stage_data.loc[:,['studyid','subjectid', 'visitid']].drop_duplicates()
    full_cont = []
    got_features = sub_and_vis.shape[0]
    for idx, row in sub_and_vis.iterrows():
        # power
        vis_addon = ' and visitid=' + str(row['visitid']) if row['visitid']!=-1 else ''
        files = med_api.get_files('studyid='+row['studyid'] + ' and fileformat=band_power and subjectid='
                                  + str(row['subjectid']) + vis_addon)
        if len(files) == 0:
            got_features -= 1
            continue
        assert len(files) == 1
        band_power = pd.read_csv(BytesIO(med_api.download_file(fid=files[0]['_id'])))
        band_power['power'] = np.log(band_power['power'])
        band_power = band_power.loc[band_power['stage'].isin(stages_to_consider), :]
        if 'Unnamed: 0' in band_power.columns:
            band_power = band_power.drop('Unnamed: 0', axis=1)
        if 'quartile' in band_power.columns:
            band_power = band_power.drop('quartile', axis=1)
        band_power = band_power.loc[band_power['chan']=='C3'].drop(['chan'], axis=1)
        band_power = band_power.pivot_table(values='power', columns='band', index=['stage', 'stage_idx']).reset_index()

        #sleep features
        files = med_api.get_files('studyid='+row['studyid'] + ' and fileformat=sleep_features and subjectid='
                                  + str(row['subjectid']) + vis_addon)
        if len(files) == 0:
            got_features -= 1
            continue
        assert len(files) == 1
        sleep_features = pd.read_csv(BytesIO(med_api.download_file(fid=files[0]['_id'])))
        if sleep_features.shape[0] > 0:
            if 'Unnamed: 0' in sleep_features.columns:
                sleep_features = sleep_features.drop('Unnamed: 0', axis=1)
            if 'quartile' in sleep_features.columns:
                sleep_features = sleep_features.drop('quartile', axis=1)
            sleep_features = sleep_features.loc[(sleep_features['chan'] == 'C3') | (sleep_features['description']=='rem_event')].drop(['chan', 'onset'], axis=1)
        if sleep_features.shape[0] == 0:
            print('Sub',row['studyid'],'-',row['subjectid'],'has no C3 features')
            sub_data = band_power
        else:
            features_cont = []
            for feature_name, feature in sleep_features.groupby('description'):
                # if feature_name == 'spindle':
                #     dat = med_api.get_data('studyid=' + row['studyid'] + ' and filetype=sleep_features and subjectid='
                #                      + str(row['subjectid']) + vis_addon)
                #     dat_spinds = np.nansum(dat.loc[:, ['sleep_features.'+p+'_C3_spindle_count' for p in ['n1','n2','n3']]].values[0])
                #     if feature.shape[0] != dat_spinds:
                #         print('spindle missmatch for ', row['studyid'], row['subjectid'], 'resetting')
                #         print('diff is', feature.shape[0], ' vs data:',dat_spinds)
                #         files = med_api.get_files(
                #             'studyid=' + row['studyid'] + ' and filetype=std_sleep_eeg and subjectid='
                #             + str(row['subjectid']) + vis_addon)
                #         # if len(files)>0:
                #         #     med_api.update_parsed_status(fid=files[0]['_id'], status=False)
                feature = feature.dropna(how='all', axis=1)
                feature = feature.drop(['description'], axis=1)
                if 'coupled_before' in feature.columns:
                    feature['coupled_before'] = feature['coupled_before'].map({False:0, True:1})
                    feature['coupled_after'] = feature['coupled_after'].map({False:0, True:1})
                feature_gb = feature.groupby(['stage_idx','stage'])
                features_agg = feature_gb.agg(np.nanmean)
                features_agg['count'] = feature_gb.count()['duration']
                features_agg.columns = [feature_name + '_'+ col for col in features_agg.columns]
                features_cont.append(features_agg.reset_index())
            if len(features_cont) == 3:
                feature_data_processed = pd.merge(features_cont[0], features_cont[1], on=['stage_idx','stage'], how='outer')
                feature_data_processed = pd.merge(feature_data_processed, features_cont[2], on=['stage_idx','stage'], how='outer')
            elif len(features_cont) == 2:
                feature_data_processed = pd.merge(features_cont[0], features_cont[1], on=['stage_idx','stage'], how='outer')
            else:
                feature_data_processed = features_cont[0]
            sub_data = pd.merge(band_power, feature_data_processed, on=['stage_idx'], how='left')
            assert np.all(sub_data.dropna()['stage_x'] == sub_data.dropna()['stage_y'])
            sub_data['stage'] = sub_data['stage_x']
            sub_data = sub_data.drop(['stage_x','stage_y'], axis=1)
        sub_data['subjectid'] = row['subjectid']
        sub_data['visitid'] = row['visitid']
        sub_data['studyid'] = row['studyid']
        full_cont.append(sub_data)
    print('Got features for', got_features,'records')
    if len(full_cont) == 0:
        raise ValueError('All subjects are missing features :(')
    all_data = pd.concat(full_cont, axis=0, sort=True)
    merged_all = pd.merge(stage_data, all_data,  on=['studyid', 'subjectid','visitid','stage_idx'], how='left')
    assert np.all(merged_all.dropna()['stage'] == merged_all.dropna()['current_epoch'])
    merged_all = merged_all.drop('stage', axis=1)
    merged_all.loc[~merged_all['bad_epoch'], 'slow_osc_count'] = merged_all.loc[~merged_all['bad_epoch'], 'slow_osc_count'].fillna(0)
    merged_all.loc[~merged_all['bad_epoch'], 'spindle_count'] = merged_all.loc[~merged_all['bad_epoch'], 'spindle_count'].fillna(0)
    merged_all.loc[~merged_all['bad_epoch'], 'rem_event_count'] = merged_all.loc[~merged_all['bad_epoch'], 'rem_event_count'].fillna(0)
    merged_all = merged_all.sort_values(['studyid','subjectid','visitid','stage_idx'])
    return merged_all


if __name__ == "__main__":
    download_and_format_all_data()


