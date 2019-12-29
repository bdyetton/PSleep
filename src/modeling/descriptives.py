from mednickdb_pyapi.pyapi import MednickAPI
import os
import warnings
file_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(file_dir, '../../data/')
import pandas as pd
import numpy as np
from datetime import timedelta
import sys
import datetime
import seaborn as sns
sns.set_palette(sns.color_palette("Set1", n_colors=8, desat=.5))
import matplotlib.pyplot as plt
from mednickdb_pyapi import parsing_helper
from mednickdb_pysleep import sleep_dynamics

specifiers = ['studyid', 'versionid', 'subjectid', 'visitid', 'sessionid']
epoch_len = 0.5
studyids_to_download = ('cfs','shhs','mros','sof','homepap')
stages_to_consider = ('n1','n2','n3','rem','waso')


def download_and_filter_data():
    med_api = MednickAPI(username=os.environ['mednickapi_username'], password=os.environ['mednickapi_password'])

    # Get the data, so easy :)
    query = 'studyid in [ ' + ','.join(studyids_to_download) + ']'
    sleep_scoring_only = ' and filetype=sleep_scoring'
    healthy_only = ' and health.healthy=true'
    raw_data = med_api.get_data(query + sleep_scoring_only + healthy_only,
                                format_as='dataframe_single_index')

    # plt.figure()
    # sns.regplot(data=raw_data, x='demographics.age', y='sleep_scoring.sleep_efficiency')
    # plt.title('Before')

    raw_data['eff_cutoff'] = -0.15/70*(raw_data['demographics.age']-20) + 0.85
    raw_data = raw_data.loc[raw_data['eff_cutoff'] < raw_data['sleep_scoring.sleep_efficiency'],:]


    # plt.figure()
    # sns.regplot(data=raw_data, x='demographics.age', y='sleep_scoring.sleep_efficiency')
    # plt.title('After')
    # plt.show()

    #subsample mros:
    raw_data = raw_data.loc[~((raw_data['studyid']=='sof') & (np.mod(raw_data['subjectid'],3)==0)),:]
    raw_data = raw_data.loc[~((raw_data['studyid']=='shhs') & (raw_data['demographics.sex']=='F') & (np.mod(raw_data['subjectid'],2)==0)),:]

    print('Got', raw_data.shape[0], 'records')

    return raw_data

def simple_descriptives(raw_data):
    data = raw_data.drop([c for c in raw_data.columns if 'health' in c], axis=1)
    data.columns = [col.split('.')[-1] for col in data.columns]
    demo_data = data.loc[:, ['sex', 'age', 'ethnicity', 'bmi', 'subjectid']]
    print(demo_data['ethnicity'].value_counts())
    plt.figure()
    sns.violinplot(data=demo_data, x='sex', y='age', scale='count')
    plt.figure()
    sns.violinplot(data=demo_data, x='sex', y='bmi', scale='count')
    plt.figure()
    sns.violinplot(data=demo_data, x='ethnicity', y='age', scale='count')
    plt.figure()
    sns.violinplot(data=demo_data, x='ethnicity', y='bmi', scale='count')
    plt.show()

    mins = ['sleep_scoring.mins_in_' + s for s in ('waso', 'n1', 'n2', 'n3', 'rem')] + ['studyid', 'subjectid']
    mins_data = raw_data.loc[:, mins]
    mins_long = parsing_helper.wide_to_long(mins_data, {
        'stage': {'waso': 'waso', 'n1': 'n1', 'n2': 'n2', 'n3': 'n3', 'rem': 'rem'}})
    sns.violinplot(data=mins_long, x='stage', y='sleep_scoring.mins_in')
    plt.show()

    plt.figure()
    sns.regplot(data=raw_data, x='demographics.age', y='sleep_scoring.sleep_efficiency')
    plt.title('After')
    plt.show()


def remove_outliers(data, idxs=None):
    if idxs is not None:
        data = data.set_index(idxs)
    mean, std = data.agg(np.nanmean), data.agg(np.nanstd)
    return data.where(~(data < (mean + 3 * std)), other=np.nan).reset_index()


def get_clean_data():
    epoch_data = pd.read_pickle('../../data/processed/epoch_x_epoch.pkl')
    epoch_data = epoch_data.loc[~epoch_data['bad_epoch'], :]
    epoch_data = epoch_data.loc[epoch_data['current_epoch'].isin(stages_to_consider), :]
    return epoch_data

def plot_freq_across_time(epoch_data):
    epoch_data = epoch_data.loc[epoch_data['age']<45,:]
    band_data = epoch_data.loc[:,
                ['beta', 'delta', 'sigma', 'theta', 'SWA', 'alpha'] + ['current_epoch', 'datetime', 'tau', 'studyid',
                                                                       'subjectid']]

    for stage, stage_data in band_data.groupby('current_epoch'):
        clock_data = stage_data.groupby(['studyid', 'subjectid']).apply(
            lambda x: x.set_index('datetime').resample('15Min').mean().reset_index())
        clock_data = clock_data.drop(['tau', 'subjectid'], axis=1)
        clock_data['datetime'] = clock_data['datetime'].apply(lambda x: x.timestamp())
        n_per_timebin = clock_data.groupby(['datetime']).size()
        n_limit = np.quantile(n_per_timebin, q=0.15)
        n_ok = n_per_timebin[n_per_timebin > n_limit]
        ok_timevals = n_ok.index.tolist()
        clock_data = clock_data.loc[clock_data['datetime'].isin(ok_timevals), :]
        clock_data = clock_data.melt(id_vars=['datetime'],
                                     value_vars=['beta', 'delta', 'sigma', 'theta', 'SWA', 'alpha'],
                                     value_name='power', var_name='band')

        plt.figure()
        sns.lineplot(data=clock_data, x='datetime', y='power', hue='band')
        plt.xlim((ok_timevals[0], ok_timevals[-1]))
        ax = plt.gca()
        xticks = ax.get_xticks()
        ax.set_xticklabels([pd.to_datetime(tm, unit='s').strftime('%H:%M:%S') for tm in xticks],
                           rotation=50)
        plt.title(stage + ': Band Power vs Clocktime')
        plt.savefig(data_dir + '../figures/descriptives/' + stage + ' Band Power vs Clocktime.png',
                    dpi=300, bbox_inches='tight', pad_inches=1)

        tau_data = stage_data.drop('datetime', axis=1)
        n_per_timebin = tau_data.groupby(['tau']).size()
        n_limit = np.quantile(n_per_timebin, q=0.15)
        n_ok = n_per_timebin[n_per_timebin > n_limit]
        ok_timevals = n_ok.index.tolist()
        tau_data = tau_data.loc[tau_data['tau'].isin(ok_timevals), :]
        tau_data = tau_data.melt(id_vars=['tau'], value_vars=['beta', 'delta', 'sigma', 'theta', 'SWA', 'alpha'],
                                 value_name='power', var_name='band')
        plt.figure()
        sns.lineplot(data=tau_data, x='tau', y='power', hue='band')
        plt.title(stage + ': Band Power vs Minutes in Stage (Tau)')
        plt.savefig(data_dir+'../figures/descriptives/'+stage+' Band Power vs Minutes in Stage.png', dpi=300, bbox_inches='tight', pad_inches=1)
    plt.show()

def plot_sleep_features_across_time(epoch_data):
    for feature_groups, stages_for_group in zip([['spindle_count','slow_osc_count'],['rem_event_count']],[['n1','n2','n3'],['rem']]):
        feature_data = epoch_data.loc[epoch_data['current_epoch'].isin(stages_for_group), feature_groups+['current_epoch', 'datetime', 'tau', 'studyid','subjectid']]
        for stage, stage_data in feature_data.groupby('current_epoch'):
            clock_data = stage_data.groupby(['studyid', 'subjectid']).apply(
                lambda x: x.set_index('datetime').resample('15Min').mean().reset_index())
            clock_data = clock_data.drop(['tau','subjectid'], axis=1)
            clock_data['datetime'] = clock_data['datetime'].apply(lambda x: x.timestamp())
            n_per_timebin = clock_data.groupby(['datetime']).size()
            n_limit = np.quantile(n_per_timebin, q=0.15)
            n_ok = n_per_timebin[n_per_timebin > n_limit]
            ok_timevals = n_ok.index.tolist()
            clock_data = clock_data.loc[clock_data['datetime'].isin(ok_timevals),:]
            clock_data = clock_data.melt(id_vars=['datetime'], value_vars=feature_groups, value_name='count', var_name='feature')
            plt.figure()
            sns.lineplot(data=clock_data, x='datetime', hue='feature', y='count')
            plt.xlim((ok_timevals[0],ok_timevals[-1]))
            ax = plt.gca()
            xticks = ax.get_xticks()
            ax.set_xticklabels([pd.to_datetime(tm, unit='s').strftime('%H:%M:%S') for tm in xticks],
                               rotation=50)
            plt.title(stage+': Feature Density vs Clocktime')
            plt.savefig(data_dir + '../figures/descriptives/' + stage + ' Feature Density vs Clocktime.png',
                        dpi=300, bbox_inches='tight', pad_inches=1)

            tau_data = stage_data.drop('datetime', axis=1)
            n_per_timebin = tau_data.groupby(['tau']).size()
            n_limit = np.quantile(n_per_timebin, q=0.15)
            n_ok = n_per_timebin[n_per_timebin > n_limit]
            ok_timevals = n_ok.index.tolist()
            tau_data = tau_data.loc[tau_data['tau'].isin(ok_timevals), :]
            tau_data = tau_data.melt(id_vars=['tau'], value_vars=feature_groups, value_name='count',
                            var_name='feature')
            plt.figure()
            sns.lineplot(data=tau_data, x='tau', hue='feature', y='count')
            plt.title(stage+': Feature Density vs Time in Stage (Tau)')
            plt.savefig(data_dir+'../figures/descriptives/'+stage+' Feature Density vs Minutes in Stage.png', dpi=300, bbox_inches='tight', pad_inches=1)
        plt.show()


def plot_durations_and_trans_p_across_clocktime(data):
    datetime_seconds = data['datetime'].apply(lambda x: x.timestamp())
    time_slices = np.linspace(datetime_seconds.min(), datetime_seconds.max(), 6)
    data['datetime_cat'] = pd.cut(datetime_seconds, bins=time_slices, labels=range(1,len(time_slices)))
    sub_cont = []
    trans_cols = set()
    for idx, data_cut in data.groupby('datetime_cat'):
        for (studyid, subjectid), subject_data in data_cut.groupby(['studyid','subjectid']):
            if len(np.unique(subject_data['current_epoch'])) < 3:
                continue
            _, trans_p, _ = sleep_dynamics.transition_counts(subject_data['current_epoch'], normalize=True)
            bout_duration = sleep_dynamics.bout_durations(subject_data['current_epoch'])
            sub_data_out = pd.Series({'subjectid':subjectid,
                                      'studyid':studyid,
                                      'datetime_cat': idx,
                                      'age': subject_data['demographics.age'].iloc[0]})
            for (c,n), p in np.ndenumerate(trans_p):
                sub_data_out['trans_p_'+str(c)+'->'+str(n)] = p
                trans_cols.add('trans_p_'+str(c)+'->'+str(n))

            for i, b in bout_duration.items():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                sub_data_out['bout_duration_'+i] = np.nanmean(b)
            sub_cont.append(sub_data_out)
    all_data = pd.concat(sub_cont, axis=1).T
    durations_data = all_data.drop(trans_cols, axis=1)
    durations_data_piv = parsing_helper.wide_to_long(durations_data, {'stage': {'waso': 'waso', 'n1': 'n1', 'n2': 'n2', 'n3': 'n3', 'rem': 'rem'}})
    durations_data_piv = durations_data_piv.apply(lambda x: pd.to_numeric(x, errors='ignore'))
    sns.lineplot(data=durations_data_piv.dropna(), x='datetime_cat', hue='stage', y='bout_duration')
    plt.show()

    n3_data = durations_data_piv.loc[durations_data_piv['stage']=='n3',:]
    sns.lineplot(data=n3_data.dropna(), x='datetime_cat', hue='studyid', y='bout_duration')
    plt.title('n3')
    plt.show()

    n3_data = durations_data_piv.loc[durations_data_piv['stage']=='rem',:]
    sns.lineplot(data=n3_data.dropna(), x='datetime_cat', hue='studyid', y='bout_duration')
    plt.title('rem')
    plt.show()

    # cfs = durations_data_piv.loc[(durations_data_piv['studyid']=='cfs') & (durations_data_piv['stage']=='n3'),:]
    # age_slices = np.linspace(cfs['age'].min()-0.01, cfs['age'].max(), 4)
    # cfs['age_cat'] = pd.cut(cfs['age'], bins=age_slices, labels=range(1,len(age_slices)))
    # cfs = cfs.apply(lambda x: pd.to_numeric(x, errors='ignore'))
    # sns.lineplot(data=cfs.dropna(), x='datetime_cat', hue='age_cat', y='bout_duration')
    # plt.show(block=True)


def simple_descriptives(data, vars_to_plot, hue, n_cols=2):
    side_len = int(np.ceil(len(vars_to_plot)/n_cols))
    f, axes = plt.subplots(side_len, n_cols, figsize=(7, 7))
    if len(axes.shape)==1:
        axes = axes[np.newaxis,:]
    bins = [list(range(18,90,2)), [a/2 for a in range(20,28)]]


    for i, var in enumerate(vars_to_plot):
        cols = [c for c in [var, 'studyid', 'subjectid', hue] if c is not None]
        data_to_plot = data.loc[:,cols].drop_duplicates().dropna()
        if hue is not None:
            for hue_i, data_hue in data_to_plot.groupby(hue):
                ax = axes[int(np.floor(i / n_cols)), int(np.mod(i, n_cols))]
                sns.distplot(data_hue[var], ax=ax, kde=False, label=hue_i, norm_hist=True, bins=bins[i]) # bins=list(range(0,20))
            ax.legend()
        else:
            sns.distplot(data_to_plot[var], ax=axes[int(np.floor(i / n_cols)), int(np.mod(i, n_cols))])
    return axes


if __name__ == '__main__':
    ok_studies= ['mass_ss2', 'mass_ss3', 'mass_ss4', 'mass_ss5', 'shhs', 'wamsley_future', 'wamsley_ken', 'wamsley_r21']
    #raw_data = download_and_filter_data()
    #epoch_data = get_clean_data()
    #epoch_data = epoch_data.loc[epoch_data['studyid'].isin(ok_studies),:]
    #sns.stripplot(data=epoch_data, x='slow_osc_count',y='studyid')
    #simple_descriptives(raw_data)
    #plt.show()
    # epoch_data = epoch_data.loc[epoch_data['bout_idx']=='1',:]
    # #epoch_data['lights_off_datetime'] += raw_data['sleep_scoring.sleep_latency_mins'].apply(lambda x: datetime.timedelta(seconds=x*60))
    # #epoch_data['lights_off_datetime']  = epoch_data['lights_off_datetime'].apply(lambda x: x.hour+x.minute/60 if x.hour+x.minute/60 > 12 else x.hour+x.minute/60 + 24)
    # axes = simple_descriptives(epoch_data, ['age','clocktime'], hue='sex')
    # axes[0, 0].set_xlabel('Age (years)')
    # labels = axes[0, 1].get_xticks()
    # labels[labels > 12] = labels[labels > 12] - 12
    # axes[0, 1].set_xticklabels(labels=['%0.0f' % l for l in labels])
    # axes[0, 1].set_xlabel('Sleep Onset Time (hour of day)')
    # plt.show()

    # epoch_data = get_clean_data()
    # epoch_data.loc[]
    # epoch_data = epoch_data.loc[epoch_data['bout_idx'] == '1', :]
    # axes = simple_descriptives(epoch_data, ['timeslept'], hue='sex')
    # axes[0, 0].set_xlabel('Age (years)')

    # cols = ['studyid','subjectid','current_epoch','spindle_count','slow_osc_count','rem_event_count']
    # density_data = epoch_data[cols].groupby(['studyid', 'subjectid','current_epoch']).mean()*2
    # density_data.columns = ['spindle_density','slow_osc_density', 'rem_density']
    # density_data = density_data.reset_index()
    # density_data.loc[~density_data['current_epoch'].isin(['n2','n3']),'spindle_density'] = np.nan
    # density_data.loc[~density_data['current_epoch'].isin(['n2','n3']),'slow_osc_density'] = np.nan
    # density_data.loc[density_data['current_epoch'] != 'rem','rem_density'] = np.nan
    # density_data = density_data.loc[density_data['current_epoch'].isin(['n2','n3','rem']),:]
    # axes = simple_descriptives(density_data.reset_index(), ['spindle_density','slow_osc_density', 'rem_density'], hue='current_epoch')
    # axes[0,0].set_xlabel('Spindle Density')
    # axes[0,1].set_xlabel('Slow Oscillation Density')
    # axes[1,0].set_xlabel('REM Density')
    # plt.show()
    # print(epoch_data['studyid'].value_counts())
    # shhs_data = epoch_data.loc[epoch_data['studyid']=='shhs',:]
    # plot_durations_and_trans_p_across_clocktime(shhs_data)

    # plot_freq_across_time(epoch_data)
    #plot_sleep_features_across_time(epoch_data)









