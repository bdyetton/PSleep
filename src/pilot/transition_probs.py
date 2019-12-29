from mednickdb_pyapi.mednickdb_pyapi import MednickAPI
from mednickdb_pysleep import defaults
import os
import matplotlib.pyplot as plt
import seaborn as sns
import bootstrapped.bootstrap as bs
import bootstrapped.compare_functions as bs_compare
import bootstrapped.stats_functions as bs_stats
import numpy as np
import pandas as pd

sleep_stages = {
    0:'wake',
    1:'stage1',
    2:'stage2',
    3:'sws',
    4:'rem'
}


def compare_dists(data, y_var, by_var, y_level=None, by_levels=None, ax=None):
    levels = data[by_var].unique()
    if by_levels is not None:
        levels = [lev for lev in levels if lev in by_levels]
    levels_data = []
    for lev in levels:
        level_data = data.loc[data[by_var] == lev, y_var].dropna()
        if y_level is not None:
            level_data = level_data.apply(lambda x: x[y_level]).dropna()
        levels_data.append(level_data.astype(float).values)

    #Runs boostrapped stats test
    is_diff = False
    if len(levels) == 2:
        diff = bs.bootstrap_ab(*levels_data, stat_func=bs_stats.mean, compare_func=bs_compare.percent_change)
        is_diff = (diff.lower_bound > 0 or diff.upper_bound < 0)
        if is_diff:
            sns.set_style("dark")
        else:
            sns.set_style("white")
        diff_msg = 'Difference: \nZero not in CI' if is_diff else 'No Difference: \nZero in CI'
        print(diff, '\n', diff_msg)

    # Plotting
    for lev in levels_data:
        sns.distplot(a=lev, ax=ax)
    ax.text(0.3, 0.5, diff_msg, transform=ax.transAxes, size=16, color='r' if is_diff else 'k')
    plt.title(y_var.split('.')[-1]+' to '+sleep_stages[y_level]+' for the Cleveland Family Study by '+by_var.split('.')[-1])
    plt.ylabel('Probability Density')
    plt.legend(levels)


def investigate_trans_probs_by_demographics(data, sleep_stages_to_consider=defaults.stages_to_consider):
    data = data.drop(['_id', 'sleep_scoring.sourceid', 'visitid', 'datemodified', 'expired'], axis=1)
    data['demographics.age_cat'] = (data['demographics.age'] > 55).map({True: 'Older', False: 'Younger'})
    data['demographics.ethnicity'] = data['demographics.ethnicity'].map({'white ': 'white', 'black ': 'black'}) #anything else will get nan
    demo_cols = ['subjectid', 'demographics.age_cat', 'demographics.ethnicity', 'demographics.sex']
    trans_probs_cols = ['sleep_scoring.trans_prob_from_' + s for s in sleep_stages_to_consider]
    cols_we_care_about = demo_cols + trans_probs_cols
    data = data.loc[:, cols_we_care_about]
    data = data.set_index(demo_cols)

    from_and_to_data_cont = []
    for trans_probs_col in trans_probs_cols:
        from_data = data.loc[:, trans_probs_col]  # keep index
        from_data = from_data.dropna()
        from_and_to_data_np = np.array(from_data.tolist()).astype(float) #not sure why need to conver
        from_and_to_data = pd.DataFrame(from_and_to_data_np, columns=sleep_stages_to_consider)
        from_and_to_data['from_stage'] = trans_probs_col.split('_')[-1]
        from_and_to_data.index = from_data.index
        from_and_to_data = from_and_to_data.reset_index()
        from_and_to_data = from_and_to_data.melt(id_vars=demo_cols+['from_stage'], value_vars=sleep_stages_to_consider, var_name='to_stage', value_name='prob')
        from_and_to_data_cont.append(from_and_to_data)
    all_trans_data = pd.concat(from_and_to_data_cont).reset_index(drop=True)

    # Plot some data
    for by_var in ['demographics.sex', 'demographics.ethnicity', 'demographics.age_cat']:
        data_to_plot = all_trans_data.drop(set(demo_cols)-set([by_var]), axis=1).dropna().reset_index(drop=True)
        sns.catplot(x='to_stage', y='prob', hue=by_var, row="from_stage",
                    data=data_to_plot, kind="violin", split=True, height=1.5, aspect=2.5, legend=False)
        plt.legend(loc='lower right')
        plt.ylim((0, 1))

        for to_and_from_stage, data in data_to_plot.groupby(['from_stage', 'to_stage']):
            from_stage, to_stage = to_and_from_stage[0], to_and_from_stage[1]
            by_data = list(data.groupby(by_var))
            diff = bs.bootstrap_ab(by_data[0][1]['prob'].values, by_data[1][1]['prob'].values,
                                   stat_func=bs_stats.mean, compare_func=bs_compare.percent_change)
            is_diff = (diff.lower_bound > 0 or diff.upper_bound < 0)
            if is_diff:
                plt.gcf().axes[sleep_stages_to_consider.index(from_stage)].text(y=0, x=sleep_stages_to_consider.index(to_stage) - 0.1, s='*', color='r', fontsize=18)
        plt.show()

if __name__ == '__main__':
    med_api = MednickAPI(username=os.environ['mednickapi_username'], password=os.environ['mednickapi_password'])

    #Get the data, so easy :)
    data = med_api.get_data('studyid=NSRR_CFS', format_as='dataframe_single_index')
    print('Got', data.shape[0], 'records')

    investigate_trans_probs_by_demographics(data)
