import pymc3 as pm
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import mode
import pandas as pd
import seaborn as sns
sns.set(font_scale=1)
sns.set_style("whitegrid", {'axes.grid' : False})
from mednickdb_pysleep.pysleep_utils import pd_to_xarray_datacube
sns.set_palette(sns.color_palette("Set1", n_colors=8, desat=.5))
from seaborn.relational import scatterplot
# Bigger than normal fonts


def model_parameters(trace, varnames=None):
    summary_df = pm.summary(trace, varnames=varnames)
    print(summary_df)
    axs = pm.traceplot(trace, varnames=varnames)
    return summary_df, axs


def stage_parameters(trace, stage_param_names, stage_map, label_plot=True):
    stage_map = {v:k for k,v in stage_map.items()}
    _, axs = model_parameters(trace, stage_param_names)
    for param in stage_param_names:
        if trace[param].dtype == np.float64:
            means = extract_mean_as_array(trace, param, 'df')
            print(param, ':\n', sep='')
            for idx, row in means.iterrows():
                stage_str = [stage_map[row[level]] for level in row.index if level != param]
                print(stage_str, row[param])
                if label_plot:
                    axs[0, 0].axvline(row[param], linewidth=0.5, linestyle='--', color='r')
                    axs[0,0].text(row[param],
                                  (axs[0,0].get_ylim()[1] - axs[0,0].get_ylim()[0])/np.random.normal(loc=2, scale=0.5),
                                  '_'.join(stage_str), rotation=45)

    plt.show()




def extract_mode_as_array(trace, var='z', astype='array'):

    def trace_mode(x):
        return pd.Series(mode(x).mode[0], name='mode')

    df = pm.summary(trace, stat_funcs=[trace_mode], varnames=[var])
    df = df.reset_index()

    def split_fun(x):
        if '__' in x:
            return [int(x) for x in x.split('__')[1].split('_')]
        else:
            return [0]

    df['var type'] = df['index'].apply(lambda x: x.split('__')[0])
    df = df.loc[df['var type'] == var, :]
    var_idxs = df['index'].apply(split_fun)
    indexs = np.stack(var_idxs)
    if astype == 'array':
        sizes = indexs.max(axis=0) + 1
        var_array = df['mode'].copy().values.reshape(sizes)
        return var_array
    else:
        df_out = pd.DataFrame(np.concatenate([indexs, np.expand_dims(df['mode'].values, -1)], axis=1))
        df_out.columns = list(df_out.columns[:-1]) + [var]
        return df_out


def extract_mean_as_array(trace, var='z', astype='array'):
    df = pm.summary(trace)
    df = df.reset_index()

    def split_fun(x):
        if '__' in x:
            return [int(x) for x in x.split('__')[1].split('_')]
        else:
            return [0]

    df['var type'] = df['index'].apply(lambda x: x.split('__')[0])
    df = df.loc[df['var type'] == var, :]
    var_idxs = df['index'].apply(split_fun)
    indexs = np.stack(var_idxs)
    if astype == 'array':
        sizes = indexs.max(axis=0)+1
        var_array = df['mean'].copy().values.reshape(sizes)
        return var_array
    else:
        df_out = pd.DataFrame(np.concatenate([indexs, np.expand_dims(df['mean'].values, -1)], axis=1))
        idx_cols = [str(i) for i in df_out.columns[:-1]]
        df_out.columns = idx_cols+[var]
        if astype == 'xarray':
            return pd_to_xarray_datacube(df_out, idx_cols, value_col=var)
        else:
            return df_out


def gen_data_for_plot(data, x, z=None, rand_sample_vars=[], mean_sample_vars=[], const_vars={}, stages='balanced', nstages=5, samples_per_x_range=500, truncate_to_percentile=0):
    """
    Generate some data that we can use to plot poterior/param values for
    :param data: data used to train model, so that levels of x are known
    :param x: continous data for x axis
    :param z: catergorical data for y axis
    :param rand_sample_vars:
    :return:
    """
    data_points = data.copy()
    unq_x = data[x].unique()
    if len(unq_x) < 7: #catergorical
        x_data = data[x].sample(samples_per_x_range).values
    else:
        if truncate_to_percentile:
            x_data = np.linspace(np.percentile(data[x],truncate_to_percentile), np.percentile(data[x],100-truncate_to_percentile), samples_per_x_range)
        else:
            x_data = np.linspace(data[x].min(), data[x].max(), samples_per_x_range)
    df = pd.DataFrame({x:x_data})
    for var in mean_sample_vars:
        var_mean = data[var].mean(skipna=True)
        var_std = data[var].std(skipna=True)
        df[var] = var_mean
        data_points = data_points.loc[(var_mean-var_std<data_points[var]) & (data_points[var]<var_mean+var_std),:]

    for var in rand_sample_vars:
        df[var] = np.random.choice(data[var], size=(samples_per_x_range, ))

    for var, val in const_vars.items():
        df[var] = [val] * samples_per_x_range
        if 'consider' not in var:
            var_std = data[var].std(skipna=True)
            data_points = data_points.loc[(val - var_std < data_points[var]) & (data_points[var] < val + var_std), :]

    if stages == 'balanced':
        df_stages = pd.DataFrame({'current_epoch':list(range(nstages))})
        n_reps = int(np.ceil(df.shape[0]/df_stages.shape[0]))
        df_stages = pd.concat([df_stages]*n_reps, axis=0).iloc[0:samples_per_x_range,:].reset_index(drop=True)
        df_stages = df_stages.sample(frac=1).reset_index(drop=True)
        df = pd.concat([df, df_stages], axis=1, sort=False)

    if z is not None:
        data_cont = []
        unique_z = data[z].unique()
        if len(unique_z) >= 7:  # make cont into categorical
            unique_z = np.linspace(data[z].min(), data[z].max(), 7)
            unique_z += (unique_z[1] - unique_z[0])/2
            unique_z = unique_z[:-1]

        for z_val in unique_z:
            new_df = df.copy()
            new_df[z] = z_val
            data_cont.append(new_df)
        df = pd.concat(data_cont, axis=0)

    return df, data_points



def pairplot_divergence(trace, ax=None, divergence=True, color='C3', divergence_color='C2'):
    theta = trace.get_values(varname='theta', combine=True)[:, 0]
    logtau = trace.get_values(varname='tau_log__', combine=True)
    if not ax:
        _, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(theta, logtau, 'o', color=color, alpha=.5)
    if divergence:
        divergent = trace['diverging']
        ax.plot(theta[divergent], logtau[divergent], 'o', color=divergence_color)
    ax.set_xlabel('theta[0]')
    ax.set_ylabel('log(tau)')
    ax.set_title('scatter plot between log(tau) and theta[0]');
    return ax


def plot_vars(mod, data, x, y, facet_row=None, facet_col=None, hue=None, style=None, y_levels=None, y_level_name='set_y_level_name',
              maps=None, data_points=None, mean_center_means=None, vars_to_label=None,
              num_draws_from_params=100, out_of_sample=True, combine_trace=False, legend='full', points_alpha=0.01):
    for var_name in mod.input_vars:
        if 'consider' in var_name:
            mod.input_vars[var_name].set_value(data[var_name].iloc[0])
        else:
            mod.input_vars[var_name].set_value(data[var_name])


    vars_ppc = [v for v in [x, y, hue, facet_col, facet_row, style] if v is not None and v != y_level_name]

    pps = mod.sample_posterior_predictive(vars=vars_ppc, num_draws_from_params=num_draws_from_params, out_of_sample=out_of_sample)

    df_ppc_cont = []
    for var in vars_ppc:
        label = [var] if (y_levels is None) or (var!=y) else y_levels
        df_ppc_var_cont = []
        for ppc_idx, ppc_sample in enumerate(pps[var]):
            df_ppc_var = pd.DataFrame(ppc_sample, columns=label)
            df_ppc_var['ppc_idx'] = ppc_idx
            df_ppc_var_cont.append(df_ppc_var)
        df_ppc = pd.concat(df_ppc_var_cont, axis=0)
        if var != vars_ppc[-1]:
            df_ppc = df_ppc.drop('ppc_idx', axis=1)
        df_ppc_cont.append(df_ppc)
    df = pd.concat(df_ppc_cont, axis=1)

    if maps:
        for col in df.columns:
            if col in maps:
                df[col] = df[col].map({v:k for k,v in maps[col].items()})

    if y_levels is not None:
        vars_ppc.remove(y)
        df = df.melt(id_vars=['ppc_idx']+vars_ppc, value_vars=y_levels, var_name=y_level_name, value_name=y).reset_index()
        hue = hue if y_level_name == facet_row or y_level_name == facet_col else y_level_name

    # if mean_center_means is not None:
    #     for var in mean_center_means:
    #         df[var] += df[var]*mean_center_means[var]['sd']+mean_center_means['mean']

    # df_prev = df.drop(['index', 'ppc_idx'], axis=1).groupby(
    #     ['previous_bout', 'current_epoch', 'feature']).mean().reset_index()
    # df_prev.to_csv(
    #     '../../data/processed/previous_bout_feature.csv')

    # df_prev = pd.read_csv('../../data/processed/previous_bout_feature.csv')
    #
    # df_current = df.drop(['index', 'ppc_idx'], axis=1).groupby(
    #     ['current_epoch', 'feature']).mean().reset_index()
    # df_current.to_csv('../../data/output/current_bout_feature.csv')
    #
    # df_merged = pd.merge(df_current,df_prev, on=['current_epoch','feature'])
    # df_merged['Difference when inc previous stage'] = df_merged['feature_rate_p_x'] - df_merged['feature_rate_p_y']
    # df_merged['Trans P when marginalizing over previous stage'] = df_merged['feature_rate_p_x']
    # df_merged['Trans P inc previous stage'] = df_merged['feature_rate_p_y']
    # df_merged.drop(['feature_rate_p_y','feature_rate_p_x','Unnamed: 0'], axis=1).to_csv('../../data/output/full_feature_p.csv')


    if combine_trace:
        g = sns.relplot(data=df, x=x, y=y, hue=hue, kind='line', col=facet_col, row=facet_row, style=style, facet_kws={'sharex':False, 'sharey':False},
                        legend=legend)#hue_order=['F','M'], hue_order=['waso','wbso','n1','n2'])#, row_order=['waso','n1','n2','n3','rem'], col_order=['waso','n1','n2','n3','rem'])
    else:
        g = sns.relplot(data=df, x=x, y=y, hue=hue, kind='line', col=facet_col, row=facet_row, style=style, estimator=None, units='ppc_idx', alpha=0.1,
                        facet_kws={'sharex': False, 'sharey': False}, legend=legend, row_order=['waso','n1','n2','n3','rem'], col_order=['F','M'])#, hue_order=['n3','waso','n1','n2']) #for running with rem & previous bout
    if mean_center_means is not None and x in mean_center_means:
        for ax in g.axes.flatten():
            labels = np.round(ax.get_xticks()*mean_center_means[x]['std']+mean_center_means[x]['mean'],0)
            if x=='clocktime':
                labels[labels>12] = labels[labels>12]-12
                ax.set_xticklabels(labels=['%0.0f' % l for l in labels])
            else:
                ax.set_xticklabels(labels=labels)


    #plt.ylim([0, 10])
    if data_points is not None:
        if y_levels is not None:
            data_points = data_points.melt(id_vars=vars_ppc, value_vars=y_levels, var_name=y_level_name,
                         value_name=y).reset_index()

        data_points = data_points.sample(df.shape[0], replace=True) #FIXME only plot real points here, dont sample more, turn oversamples to alpha=0
        g.data = data_points
        g.map_dataframe(scatterplot, x, y, hue=hue, size=0.02, alpha=points_alpha) #need to deal with style and coloring datapoints.
        # to_groupby = [v for v in [facet_row, facet_col] if v is not None]
        # if len(to_groupby) > 0:
        #     df = df.set_index(to_groupby)
        #     for idxs, data_slice in data_points.groupby(to_groupby):
        #         iloc_idx = []
        #         if ~isinstance(idxs,list): #FIXME maybe this need to be numpy array
        #             idxs = [idxs]
        #         for i in idxs:
        #             if hasattr(g, 'row_names') and i in g.row_names:
        #                 iloc_idx.append(g.row_names.index(i))
        #             else:
        #                 iloc_idx.append(0)
        #             if hasattr(g, 'col_names') and i in g.col_names:
        #                 iloc_idx.append(g.col_names.index(i))
        #
        #         if len(iloc_idx) < len(g.axes.shape):
        #             iloc_idx.append(0)
        #         idxs = tuple(iloc_idx)
        #         #g.axes[idxs].set_xlim((data_slice.loc[:,x].min(), data_slice.loc[:,x].max()))
        #         #g.axes[idxs].set_ylim((data_slice.loc[:,y].min(), data_slice.loc[:,y].max()))
        #         if vars_to_label is not None:
        #             df_slice = df[idxs]
        #             x_text = (data_slice.loc[:,x].max()-data_slice.loc[:,x].min())/2
        #             x_sd = (data_slice.loc[:,x].max()-data_slice.loc[:,x].min())/10
        #             y_sd = (data_slice.loc[:,y].max()-data_slice.loc[:,y].min())/10
        #             for var_to_label in vars_to_label:
        #                 for style_i, style_data in df_slice.groupby(var_to_label):
        #                     y_text = style_data.loc[(style_data[x]<x_text+x_sd) & (style_data[x]>x_text-x_sd), y].iloc[0]
        #                     g.axes[idxs].annotate(style,
        #                                          xy=(x_text, y_text), xycoords='data',
        #                                          xytext=(x_text+0.1*x_sd, y_text+0.1*y_sd),
        #                                          textcoords='data',
        #                                          size=16, va="center", ha="center",
        #                                          bbox=dict(boxstyle="round", fc="w"),
        #                                          arrowprops=dict(arrowstyle="-|>",
        #                                                          connectionstyle="arc3,rad=-0.2",
        #                                                          fc="w"))
    return g





