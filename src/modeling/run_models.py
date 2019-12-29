import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from src.modeling import sleep_stage_models, duration_models, base_model, meta_model, visualization, model_fitting
import sys
import os
import pymc3 as pm
import theano.tensor as tt



from theano.printing import Print
def pm_print(x, attrs=None, on=True, name=None):
    if name is not None:
        x.name = name
    if on:
        if attrs is not None:
            return Print(x.name, attrs=attrs)(x)
        else:
            return Print(x.name)(x)
    else:
        return x

def pm_shape(x, on=True):
    if on:
        return Print(x.name, attrs=['shape'])(x)
    else:
        return x



def format_data_for_modeling(raw_data, stages_to_consider=('waso','n1','n2','n3','rem'),
                             consider_trans_from=('waso','n1','n2','n3','rem','wbso'),
                             mean_center_vars=[],
                             standardized_vars=[],
                             half_normal_vars=[]):
    """
    Load and format epoch by epoch data into data that can be used in pymc3
    :param raw_data: epoch_by_epoch data from download_dataset functions
    :return: data: the data formatted for modeling,
             maps: maps levels of variables from modeling data back to human readable levels
    """
    raw_data = raw_data.loc[~raw_data['bad_epoch'],:]
    if stages_to_consider is None:
        stages_to_consider = ('waso','n1','n2','n3','rem')
    if consider_trans_from is None:
        consider_trans_from = ('waso','n1','n2','n3','rem','wbso')
    cols_to_skip = ['std_sleep_eeg.epochs_with_artifacts', 'lights_off_datetime',  'datetime', 'fastsigma', 'rem_event_duration', 'slow_osc_duration', 'slow_osc_peak_time', 'slow_osc_peak_uV', 'slow_osc_trough_time', 'slow_osc_trough_uV', 'slow_osc_zero_time', 'spindle_duration', 'spindle_freq_peak', 'spindle_peak_time', 'spindle_peak_uV']
    cols_to_drop = [col for col in raw_data.columns if col in cols_to_skip]
    raw_data = raw_data.drop(cols_to_drop, axis=1)
    #raw_data['tau'] = raw_data['tau']/60 #scale to hours

    catergorical_cols = ['studyid', 'subjectid', 'current_epoch', 'next_epoch', 'previous_bout',
                         'sex', 'ethnicity']

    raw_data.columns = [a.split('.')[-1] for a in raw_data.columns]

    cols_to_model =  catergorical_cols + ['t', 'tau']
    data = raw_data.loc[:, cols_to_model]

    data = raw_data.loc[data['next_epoch'].isin(stages_to_consider)
                        & data['current_epoch'].isin(stages_to_consider)
                        & data['previous_bout'].isin(consider_trans_from),:]

    stage_map = {v:k for k,v in enumerate(stages_to_consider)}
    previous_bout_stagemap = {v:k for k,v in enumerate(consider_trans_from)}

    maps = {'current_epoch': stage_map,
            'next_epoch':stage_map,
            'previous_bout': previous_bout_stagemap}
    for col in catergorical_cols:
        if col not in maps:
            col_map = {val:idx for idx, val in enumerate(data[col].unique())}
            maps[col] = col_map
        data[col] = data[col].map(maps[col])

    mean_center_means = {}
    for var in mean_center_vars:
        mean_center_means[var] = {'mean':data[var].mean(skipna=True), 'std':1}
        data[var] = data[var]-mean_center_means[var]['mean']
    for var in standardized_vars:
        mean_center_means[var] = {'mean':data[var].mean(skipna=True), 'std':data[var].std(skipna=True)}
        data[var] = (data[var]-mean_center_means[var]['mean'])/mean_center_means[var]['std']
    for var in half_normal_vars:
        mean_center_means[var] = {'mean': 0, 'std': np.sqrt((1-2/np.pi)*data[var].std(skipna=True)**2)}
        data[var] = (data[var]-mean_center_means[var]['mean'])/mean_center_means[var]['std']

    return data, maps, mean_center_means


def run_meta_models(data, maps, terms_to_try, outcome=None, force_refit=False): #TODO add checks if model already run, and rebuilt
    #data = data.loc[data['ethnicity'].isin([0,1]), :
    models_to_compare = []
    outcome = outcome if 'outcome' is not None else 'next_epoch'
    for name, terms in terms_to_try.items():
        refit = force_refit
        try:
            mod = base_model.load_model(name)
            if hasattr(mod, 'data_name') and data.name != mod.data_name:
                raise FileNotFoundError
            print('Loaded', name)
        except FileNotFoundError:
            refit = True
        if refit:
            print('Fitting', name)
            mod = meta_model.MetaModel(name=name, terms=terms, maps=maps, outcome=outcome)
            mod.fit(data)
            mod.save_model(model_name=name)
        models_to_compare.append(mod)
    return models_to_compare


def make_data_str(n, studies_to_inc=('shhs',), stages_to_consider=None):
    studies_to_inc_str = '_'.join(studies_to_inc) if studies_to_inc is not None and len(studies_to_inc) > 0 else 'all_studies'
    stages_str = '_'.join(stages_to_consider) if stages_to_consider is not None and len(stages_to_consider) > 0 else 'all_stages'
    return 'data_n' + str(n) + '_' + studies_to_inc_str + '_'+stages_str


def make_or_load_regular_data(n, studies_to_inc=('shhs',), age_cutoff=None, stages_to_consider=None,
                              force_refesh=False,
                              mean_center_vars=[],
                              standardized_vars=[],
                              half_normal_vars=[],
                              load_from_name=None):
    if load_from_name is not None:
        return pickle.load(open('../../data/models/' + load_from_name + '/' + load_from_name + '.pkl', 'rb'))

    data_str = make_data_str(n, studies_to_inc, stages_to_consider)
    if not force_refesh:
        try:
            (data, maps, mean_center_means) = pickle.load(open('../../data/models/'+data_str+'/'+data_str+'.pkl', 'rb'))
        except FileNotFoundError:
            force_refesh = True
    if force_refesh:
        input('refresh data?')
        raw_data = pd.read_pickle('../../data/processed/epoch_x_epoch.pkl')
        print('Raw Data N', raw_data.shape[0])
        if studies_to_inc is not None and len(studies_to_inc) > 0:
            raw_data = raw_data.loc[raw_data['studyid'].isin(studies_to_inc),:]
        if stages_to_consider is not None:
            raw_data = raw_data.loc[raw_data['current_epoch'].isin(stages_to_consider), :]
        if age_cutoff is not None:
            raw_data = raw_data.loc[raw_data['age']<age_cutoff, :]
        data, maps, mean_center_means= format_data_for_modeling(raw_data, stages_to_consider=stages_to_consider,
                                                                mean_center_vars=mean_center_vars,
                                                                standardized_vars=standardized_vars,
                                                                half_normal_vars=half_normal_vars)
        print('samples:', data.shape[0])
        if n is None or n>data.shape[0]:
            print('N larger than dataset, not subsampling.')
        else:
            data = data.sample(n=n)
        if not os.path.isdir('../../data/models/'+data_str):
            os.mkdir('../../data/models/'+data_str)
        pickle.dump((data, maps, mean_center_means), open('../../data/models/'+data_str+'/'+data_str+'.pkl', 'wb'))

    print('Num samples', data['subjectid'].shape)
    print('Num subs', len(data['subjectid'].unique()))
    print('Num studies', len(data['studyid'].unique()))
    print('Num subs in studies:\n', data['studyid'].value_counts())
    #print(maps)
    data.name = data_str
    return data, maps, mean_center_means


def fit_and_compare_rem_models(n=30008): #GOOD as is.

    priors = {
        'baserate_p':2, #double check rem properties
        'previous_bout_p': 0,
        'clocktime_p':0.05, #assuming hours
        'clocktime^2_p':0, #assuming hours
        'clocktime_phase_p': 0,
        'clocktime_rate_p': 0,
        'clocktime_alpha_exp_p': 0.05,
        'clocktime_lam_exp_p': -0.05,
        # 'timeslept_p': -0.03,
        # 'timeslept^2_p': 0,  # assuming hours
        # 'timeslept_phase_p': 0,
        # 'timeslept_rate_p': 0,
        # 'timeslept_alpha_exp_p': 0.05,
        # 'timeslept_lam_exp_p': 0,
        'tau_p':0.005,
        'tau^2_p':-0.0005,
        'tau_alpha_exp_p': 0.5,
        'tau_lam_exp_p': -0.05,
        'age_p':-0.05,
        'age^2_p':0,
        'sex_p': -0.5,
        'ageXsex_p': 0.05,
    }

    sd_priors = {
        'baserate_p': 1,  # double check rem properties
        'previous_bout_p': 0.5,
        'clocktime_p':0.5, #assuming hours
        'clocktime^2_p': 0.1,  # assuming hours
        'clocktime_rate_p':3,
        'clocktime_kappa_p': 0.5,
        'clocktime_alpha_exp_p': 0.25,
        'clocktime_lam_exp_p': 0.1,
        # 'timeslept_p': 0.5,  # assuming hours
        # 'timeslept^2_p': 0.1,  # assuming hours
        # 'timeslept_rate_p': 3,
        # 'timeslept_kappa_p': 0.5,
        # 'timeslept_alpha_exp_p': 0.2,
        # 'timeslept_lam_exp_p': 0.05,
        'tau_p':0.01,
        'tau^2_p':0.001,
        'age_p':0.1,
        'age^2_p':0.001,
        'sex_p': 3,
        'ageXsex_p': 0.1,
        'tau_alpha_exp_p': 1,
        'tau_lam_exp_p': 0.1,
    }

    studies_to_inc = ['mass_ss2', 'shhs', 'wamsley_r21', 'mass_ss4', 'mass_ss1', 'sof', 'mros', 'mass_ss5', 'mass_ss3']
    print('Running REM')
    data, maps, mean_center_means = make_or_load_regular_data(n, studies_to_inc, stages_to_consider=('rem',),
                                           force_refesh=False, mean_center_vars=['age','clocktime'])
    outcome = ['rem_event_count']

    fitter = model_fitting.ModelFitter(data=data,
                                       maps=maps,
                                       outcome=outcome,
                                       priors=priors,
                                       sd_priors=sd_priors,
                                       inital_set = [{'baserate': {'type': 'fixed', 'nested':True}},{'baserate': {'type': 'fixed'}}])
    fitter.add_regular_set(remove_pareto=True)
    fitter.run_fitter(force_refresh=False)
    fitter.compare_all().to_csv('../../data/output/rem_results.csv')

def fit_rem_mega_model():
    #best model:
    studies_to_inc = ['mass_ss2', 'shhs', 'wamsley_r21', 'mass_ss4', 'mass_ss1', 'sof', 'mros', 'mass_ss5', 'mass_ss3']
    data, _, mean_center_means = make_or_load_regular_data(200000, studies_to_inc=studies_to_inc, stages_to_consider=('rem',),
                                                           force_refesh=False, mean_center_vars=['age','clocktime'])

    priors = {
        'baserate_p':2, #double check rem properties
        'previous_bout_p': 0,
        'clocktime_p':0.05, #assuming hours
        'clocktime^2_p':0, #assuming hours
        'clocktime_phase_p': 0,
        'clocktime_rate_p': 0,
        'clocktime_alpha_exp_p': 0.05,
        'clocktime_lam_exp_p': -0.05,
        # 'timeslept_p': -0.03,
        # 'timeslept^2_p': 0,  # assuming hours
        # 'timeslept_phase_p': 0,
        # 'timeslept_rate_p': 0,
        # 'timeslept_alpha_exp_p': 0.05,
        # 'timeslept_lam_exp_p': 0,
        'tau_p':0.005,
        'tau^2_p':-0.0005,
        'tau_alpha_exp_p': 0.5,
        'tau_lam_exp_p': -0.05,
        'age_p':-0.05,
        'age^2_p':0,
        'sex_p': -0.5,
        'ageXsex_p': 0.05,
    }

    sd_priors = {
        'baserate_p': 1,  # double check rem properties
        'previous_bout_p': 0.5,
        'clocktime_p':0.5, #assuming hours
        'clocktime^2_p': 0.1,  # assuming hours
        'clocktime_rate_p':3,
        'clocktime_kappa_p': 0.5,
        'clocktime_alpha_exp_p': 0.25,
        'clocktime_lam_exp_p': 0.1,
        # 'timeslept_p': 0.5,  # assuming hours
        # 'timeslept^2_p': 0.1,  # assuming hours
        # 'timeslept_rate_p': 3,
        # 'timeslept_kappa_p': 0.5,
        # 'timeslept_alpha_exp_p': 0.2,
        # 'timeslept_lam_exp_p': 0.05,
        'tau_p':0.01,
        'tau^2_p':0.001,
        'age_p':0.1,
        'age^2_p':0.001,
        'sex_p': 3,
        'ageXsex_p': 0.1,
        'tau_alpha_exp_p': 1,
        'tau_lam_exp_p': 0.1,
    }

    data_name = 'data_n30008_mass_ss2_shhs_wamsley_r21_mass_ss4_mass_ss1_sof_mros_mass_ss5_mass_ss3_rem/'
    best_model = 'rem_event_count_baserate-nested_previous_bout_tau-l_tau^2-q_clocktime-s_age-l_age^2-q_ageXsex-i_sex-l_'
    selected_mod = base_model.load_model(data_name=data_name, model_name=best_model)
    mega_mod = meta_model.MetaModel('final_rem', outcome=['rem_event_count'],
                                    terms=selected_mod.terms,
                                    maps=selected_mod.maps,
                                    priors=priors,
                                    sd_priors=sd_priors)
    mega_mod.fit(data)
    mega_mod.save_model(data_name=data.name)
    mega_mod.summary()
    mega_mod.traceplot()
    plt.show()


def fit_and_compare_stage_models(n=40007, fit_mega=False):
    studies_to_inc = None #['shhs']
    print('Running Next Epoch Models')
    data, maps, mean_center_means = make_or_load_regular_data(n, studies_to_inc,
                                                              stages_to_consider=None,
                                                              force_refesh=False,
                                                              mean_center_vars=['clocktime','age'])
    outcome = ['next_epoch']
    current_epoch_p = np.array([[0.826678401,0.11204628,0.05777386,0,0.003501446],
                           [0.079971963,0.814018692, 0.09671028,0,0.009299065],
                           [0.071140187,0.00646729, 0.838317757,0.050121495,0.033953271],
                           [0.025654206,0,0.13682243,0.828971963,0.008551402],
                           [0.063925234,0.009588785,0.033028037,0,0.893457944]])
    baserate_p = np.array([0.2, 0.17, 0.38, 0.22, 0.05])
    priors = {
        'baserate_p': baserate_p,  # double check rem properties
        'current_epoch_p': current_epoch_p-baserate_p,
        'previous_bout_p': 0,
        'clocktime_p': [0.07/6, 0.05/6, -0.07/6, -0.11/6, 0.05/6],  # assuming hours
        'clocktime^2_p': [0.01/6],  # assuming hours
        'clocktime_alpha_exp_p': 0.05,
        'clocktime_lam_exp_p': [-0.03/6, -0.02/6, 0.03/6, 0.06/6, -0.01/6],
        'clocktime_phase_p': [1, 0.15, -0.5, 0.5, 0.8],
        'clocktime_rate_p': 0,
        # 'timeslept_p': [0.0001, ],  # not to sure how to decouple...
        # 'timeslept^2_p': [0.0001, ],  # not to sure how to decouple...
        # 'timeslept_lam_exp_p': [-0.0001, ],
        # 'timeslept_alpha_exp_p': 0.05,
        # 'timeslept_phase_p': 0,
        # 'timeslept_rate_p': 0,
        'tau_par_p': -0.05,
        'tau_p': 0.1/50,
        'tau^2_p': 0,
        'age_p': [0.5/100, 0.25/100, 0.2/100, -0.7/100, 0],
        'age^2_p': [0.01/100, 0, 0.005/100, 0, -0.01/100],
        'sex_p': [0.1, 0.05, 0.05, -0.1, 0],
        'ageXsex_p': [0.1/100,0,0.1/100,-0.2/100,0],
        'tau_alpha_exp_p': 0.05,
        'tau_lam_exp_p': -0.1,
    }

    sd_priors = {
        'baserate_p': 0.25,  # double check rem properties
        'current_epoch_p': 0.4,
        'previous_bout_p': 0.1,
        'clocktime_p': 0.1,  # assuming hours
        'clocktime^2_p': 0.01,  # assuming hours
        'clocktime_rate_p': 0.1,
        'clocktime_kappa_p': 0.5,
        'clocktime_alpha_exp_p': 0.25,
        'clocktime_lam_exp_p': 0.1,
        # 'timeslept_p': 0.1,
        # 'timeslept^2_p': 0.01,
        # 'timeslept_lam_exp_p': 0.1,
        # 'timeslept_alpha_exp_p': 0.25,
        # 'timeslept_rate_p': 0.1,
        # 'timeslept_kappa_p': 0.5,
        'tau_par_p':0.03,
        'tau_p': 0.1/50,
        'tau^2_p': 0.05/50,
        'age_p': 0.2/70,
        'age^2_p': 0.02/70,
        'sex_p': 0.3,
        'ageXsex_p': 0.25/70,
        'tau_alpha_exp_p': 1,
        'tau_lam_exp_p': 0.5,
    }

    #Plan of attack. Try with larger SD priors, and more samples
    #Then remove timeslept or clocktime
    if not fit_mega:
        inital_set = [
                      {'baserate': {'type': 'fixed'}, 'current_epoch': {'type': '', 'nested':True}},
                      {'baserate': {'type': 'fixed'}, 'current_epoch': {'type': ''}},
                      {'baserate': {'type': 'fixed', 'nested':True}, 'current_epoch':{'type':''}},
                      {'baserate': {'type': 'fixed', 'nested':True}, 'current_epoch': {'type': '', 'nested': True}} #FIXME consider dropping this...
                     ]
        fitter = model_fitting.ModelFitter(data=data, maps=maps, outcome=outcome, inital_set=inital_set, sd_priors=sd_priors, priors=priors)
        tau_special = [{'tau':{'type':'linear','self_trans_only':False}},
                       {'tau':{'type':'linear', 'self_trans_only':False},'tau^2':{'type':'quadratic', 'self_trans_only':False}},
                       {'tau':{'type':'pareto', 'self_trans_only':False}}
                       ]
        fitter.add_regular_set(replace={1:tau_special})
        fitter.run_fitter(force_refresh=False)
        fitter.compare_all().to_csv('../../data/output/next_epoch_results.csv')
    else:
        data_name = 'data_n40007_all_studies_all_stages/'
        best_model = 'next_epoch_baserate-nested_current_epoch-nested_previous_bout_tau-p_clocktime-e_age-l_sex-l_'
        selected_mod = base_model.load_model(data_name=data_name, model_name=best_model)
        mega_mod = meta_model.MetaModel('final_stage', outcome=outcome,
                                        terms=selected_mod.terms,
                                        maps=selected_mod.maps,
                                        priors=priors,
                                        sd_priors=sd_priors)
        mega_mod.fit(data)
        mega_mod.save_model(data_name=data.name)
        mega_mod.summary()
        mega_mod.traceplot()
        plt.show()


def interpret_rem_models():
    outcome = ['rem_event_count']
    studies_to_inc = None #['shhs']
    data_name = 'data_n200000_mass_ss2_shhs_wamsley_r21_mass_ss4_mass_ss1_sof_mros_mass_ss5_mass_ss3_rem'
    best_mod_name = 'final_rem'
    data, maps, mean_center_means = make_or_load_regular_data(0, studies_to_inc, stages_to_consider=('rem',), age_cutoff=40,
                                           force_refesh=False, mean_center_vars=['age'], load_from_name=data_name)

    # real_comp = 'data_n30008_mass_ss2_shhs_wamsley_r21_mass_ss4_mass_ss1_sof_mros_mass_ss5_mass_ss3_rem'
    # models_to_compare = model_fitting.load_models(data_name=real_comp)
    # model_fitting.compare_models(models=models_to_compare)

    best_mod = base_model.load_model(data_name, model_name=best_mod_name)
    data = data.loc[data['previous_bout']!=3,:]
    data_viz, data_points = visualization.gen_data_for_plot(data, samples_per_x_range=200, x='tau', mean_sample_vars=['clocktime','age'], const_vars={'studyid':5, 'sex':0, 'consider_current_epoch':1, 'consider_previous_bout':0, 'previous_bout':0}, nstages=len(maps['current_epoch']))
    g = visualization.plot_vars(best_mod, data_viz, num_draws_from_params=100,
                            x='tau', y='feature_rate_p', y_levels=['rem_event_count'], y_level_name='rem_event_count',
                            facet_row='rem_event_count',
                            combine_trace=True, out_of_sample=True, maps=maps)

    g.axes[0, 0].set_ylim((0,5))
    g.axes[0, 0].set_xlim((0,45))
    g.axes[0, 0].set_ylabel('Feature Rate (counts/epoch)')
    g.axes[0, 0].set_xlabel('Tau (Minutes in stage)')
    g.axes[0, 0].set_title('REM Events')
    plt.show()

    # data_viz, data_points = visualization.gen_data_for_plot(data, samples_per_x_range=200, x='clocktime',
    #                                                         mean_sample_vars=['tau','age'],
    #                                                         rand_sample_vars=['previous_bout'],
    #                                                         const_vars={'studyid':5, 'sex':0,
    #                                                                     'consider_current_epoch':1,
    #                                                                     'consider_previous_bout':0},
    #                                                         nstages=len(maps['current_epoch']))
    # g = visualization.plot_vars(best_mod, data_viz, num_draws_from_params=100,
    #                         x='clocktime', y='feature_rate_p', y_levels=['rem_event_count'],
    #                         combine_trace=True,
    #                         data_points=None, out_of_sample=True, legend=False, maps=maps, mean_center_means=mean_center_means)
    # g.axes[0, 0].set_ylim((0,4))
    # #g.axes[0, 0].set_xlim((0,45))
    # g.axes[0, 0].set_ylabel('Feature Rate (counts/epoch)')
    # g.axes[0, 0].set_xlabel('Clocktime (Time of Day)')
    # g.axes[0, 0].set_title('REM Events')
    # plt.show()

    data_viz, data_points = visualization.gen_data_for_plot(data, samples_per_x_range=200, x='age', z='sex', mean_sample_vars=['tau','clocktime'], rand_sample_vars=['previous_bout'], const_vars={'studyid':5, 'consider_current_epoch':1, 'consider_previous_bout':0}, nstages=len(maps['current_epoch']))
    g = visualization.plot_vars(best_mod, data_viz, num_draws_from_params=100,
                            x='age', hue='sex', y='feature_rate_p', y_levels=['rem_event_count'], y_level_name='rem',
                            combine_trace=True,
                            facet_row='rem',
                            data_points=None, points_alpha=0, out_of_sample=True, legend='full', maps=maps, mean_center_means=mean_center_means)
    g.axes[0, 0].set_ylim((0,5))
    #g.axes[0, 0].set_xlim((0,45))
    g.axes[0, 0].set_ylabel('Feature Rate (counts/epoch)')
    g.axes[0, 0].set_xlabel('Age (years)')
    g.axes[0, 0].set_title('REM Events')
    plt.show()


def interpret_feature_models(n=1000):

    outcome = ['spindle_count', 'slow_osc_count']
    data_name = 'data_n200000_mass_ss2_mass_ss3_mass_ss4_mass_ss5_shhs_wamsley_future_wamsley_ken_wamsley_r21_n2_n3'
    best_mod_name = 'final_features'
    data, maps, mean_center_means = make_or_load_regular_data(n, studies_to_inc=None,
                                           force_refesh=False, mean_center_vars=['age','clocktime'], load_from_name=data_name)

    # real_comp = 'data_n30003_mass_ss2_mass_ss3_mass_ss4_mass_ss5_shhs_wamsley_future_wamsley_ken_wamsley_r21_n2_n3'
    # models_to_compare = model_fitting.load_models(data_name=real_comp)
    # model_fitting.compare_models(models=models_to_compare)

    best_mod = base_model.load_model(data_name, model_name=best_mod_name)
    best_mod.summary().to_csv('../../data/output/final_features_summary.csv')

    data_viz, data_points = visualization.gen_data_for_plot(data, samples_per_x_range=200, x='tau', z='current_epoch',
                                                            mean_sample_vars=['clocktime', 'age'],
                                                            const_vars={'studyid': 5, 'sex':0, 'previous_bout':0, 'consider_current_epoch': 1,
                                                                        'consider_previous_bout': 0},
                                                            nstages=len(maps['current_epoch']))

    g = visualization.plot_vars(best_mod, data_viz, num_draws_from_params=100,
                            x='tau', y='feature_rate_p', y_levels=outcome, y_level_name='feature',
                            facet_col='feature',
                            combine_trace=True,
                            hue='current_epoch', out_of_sample=True, maps=maps)
    g.axes[0, 1].set_ylim((0, 10))
    g.axes[0, 0].set_ylim((0, 5))
    g.axes[0, 0].set_xlim((0, 40))
    g.axes[0, 1].set_xlim((0, 40))
    g.axes[0, 0].set_ylabel('Feature Rate (counts/epoch)')
    g.axes[0, 0].set_xlabel('Tau (mins in stage)')
    g.axes[0, 1].set_xlabel('Tau (mins in stage)')
    g.axes[0, 0].set_title('Spindles Events')
    g.axes[0, 1].set_title('Slow Oscillation Events')
    plt.show()

    data_viz, data_points = visualization.gen_data_for_plot(data, samples_per_x_range=200, x='tau', z='current_epoch',
                                                            mean_sample_vars=['clocktime', 'age'],
                                                            rand_sample_vars=['previous_bout'],
                                                            const_vars={'studyid': 5, 'sex':0, 'consider_current_epoch': 1,
                                                                        'consider_previous_bout': 1},
                                                            nstages=len(maps['current_epoch']))

    g = visualization.plot_vars(best_mod, data_viz, num_draws_from_params=100,
                            x='previous_bout', y='feature_rate_p', y_levels=outcome, y_level_name='feature',
                            facet_col='feature',
                            combine_trace=True,
                            hue='current_epoch',
                            data_points=data_points, out_of_sample=True, maps=maps)

    data_viz, data_points = visualization.gen_data_for_plot(data, samples_per_x_range=200, x='clocktime',
                                                            z = 'current_epoch',
                                                            mean_sample_vars=['tau', 'age'],
                                                            rand_sample_vars=['previous_bout'],
                                                            const_vars={'studyid': 5, 'consider_current_epoch': 1,
                                                                        'consider_previous_bout': 0, 'sex':0},
                                                            nstages=len(maps['current_epoch']))
    g = visualization.plot_vars(best_mod, data_viz, num_draws_from_params=100,
                            x='clocktime', y='feature_rate_p', y_levels=outcome, y_level_name='Feature',
                            combine_trace=True,
                            facet_col='Feature',
                            hue='current_epoch',
                            #data_points=data_points,
                            out_of_sample=True, legend='full', maps=maps,
                            mean_center_means=mean_center_means)
    g.axes[0, 0].set_ylim((0, 3))
    g.axes[0, 1].set_ylim((0, 9))
    g.axes[0, 0].set_ylabel('Feature Rate (counts/epoch)')
    g.axes[0, 0].set_xlabel('Clocktime (Time of Day)')
    g.axes[0, 1].set_xlabel('Clocktime (Time of Day)')
    g.axes[0, 0].set_title('Spindles Events')
    g.axes[0, 1].set_title('Slow Oscillation Events')
    plt.show()

    data_viz, data_points = visualization.gen_data_for_plot(data, samples_per_x_range=200, x='age',
                                                            z = 'sex',
                                                            mean_sample_vars=['tau','clocktime'],
                                                            rand_sample_vars=['previous_bout','sex'],
                                                            const_vars={'studyid': 5, 'consider_current_epoch': 1,
                                                                        'consider_previous_bout': 0},
                                                            nstages=len(maps['current_epoch']))

    for current_epoch, stage_str in zip([1,0],['N3','N2']):
        age_min = (18-mean_center_means['age']['mean'])/mean_center_means['age']['std']
        age_max = (40-mean_center_means['age']['mean'])/mean_center_means['age']['std']
        data_viz_temp = data_viz.loc[(data_viz['current_epoch']==current_epoch) & (data_viz['age']>=age_min) & (data_viz['age']<=age_max), :]
        data_points_temp = data_points.loc[(data_points['current_epoch']==current_epoch) & (data_points['age']>=age_min) & (data_points['age']<=age_max), :]
        #data_viz_temp = data_viz.loc[(data_viz['age']>=age_min) & (data_viz['age']<=age_max), :]
        #data_points_temp = data_points.loc[(data_points['age']>=age_min) & (data_points['age']<=age_max), :]
        g = visualization.plot_vars(best_mod, data_viz_temp, num_draws_from_params=100,
                                x='age', y='feature_rate_p', y_levels=outcome, y_level_name='Feature',
                                combine_trace=True,
                                facet_col='Feature',
                                hue='sex',
                                points_alpha=0.0001,
                                data_points=data_points_temp, out_of_sample=True, legend='full', maps=maps,
                                mean_center_means=mean_center_means)
        g.axes[0,0].set_ylim((0,3))
        if current_epoch == 1:
            g.axes[0,1].set_ylim((0,15))
        else:
            g.axes[0, 1].set_ylim((0, 3))
        g.axes[0,0].set_ylabel('Feature Rate (counts/epoch)')
        g.axes[0,0].set_xlabel('Age (years)')
        g.axes[0,1].set_xlabel('Age (years)')
        g.axes[0,0].set_title(stage_str + ' Spindles Events')
        g.axes[0,1].set_title(stage_str + ' Slow Oscillation Events')
        plt.show()


def interpret_stage_models(n=1000):

    # real_comp = 'data_n40007_all_studies_all_stages'
    # models_to_compare = model_fitting.load_models(data_name=real_comp)
    # model_fitting.compare_models(models=models_to_compare)

    outcome = ['next_epoch']
    data_name = 'data_n200000_all_studies_all_stages'
    #data_name = 'data_n40007_all_studies_all_stages'
    best_mod_name = 'final_stage'
    #best_mod_name = 'next_epoch_baserate-nested_current_epoch-nested_previous_bout_tau-p_clocktime-e_age-l_sex-l_ageXsex-i_'
    studies_to_inc = None
    data, maps, mean_center_means = make_or_load_regular_data(n, studies_to_inc,
                                                              stages_to_consider=None,
                                                              force_refesh=False,
                                                              mean_center_vars=['clocktime','age'],
                                                              load_from_name=data_name)
    # print('real_trans -------------')
    # for current, current_data in data[['current_epoch', 'next_epoch']].groupby('current_epoch'):
    #     print(current)
    #     print(current_data['next_epoch'].value_counts() / current_data.shape[0])
    # print('------------------------')

    print(mean_center_means)

    best_mod = base_model.load_model(data_name, model_name=best_mod_name)
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     csv = best_mod.summary()
    #     print(csv.to_csv('../../data/processed/parameters_final_stage.csv'))

    # data_viz, data_points = visualization.gen_data_for_plot(data, samples_per_x_range=200, x='tau', z='current_epoch',
    #                                                         mean_sample_vars=['age','clocktime'],
    #                                                         rand_sample_vars=['previous_bout'],
    #                                                         const_vars={'studyid': 2, 'consider_current_epoch': 1,
    #                                                                     'consider_previous_bout': 0, 'sex':0},
    #                                                         nstages=len(maps['current_epoch']))
    # g = visualization.plot_vars(best_mod, data_viz, num_draws_from_params=100,
    #                         x='tau', y='trans_p', y_levels=maps['next_epoch'], y_level_name='Next Stage',
    #                         facet_col='current_epoch',
    #                         combine_trace=True,
    #                         hue='Next Stage',
    #                         data_points=None, out_of_sample=False, maps=maps)
    #
    # stage_names = ['WASO','N1','N2','N3','REM']
    # xlims = [50,20,60,60,60]
    # g.axes[0, 0].set_ylabel('Transition Probability')
    # for j in range(5):
    #     g.axes[0,j].set_xlim((0, xlims[j]))
    #     g.axes[0,j].set_ylim((0, 1))
    #     g.axes[0, j].set_xlabel('Tau (mins in stage)')
    #     g.axes[0, j].set_title('Current epoch is '+stage_names[j])
    # plt.show()

    # data_viz, data_points = visualization.gen_data_for_plot(data, samples_per_x_range=200, x='clocktime', z='current_epoch',
    #                                                         mean_sample_vars=['age','tau'],
    #                                                         rand_sample_vars=['previous_bout'],
    #                                                         const_vars={'studyid': 2, 'consider_current_epoch': 1,
    #                                                                     'consider_previous_bout': 0, 'sex':0},
    #                                                         nstages=len(maps['current_epoch']))
    # g = visualization.plot_vars(best_mod, data_viz, num_draws_from_params=100,
    #                         x='clocktime', y='trans_p', y_levels=maps['next_epoch'], y_level_name='Next Stage',
    #                         facet_col='current_epoch',
    #                         combine_trace=True,
    #                         hue='Next Stage',
    #                         data_points=None, out_of_sample=False, maps=maps, mean_center_means=mean_center_means)
    #
    # stage_names = ['WASO','N1','N2','N3','REM']
    # #xlims = [50,20,60,60,60]
    # g.axes[0, 0].set_ylabel('Transition Probability')
    # for j in range(5):
    #     #g.axes[0,j].set_xlim((0, xlims[j]))
    #     g.axes[0,j].set_ylim((0, 1))
    #     g.axes[0, j].set_xlabel('Time of Day')
    #     g.axes[0, j].set_title(stage_names[j])
    # plt.show()



    data_viz, data_points = visualization.gen_data_for_plot(data, samples_per_x_range=200, x='age', z='current_epoch',
                                                            mean_sample_vars=['clocktime','tau'],
                                                            rand_sample_vars=['previous_bout','sex'],
                                                            const_vars={'studyid': 2, 'consider_current_epoch': 1,
                                                                        'consider_previous_bout': 0},
                                                            nstages=len(maps['current_epoch']))
    g = visualization.plot_vars(best_mod, data_viz, num_draws_from_params=100,
                            x='age', y='trans_p', y_levels=maps['next_epoch'], y_level_name='Next Stage',
                            facet_row='current_epoch',
                            facet_col='Next Stage',
                            combine_trace=True,
                            hue='sex',
                            data_points=None, out_of_sample=False, maps=maps, mean_center_means=mean_center_means)

    stage_names = ['WASO','N1','N2','N3','REM']
    #xlims = [50,20,60,60,60]
    g.axes[0, 0].set_ylabel('Transition Probability')
    for i in range(5):
        for j in range(5):
            if i==j:
                g.axes[i, j].set_ylim((0.5, 1))
            else:
                g.axes[i, j].set_ylim((0, 0.4))
            g.axes[i, j].set_xlim((20-56.5, 85-56.5))
            g.axes[i, j].set_title('')
            if i != 4:
                g.axes[i,j].set_xticklabels([])
        g.axes[i, 0].set_ylabel('Trans Prob '+ stage_names[i])
        g.axes[0, i].set_xlabel('Age')
        g.axes[0, i].set_title(stage_names[i])
    plt.tight_layout()
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.3, top=0.9)
    plt.show()
    print()


def fit_and_compare_feature_models(n=30003, mega_model=False):
    studies_to_inc = ['mass_ss2', 'mass_ss3', 'mass_ss4', 'mass_ss5', 'shhs', 'wamsley_future', 'wamsley_ken', 'wamsley_r21']
    data, maps, mean_center_means = make_or_load_regular_data(n, studies_to_inc, stages_to_consider=('n2', 'n3'), age_cutoff=40,
                                           force_refesh=False, mean_center_vars=['age','clocktime'])
    outcome = ['spindle_count', 'slow_osc_count']

    priors = {
        'baserate_p':[2.5, 8], #double check rem properties
        'current_epoch_p': np.array([[3,4],
                                     [2,16]])-np.array([2.5, 8]),
        'previous_bout_p': 0,
        'clocktime_p':[[0, -4/60],
                       [0, -12/60]], #assuming hours
        'clocktime^2_p':0, #assuming hours
        'clocktime_phase_p': 0,
        'clocktime_rate_p': [[1,2],
                             [1,3]],
        'clocktime_alpha_exp_p': 0.05,
        'clocktime_lam_exp_p': -0.05,
        # 'timeslept_p': -0.03,
        # 'timeslept^2_p': 0,  # assuming hours
        # 'timeslept_phase_p': 0,
        # 'timeslept_rate_p': 0,
        # 'timeslept_alpha_exp_p': 0.05,
        # 'timeslept_lam_exp_p': 0,
        'tau_p':[[0,0],
                 [0,0]],
        'tau^2_p':[[-1/900,0],
                   [-1/900,0]],
        'tau_alpha_exp_p': 0.5,
        'tau_lam_exp_p': -0.05,
        'age_p':[[-1.5/60,0],
                 [-1/60  ,-4/40]],
        'age^2_p':0,
        'sex_p': [[1,0],[1.2,-2]],
        'ageXsex_p': 0,
    }

    sd_priors = {
        'baserate_p': 10,  # double check rem properties
        'current_epoch_p':10,
        'previous_bout_p': 1,
        'clocktime_p':5/6, #assuming hours
        'clocktime^2_p': 0.1/6,  # assuming hours
        'clocktime_rate_p':3,
        'clocktime_kappa_p': 0.5,
        'clocktime_alpha_exp_p': 0.25,
        'clocktime_lam_exp_p': 0.1,
        # 'timeslept_p': 0.5,  # assuming hours
        # 'timeslept^2_p': 0.1,  # assuming hours
        # 'timeslept_rate_p': 3,
        # 'timeslept_kappa_p': 0.5,
        # 'timeslept_alpha_exp_p': 0.2,
        # 'timeslept_lam_exp_p': 0.05,
        'tau_p':6/60,
        'tau^2_p':0.5/60,
        'age_p':2/60,
        'age^2_p':0.001,
        'sex_p': 3,
        'ageXsex_p': 2/60,
        'tau_alpha_exp_p': 1,
        'tau_lam_exp_p': 0.1,
    }
    if not mega_model:
        fitter = model_fitting.ModelFitter(data=data, maps=maps, outcome=outcome, priors=priors, sd_priors=sd_priors,
                                           inital_set = [
                                               {'baserate': {'type': ''}},
                                               {'baserate': {'type': '', 'correl': True}},
                                               {'baserate': {'type': '', 'nested':True}},
                                               {'baserate': {'type': '', 'nested':True, 'correl': True}}
                                                        ])
        fitter.add_regular_set(remove_pareto=True)
        fitter.run_fitter()
        fitter.compare_all().to_csv('../../data/output/feature_results.csv')
    else:
        data_name = 'data_n30003_mass_ss2_mass_ss3_mass_ss4_mass_ss5_shhs_wamsley_future_wamsley_ken_wamsley_r21_n2_n3'
        best_model =  'spindle_count-slow_osc_count_baserate-nested_previous_bout_tau-l_tau^2-q_clocktime-l_clocktime^2-q_age-l_age^2-q_sex-l_ageXsex-i_'
        selected_mod = base_model.load_model(data_name=data_name, model_name=best_model)
        mega_mod = meta_model.MetaModel('final_features', outcome=outcome,
                                        terms=selected_mod.terms,
                                        maps=selected_mod.maps,
                                        priors=priors,
                                        sd_priors=sd_priors)
        mega_mod.fit(data)
        mega_mod.save_model(data_name=data.name)
        mega_mod.summary()
        mega_mod.traceplot()
        plt.show()

def fit_and_compare_band_models(n=3001):
    studies_to_inc = None #['shhs']
    data, maps, mean_center_means = make_or_load_regular_data(n, studies_to_inc, stages_to_consider=('n1', 'n2', 'n3'), age_cutoff=40,
                                           force_refesh=False, standardized_vars=['age','SWA','delta','theta','alpha','sigma','beta'], mean_center_vars=['clocktime','timeslept'])
    outcome = ['SWA','delta','theta','alpha','sigma','beta']

    # mod = meta_model.MetaModel('testing_previous', maps=maps, terms={'baserate':{'type':'fixed'}}, outcome=outcome)
    # mod.fit(data)
    # mod.traceplot()
    # plt.show()

    fitter = model_fitting.ModelFitter(data=data, maps=maps, outcome=outcome,
                                       inital_set = [
                                           {'baserate': {'type': 'fixed'}},
                                           {'baserate': {'type': 'fixed', 'nested':True}},
                                           {'baserate': {'type': 'fixed', 'nested':True, 'correl': True}}
                                                    ])
    fitter.add_regular_set()
    fitter.run_fitter(force_refresh=True)
    fitter.compare_all().to_csv('../../data/output/band_results.csv')


if __name__ == "__main__":
    to_run = sys.argv[1]

    if to_run == 'fit_rem':
        fit_and_compare_rem_models()
    if to_run == 'interp_rem':
        interpret_rem_models()
    if to_run == 'mega_rem':
        fit_rem_mega_model()
    if to_run == 'fit_stage':
        fit_and_compare_stage_models()
    if to_run == 'interp_stage':
        interpret_stage_models()
    if to_run == 'mega_stage':
        fit_and_compare_stage_models(n=200000, fit_mega=True)
    if to_run == 'fit_band':
        fit_and_compare_band_models()
    if to_run == 'fit_feature':
        fit_and_compare_feature_models()
    if to_run == 'interp_feature':
        interpret_feature_models()
    if to_run == 'mega_feature':
        fit_and_compare_feature_models(n=200000, mega_model=True)

    # mod = meta_model.MetaModel('testing_band', outcome=['sigma','SWA','delta','alpha','theta','beta'],
    #                            terms={'baserate':{'type':'fixed'}}, maps=maps)
    # mod.fit(data)
    # mod.save_model()
    # mod.traceplot()
    # plt.show()

    # clocktime_terms = {
    #     'baserate_lin_clocktime': {'clocktime': {'stages':['rem','n3', 'n2'], 'type':'linear'}},
    #     'baserate_quad_clocktime': {'clocktime': {'stages':['rem','n3', 'n2'], 'type':'quadratic'}},
    #     'baserate_only': {},
    # }

    #run_meta_models(data, maps, terms_to_try)
    #models_to_compare = load_and_compare_models(terms_to_try)
    #base_model.compare_models(models_to_compare)

    # mod = base_model.load_model('testing_features')
    # # mod.traceplot()
    # # plt.show()
    # sd_log = mod.trace.get_values(varname='baserate_sd_log__', combine=True)
    # a_prob = mod.trace.get_values(varname='baserate_global_p_interval__', combine=True)[:,1,2]
    # _, ax = plt.subplots(1, 1, figsize=(10, 5))
    # ax.plot(a_prob, sd_log, 'o', color='C3', alpha=.5)
    # divergent = mod.trace['diverging']
    # ax.plot(a_prob[divergent], sd_log[divergent], 'o', color='C2')
    # ax.set_xlabel('a prob')
    # ax.set_ylabel('log(sd)')
    # ax.set_title('scatter plot between log(sd) and a prob]')
    #
    # divergent_point = defaultdict(list)
    # chain_warn = mod.trace.report._chain_warnings
    # for i in range(len(chain_warn)):
    #     for warning_ in chain_warn[i]:
    #         if warning_.step is not None and warning_.extra is not None:
    #             for RV in mod.model.free_RVs:
    #                 para_name = RV.name
    #                 if para_name == 'baserate_p_interval__':
    #                     continue
    #                 divergent_point[para_name].append(warning_.extra[para_name])
    #
    # for RV in mod.model.free_RVs:
    #     para_name = RV.name
    #     divergent_point[para_name] = np.asarray(divergent_point[para_name])
    # ii = 5
    #
    # sd_log_d = divergent_point['baserate_sd_log__']
    # a_prob_d = divergent_point['baserate_global_p_interval__'][:,1,2]
    # Ndiv_recorded = len(a_prob_d)
    #
    # ax.plot([a_prob[divergent == 1][:Ndiv_recorded], a_prob_d],
    #            [sd_log[divergent == 1][:Ndiv_recorded], sd_log_d],
    #            'k-', alpha=.5)
    #
    # ax.scatter(a_prob_d, sd_log_d,
    #               color='C3',
    #               label='Location of Energy error (start location of leapfrog)')
    #
    # plt.show()
    #
    # data_viz = visualization.gen_data_for_plot(data, x='tau', cont_vars={'studyid':0, 'clocktime':0})
    #
    # terms = {
    #     'testing_tau':{'baserate': {'type': 'fixed'}, 'tau': {'stages': 'all'}},
    #     'testing_clocktime':{'baserate': {'type': 'fixed'}, 'tau': {'stages': 'all'}, 'clocktime':{'stages':'all', 'type':'linear'}},
    # }
    #
    # mods = run_meta_models(terms_to_try=terms, maps=maps, data=data)
    # base_model.compare_models(mods)
    # for mod in mods:
    #     mod.plot_vars(data_viz, x='tau', y='trans_p', facet_col='current_epoch', y_level_name='stage', y_levels=list(maps['next_epoch'].keys()))
    #     mod.run_ppcs(data)
    # plt.show()

    #models_to_compare = load_and_compare_models()
    #base_model.compare_models(models_to_compare)

    #run_meta_models(data, maps)
    sys.exit()



# data = np.random.randint(0,3,size=(1000,2))
#
# with pm.Model() as model:
#     tp1 = pm.Dirichlet('tp1', a=np.array([0.25]*4), shape=(4,4))
#     obs = pm.Categorical('obs', p=tp1[data[:,0],:], observed=data[:,1])
#     trace = pm.sample()
#
# data = np.random.randint(0, 3, size=(1000, 1))
#
# with pm.Model() as model:
#     tp1 = pm.Dirichlet('tp1', a=np.array([0.25]*4), shape=(4,))
#     obs = pm.Categorical('obs', p=tp1, observed=data)
#     trace = pm.sample()

# a_predicted_value = np.random.normal(loc=10, size=(1000,))
# a_predictor_value = shared(np.random.normal(loc=0, size=(1000,)))
# with pm.Model() as model:
#     a_parameter = pm.Normal('mu', 0, sd=1)
#     a_predictor_value_det = pm.Deterministic('a_predictor_value', a_predictor_value)
#     sd = pm.HalfNormal('sd', sd=1)
#     a_predicted_value_obs = pm.Normal('a_predicted_value', mu=a_parameter + a_predictor_value_det, sd=sd,
#                                       observed=a_predicted_value)
#     trace = pm.sample()
#
# pred_values_before_setting = pm.sample_posterior_predictive(trace=trace, model=model,
#                                                             vars=[a_predictor_value_det, a_predicted_value_obs])
# a_predictor_value.set_value(np.random.normal(loc=10, size=(1000,)))
# pred_values_after_setting = pm.sample_posterior_predictive(trace=trace, model=model,
#                                                            vars=[a_predictor_value_det, a_predicted_value_obs])
#
# print(pred_values_before_setting['a_predictor_value'].flatten().mean())
# print(pred_values_after_setting['a_predictor_value'].flatten().mean())
#
# sys.exit()

#TODO
# previous power as a predictor of current power, power to predict stage
# Slow Osc changes with age and clocktime, into quartiles and agebins.
