import subprocess
import os
from mednickdb_pyapi import upload_helper
file_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(file_dir, '../../data/')
mednickdb_user = os.environ['mednickapi_username']
mednickdb_pass = os.environ['mednickapi_password']
import pandas as pd
import numpy as np
import mne
import glob
from mednickdb_pysleep import edf_tools, scorefiles
from mednickdb_pyapi import upload_helper
import warnings
import sys
import yaml
import pickle
import datetime
from wonambi import Dataset

debug=False


def prep_data(edf_input: str,
             edf_output: str,
             health_file: str,
             health_output: str,
             study_settings_path: str,
             scorefile_fix_offset=False,
             scorefiles_edf_path=None,
             scorefile_output=None):


    health_data = pd.read_excel(health_file)
    health_class = classify_health_status(health_data)
    health_class.to_csv(health_output)

    study_settings = yaml.safe_load(open(study_settings_path, 'rb'))

    if scorefile_fix_offset:
        check_edf_scorefiles(edf_input, scorefiles_edf_path, scorefile_output, study_settings)
    elif scorefiles_edf_path is not None:
        compress_edf_scorefiles(scorefiles_edf_path, scorefile_output, study_settings)

    compress_edfs(edf_input, edf_output, study_settings)

def classify_health_status(health_vars: pd.DataFrame):
    """
    Take a df of health variables (one row per subject) and format into a standard health dataframe, where each diagnosis can be true or false,
    and any diagnosis leads to the subjects begin classified as unhealthy.

    :param health_vars: dataframe of health variables, with one row per subject, and columns for each health variable.
    Varibles should be bool if they are a diagnosis/meds, or continous if they are some measurement of bmi or apnea
    :return: A dataframe where each row summarizes the healthiness of a subject (what meds they are taking, illness they have, etc)
    """
    diagnosis = ['apnea', 'treated_apnea', 'high_alcohol', 'narcolepsy',
                 'restless_leg_syndrome', 'excessive_sleepiness', 'depression', 'alzheimers',
                 'parkinsons', 'insomnia', 'heart_attack', 'stroke', 'other_sleep_disorder', 'other_health_concern']
    meds = ['psych_meds', 'sleep_meds']

    # levels
    apnea_levels = {'rdi': (15, np.inf), 'ahi': (15, np.inf), 'oahi': (15, np.inf), 'oai': (5, np.inf)}
    bmi_levels = {'bmi_obese': (30, np.inf), 'bmi_underweight': (-np.inf, 18.5)}
    plms_levels = {'plms':(60, np.inf)}
    levels = apnea_levels.copy()
    levels.update(bmi_levels)
    levels.update(plms_levels)

    identifiers = ['subjectid']
    if 'visitid' in health_vars.columns:
        identifiers.append('visitid')

    if 'bmi' not in health_vars.columns:
        if 'weight_kg' in health_vars.columns and 'height_cm' in health_vars.columns:
            health_vars['weight_kg'] = health_vars['weight_kg'].apply(lambda x: np.nan if isinstance(x, str) else float(x))
            health_vars['height_cm'] = health_vars['height_cm'].apply(lambda x: np.nan if isinstance(x, str) else float(x))
            health_vars['bmi'] = health_vars['weight_kg'] / (health_vars['height_cm']/100)**2

    health_vars_processed_cont = []
    default_health = {k: False for k in diagnosis + meds}
    for idx, row in health_vars.iterrows():
        current_health = default_health.copy()
        for id in identifiers:
            current_health[id] = row[id]

        for var_name, var_value in row.iteritems():
            if var_name in default_health:
                if var_value == 'A':
                    var_value = 1
                var_value = 0 if var_value in ['M','K','D','O'] else float(var_value)
                # if diag and already diag, then keep, if diag and not diag then add, if not diag then then do nothing
                var_value = False if np.isnan(var_value) else bool(var_value)
                current_health[var_name] |= var_value
            if var_name in apnea_levels:
                if apnea_levels[var_name][0] < var_value < apnea_levels[var_name][1]:
                    current_health['apnea'] = True
            if var_name == 'bmi':
                for bmi_var_name, bmi_limits in bmi_levels.items():
                    if bmi_limits[0] < var_value < bmi_limits[1]:
                        current_health[bmi_var_name] = True
                    else:
                        current_health[bmi_var_name] = False
        health_vars_processed_cont.append(pd.Series(current_health))

    health_vars_processed = pd.concat(health_vars_processed_cont, axis=1).T
    health_vars_processed = health_vars_processed.set_index(identifiers)
    if health_vars_processed.shape[1] == 0:
        health_vars_processed['healthy'] = True
    else:
        health_vars_processed['healthy'] = ~health_vars_processed.any(axis=1)

    return health_vars_processed.reset_index()


def compress_edfs(input_path, output_path, study_settings):
    files_to_convert = glob.glob(os.path.join(input_path,'*.edf'))
    chans_ok = list(study_settings['known_eeg_chans'].keys()) + list(study_settings['known_eog_chans'].keys())
    print('Copying EDFs')
    for edf_file in files_to_convert:
        _, file_base = os.path.split(edf_file)
        output_filepath = output_path+file_base
        if os.path.exists(output_filepath):
            print('File', output_filepath, 'exists. Skipping.')
            continue
        edf = mne.io.read_raw_edf(edf_file, stim_channel=None, verbose=False, exclude=['EMG Ant Tibial R', 'EMG Ant Tibial L', 'Resp themistance'])
        chans_to_rem = set(edf.ch_names) - set(chans_ok)
        chans_to_rem = list(chans_to_rem)+['EMG Ant Tibial R', 'EMG Ant Tibial L', 'Resp themistance']
        if len(chans_to_rem) > 0:
            print('shrinking',edf_file)
            del edf
            edf = mne.io.read_raw_edf(edf_file, stim_channel=None, verbose=False, exclude=chans_to_rem)
            edf.load_data()
            print(edf.ch_names)
            if edf.info['sfreq'] > 256:
                edf.resample(256)
            edf_tools.write_edf_from_mne_raw_array(edf, output_filepath, overwrite=True)
    print('Done copying EDFs')


def compress_edf_scorefiles(input_path, output_path, study_settings):

    files_to_convert = glob.glob(os.path.join(input_path,'*.edf'))
    print('Copying EDFs')
    for edf_file in files_to_convert:
        _, file_base = os.path.split(edf_file)
        output_filepath = (output_path + file_base).replace('.edf','.pkl')
        if os.path.exists(output_filepath):
            print('File', file_base, 'exists. Skipping.')
            epochstages, _, _ = pickle.load(open(output_filepath,'rb'))
            if not all([a == 'unknown' for a in epochstages]):
                continue
            else:
                print('All unknown, redoing')
        print('pickling', file_base)
        scorefile_output = scorefiles.extract_epochstages_from_scorefile(edf_file, study_settings['stage_map'])
        assert not all([a=='unknown' for a in scorefile_output[0]])
        pickle.dump(scorefile_output, open(output_filepath,'wb'))
    print('Done pickling scorefiles')


def check_edf_scorefiles(edf_input, input_path, output_path, study_settings):
    files_to_convert = glob.glob(os.path.join(edf_input,'*.edf'))
    print('Copying EDFs')
    for edf_file in files_to_convert:
        _, file_base = os.path.split(edf_file)
        edf_file = edf_file
        scorefile = input_path + file_base.split('_')[-1]
        output_filepath = (output_path + file_base).replace('.edf','.pkl')
        if os.path.exists(output_filepath):
            print('File', file_base, 'exists. Skipping.')
            continue
        print('pickling', file_base)
        raw_score = Dataset(scorefile)
        raw_edf = Dataset(edf_file)
        assert raw_score.header['n_samples']/raw_score.header['s_freq'] - raw_edf.header['n_samples']/raw_edf.header['s_freq'] < 0.0001
        start_diff = raw_score.header['start_time'] - raw_edf.header['start_time']
        epochstages, epochoffset, starttime = scorefiles.extract_epochstages_from_scorefile(scorefile, study_settings['stage_map'])
        starttime -= start_diff
        assert starttime == raw_edf.header['start_time']
        assert not all([a=='unknown' for a in epochstages])
        pickle.dump((epochstages, epochoffset, starttime), open(output_filepath,'wb'))
    print('Done pickling scorefiles')


def upload_data(base_folder):
    print('Uploading for ',base_folder)
    files = glob.glob(base_folder+'*/*', recursive=True)
    files = [file for file in files if os.path.getmtime(file) < (datetime.datetime.utcnow().timestamp()-24*60*60)]
    upload_helper.upload_via_upload_spec(folder_to_search_in=base_folder,
                                         password=mednickdb_pass,
                                         username=mednickdb_user,
                                         filetypes_to_ignore=['demographics','health','raw_sleep_eeg'],
                                         #files_to_ignore=files,
                                         auto_mode=True)



if __name__ == "__main__":

    # for mass_num in range(1,6):
    #     prep_data(
    #         edf_input='/data/DataToUpload/MASS_SS'+str(mass_num)+'/edfs/',
    #         edf_output=data_dir+'non_nsrr_studies/mass_ss'+str(mass_num)+'/raw_sleep_eeg/',
    #         scorefiles_edf_path='/data/DataToUpload/MASS_SS'+str(mass_num)+'/scorefiles/',
    #         scorefile_output=data_dir+'non_nsrr_studies/mass_ss'+str(mass_num)+'/sleep_scoring/',
    #         scorefile_fix_offset=True,
    #         study_settings_path=data_dir+'non_nsrr_studies/mass_ss'+str(mass_num)+'/study_settings/MASS_study_settings.yaml',
    #         health_file=data_dir+'non_nsrr_studies/mass_ss'+str(mass_num)+'/demographics/Demographics_MASS_SS'+str(mass_num)+'.xlsx',
    #         health_output=data_dir+'non_nsrr_studies/mass_ss'+str(mass_num)+'/health/health_mass_ss'+str(mass_num)+'.csv',
    #     )
    #
    #
    #
    # prep_data(
    #     edf_input='/data/DataToUpload/WamsleyLab_KEN/scorefiles/',
    #     edf_output=data_dir + 'non_nsrr_studies/wamsley_ken/raw_sleep_eeg/',
    #     scorefiles_edf_path='/data/DataToUpload/WamsleyLab_KEN/scorefiles/',
    #     scorefile_output=data_dir + 'non_nsrr_studies/wamsley_ken/sleep_scoring/',
    #     study_settings_path=data_dir + 'non_nsrr_studies/wamsley_ken/study_settings/wamsley_ken_study_settings.yaml',
    #     health_file=data_dir + 'non_nsrr_studies/wamsley_ken/demographics/Demographics_WamsleyLab_KEN.xlsx',
    #     health_output=data_dir + 'non_nsrr_studies/wamsley_ken/health/health_wamsley_ken.csv',
    # )


    # prep_data(
    #     edf_input='/data/DataToUpload/WamsleyLab_R21/scorefiles/',
    #     edf_output=data_dir + 'non_nsrr_studies/wamsley_r21/raw_sleep_eeg/',
    #     scorefiles_edf_path='/data/DataToUpload/WamsleyLab_R21/scorefiles/',
    #     scorefile_output=data_dir + 'non_nsrr_studies/wamsley_r21/sleep_scoring/',
    #     study_settings_path=data_dir + 'non_nsrr_studies/wamsley_r21/study_settings/wamsley_r21_study_settings.yaml',
    #     health_file=data_dir + 'non_nsrr_studies/wamsley_r21/demographics/Demographics_WamsleyLab_R21.xlsx',
    #     health_output=data_dir + 'non_nsrr_studies/wamsley_r21/health/health_wamsley_r21.csv',
    # )


    prep_data(
        edf_input='/data/DataToUpload/WamsleyLab_Future/scorefiles/',
        edf_output=data_dir + 'non_nsrr_studies/wamsley_future/raw_sleep_eeg/',
        scorefiles_edf_path='/data/DataToUpload/WamsleyLab_Future/scorefiles/',
        scorefile_output=data_dir + 'non_nsrr_studies/wamsley_future/sleep_scoring/',
        study_settings_path=data_dir + 'non_nsrr_studies/wamsley_future/study_settings/wamsley_future_study_settings.yaml',
        health_file=data_dir + 'non_nsrr_studies/wamsley_future/demographics/Demographics_WamsleyLab_Future.xlsx',
        health_output=data_dir + 'non_nsrr_studies/wamsley_future/health/health_wamsley_future.csv',
    )
    #
    # # sys.exit(0)
    # upload_data(base_folder=data_dir + 'non_nsrr_studies/mass_ss1/')
    # upload_data(base_folder=data_dir + 'non_nsrr_studies/mass_ss2/')
    # upload_data(base_folder=data_dir + 'non_nsrr_studies/mass_ss3/')
    # upload_data(base_folder=data_dir + 'non_nsrr_studies/mass_ss4/')
    # upload_data(base_folder=data_dir + 'non_nsrr_studies/mass_ss5/')
    # # #
    # upload_data(base_folder=data_dir + 'non_nsrr_studies/wamsley_ken/')
    # upload_data(base_folder=data_dir + 'non_nsrr_studies/wamsley_future/')
    # upload_data(base_folder=data_dir + 'non_nsrr_studies/wamsley_r21/')
    # upload_data(base_folder=data_dir + 'non_nsrr_studies/ucddb/')



