import subprocess
import os
from mednickdb_pyapi import upload_helper, pyapi
file_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(file_dir, '../../data/nsrr/')
mednickdb_user = os.environ['mednickapi_username']
mednickdb_pass = os.environ['mednickapi_password']
import pandas as pd
import numpy as np
import mne
import wonambi.ioeeg.edf as wnbi_edf_reader
import glob
from mednickdb_pysleep import edf_tools
import warnings
import sys

debug=False


class NsrrData():
    def __init__(self,
                 mednickdb_upload_specifiers: dict,
                 nsrr_edf_path: str,
                 edf_file_specifiers_template: str,
                 nsrr_scoring_path: str,
                 scoring_specifiers_template: str,
                 nsrr_variables_path: str,
                 demographics_map: dict,
                 shrink_edfs: bool = True,
                 subjectid_format='{:d}',
                 to_upload: list = ('sleep_scoring', 'health', 'demographics', 'study_settings', 'raw_sleep_eeg'),
                 to_download: list = ('sleep_scoring', 'health', 'demographics', 'raw_sleep_eeg')):
        """
        Init a NSRR dataset with all required args to download, process_datasets and upload to mednickdb
        :param mednickdb_upload_specifiers: key:value pairs that should be attached to this data when uploaded to mednickdb
        :param nsrr_edf_path: where on the NSRR server to get the data edf files from
        :param edf_file_specifiers_template: How to parse mednickdb specifiers from the edf filenames. See upload_helper for format.
        :param nsrr_scoring_path: where on the NSRR server to get the scoring files from
        :param scoring_specifiers_template: how to parse mendnickdb specifiers from the scoring filenames
        :param nsrr_variables_path: where on the NSRR server to find variables
        :param demographics_map: How to map variables in the nsrr_variables file to demographics. Format is:
            {nsrr_variable_name:(mednickdb_variable_name,
                                {nsrr_value_name1:mednickdb_value_name1, nsrr_value_name2:mednickdb_value_name2})}.
            See slice_rename_and_map_df_vars for more info.
        """
        self.mednickdb_upload_specifiers = mednickdb_upload_specifiers
        self.nsrr_edf_path = nsrr_edf_path
        self.edf_file_specifiers_template = edf_file_specifiers_template
        self.nsrr_scoring_path = nsrr_scoring_path
        self.nsrr_variables_path = nsrr_variables_path
        self.scoring_template = scoring_specifiers_template
        self.demographics_map = demographics_map
        self.data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../data/nsrr/')
        self.mednickdb_user = os.environ['mednickapi_username']
        self.mednickdb_pass = os.environ['mednickapi_password']
        self.nsrr_token = b'946-bemqvUJMcVWJqpyZfUz3\n'
        self.shrink_edfs = shrink_edfs
        self.to_upload = to_upload
        self.to_download = to_download
        self.bad_sub_visits = []
        self.good_sub_visits = None
        self.subjectid_format = subjectid_format
        self.overwrite_db_files = ['health', 'demographics', 'study_settings']

    @staticmethod
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
        levels = apnea_levels.copy()
        levels.update(bmi_levels)

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
        health_vars_processed['healthy'] = ~health_vars_processed.any(axis=1)

        return health_vars_processed.reset_index()

    @staticmethod
    def slice_rename_and_map_df_vars(df: pd.DataFrame, name_val_remap: dict):
        """
        Slice given dataframe to extract a set of variables. Rename these variables if required, and then remap the variables values if desired.
        :param df: dataframe to slice, rename and value map
        :param name_val_remap: a dict of format {existing_var_name:(new_var_name, {existing_value1:new_value1, existing_value1:new_value1})}
            Every var desired should be entered, enter None when no var name remap is required, and None if no value map is required.
            Values that do not have a mapping in the value map will be passed through unaffected.
        :return:
        """
        var_map = {k: v[0] for k, v in name_val_remap.items() if v[0] is not None}
        val_map = {(k if v[0] is None else v[0]): v[1] for k, v in name_val_remap.items() if v[1] is not None}

        df = df.loc[:, list(name_val_remap.keys())]
        df.columns = [var_map[col] if col in var_map else col for col in df.columns]
        for k, v in val_map.items():
            df[k] = df[k].map(v)
        return df


    def nsrr_download(self, download_path):
        """Download data from the NSRR. Writes data to data dir + download path.
        :param download_path: the path of files on www.sleepdata.org that you want to download

        """
        print('Downloading from', download_path)
        with subprocess.Popen(['nsrr download '+download_path + ' --fast'],
                              stdin=subprocess.PIPE,
                              shell=True,
                              cwd=data_dir) as nsrr_shell:
            nsrr_shell.communicate(self.nsrr_token)


    def run(self):
        """Run all tasks for the Cleavland Family Study data"""
        if ('health' in self.to_download) or ('demographics' in self.to_download):
            self.nsrr_download(self.nsrr_variables_path)
        self.process_datasets()
        self.pick_healthy_subs()
        if 'raw_sleep_eeg' in self.to_download:
            self.nsrr_download_good_subs(self.nsrr_edf_path, self.edf_file_specifiers_template)
        if 'sleep_scoring' in self.to_download:
            self.nsrr_download_good_subs(self.nsrr_scoring_path, self.scoring_template)

        self.upload()


    def nsrr_download_good_subs(self, base_path, template):
        compression_ticker = 0
        self.compress_edfs()
        if self.good_sub_visits is not None:
            for i, sub_visit in enumerate(self.good_sub_visits):
                print(i,'of',len(self.good_sub_visits))
                if sub_visit['visitid'] is None or np.isnan(sub_visit['visitid']):
                    sub_visit['visitid'] = self.mednickdb_upload_specifiers['visitid']
                path = base_path + template.replace('{subjectid}',
                                                    self.subjectid_format.format(int(sub_visit['subjectid']))).replace('{visitid}',
                                                                                         str(sub_visit['visitid']))
                if os.path.exists(self.data_dir + path):
                    print('Sub already downloaded. Skipping.')
                    compression_ticker -= 1
                    continue
                self.nsrr_download(path)
                compression_ticker += 1
                if compression_ticker > 50:
                    self.compress_edfs()
                    compression_ticker = 0
        else:
            self.nsrr_download(self.nsrr_edf_path)
        self.compress_edfs()


    def process_datasets(self):
        """Process the variable data from the Cleavland Family Study into demographics and health data.
        write this data locally to disk (so the upload helper can be called on them)"""
        cfs_vars_og = pd.read_csv(self.data_dir + self.nsrr_variables_path)
        if debug:
            cfs_vars_og = cfs_vars_og.head(n=50)
        cfs_vars_og.columns = [col.lower() for col in cfs_vars_og.columns]
        demo_vars = self.slice_rename_and_map_df_vars(cfs_vars_og, self.demographics_map)
        demo_vars['ethnicity'] = demo_vars.apply(lambda x: x['race'] if x['ethnicity'] is None else x['ethnicity'], axis=1)
        if 'race' in demo_vars.columns:
            demo_vars = demo_vars.drop(['race'], axis=1)

        # fix datatype of cols
        for spec in demo_vars.columns:  # FIXME: this should be better baked into the mednickdb_pyapi, explicit for now
            dtype_wanted = type(demo_vars[spec][0])
            demo_vars[spec] = demo_vars[spec].astype(dtype_wanted)

        demo_vars.to_csv(self.data_dir +'/' + self.mednickdb_upload_specifiers['studyid'] + '/datasets/demographics.csv', index=False)

        health_var_map = pd.read_excel(self.data_dir +'/' + self.mednickdb_upload_specifiers['studyid'] + '/mednickdb/health_map.xlsx', index_col=0).T
        health_var_map = health_var_map.drop('acceptable_level', axis=1).squeeze().to_dict()
        health_var_map = {k:(v,None) for k,v in health_var_map.items()}
        health_var_map.update(self.demographics_map)

        health_vars = self.slice_rename_and_map_df_vars(cfs_vars_og, health_var_map)
        health_vars = self.classify_health_status(health_vars)

        # fix datatype of cols
        for spec in health_vars.columns:  # FIXME: this should be better baked into the mednickdb_pyapi, explicit for now
            dtype_wanted = type(health_vars[spec][0])
            health_vars[spec] = health_vars[spec].astype(dtype_wanted)

        health_vars.to_csv(self.data_dir +'/' + self.mednickdb_upload_specifiers['studyid'] + '/datasets/health.csv', index=False)

    def compress_edfs(self):
        edf_path = self.data_dir+os.path.dirname(self.nsrr_edf_path)
        files_to_convert = glob.glob(os.path.join(edf_path,'*.edf'))
        print('Checking EDFs')
        for edf_file in files_to_convert:
            edf = mne.io.read_raw_edf(edf_file, stim_channel=None, verbose=False)
            chans_to_rem = set(edf.ch_names) - {'C3','C4','LOC','ROC','M1','M2','A1','A2','EOG(L)','EOG(R)','EEG','EEG(sec)','C3-M2','C4-M1'}
            chans_to_rem = {a.replace('-0','').replace('-1','').replace('-2','') for a in chans_to_rem} #remove running names
            if set(edf.ch_names) == chans_to_rem:
                print('File is missing key channels, ignoring')
                continue
            elif len(chans_to_rem) > 0:
                print('shrinking',edf_file)
                del edf
                edf = mne.io.read_raw_edf(edf_file, verbose=False, exclude=chans_to_rem)
                edf.load_data()
                print(edf.ch_names)
                if not (1 < 1e6*np.sum(np.abs(edf.get_data()))/edf.get_data().size < 200):
                    print('Removing a mV error file')
                    os.remove(edf_file)
                    continue
                if edf.info['sfreq'] > 256:
                    edf.resample(256)
                edf_tools.write_edf_from_mne_raw_array(edf, edf_file, overwrite=True)
            else:
                pass
                #print('Already compressed', edf_file)
        print('Done Checking EDFs')

    def pick_healthy_subs(self):
        health_data = pd.read_csv(
            self.data_dir + '/' + self.mednickdb_upload_specifiers['studyid'] + '/datasets/' + 'health.csv')
        itentifiers = ['subjectid']
        if 'visitid' in health_data:
            itentifiers.append('visitid')
        elif 'visitid' in self.mednickdb_upload_specifiers:
            health_data['visitid'] = self.mednickdb_upload_specifiers['visitid']
            itentifiers.append('visitid')

        self.bad_sub_visits = health_data.loc[~health_data['healthy'], itentifiers].to_dict(orient='records')
        self.good_sub_visits = health_data.loc[health_data['healthy'], itentifiers].to_dict(orient='records')

        healthy_data_only = health_data.loc[health_data['healthy'], :]
        healthy_data_only.to_csv(
            self.data_dir + '/' + self.mednickdb_upload_specifiers['studyid'] + '/datasets/' + 'health.csv', index=False)

        demo_data = pd.read_csv(
            self.data_dir + '/' + self.mednickdb_upload_specifiers['studyid'] + '/datasets/' + 'demographics.csv')

        if 'visitid' in self.mednickdb_upload_specifiers:
            demo_data['visitid'] = self.mednickdb_upload_specifiers['visitid']

        demo_data = pd.merge(healthy_data_only, demo_data, on=itentifiers)
        cols_to_drop = list(healthy_data_only.columns)
        for itentifier in itentifiers:
            cols_to_drop.remove(itentifier)
        demo_data = demo_data.drop(cols_to_drop, axis=1)

        demo_data.to_csv(
            self.data_dir + '/' + self.mednickdb_upload_specifiers['studyid'] + '/datasets/' + 'demographics.csv', index=False)


    def existing_files_to_ignore(self, med_api, query):
        if not any([a in query for a in self.overwrite_db_files]):
            ignore = med_api.extract_var(med_api.get_files(query), 'filename')
            return ignore + self.bad_sub_visits
        else:
            return self.bad_sub_visits


    def upload(self):
        """Upload all data for the Cleavland Family Study to the mednickdb via the upload_helper tool"""
        base_specifiers = self.mednickdb_upload_specifiers.copy()
        base_specifiers['versionid'] = 1
        base_specifiers['fileformat'] = 'study_settings'
        base_specifiers['filetype'] = 'study_settings'

        med_api = pyapi.MednickAPI(username=mednickdb_user, password=mednickdb_pass)


        # upload study settings
        if base_specifiers['filetype'] in self.to_upload:
            ignore = self.existing_files_to_ignore(med_api, 'filetype=study_settings')
            upload_helper.run_upload_helper(self.data_dir +'/' + self.mednickdb_upload_specifiers['studyid'] + '/mednickdb/',
                                            '*study_settings.yaml',
                                            base_specifiers,
                                            username=mednickdb_user,
                                            password=mednickdb_pass,
                                            files_to_ignore=ignore,
                                            auto_mode=True)


        #upload scoring
        base_specifiers = self.mednickdb_upload_specifiers.copy()
        base_specifiers['versionid'] = 1
        base_specifiers['fileformat'] = 'sleep_scoring'
        base_specifiers['filetype'] = 'sleep_scoring'
        if base_specifiers['filetype'] in self.to_upload:
            ignore = self.existing_files_to_ignore(med_api, 'filetype=sleep_scoring')
            upload_helper.run_upload_helper(self.data_dir +'/' + os.path.dirname(self.nsrr_scoring_path),
                                            self.scoring_template,
                                            base_specifiers,
                                            username=mednickdb_user,
                                            password=mednickdb_pass,
                                            files_to_ignore=ignore,
                                            auto_mode=True)


        #upload edfs
        base_specifiers = self.mednickdb_upload_specifiers.copy()
        base_specifiers['versionid'] = 1
        base_specifiers['fileformat'] = 'raw_sleep_eeg'
        base_specifiers['filetype'] = 'raw_sleep_eeg'
        if base_specifiers['filetype'] in self.to_upload:
            ignore = self.existing_files_to_ignore(med_api, 'filetype=raw_sleep_eeg')
            upload_helper.run_upload_helper(self.data_dir+os.path.dirname(self.nsrr_edf_path),
                                            self.edf_file_specifiers_template,
                                            base_specifiers,
                                            username=mednickdb_user,
                                            password=mednickdb_pass,
                                            files_to_ignore=ignore,
                                            auto_mode=True)

        #upload demographics
        base_specifiers = self.mednickdb_upload_specifiers.copy()
        base_specifiers['versionid'] = 1
        base_specifiers['fileformat'] = 'tabular'
        base_specifiers['filetype'] = 'demographics'
        if base_specifiers['filetype'] in self.to_upload:
            ignore = self.existing_files_to_ignore(med_api, 'filetype=demographics')
            upload_helper.run_upload_helper(self.data_dir + '/' + self.mednickdb_upload_specifiers['studyid'] + '/datasets/',
                                            'demographics.csv',
                                            base_specifiers,
                                            username=mednickdb_user,
                                            password=mednickdb_pass,
                                            files_to_ignore=ignore,
                                            auto_mode=True)

        #upload health
        base_specifiers['fileformat'] = 'tabular'
        base_specifiers['filetype'] = 'health'
        if base_specifiers['filetype'] in self.to_upload:
            ignore = self.existing_files_to_ignore(med_api, 'filetype=health')
            upload_helper.run_upload_helper(self.data_dir +'/' + self.mednickdb_upload_specifiers['studyid'] + '/datasets/',
                                            'health.csv',
                                            base_specifiers,
                                            username=mednickdb_user,
                                            password=mednickdb_pass,
                                            files_to_ignore=ignore,
                                            auto_mode=True)


if __name__ == "__main__":

    cfs_demographics_map = {'nsrrid':('subjectid', None),
                            #'visit':('visitid', None),
                            'age':(None, None),
                            'sex':(None,{0:'F',1:'M'}),
                            'race':(None, {1:'white', 2:'black', 3:'other'}),
                            'ethnicity': (None, {0: None, 1: 'hispanic'}),
                            'bmi':(None, None)}

    cfs_data = NsrrData(
        to_upload=['sleep_scoring','health','demographics'],#['study_settings'],#['sleep_scoring','health','demographics', 'study_settings','raw_sleep_eeg'],
        to_download=[],  # ['sleep_scoring','health','demographics', 'study_settings','raw_sleep_eeg'],
        mednickdb_upload_specifiers={'studyid':'cfs', 'visitid':5},
        nsrr_edf_path='cfs/polysomnography/edfs/',
        edf_file_specifiers_template='cfs-visit{visitid}-{subjectid}.edf',
        nsrr_scoring_path='cfs/polysomnography/annotations-events-nsrr/',
        scoring_specifiers_template='cfs-visit{visitid}-{subjectid}-nsrr.xml',
        nsrr_variables_path='cfs/datasets/cfs-visit5-dataset-0.4.0.csv',
        demographics_map=cfs_demographics_map,
                    )
    cfs_data.run()

    # mros_demographics_map = {'nsrrid':('subjectid', lambda x: int(x.replace('AA',''))),
    #                          'visit': ('visitid', None),
    #                         'vsage1':('age', None),
    #                         'gender':('sex',{1:'F',2:'M'}),
    #                         'gierace':('ethnicity', {1:'white', 2:'black', 3:'asian', 4: 'hispanic', 5:'other'}),
    #                         'hwhgt':('height_cm', None),
    #                         'hwwgt':('weight_kg', None)}
    #
    # mros_data = NsrrData(
    #     shrink_edfs = False,
    #     to_upload=['raw_sleep_eeg'],#['sleep_scoring','health','demographics', 'study_settings','raw_sleep_eeg'],
    #     to_download=['raw_sleep_eeg'],  # ['sleep_scoring','health','demographics', 'study_settings','raw_sleep_eeg'],
    #     mednickdb_upload_specifiers={'studyid':'mros'},
    #     nsrr_edf_path='mros/polysomnography/edfs/visit1/',
    #     edf_file_specifiers_template='mros-visit{visitid}-aa{subjectid}.edf',
    #     nsrr_scoring_path='mros/polysomnography/annotations-events-nsrr/visit1/',
    #     scoring_specifiers_template='mros-visit{visitid}-aa{subjectid}-nsrr.xml',
    #     nsrr_variables_path='mros/datasets/mros-visit1-dataset-0.3.0.csv',
    #     subjectid_format='{:04d}',
    #     demographics_map=mros_demographics_map,
    #                 )
    # mros_data.run()
    #
    #
    # sof_demographics_map = {'sofid':('subjectid', None),
    #                         'visit': ('visitid', None),
    #                         'v8age':('age', None),
    #                         'gender':('sex',{1:'F',2:'M'}),
    #                         'race':('ethnicity', {1:'white', 2:'black'}),
    #                         'v8bmi':('bmi', None)}
    #
    # sof_data = NsrrData(
    #     shrink_edfs = False,
    #     to_upload=['sleep_scoring','health','demographics', 'study_settings','raw_sleep_eeg'],
    #     to_download=[],  # ['sleep_scoring','health','demographics', 'study_settings','raw_sleep_eeg'],
    #     mednickdb_upload_specifiers={'studyid':'sof'},
    #     nsrr_edf_path='sof/polysomnography/edfs/',
    #     edf_file_specifiers_template='sof-visit-{visitid}-{subjectid}.edf',
    #     nsrr_scoring_path='sof/polysomnography/annotations-events-nsrr/',
    #     scoring_specifiers_template='sof-visit-{visitid}-{subjectid}-nsrr.xml',
    #     nsrr_variables_path='sof/datasets/sof-visit-8-dataset-0.5.0.csv',
    #     subjectid_format ='{:05d}',
    #     demographics_map=sof_demographics_map,
    #                 )
    # sof_data.run()

    # homepap_demographics_map = {'nsrrid':('subjectid', None),
    #                         'visit': ('visitid', None),
    #                         'age':('age', None),
    #                         'gender':('sex',{1:'F',2:'M'}),
    #                         'race3':('race', {1:'white', 2:'black', 3:'other'}),
    #                         'ethnicity': (None, {0: None, 1: 'hispanic', 2:None}),
    #                         'bmi':('bmi', None)}
    #
    # homepap_data = NsrrData(
    #     shrink_edfs = False,
    #     to_upload=['sleep_scoring','health','demographics', 'study_settings','raw_sleep_eeg'],
    #     to_download=['sleep_scoring','health','demographics', 'study_settings','raw_sleep_eeg'],
    #     mednickdb_upload_specifiers={'studyid':'homepap'},
    #     nsrr_edf_path='homepap/polysomnography/edfs/lab/full/',
    #     edf_file_specifiers_template='homepap-lab-full-{subjectid}.edf',
    #     nsrr_scoring_path='homepap/polysomnography/annotations-events-nsrr/lab/full/',
    #     scoring_specifiers_template='homepap-lab-full-{subjectid}-nsrr.xml',
    #     nsrr_variables_path='homepap/datasets/homepap-baseline-dataset-0.1.0.csv',
    #     subjectid_format ='{:d}',
    #     demographics_map=homepap_demographics_map,
    #                 )
    # homepap_data.run()


    # shhs_demographics_map = {'nsrrid':('subjectid', None),
    #                         'age_s1':('age', None),
    #                         'visitnumber':('visitid', None),
    #                         'gender':('sex',{1:'M',2:'F'}),
    #                         'race':(None, {1:'white', 2:'black', 3:'other'}),
    #                         'ethnicity': (None, {0: None, 1: 'hispanic', 2: None,}),
    #                         'bmi_s1':('bmi', None)}
    #
    # shhs_data = NsrrData(
    #                     to_upload=['study_settings'],#['sleep_scoring','health','demographics', 'study_settings','raw_sleep_eeg'],
    #                     to_download = [],#['sleep_scoring','health','demographics', 'study_settings','raw_sleep_eeg'],
    #                     mednickdb_upload_specifiers= {'studyid': 'shhs'},
    #                     nsrr_edf_path='shhs/polysomnography/edfs/shhs1/',
    #                     edf_file_specifiers_template='shhs{visitid}-{subjectid}.edf',
    #                     nsrr_scoring_path='shhs/polysomnography/annotations-events-nsrr/shhs1/',
    #                     scoring_specifiers_template='shhs{visitid}-{subjectid}-nsrr.xml',
    #                     nsrr_variables_path='shhs/datasets/shhs1-dataset-0.13.0.csv',
    #                     demographics_map=shhs_demographics_map,
    #                     )
    # shhs_data.run()


