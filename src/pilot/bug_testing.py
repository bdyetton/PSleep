from mednickdb_pyapi.pyapi import MednickAPI
from mednickdb_pysleep.process_sleep_record import extract_eeg_variables
from mednickdb_pysleep.sleep_features import load_and_slice_data_for_feature_extraction
from mednickdb_pysleep.edf_tools import add_events_df_to_mne_raw
from mednickdb_pysleep.artifact_detection import epochs_with_artifacts_to_event_df
import os
import matplotlib.pyplot as plt
import numpy as np
import glob
import mne
import pandas as pd
import tempfile
from io import BytesIO
file_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(file_dir, '../../data/')
import pickle

bad_studies = ['homepap', 'cfs', 'ucddb']

def check_voltage_for_records(eeg_kind='std'):
    edf_files = glob.glob(os.path.join(data_dir,'scrap/*.edf'))
    edf_files = [f for f in edf_files if eeg_kind in f]
    for file in edf_files:
        studyid = file.split('\\')[-1].split('.')[0]
        # f = tempfile.NamedTemporaryFile(suffix='.edf', delete=False)
        # path = f.name
        # f.write(file)
        raw = mne.io.read_raw_edf(file)
        eeg_data = raw.get_data()
        print(studyid, 1e6*np.sum(np.abs(eeg_data))/eeg_data.size)


def download_some_files_for_testing(eeg_kind='std'):
    med_api = MednickAPI(username=os.environ['mednickapi_username'], password=os.environ['mednickapi_password'])
    studyids = med_api.get_unique_var_values('studyid', store='files', query='filetype='+eeg_kind+'_sleep_eeg')
    for studyid in studyids:
        files = med_api.get_files(query='studyid==' + studyid + ' and fileformat='+eeg_kind+'_sleep_eeg')
        a_file = np.random.choice(files)
        print('Downloading', a_file['filename'])
        file = med_api.download_file(fid=a_file['_id'])
        with open(os.path.join(data_dir,'scrap/'+studyid+'.edf'),'wb') as f:
            f.write(file)
        if eeg_kind == 'std':
            file_info = med_api.get_file_by_fid(fid=a_file['_id'])
            data = med_api.get_data(**{k:v for k,v in file_info.items() if k in ['studyid','visitid','subjectid']}, format='flat_dict')
            pickle.dump(data[0], open(os.path.join(data_dir,'scrap/'+studyid+'.pkl'),'wb'))

    
def test_feature_detectors(plot=True):
    edf_files = glob.glob(os.path.join(data_dir,'scrap/*.edf'))
    for file in edf_files:
        studyid = file.split('\\')[-1].split('.')[0]
        if studyid in bad_studies:
            continue
        data = pickle.load(open(file.replace('.edf','.pkl'), 'rb'))
        print('-----Working on',studyid,'------')

        features_df, power_df, _, _, epochs_with_artifacts = \
            extract_eeg_variables(edf_filepath=file,
                              epochstages=data['sleep_scoring.epochstages'],
                              epochoffset_secs=data['sleep_scoring.epochoffset_secs'],
                              do_artifacting=True,
                              do_spindles=False,
                              do_slow_osc=True,
                              do_rem=False,
                              do_band_power=False,
                              artifacting_channels=['C3'],
                              spindle_channels=['C3'],
                              slow_osc_channels=['C3'],
                              do_quartiles=False,
                              timeit=True)

        if plot:
            mne_raw = mne.io.read_raw_edf(file)
            if epochs_with_artifacts is not None:
                artifacts_df = epochs_with_artifacts_to_event_df(epochs_with_artifacts)
                mne_raw = add_events_df_to_mne_raw(mne_raw, artifacts_df, orig_time=data['sleep_scoring.epochoffset_secs'])
            mne_raw = add_events_df_to_mne_raw(mne_raw, features_df, orig_time=data['sleep_scoring.epochoffset_secs'])
            mne_raw.plot()
            plt.show()

    #bads = homepap, cfs, ucddb

def load_moda():
    moda_gs = pd.read_csv(os.path.join(data_dir,'scrap/moda/gold_standard_spindle_markers_MODA.txt'), delimiter='\t')
    def moda_to_mass_ids(x):
        parts = x.split('-')
        return int(parts[2]), 'mass_ss'+str(int(parts[1]))
    moda_gs['subjectid'], moda_gs['studyid'] = zip(*moda_gs['MASS subjectid'].apply(moda_to_mass_ids))
    moda_gs['onset'] = moda_gs['MASS Marker Start Time (s)']
    moda_gs['description'] = 'MODA GS spindle'
    moda_gs['duration'] = moda_gs['MASS Marker End Time (s)'] - moda_gs['MASS Marker Start Time (s)']
    cols = [c for c in moda_gs.columns if c not in ['onset','subjectid','studyid','duration','description']]
    return moda_gs.drop(cols, axis=1)


def download_moda_mass_records(moda_df, n=5):
    med_api = MednickAPI(username=os.environ['mednickapi_username'], password=os.environ['mednickapi_password'])
    ticker = 0
    for (studyid, subjectid), moda_data in moda_df.groupby(['studyid','subjectid']):
        files_eeg = med_api.get_files(
            query='studyid=' + studyid + ' and subjectid=' + str(subjectid) + ' and fileformat=std_sleep_eeg')
        if len(files_eeg) != 1:
            continue
        files_eeg = files_eeg[0]
        print('--downloading--', studyid, subjectid)
        file = med_api.download_file(fid=files_eeg['_id'])
        with open(os.path.join(data_dir, 'scrap/moda/studyid_' + studyid + '_subjectid_' +str(subjectid) + '.edf'), 'wb') as f:
            f.write(file)
        file_info = med_api.get_file_by_fid(fid=files_eeg['_id'])
        data = med_api.get_data(**{k: v for k, v in file_info.items() if k in ['studyid', 'visitid', 'subjectid']},
                                format='flat_dict')
        pickle.dump(data[0], open(os.path.join(data_dir, 'scrap/moda/studyid_' + studyid + '_subjectid_' +str(subjectid) + '.pkl'), 'wb'))
        if ticker > 5:
            break
        else:
            ticker += 1

def compare_moda_vs_spindle_algo(moda_df, plot=True):
    edf_files = glob.glob(os.path.join(data_dir, 'scrap/moda/*.edf'))
    for file in edf_files:
        studyid = file.split('studyid_')[-1].split('_subjectid')[0]
        subjectid = int(file.split('subjectid_')[-1].split('.')[0])
        moda_slice = moda_df.loc[(moda_df['studyid']==studyid) & (moda_df['subjectid']==subjectid),:]
        data = pickle.load(open(file.replace('.edf', '.pkl'), 'rb'))
        print('-----Working on', studyid, '------')
        features_df, power_df, _, _, epochs_with_artifacts = \
            extract_eeg_variables(edf_filepath=file,
                                  epochstages=data['sleep_scoring.epochstages'],
                                  epochoffset_secs=data['sleep_scoring.epochoffset_secs'],
                                  do_artifacting=True,
                                  do_spindles=True,
                                  do_slow_osc=True,
                                  do_rem=False,
                                  do_band_power=False,
                                  artifacting_channels=['C3'],
                                  spindle_channels=['C3'],
                                  slow_osc_channels=['C3'],
                                  do_quartiles=False,
                                  timeit=True)

        if plot:
            mne_raw = mne.io.read_raw_edf(file)
            if epochs_with_artifacts is not None:
                artifacts_df = epochs_with_artifacts_to_event_df(epochs_with_artifacts)
                mne_raw = add_events_df_to_mne_raw(mne_raw, artifacts_df,
                                                   orig_time=data['sleep_scoring.epochoffset_secs'])
                mne_raw = add_events_df_to_mne_raw(mne_raw, moda_slice, orig_time=data['sleep_scoring.epochoffset_secs'])
            mne_raw = add_events_df_to_mne_raw(mne_raw, features_df, orig_time=data['sleep_scoring.epochoffset_secs'])
            mne_raw.plot()
            plt.show()

def check_units():
    raw = mne.io.read_raw_edf('C:/Users/bdyet/Desktop/unit_testing.edf')
    some_edf_channels = ['C3','C4']
    chan_data = raw.get_data(some_edf_channels)
    chans = raw.ch_names
    print(np.sum(np.abs(chan_data))/chan_data.size)

    exclude = [ch for ch in chans if ch not in some_edf_channels]
    raw = mne.io.read_raw_edf('C:/Users/bdyet/Desktop/unit_testing.edf', exclude=exclude)
    chan_data = raw.get_data(some_edf_channels)
    print(np.sum(np.abs(chan_data))/chan_data.size)

    raw = mne.io.read_raw_edf('C:/Users/bdyet/Desktop/unit_testing_eeg_only.edf')
    chan_data = raw.get_data(some_edf_channels)
    print(np.sum(np.abs(chan_data))/chan_data.size)

def check_units_2():
    import pyedflib
    import numpy as np
    from wonambi import Dataset
    import mne
    f = pyedflib.EdfReader('C:/Users/bdyet/Desktop/unit_testing_eeg_only.edf')
    n = f.signals_in_file
    pyedf_sigs = np.zeros((n, f.getNSamples()[0]))
    for i in np.arange(n):
        pyedf_sigs[i, :] = f.readSignal(i)
    print('pyedf av in mV:', np.sum(np.abs(pyedf_sigs)) / pyedf_sigs.size)

    wnbi = Dataset('C:/Users/bdyet/Desktop/unit_testing_eeg_only.edf')
    wnbi_sigs = wnbi.read_data().data[0]
    print('Wonambi av in mV:', np.sum(np.abs(wnbi_sigs))/wnbi_sigs.size)

    raw = mne.io.read_raw_edf('C:/Users/bdyet/Desktop/unit_testing_eeg_only.edf')
    mne_sigs = raw.get_data()
    print('MNE av in V:', np.sum(np.abs(mne_sigs))/mne_sigs.size)


def get_bad_fids():
    bad_files = ['5d2edd85177ab50011bff074', '5d3ddeabd12a42001bba4e62', '5d2e519e177ab50011bfe295', '5d2e519a177ab50011bfe291', '5d2e4215177ab50011bfdf3d', '5d2e5e7d177ab50011bfe4b3', '5d2e5b0e177ab50011bfe419']
    med_api = MednickAPI(username=os.environ['mednickapi_username'], password=os.environ['mednickapi_password'])
    studyid = []
    subjectid=[]
    for fid in bad_files:
        a = med_api.get_file_by_fid(fid=fid)
        studyid.append(a['studyid'])
        subjectid.append(a['subjectid'])

    b = pd.DataFrame({'studyid':studyid, 'subjectid':subjectid})
    print(b.sort_values(['studyid','subjectid']))

def mass_error():
    med_api = MednickAPI(username=os.environ['mednickapi_username'], password=os.environ['mednickapi_password'])
    file_info = med_api.get_files(query='subjectid=8 and studyid=mass_ss4 and fileformat=std_sleep_eeg', )[0]
    file = med_api.download_file(fid=file_info['_id'])
    with open('temp_edf.edf','wb') as f:
        f.write(file)
    data = med_api.get_data(query='subjectid=8 and studyid=mass_ss4',
                            format='flat_dict')[0]

    features_df, power_df, _, _, epochs_with_artifacts = \
        extract_eeg_variables(edf_filepath='temp_edf.edf',
                              epochstages=data['sleep_scoring.epochstages'],
                              epochoffset_secs=data['sleep_scoring.epochoffset_secs'],
                              do_artifacting=True,
                              do_spindles=False,
                              do_slow_osc=False,
                              do_rem=True,
                              do_band_power=False,
                              artifacting_channels=['C3'],
                              spindle_channels=['C3'],
                              slow_osc_channels=['C3'],
                              do_quartiles=False,
                              timeit=True)
    features_df.to_csv(data_dir+'temp/lala.csv')
    assert all(features_df['stage'] == 'rem')

def remove_data_with_bad_health():
    med_api = MednickAPI(username=os.environ['mednickapi_username'], password=os.environ['mednickapi_password'])
    file_info = med_api.get_data(query='studyid=cfs and health.healthy not in [true,false]', data_only=False)
    for _, row in file_info.iterrows():
        med_api.delete_data(id=row['_id'])

def check_spindles():
    med_api = MednickAPI(username=os.environ['mednickapi_username'], password=os.environ['mednickapi_password'])
    data = med_api.get_data(query='filetype=sleep_features and health.healthy=True and demographics.age<40')
    print(data['sleep_features.spindle_algo'].value_counts())
    print(data['sleep_features.spindle_algo'].shape[0])
    data = data.loc[data['sleep_features.spindle_algo']=='Wamsley2012',:]
    print('num missing', np.sum(pd.isna(data['sleep_features.n3_C3_spindle_density'])))
    plt.hist(data['sleep_features.n3_C3_spindle_density'])
    plt.show()
    pass




if __name__ == '__main__':
    check_spindles()
    #remove_data_with_bad_health()
    # a,b,c = pickle.load(open('C:/Users/bdyet/Downloads/subjectid46__1563301995088.pkl','rb'))
    # mass_error()
    #get_bad_fids()
    #check_units_2()
    #moda = load_moda()
    #download_moda_mass_records(moda)
    #compare_moda_vs_spindle_algo(moda)
    #download_some_files_for_testing()
    # check_voltage_for_records()
    # test_feature_detectors()

    #On axon:
    # download std for each good file
    # run detector
    # plot spindles, so, bad epochs and rem for each - check consistancy
    # compare spindles on mass...
