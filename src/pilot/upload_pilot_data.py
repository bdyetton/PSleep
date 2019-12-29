from mednickdb_pyapi.upload_helper import run_upload_helper

if __name__ == '__main__':

    #Upload some scorefiles/
    pilot_data_loc = '../../../DatabaseData/NSRR_CFS/'
    spec_for_all = {'studyid':'NSRR_CFS', 'versionid':1, 'filetype': 'stage_map', 'fileformat': 'stage_map'}
    run_upload_helper(pilot_data_loc, '*stagemap*.xlsx', spec_for_all)

    spec_for_all.update({'filetype':'sleep_scoring', 'fileformat':'sleep_scoring'})
    files_to, files_uploaded = run_upload_helper(pilot_data_loc+'scorefiles/', 'subjectid{subjectid}_visitid{visitid}*.xml', spec_for_all)

    #Upload demographics
    spec_for_all.update({'filetype':'demographics', 'fileformat':'tabular'})
    run_upload_helper(pilot_data_loc, '*Demographics*.xlsx', spec_for_all)