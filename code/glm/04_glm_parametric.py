# General Linear Model (GLM) for Parametric Modulation Beta Maps
# Author: Frederic R. Hopp
# April 2024

import os
import time, random
import glob
import numpy as np
import pandas as pd
from nltools.file_reader import onsets_to_dm
from nltools.stats import zscore
from nltools.data import Brain_Data, Design_Matrix
import argparse
from hemodynamic_models import * 

time.sleep(random.randint(5, 20))

parser = argparse.ArgumentParser()
parser.add_argument('-subject', type=str, required=True, dest='subject')

args = parser.parse_args()
subject = 'sub-' + args.subject

scale = True
study = 'ucsb_1'

fwhm = 8
spike_cutoff = 3
first_run = 1
output_dir = f'/srv/lab/fmri/mft/fhopp_diss/analysis/signature/results/glm/{study}/'

if study == 'ucsb_1' or study=='ucsb_2' or study=='uva':
    tr = 0.72
if study == 'duke':
    tr = 2

sampling_freq = 1./tr

conv_func = spm_hrf(tr, oversampling=1.0)
conv_deriv = spm_time_derivative(tr, oversampling=1.0)

def load_bids_events(onset_path, beh_path, n_tr):
    '''Create a design_matrix instance from BIDS event file'''

    onsets = pd.read_csv(onset_path, sep='\t',usecols=['onset','duration','trial_type'])

    if study == 'duke':
        onsets = pd.read_csv(onset_path, sep='\t',usecols=['trialstart_zeroed','duration','trial_type'])
        onsets = onsets.rename(columns={'trialstart_zeroed':'onset'})
        onsets = onsets[['onset','duration','trial_type']]
    
    if study == 'uva':
        onsets = pd.read_csv(onset_path, sep='\t',usecols=['onset','duration','trial_type'])
        onsets['onset'] = onsets['onset'] - (11*tr) # account for 11 dummy scans at beginning
    
    behavior = pd.read_csv(beh_path, sep='\t', usecols=['moral_decision'])
    onsets = onsets.join(behavior)
    onsets = onsets[onsets['moral_decision'] != 'n/a']
    onsets.dropna(subset=['moral_decision'], inplace=True)
    judgment_mean = onsets['moral_decision'].mean()

    onsets.reset_index(drop=True, inplace=True)
    onsets['trial_type'] = onsets['moral_decision'].astype(int).astype(str)
    onsets = onsets[['onset','duration','trial_type']]
    onsets.columns = ['Onset', 'Duration', 'Stim']
    ratings_dm = onsets_to_dm(onsets, sampling_freq=1./tr, run_length=n_tr)
  
    return ratings_dm, judgment_mean

def make_motion_covariates(mc):
    z_mc = zscore(mc)
    z_mc.columns = [f"z_{col}" for col in z_mc.columns]

    z_mc_sq = z_mc**2
    z_mc_sq.columns = [f"sq_{col}" for col in z_mc_sq.columns]

    z_mc_diff = z_mc.diff()
    z_mc_diff.columns = [f"diff_{col}" for col in z_mc_diff.columns]

    z_mc_diff_sq = z_mc_diff**2
    z_mc_diff_sq.columns = [f"diff_sq_{col}" for col in z_mc_diff_sq.columns]

    all_mc = pd.concat([z_mc, z_mc_sq, z_mc_diff, z_mc_diff_sq], axis=1)
    all_mc.fillna(value=0, inplace=True)
    
    return Design_Matrix(all_mc, sampling_freq=1/tr)

num_runs = 3

if study == 'ucsb_1' and subject == 'sub-35':
    num_runs = 2

if study == 'ucsb_2':
    if subject == 'sub-13' or subject == 'sub-21':
        first_run = 2
        
all_runs = Design_Matrix(sampling_freq = sampling_freq)
fmri_data = Brain_Data()

for run in range(first_run, num_runs+1):
    print('Loading run:', run)
    nifti_path = f'/srv/lab/fmri/mofomic/{study}/derivatives/fmriprep/{subject}/{subject}/func/{subject}_task-mfv_run-0{str(run)}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'
    cov_path = f'/srv/lab/fmri/mofomic/{study}/derivatives/fmriprep/{subject}/{subject}/func/{subject}_task-mfv_run-0{str(run)}_desc-confounds_timeseries.tsv'
    onset_path = f'/srv/lab/fmri/mofomic/{study}/{subject}/func/{subject}_task-mfv_run-0{str(run)}_events.tsv'
    beh_path = f'/srv/lab/fmri/mofomic/{study}/{subject}/beh/{subject}_task-mfv_run-0{str(run)}_beh.tsv'

    # 1) Load data and onsets for this run
    if study == 'ucsb_1' or study == 'uva':
        data = Brain_Data(nifti_path)[11:-11].smooth(fwhm=fwhm) # drop 11 dummy scans at beginning and end
    if study == 'ucsb_2':
        data = Brain_Data(nifti_path).smooth(fwhm=fwhm)
    if study == 'duke':
        data = Brain_Data(nifti_path).smooth(fwhm=fwhm)
    
    if scale == True:
        # Rescale Data so that each voxel is scaled proportional to percent signal change
        fmri_data = fmri_data.append(data.scale())
    else:
        fmri_data = fmri_data.append(data)

    n_tr = data.shape()[0]
    

    dm, judgment_mean = load_bids_events(onset_path, beh_path, n_tr)
    dm.insert(0, 'vig', 0.0)
    '''Below part is a bit clunky; we add the moral judgment as a parametric modulator'''
    # Because the values are stored in columns, we need to convert to a single column
    moral_judgments = []
    trials = []
    for i,row in dm.iloc[:,1:].iterrows():
        if row.sum() < 1:
            moral_judgments.append(0.0)
            trials.append(0)
        else:
            for c,v in row.items():
                if int(v) != 0:
                    moral_judgments.append(float(c) - judgment_mean)
                    trials.append(1)
    dm['moral_decision'] = pd.Series(moral_judgments)
    dm['vig'] = pd.Series(trials)
    dm = dm[['vig','moral_decision']]
    dm_conv = dm.convolve(conv_func)
    dm_deriv = dm.convolve(conv_deriv)
    dm_deriv.columns = ['deriv_'+c for c in dm_deriv.columns]
    dm_conv = dm_conv.append(dm_deriv, axis=1)

    # 2) Load in covariates for this run
    covariates = pd.read_csv(cov_path, sep='\t')[11:-11].reset_index(drop=True)
    cov = make_motion_covariates(covariates[['trans_x','trans_y','trans_z','rot_x', 'rot_y', 'rot_z']])
    spikes = data.find_spikes(global_spike_cutoff=spike_cutoff, diff_spike_cutoff=spike_cutoff)
    cov = cov.add_dct_basis(duration=90).add_poly(order=1, include_lower=True)
    
    full = dm_conv.append(cov,axis=1).append(Design_Matrix(spikes.iloc[:, 1:], sampling_freq=1/tr), axis=1)

    all_runs = all_runs.append(full,axis=0,unique_cols=cov.columns)
    print('Finished run:', run)

fmri_data.X = all_runs
stats = fmri_data.regress()

# Store beta maps for parametric effect of this run
for i, col in enumerate(fmri_data.X.columns):
    if col.startswith('moral_decision'):
        stats['beta'][i].write(output_dir + f"{subject}_parametric_smooth{fwhm}.nii.gz")