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

# Define arguments
parser = argparse.ArgumentParser()
parser.add_argument('-subject', type=str, required=True, dest='subject')

args = parser.parse_args()
sub = 'sub-' + args.subject

output_dir = "/srv/lab/fmri/mft/intention/derivatives/glm/"

tr = 2
fwhm = 8
spike_cutoff = 3
sampling_freq = 1./tr

conv_func = spm_hrf(tr, oversampling=1.0)
conv_deriv = spm_time_derivative(tr, oversampling=1.0)
# select either "whole-trial" or "intention" for modeling
trial_window = 'intention'

def load_bids_events(onset_path, n_tr):

    onsets = pd.read_csv(onset_path, sep='\t')
    onsets['condition'] = onsets['condition'].astype(str)
    onsets['trial_type'] = onsets['condition']
    if trial_window == 'intention':
        onsets['onset'] = onsets['onset'] + 14
        onsets['duration'] = 4

    onsets = onsets[['onset','duration','trial_type']]
    onsets.columns = ['Onset', 'Duration', 'Stim']

    return onsets_to_dm(onsets, sampling_freq=1./tr, run_length=n_tr)

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

all_runs = Design_Matrix(sampling_freq = sampling_freq)
fmri_data = Brain_Data()

num_runs = [1,2,3,4,5,6]

# sub-07 has faulty run-2 with only 22 TRs; skip! 
if sub == 'sub-07':
    num_runs = [1,3,4,5,6]

for run in num_runs:
    print('Loading run:', run)
    nifti_path = f"/srv/lab/fmri/mft/intention/derivatives/fmriprep/{sub}/func/{sub}_task-dis_run-0{str(run)}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    cov_path =  f'/srv/lab/fmri/mft/intention/derivatives/fmriprep/{sub}/func/{sub}_task-dis_run-0{str(run)}_desc-confounds_timeseries.tsv'
    onset_path = f"/srv/lab/fmri/mft/intention/{sub}/func/{sub}_task-dis_run-0{str(run)}_events.tsv"

    data = Brain_Data(nifti_path).smooth(fwhm=fwhm)
    fmri_data = fmri_data.append(data.scale())

    n_tr = data.shape()[0]

    dm = load_bids_events(onset_path, n_tr)
    dm_conv = dm.convolve(conv_func)
    dm_deriv = dm.convolve(conv_deriv)
    dm_deriv.columns = ['deriv_'+c for c in dm_deriv.columns]
    dm_conv = dm_conv.append(dm_deriv, axis=1)
    
    covariates = pd.read_csv(cov_path, sep='\t')

    cov_unique = make_motion_covariates(covariates[['trans_x','trans_y','trans_z','rot_x', 'rot_y', 'rot_z']])
    spikes = data.find_spikes(global_spike_cutoff=spike_cutoff, diff_spike_cutoff=spike_cutoff)
    cov = cov_unique.add_dct_basis(duration=128).add_poly(order=1, include_lower=True)
    
    full = dm_conv.append(cov,axis=1).append(Design_Matrix(spikes.iloc[:, 1:], sampling_freq=1/tr), axis=1)
    
    all_runs = all_runs.append(full,axis=0,unique_cols=cov_unique.columns)
    print('Finished run:', run)

fmri_data.X = all_runs
stats = fmri_data.regress()

conds = ["F_PHI", "G_PSI","H_II","I_PI", "A_PHA", "B_PSA","C_IA","D_PA", "E_NA", "J_NI"]

for cond in conds:
        for i, col in enumerate(fmri_data.X.columns):
            if col.startswith(cond):
                print(f'Rating: {cond}, design matrix column: {col, i}')
                stats['beta'][i].write(output_dir + f"{sub}_{cond}_{trial_window}_smooth{fwhm}_scaled.nii.gz")