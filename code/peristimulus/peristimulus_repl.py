# Application of the MJS to the Time Series of each Vignette Trial.
# Replication Cohort
# Author: Frederic R. Hopp
# May 2024

import pandas as pd

from nltools.stats import zscore
from nltools.data import Brain_Data
import argparse
from nltools.stats import zscore

parser = argparse.ArgumentParser()
parser.add_argument('-subject', type=str, required=True, dest='subject')

args = parser.parse_args()
subject = 'sub-' + args.subject

# Load MJS
mjs = Brain_Data("/srv/lab/fmri/mft/fhopp_diss/analysis/signature/weightmaps/mjs/mjs_weights.nii.gz")

tr = 2
n_tr = 230

def load_bids_events(onset_path, beh_path):
    '''Create a design_matrix instance from BIDS event file'''
     
    onsets = pd.read_csv(onset_path, sep='\t',usecols=['trialstart_zeroed','duration','trial_type', 'cond_id'])
    ratings = pd.read_csv(beh_path, sep='\t', usecols=['moral_decision','RT'])
    
    # Extend trial duration to capture HRF lag
    onsets['onset'] = onsets['trialstart_zeroed']
    onsets['duration'] = 18
    onsets['onset_TR'] = onsets['onset'] / 2
    onsets['onset_TR'] =  onsets['onset_TR'].floordiv(1).astype(int)
    onsets['duration_TR'] = 9
    onsets['end_TR'] = onsets['onset_TR'] + onsets['duration_TR']
    onsets['end_TR'] = onsets['end_TR'].astype(int)

    onsets = onsets.join(ratings)
    onsets = onsets.dropna(subset=['RT'])
    ratings = onsets.copy(deep=True)
    ratings['trial_type'] = ratings['moral_decision'].astype(int).astype(str)
    ratings['cond_id'] = ratings['cond_id']
    
    return ratings

pexp_runs = []

runs = range(1,4)
    
for run in runs:
    onset = f'/srv/lab/fmri/mofomic/duke/{subject}/func/{subject}_task-mfv_run-0{str(run)}_events.tsv'
    beh = f'/srv/lab/fmri/mofomic/duke/{subject}/beh/{subject}_task-mfv_run-0{str(run)}_beh.tsv'
    # Get trial information for run and subject
    dm = load_bids_events(onset, beh)
    # Load brain and smooth
    brain_data = Brain_Data(f'/srv/lab/fmri/mofomic/duke/derivatives/fmriprep/{subject}/{subject}/func/{subject}_task-mfv_run-0{str(run)}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz').smooth(8)
    # Build DF to store pattern expression of MJS
    pexp = pd.DataFrame(columns=['TR','rating','item','pexp'])
    row_count = 0 
    for i,row in dm.iterrows():
        # Select trial from entire brain time series
        trial_brain = brain_data[int(row.onset_TR) : int(row.end_TR)]
        # Compute dot product between each TR of trial and MJS
        for tr_ix, tr_brain in enumerate(trial_brain):
            pexp.at[row_count, 'pexp'] = tr_brain.similarity(mjs, method='dot_product')
            pexp.at[row_count, 'TR'] = tr_ix
            pexp.at[row_count, 'rating'] = row.trial_type
            pexp.at[row_count, 'item'] = row.cond_id
            row_count += 1
    # Z-score pexp and store in DF
    pexp['pexp'] = zscore(pexp['pexp'])
    pexp_runs.append(pexp)
    print(f'Finished run: {run}')
    
df = pd.concat(pexp_runs)
df.to_csv(f'/srv/lab/fmri/mft/fhopp_diss/analysis/signature/results/peristimulus/replication/{subject}.csv')