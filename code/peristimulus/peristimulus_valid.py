# Application of the MJS to the Time Series of each Vignette Trial.
# Author: Frederic R. Hopp
# June 2024

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

tr = 0.72

def load_bids_events(onset_path, beh_path):
    '''Create a design_matrix instance from BIDS event file'''
     
    onsets = pd.read_csv(onset_path, sep='\t',usecols=['onset','duration','trial_type', 'cond_id'])
    ratings = pd.read_csv(beh_path, sep='\t', usecols=['moral_decision','RT'])
    # Extend trial duration to capture HRF lag
    onsets['duration'] = 15.12
    onsets['onset_TR'] = onsets['onset'] / 0.72
    onsets['duration_TR'] = onsets['duration'] / 0.72
    onsets['end_TR'] = onsets['onset_TR'] + onsets['duration_TR']

    onsets = onsets.join(ratings)
    onsets = onsets.dropna(subset=['moral_decision'])
    ratings = onsets.copy(deep=True)
    ratings['trial_type'] = ratings['moral_decision'].astype(int).astype(str)
    ratings['cond_id'] = ratings['cond_id']
    
    return ratings

pexp_runs = []
if subject == 'sub-13' or subject == 'sub-21':
    runs = range(2,4)
else:
    runs = range(1,4)
    
pexp_runs = []
    
for run in runs:
    onset = f'/srv/lab/fmri/mofomic/ucsb_2/{subject}/func/{subject}_task-mfv_run-0{str(run)}_events.tsv'
    beh = f'/srv/lab/fmri/mofomic/ucsb_2/{subject}/beh/{subject}_task-mfv_run-0{str(run)}_beh.tsv'
    dm = load_bids_events(onset, beh)
    # Load brain and smooth
    brain_data = Brain_Data(f'/srv/lab/fmri/mofomic/ucsb_2/derivatives/fmriprep/{subject}/{subject}/func/{subject}_task-mfv_run-0{str(run)}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz').smooth(8)
    # Build DF to store pattern expression of MJS
    pexp = pd.DataFrame(columns=['TR','rating','pexp'])
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
df.to_csv(f'/srv/lab/fmri/mft/fhopp_diss/analysis/signature/results/peristimulus/validation/{subject}.csv')