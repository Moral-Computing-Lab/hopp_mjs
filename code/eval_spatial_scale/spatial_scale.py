import os
import sys
import glob
import numpy as np
import pandas as pd 
import pickle

from nltools.data import Brain_Data
from nltools import expand_mask
from joblib import Parallel, delayed
from sklearn.metrics import mean_squared_error
from scipy.stats import sem, pearsonr
import scipy.stats as stats
import argparse

# Define arguments
parser = argparse.ArgumentParser()
parser.add_argument('-iteration', type=int, required=True, dest='iteration')

args = parser.parse_args()
iteration = args.iteration
print('Iteration:', iteration)
num_vox = [ 50, 150, 250, 500, 750, 1000, 1500, 3000, 5000, 6500, 10000, 15000, 25000, 50000, 100000]

# Load data
moral_data = Brain_Data("/srv/lab/fmri/mft/fhopp_diss/analysis/signature/results/activations/discovery_betas.hdf5")
subject_id = moral_data.X['subject']

# Do we run whole-brain or networks? 
MODE = 'networks' # or whole-brain

def eval_whole_brain(n_voxels):
    # Get the voxel indices of the brain image
    vox_idx = np.arange(len(moral_data.data[1]))
    predictions = {}

    # Randomly sample N-voxels
    sampled_vox = np.random.choice(vox_idx, n_voxels, replace=False)

    # Create a mask from the sampled voxels
    mask_helper = moral_data.copy()
    mask_helper.data = np.zeros(mask_helper.data[1].shape)
    mask_helper.data[sampled_vox] = 1
    sample_mask = mask_helper.threshold(binarize=True)

    # Run prediction
    stats = moral_data.apply_mask(sample_mask).standardize(axis=1).predict(algorithm='svr', 
                                cv_dict={'type': 'loso','subject_id': subject_id}, verbose=0,
                                                    plot=False, **{'kernel':"linear"})

    predictions[n_voxels] = stats['r_xval']

    sys.stdout.write(f'Number of voxels: {n_voxels}, r: {stats["r_xval"]}\n')

    return predictions

def eval_network(mask_ix):
    
    if mask_ix == 8:
        mask_network = moral_data.apply_mask(ledoux)
    else:
        mask_network = moral_data.apply_mask(bna_x[mask_ix])
    
    # Get number of voxels of network to avoid oversampling
    n_voxels_mask = len(mask_network.data[1])

    # Set seed for reproducibility
    num_vox = [ 50, 150, 250, 500, 750, 1000, 1500, 3000, 5000, 6500, 10000, 15000, 25000, 50000, 100000]

    # Get the voxel indices of the brain image
    vox_idx = np.arange(len(mask_network.data[1]))
    predictions = {}

    for n_voxels in num_vox:
        if n_voxels_mask < n_voxels:
            continue
        else:
            # Randomly sample N-voxels
            sampled_vox = np.random.choice(vox_idx, n_voxels, replace=False)

            # Create a mask from the sampled voxels
            mask_helper = mask_network.copy()
            mask_helper.data = np.zeros(mask_helper.data[1].shape)
            mask_helper.data[sampled_vox] = 1
            sample_mask = mask_helper.threshold(binarize=True)

            # Run prediction
            stats = mask_network.apply_mask(sample_mask).standardize(axis=1).predict(algorithm='svr', 
                                        cv_dict={'type': 'loso','subject_id': subject_id}, verbose=0,
                                                            plot=False, **{'kernel':"linear"})

            predictions[n_voxels] = stats['r_xval']
            
            print(f"Network {mask_ix} - {net_labels[mask_ix]}: {n_voxels} voxels done", 'Correlation:', stats['r_xval'])

    return predictions

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> # 

print('Starting spatial evaluation...')
if MODE == 'whole-brain':
    spatial_scale = Parallel(n_jobs=-1)(delayed(eval_spatial_scale)(i) for i in num_vox)

    # Save results as pickle
    with open(f'/srv/lab/fmri/mft/fhopp_diss/analysis/signature/results/spatial_scale/whole_brain/spatial_scale_{iteration}.pkl', 'wb') as f:
        pickle.dump(spatial_scale, f)

if MODE == 'networks':
    # Load Yeo networks
    bna = Brain_Data('/srv/lab/fmri/mft/fhopp_diss/analysis/signature/masks/Fan_et_al_atlas_nine_networks.nii')
    bna_x = expand_mask(bna)
    
    # Load Consciousness network
    ledoux = Brain_Data('/srv/lab/fmri/mft/fhopp_diss/analysis/signature/masks/consciousness_mask.nii.gz')

    net_labels = { 0: 'Visual',
                1: 'Somatomotor', 
                2: 'dAttention',
                3: 'vAttention',
                4: 'Limbic',
                5: 'Frontoparietal',
                6: 'Default Mode',
                7: 'Subcortical',
                8: 'Consciousness'}

    
    for mask_ix in net_labels.keys():
        spatial_scale = eval_network(mask_ix)
        # Save results as pickle
        with open(f'/srv/lab/fmri/mft/fhopp_diss/analysis/signature/results/spatial_scale/networks/spatial_scale_{net_labels[mask_ix]}_{iteration}.pkl', 'wb') as f:
            pickle.dump(spatial_scale, f)