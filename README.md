# Moral Judgment Signature 

Code accompanying the paper: A sensitive and specific neural signature robustly predicts graded computations of moral wrongness  

Authors: Frederic R. Hopp, Sungbin Youk, Walter Sinnott-Armstrong, & René Weber

Correspondence should be addressed to René Weber (renew@comm.ucsb.edu)

## Contents: 

### code:
    - bootstrap:
        1. bootstrap_mjs.py for running the bootstrap prediction

    - eval_spatial_scale:
        1. spatial_scale.py for sampling increasing number of voxels across networks for model predictions

    - glm:
        1. 01_glm_common.py GLM analysis for obtaining beta maps per wrongness rating (1-4)
        2. 02_glm_conditions.py GLM analysis for obtaining condition-wise beta maps for each wrongness rating 
        3. 03_glm_trial.py LSA GLM analysis for obtaining beta maps for each vignette item
        4. 04_glm_parametric.py GLM analysis with wrongness ratings as parametric modulator 
        5. 05_glm_smid.py GLM analysis for obtaining beta maps per morality rating (1-5)
        6. 06_glm_intention_condition.py GLM analysis for obtaining beta maps for intentional vs. accidental harm scenarios

    - matlab:
        - Note: requires CanLabCore tools (https://github.com/canlab/CanlabCore)
        - create_moral_atlas.m code generates a brain atlas based on pre-defined moral ROIs (see roi_masks)
        - load_patterns_thr.m loads neural signatures for riverplots 
        - mediation_analysis.m runs the multi-level mediation analysis 
        - model_encode.m creates the model encoding maps (structure coefficients)
        - plot_brain.m code used to generate visualiatzions of statistical brain maps
        - plot_riverplot_miniature.m creates the inlays below the riverplots
        - plot_riverplot_networks.m creates riverplots for brain networks 
        - plot_riverplot_rois. creates riverplots for individual ROIs 

    - misc:
        1. bna_create_rois.py mappings of parcels -> ROIs for Fan Brainnetome atlas
        2. run_slurm.sh generic script for distributing code across SLURM cluster

    - notebooks:
        1. 01_behavior.ipynb analyses of behavioral data across studies 1­-4.
        2. 02_train_eval_mjs.ipynb training and evaluation of MJS across studies 1-4.
        3. 03_bootstrap_mjs.ipynb thresholding of MJS bootstrap map (see bootstrap/boostrap_mjs.py for running the bootstrap prediction)
        4. 04_validate_mjs_vignettes.ipynb peristimulus plots, trial-wise prediction across MJS, PINES, and VIDS, out-of-sample correlations 
        5. 05_cross_decoding.ipynb documents the PLS-R analyses.
        6. 06_forward_backward.ipynb for the univariate parametric t-test, within-subject classifiers, and model-encoding maps thresholding.
        7. 07_alternative_models.ipynb searchlights, parcellation, and network-based predictions. 
        8. 08_sensitivity.ipynb functional comparison of MJS, PINES, and VIDS.

    - peristimulus
        1. peristimulus_study.py applies the MJS to the TRs of each trial via dot-product.

### masks:
    - roi_masks contain the ROI masks for riverplots
    - talairach_atlas is used to create the mask for the occipital lobe (occ_talairach_nii.gz)
    - Fan* are the Brainnetome atlases (networks and ROIs)
    - moral_uniformity is the "moral" map from Neurosynth

## Dependencies

The above code was tested and executed using:
- Python (version 3.11.12) and the nltools package (v.0.5.1), in an Anaconda environment.
- MATLAB www.mathworks.com (R2024a)
- CANLab Core tools CANlab CanlabCore and Neuroimaging_Pattern_Masks repositories (https://github.com/canlab)

## Data availability
fMRI data used to train and validate the signature are available at https://figshare.com/articles/dataset/Discovery_dataset_mjs/29423726?file=55720082 (study 1); https://figshare.com/articles/dataset/Validation_dataset_mjs/29423789?file=55721972 (study 2), https://figshare.com/articles/dataset/Replication_dataset_mjs/29423966?file=55724255 (study 3), and https://figshare.com/articles/dataset/Generalization_dataset_mjs/29423981?file=55724291 (study 4). The data of study 5 are from a previous study (https://doi.org/10.1371/journal.pbio.1002180) and are available at https://neurovault.org/collections/1964. The data of study 6 are from a previous study (https://doi.org/10.1038/s41562-024-01868-x) and are available at https://figshare.com/articles/dataset/validation_dataset_disgust/22841117. The data of study 7 are from a previous study (https://doi.org/10.1073/pnas.1207992110) and are available at https://openneuro.org/datasets/ds000212/versions/1.0.0. The data of the visual scenes moral judgment paradigms are available at  https://figshare.com/articles/dataset/Visualscenes1_dataset_mjs/29424164?file=55725344 (study 8) and at https://figshare.com/articles/dataset/Visualscenes2_dataset_mjs/29424176?file=55725368 (study 9). The data from the ultimatum game (study 10) were provided by the authors of a previous study (https://doi.org/10.1038/s42003-025-07561-7). The moral judgment signature and the thresholded statistical maps are available via figshare at https://figshare.com/articles/dataset/Brain_models_and_maps/29424206?file=55725515
