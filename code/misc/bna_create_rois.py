from nltools.data import Brain_Data
from nltools.mask import expand_mask, collapse_mask

bna = Brain_Data('/srv/lab/fmri/mft/fhopp_diss/analysis/signature/masks/Fan_et_al_atlas_r279_MNI.nii')
bna_x = expand_mask(bna)

dmpfc_idx = [0,1, 10, 11, 12,13]
dmpfc_mask = collapse_mask(bna_x[dmpfc_idx]).threshold(binarize=True)
dmpfc_mask.write('/srv/lab/fmri/mft/fhopp_diss/analysis/signature/masks/rois/dmpfc_mask.nii.gz')

vmpfc_idx = [40, 41, 44, 45, 46, 47]
vmpfc_mask = collapse_mask(bna_x[vmpfc_idx]).threshold(binarize=True)
vmpfc_mask.write('/srv/lab/fmri/mft/fhopp_diss/analysis/signature/masks/rois/vmpfc_mask.nii.gz')

precun_idx = [146, 147, 148, 149, 150, 151, 152, 153]
precun_mask = collapse_mask(bna_x[precun_idx]).threshold(binarize=True)
precun_mask.write('/srv/lab/fmri/mft/fhopp_diss/analysis/signature/masks/rois/precun_mask.nii.gz')

thal_idx = range(230, 246)
thal_mask = collapse_mask(bna_x[thal_idx]).threshold(binarize=True)
thal_mask.write('/srv/lab/fmri/mft/fhopp_diss/analysis/signature/masks/rois/thal_mask.nii.gz')

ins_idx = [165, 166, 167,  174]
ins_mask = collapse_mask(bna_x[ins_idx]).threshold(binarize=True)
ins_mask.write('/srv/lab/fmri/mft/fhopp_diss/analysis/signature/masks/rois/ins_mask.nii.gz')

dlpfc_idx = [14,15, 22,23 ]
dlpfc_mask = collapse_mask(bna_x[dlpfc_idx]).threshold(binarize=True)
dlpfc_mask.write('/srv/lab/fmri/mft/fhopp_diss/analysis/signature/masks/rois/dlpfc_mask.nii.gz')

pSTS_idx = [120, 121, 122, 123]
pSTS_mask = collapse_mask(bna_x[pSTS_idx]).threshold(binarize=True)
pSTS_mask.write('/srv/lab/fmri/mft/fhopp_diss/analysis/signature/masks/rois/pSTS_mask.nii.gz')

amy_idx = [210, 211, 212, 213]
amy_mask = collapse_mask(bna_x[amy_idx]).threshold(binarize=True)
amy_mask.write('/srv/lab/fmri/mft/fhopp_diss/analysis/signature/masks/rois/amy_mask.nii.gz')