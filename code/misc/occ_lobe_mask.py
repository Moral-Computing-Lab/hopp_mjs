import numpy as np
from nilearn import datasets, image, plotting

# Load the AAL atlas and labels
aal_atlas = datasets.fetch_atlas_aal()
atlas_filename = aal_atlas['maps']
atlas_labels = aal_atlas['labels']

# Define the labels corresponding to the occipital lobe regions
occipital_labels = [
    'Occipital_Sup_L', 'Occipital_Sup_R',
    'Occipital_Mid_L', 'Occipital_Mid_R',
    'Occipital_Inf_L', 'Occipital_Inf_R',
    'Cuneus_L', 'Cuneus_R',
    'Calcarine_L', 'Calcarine_R',
    'Lingual_L', 'Lingual_R'
]

# Print all labels to ensure we have the correct names
print("AAL Labels:", atlas_labels)

# Load the atlas image
atlas_img = image.load_img(atlas_filename)
atlas_data = atlas_img.get_fdata()

# Identify the numeric values corresponding to the occipital regions
occipital_indices = aal_labels['index'].values

# Create a mask image where occipital regions are set to 0 and others to 1
mask_data = np.ones(atlas_data.shape, dtype=np.int32)

# Set occipital regions to 0 in the mask
for idx in occipital_indices:
    mask_data[atlas_data == idx] = 0  # Use the correct numeric values

# Check unique values in mask data to confirm the mask creation
unique_mask_values = np.unique(mask_data)
print("Unique values in mask data:", unique_mask_values)

# Create a new Nifti image for the mask
mask_img = image.new_img_like(atlas_img, mask_data)

# Save the mask image (optional)
mask_img.to_filename('masks/aal_occipital.nii.gz')