import glob
import pandas as pd
from nltools.data import Brain_Data, Design_Matrix

betas_dir = '/srv/lab/fmri/mft/fhopp_diss/bids/derivatives/glm/'

sub_list = [x.split('/')[-1].split('_')[0] for x in glob.glob(betas_dir+'sub*')]
sub_list.sort()

betas = [x for x in glob.glob(betas_dir+'sub*/*') if 'common' in x]
betas.sort()

sub_info = pd.DataFrame()
sub_info['subject'] = pd.Series([int(x.split('/')[-1].split('_')[0].split('-')[1]) for x in betas])
sub_info['ratings'] = pd.Series([int(x.split('/')[-1].split('_')[3]) for x in betas])

moral_data = Brain_Data(betas).standardize(axis=1)
moral_data.X['subject'] = sub_info['subject']
moral_data.Y = sub_info['ratings']
subject_id = moral_data.X['subject']

b = moral_data.bootstrap('predict', n_samples=10000, algorithm='svr', save_weights=True, n_jobs=-1, **{'kernel':"linear"})
b['samples'].write(f'/srv/lab/fmri/mft/fhopp_diss/analysis/signature/analysis/weightmaps/mjs/mjs_bootstrap_n10000.hdf5')