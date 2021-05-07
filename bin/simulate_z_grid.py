import logging
#logger=logging.getLogger()
#logger.setLevel(logging.DEBUG)
import os
import argparse
import yaml
import numpy as np
import copy
import lsst.desc.bfd as dbfd
import multiprocessing
from bfd_desc_sims import (generate_pqr_z, read_prior, generate_pqr,
                           generate_grid_catalog_z, generate_grid_prior_z)


def generate_z_grid_catalog_write(args: dict):
    """Generate catalog and wrie out result.  This is useful in multiprocessing"""
    try:
        cat = generate_grid_catalog_z(args)
    except Exception as e:
        print("Failed to generate catalog:", e)
        return
    cat.writeFits(f"{args['outdir']}/{args['galfile']}_{args['index']}.fits")
    # if args.get('write_image'):
    #     for band,imlist in result.items():
    #         for ii,im in enumerate(imlist):
    #             im.image.write(f"{args['outdir']}/image_{args['galfile']}_{args['index']}_{band}_{ii}.fits")
    #             exp.writeFits(f"{args['outdir']}/exp_{args['galfile']}_{args['index']}_{band}_{ii}.fits")

    # if args.get('write_truth'):

    #     catalog = sim._object_data
    #     data = defaultdict(list)
    #     for obj in catalog:
    #         for band in obj.keys():
    #             data[f'{band}_x'].append(obj[band]['pos'][0].x)
    #             data[f'{band}_y'].append(obj[band]['pos'][0].y)
    #             data[f'{band}_flux'].append(obj[band]['obj'].flux)
    #             data[f'{band}_radius'].append(obj[band]['obj'].scale_radius)

    #     table = Table(data)
    #     table.write(f"{args['outdir']}/truth_{args['galfile']}_{args['index']}.fits", overwrite=True)



def generate_z_grid_prior_write(args: dict):
    """Generate catalog and wrie out result.  This is useful in multiprocessing"""
    try:
        cat, prior, prior_sel = generate_grid_prior_z(args)
    except Exception as e:
        print('prior failed: ',e)
        return
    cat.writeFits(args['write_prior_file'])


parser = argparse.ArgumentParser(description='Run image with multiple filters')
parser.add_argument('config', default='config.yaml', help='config file')


logging.basicConfig(level=logging.INFO)

args, unknown = parser.parse_known_args()

data = yaml.safe_load(open(args.config))

for entry in unknown:
    key, value = entry.split('=')
    data.update(yaml.safe_load(f"{key}: {value}"))

if data['seed'] == 0:
    np.random.seed()
else:
    np.random.seed(data['seed'])


seed_init = np.random.choice(1000000, 1)[0]
seed_prior_init = np.random.choice(1000000, 1)[0]


if os.path.exists(data['outdir']) is False:
    os.makedirs(data['outdir'])

prior_threads = min(data['n_threads'], data['njobs_prior'])
if prior_threads > 1:
    prior_pool = multiprocessing.Pool(processes=prior_threads)

prior_list_args = []
if data['run_prior']:
    for i in range(data['njobs_prior']):
        # Create moment noise catalog for prior and a single galaxy
        if i == 0:
            cat_args = copy.deepcopy(data)
            cat_file = f"{cat_args['outdir']}/{cat_args['prior_galfile']}.fits"
            ngal = 1
            spacing = cat_args['stamp_size']
            sims_dict = cat_args['sims_dict']
            
            sims_dict['layout_kws'] = {'dim': ngal}
            sims_dict['coadd_dim'] = int(ngal*spacing)
            sims_dict['g1'] = 0.0
            sims_dict['g2'] = 0.0
            sims_dict['noise_per_band'] = cat_args['noise_sigma']
            cat_args['sim'] = sims_dict
            cat_args['n_grid'] = 1

            sims_dict['layout_type'] = 'random'
            cat = generate_grid_catalog_z(cat_args)
            cat.writeFits(cat_file)

        prior_args = copy.deepcopy(data)
        prior_args['index'] = i + prior_args['index_prior']
        prior_args['seed'] = seed_prior_init + prior_args['index']
        sims_dict = prior_args['sims_dict']

        sims_dict['layout_type'] = 'random'
        sims_dict['coadd_dim'] = prior_args['image_size_prior']
        if prior_args.get('gal_density_prior'):
            sims_dict['gals_kws']['density'] = prior_args['gal_density_prior']

        prior_args['g1'] = 0.0
        prior_args['g2'] = 0.0
        sims_dict['noise_per_band'] = prior_args['template_noise_sigma']
        prior_args['sim'] = sims_dict
        prior_args['write_prior_file'] = f"{prior_args['outdir']}/{prior_args['priorfile']}_{prior_args['index']}.fits"
        prior_list_args.append(prior_args)

if prior_threads > 1:
    prior_results = prior_pool.map(generate_z_grid_prior_write, prior_list_args)
else:
    for arg in prior_list_args:
        generate_z_grid_prior_write(arg)

# List of processed files to pass on to the pqr function
cat_files = []
start_index = 0
if data.get('start_index'):
    start_index = data['start_index']
all_args = []
function = None
if data['n_threads'] > 1:
    pool = multiprocessing.Pool(processes=data['n_threads'])
for i in range(data['njobs']):

    local_args = copy.deepcopy(data)
    local_args['index'] = i + data['index'] + start_index
    local_args['seed'] = seed_init + local_args['index']

    spacing = local_args['stamp_size']
    sims_dict = local_args['sims_dict']
    sims_dict['noise_per_band'] = local_args['noise_sigma']

    sims_dict['layout_type'] = 'random'
    sims_dict['coadd_dim'] = local_args['image_size']
    if local_args.get('gal_density'):
        sims_dict['gals_kws']['density'] = local_args['gal_density']

    local_args['sim'] = local_args['sims_dict']

    if(data['run_catalog']):
        cat_files.append(f"{local_args['outdir']}/{local_args['galfile']}_{local_args['index']}.fits")
        if 'no_catalog' in local_args:
            continue
        function = generate_z_grid_catalog_write
        all_args.append(local_args)

if data['n_threads'] > 1:
    results = pool.map(function, all_args)
else:
    for arg in all_args:
        function(arg)

prior_match = f"{data['outdir']}/{data['priorfile']}_*"
if data['run_pqr']:

    bfd_config = dbfd.BFDConfig(use_mag=data['use_mag'], use_conc=data['use_conc'], 
                                ncolors=data['ncolors'])

    max_files = -1
    if 'max_prior_files' in data:
        max_files = data['max_prior_files']
    prior, md, desel_pqr = read_prior(prior_match, bfd_config, data['seed'], max_files, True)

    for file in cat_files:
        pqr_file = file.replace(data['galfile'], data['pqrfile'])
        #try:
        pqr_cat, pqr_sum = generate_pqr_z(data, prior, file, desel_pqr)
        pqr_cat.writeFits(pqr_file)
        #except Exception as e:
        #    print('Failed to generate pqr data: ', e)
        #    continue
