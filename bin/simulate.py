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

from bfd_desc_sims import (generate_pqr, generate_grid_catalog,
                           generate_grid_catalog,
                           generate_grid_prior, read_prior)


def generate_grid_catalog_write(args: dict):
    """Generate catalog and wrie out result.  This is useful in multiprocessing"""
    sim, result, cat = generate_grid_catalog(args)
    cat.writeFits(f"{args['outdir']}/{args['galfile']}_{args['index']}.fits")
    if args.get('write_image'):
        for band,imlist in result.items():
            for ii,im in enumerate(imlist):
                im.image.write(f"{args['outdir']}/image_{args['index']}_{band}_{ii}.fits")


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


seeds = np.random.randint(10000000, size=data['njobs'])

if os.path.exists(data['outdir']) is False:
    os.makedirs(data['outdir'])

prior_file = f"{data['outdir']}/{data['priorfile']}.fits"
if data['run_prior']:

    # Create moment noise catalog for prior
    cat_args = copy.deepcopy(data)
    cat_file = f"{cat_args['outdir']}/{cat_args['prior_galfile']}.fits"
    ngal = 1
    spacing = data['stamp_size']
    sims_dict = data['sims_dict']

    sims_dict['layout_kws'] = {'dim': ngal}
    sims_dict['coadd_dim'] = int(ngal*spacing)
    sims_dict['g1'] = 0.0
    sims_dict['g2'] = 0.0
    sims_dict['noise_per_band'] = data['noise_sigma']
    cat_args['sim'] = sims_dict
    sim, result, cat = generate_grid_catalog(cat_args)
    cat.writeFits(cat_file)

    grid = cat_args['n_grid_prior'] > -1

    if grid:
        ngal = cat_args['n_grid_prior']
        sims_dict['layout_kws'] = {'dim': ngal}
        sims_dict['coadd_dim'] = int(ngal*spacing)

    sims_dict['noise_per_band'] = data['template_noise_sigma']
    psim, result, pcat, prior, prior_sel = generate_grid_prior(cat_args)
    pcat.writeFits(prior_file)

# List of processed files to pass on to the pqr function
cat_files = []
pool = multiprocessing.Pool(processes=data['n_threads'])
all_args = []
function = None
for i in range(data['njobs']):

    local_args = copy.deepcopy(data)
    local_args['index'] = i + data['index']
    local_args['seed'] = int(seeds[i])

    spacing = local_args['stamp_size']
    sims_dict = local_args['sims_dict']
    sims_dict['noise_per_band'] = data['noise_sigma']
    grid = local_args['n_grid'] > -1
    if grid:
        ngal = local_args['n_grid']
        sims_dict['layout_kws'] = {'dim': ngal}
        sims_dict['coadd_dim'] = int(ngal*spacing)

    local_args['sim'] = local_args['sims_dict']

    if(data['run_catalog']):
        cat_files.append(f"{local_args['outdir']}/{local_args['galfile']}_{local_args['index']}.fits")
        if 'no_catalog' in local_args:
            continue
        function = generate_grid_catalog_write
        all_args.append(local_args)

results = pool.map(function, all_args)

if data['run_pqr']:

    bfd_config = dbfd.BFDConfig(use_mag=data['use_mag'], use_conc=data['use_conc'], 
                                ncolors=data['ncolors'])

    prior, md, desel_pqr  = read_prior(prior_file, bfd_config, data['seed'])

    for file in cat_files:
        pqr_file = file.replace(data['galfile'], data['pqrfile'])
        pqr_cat, pqr_sum = generate_pqr(data, prior, file, desel_pqr)
        pqr_cat.writeFits(pqr_file)
