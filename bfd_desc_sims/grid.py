import galsim
import bfd

import numpy as np
import lsst.afw.table as afwTable
import lsst.geom as geom
import lsst.daf.base as dafBase
import lsst.desc.bfd as dbfd
from descwl_shear_sims.simple_sim import Sim
from .utils import buildKGalaxy, define_cat_schema, define_prior_schema, make_templates, compress_cov, getCovariances

def generate_grid_catalog(args: dict):
    """Generate catalog with moment measurements"""
    bfd_config = dbfd.BFDConfig(use_mag=args['use_mag'], use_conc=args['use_conc'],
                                ncolors=args['ncolors'])
    weight = dbfd.KSigmaWeightF(args['weight_sigma'], args['weight_n'])

    keys, bfd_schema = define_cat_schema(bfd_config)
    cat = afwTable.SourceCatalog(bfd_schema)

    rng = np.random.RandomState(args['seed'])
    sim = Sim(rng=rng, **args['sim'])
    result = sim.gen_sim()

    # get grid distance
    gdim = sim.layout_kws.get('dim')
    frac = 1.0 - sim.buff * 2 / sim.coadd_dim
    _pos_width = sim.coadd_dim * frac * 0.5
    dg = _pos_width * 2 / gdim

    # subtract 1
    dg -= 1

    gals = sim._object_data
    ngal = len(gals)
    ref_band = 'i'

    for i in range(ngal):
        pos = gals[i][ref_band]['pos'][0]
        ref_wcs = result[ref_band][0].wcs
        xy_pos = (pos.x, pos.y)
        kgals = []
        for iband, band in enumerate(sim.bands):
            se_obs = result[band][0]
            wcs = se_obs.wcs
            image = result[band][0].image
            psf_image = se_obs.get_psf(pos.x, pos.y)
            noise = sim.noise_per_epoch[iband]
            if args.get('post_blend_noise'):
                noise = np.sqrt(sim.noise_per_epoch[iband]**2 + args['post_blend_noise']**2)
            bounds = galsim.BoundsI(int(pos.x), int(
                pos.x), int(pos.y), int(pos.y))
            bounds = bounds.withBorder(dg // 2)

            kgal = buildKGalaxy(weight, bfd_config, xy_pos,
                                image, bounds, psf_image, noise, wcs, _id=i)
            kgals.append(kgal)

        kc = dbfd.KColorGalaxy.KColorGalaxy(bfd_config, kgals)
        world = ref_wcs.toWorld(pos)
        dx, badcentering, msg = kc.recenter(2)

        outRecord = cat.addNew()
        if badcentering:
            outRecord.set(keys['flagKey'], 1)
            continue

        mom, cov = kc.get_moment(dx[0], dx[1], True)

        mom_even = mom.m
        mom_odd = mom.xy
        cov_even, cov_odd = compress_cov(cov)

        # compute final position
        final_world = galsim.CelestialCoord(world.ra + dx[0]*galsim.arcsec,
                                            world.dec + dx[1]*galsim.arcsec)
        final_xy = ref_wcs.toImage(final_world)

        outRecord.set(keys['evenKey'], np.array(mom_even, dtype=np.float32))
        outRecord.set(keys['oddKey'], np.array(mom_odd, dtype=np.float32))
        outRecord.set(keys['cov_evenKey'], np.array(cov_even, dtype=np.float32))
        outRecord.set(keys['cov_oddKey'], np.array(cov_odd, dtype=np.float32))
        outRecord.set(keys['shiftKey'], np.array(
            [dx[0], dx[1]], dtype=np.float32))
        outRecord.set(keys['xKey'], final_xy.x)
        outRecord.set(keys['yKey'], final_xy.y)
        outRecord.set('coord_ra', world.ra.rad*geom.radians)
        outRecord.set('coord_dec', world.dec.rad*geom.radians)

    return sim, result, cat


def generate_grid_catalog_write(args: dict):
    """Generate catalog and wrie out result.  This is useful in multiprocessing"""
    sim, result, cat = generate_grid_catalog(args)
    cat.writeFits(f"{args['outdir']}/{args['galfile']}_{args['index']}.fits")
    if args.get('write_image'):
        for band,imlist in result.items():
            for ii,im in enumerate(imlist):
                im.image.write(f"{args['outdir']}/image_{args['index']}_{band}_{ii}.fits")


def generate_grid_prior(args: dict):
    """Generate prior catalog"""
    bfd_config = dbfd.BFDConfig(use_mag=args['use_mag'], use_conc=args['use_conc'],
                                ncolors=args['ncolors'])
    n_even = bfd_config.BFDConfig.MSIZE
    n_odd = bfd_config.BFDConfig.XYSIZE
    weight = dbfd.KSigmaWeightF(args['weight_sigma'], args['weight_n'])

    keys, bfd_schema = define_prior_schema(bfd_config)
    cat = afwTable.SourceCatalog(bfd_schema)

    rng = np.random.RandomState(args['seed'])
    sim = Sim(rng=rng, **args['sim'])
    result = sim.gen_sim()

    # get grid distance
    gdim = sim.layout_kws.get('dim')
    frac = 1.0 - sim.buff * 2 / sim.coadd_dim
    _pos_width = sim.coadd_dim * frac * 0.5
    dg = _pos_width * 2 / gdim

    # subtract 1
    dg -= 1

    gals = sim._object_data
    ngal = len(gals)
    ref_band = 'i'

    cov_even, cov_odd = getCovariances(f"{args['outdir']}/{args['prior_galfile']}.fits", n_even, n_odd)
    sigma_flux = np.sqrt(cov_even[0, 0])
    sigma_xy = np.sqrt(cov_odd[0, 0])

    covMat = bfd_config.MomentCov(cov_even, cov_odd)
    if args['seed']:
        ud = dbfd.UniformDeviate(args['seed'])
    else:
        ud = dbfd.UniformDeviate()

    minFlux = args['minSN']*sigma_flux
    maxFlux = args['maxSN']*sigma_flux
    nSample = args['nSample']
    selectionOnly = False
    noiseFactor = args['noiseFactor']
    priorSigmaCutoff = args['priorSigmaCutoff']
    priorSigmaStep = args['priorSigmaStep']
    priorSigmaBuffer = args['priorSigmaBuffer']
    invariantCovariance = True
    momentPrior = bfd_config.KDTreePrior(minFlux, maxFlux, covMat, ud,
                                         nSample, selectionOnly,
                                         noiseFactor, priorSigmaStep, priorSigmaCutoff,
                                         priorSigmaBuffer, invariantCovariance)
    atemplates = []
    ngals = 0
    for i in range(ngal):
        pos = gals[i][ref_band]['pos'][0]
        ref_wcs = result[ref_band][0].wcs
        xy_pos = (pos.x, pos.y)
        kgals = []
        for iband, band in enumerate(sim.bands):
            se_obs = result[band][0]
            wcs = se_obs.wcs
            image = result[band][0].image
            psf_image = se_obs.get_psf(pos.x, pos.y)
            noise = sim.noise_per_epoch[iband]
            bounds = galsim.BoundsI(int(pos.x), int(
                pos.x), int(pos.y), int(pos.y))
            bounds = bounds.withBorder(dg // 2)

            kgal = buildKGalaxy(weight, bfd_config, xy_pos,
                                image, bounds, psf_image, noise, wcs, _id=i)
            kgals.append(kgal)

        kc = dbfd.KColorGalaxy.KColorGalaxy(bfd_config, kgals)
        world = ref_wcs.toWorld(pos)
        dx, badcentering, msg = kc.recenter(2)  # fixme

        if badcentering:
            continue

        ltemplates = make_templates(rng, kc, sigma_xy, sn_min=minFlux/sigma_flux,
                                    sigma_flux=sigma_flux, sigma_step=priorSigmaStep,
                                    sigma_max=priorSigmaCutoff, xy_max=2, tid=i,  # fixme
                                    weight_sigma=args['weight_sigma'])
        atemplates.extend(ltemplates)

        if len(ltemplates) > 0:
            ngals += 1

    for temp in atemplates:
        momentPrior.addTemplate(temp)

    for temp in momentPrior.templates:
        rec = cat.addNew()
        rec.set(keys['mKey'], temp.m)
        rec.set(keys['dmKey'], temp.dm.flatten())
        rec.set(keys['dxyKey'], temp.dxy.flatten())
        rec.set(keys['ndaKey'], temp.nda)
        rec.set(keys['idKey'], temp.id)

    # build selection prior and compute
    momentPriorSel = bfd_config.Prior(minFlux, maxFlux, covMat, ud,
                                      True, noiseFactor, priorSigmaStep,
                                      priorSigmaCutoff,
                                      True, False)

    for temp in atemplates:
        momentPriorSel.addTemplate(temp)

    momentPriorSel.prepare()
    pqr, _, _ = momentPriorSel.getPqr(bfd_config.TargetGalaxy())
    pqr /= ngals

    sel_pqr = np.array(pqr._pqr, dtype=float)
    no_sel_pqr = np.array(pqr._pqr, dtype=float)*(-1.)
    no_sel_pqr[0] += 1
    print('Selection', sel_pqr, no_sel_pqr)

    metadata = dafBase.PropertyList()
    metadata.set('cov_even', np.array(cov_even.flatten(), dtype=float))
    metadata.set('cov_odd', np.array(cov_odd.flatten(), dtype=float))
    metadata.set('fluxMin', minFlux)
    metadata.set('fluxMax', maxFlux)
    metadata.set('noiseFactor', noiseFactor)
    metadata.set('priorSigmaCutoff', priorSigmaCutoff)
    metadata.set('priorSigmaStep', priorSigmaStep)
    metadata.set('priorSigmaBuffer', priorSigmaBuffer)
    metadata.set('nsample', nSample)
    metadata.set('selectionOnly', selectionOnly)
    metadata.set('invariantCovariance', invariantCovariance)
    metadata.set('sigma', args['weight_sigma'])
    metadata.set('wIndex', args['weight_n'])
    metadata.set('covFile', args['prior_galfile'])
    metadata.set('sel_pqr', np.array(sel_pqr, dtype=float))
    metadata.set('desel_pqr', no_sel_pqr)
    cat.getTable().setMetadata(metadata)
    return sim, result, cat, momentPrior, momentPriorSel
