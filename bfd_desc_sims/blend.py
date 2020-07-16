import galsim
import bfd

import numpy as np
import lsst.afw.table as afwTable
import lsst.afw.image as afwImage
import lsst.afw.detection as afwDet
import lsst.afw.geom as afwGeom
import lsst.geom as geom
import lsst.daf.base as dafBase
import lsst.desc.bfd as dbfd
from collections import defaultdict
from descwl_shear_sims.simple_sim import Sim

from lsst.meas.base import NoiseReplacerConfig, NoiseReplacer
from lsst.meas.algorithms import SourceDetectionTask, SourceDetectionConfig
from lsst.afw.geom import makeSkyWcs, makeCdMatrix
from lsst.meas.deblender import SourceDeblendTask, SourceDeblendConfig
from lsst.meas.algorithms import SingleGaussianPsf
from lsst.meas.base import SingleFrameMeasurementConfig, SingleFrameMeasurementTask

from .utils import buildKGalaxy, define_cat_schema, define_prior_schema, make_templates, compress_cov, convert_to_dm, buildKGalaxyDM, getCovariances


def generate_blend_catalog(args: dict):
    """Generate catalog with moment measurements"""
    bfd_config = dbfd.BFDConfig(use_mag=args['use_mag'], use_conc=args['use_conc'],
                                ncolors=args['ncolors'])
    weight = dbfd.KSigmaWeightF(args['weight_sigma'], args['weight_n'])

    keys, bfd_schema = define_cat_schema(bfd_config)
    cat = afwTable.SourceCatalog(bfd_schema)

    rng = np.random.RandomState(args['seed'])
    sim = Sim(rng=rng, **args['sim'])
    result = sim.gen_sim()

    ref_band = 'i'
    image = result[ref_band][0]
    exp = convert_to_dm(result[ref_band][0])
    noise_sigma = np.sqrt(1./result[ref_band][0].weight[0, 0])

    schema = afwTable.SourceTable.makeMinimalSchema()

    # Setup algorithms to run
    meas_config = SingleFrameMeasurementConfig()
    meas_config.plugins.names = [
        "base_SdssCentroid",
        "base_SdssShape",
        "base_PsfFlux",
        "base_SkyCoord",
        "base_LocalBackground",
    ]
    # set these slots to none because we aren't running these algorithms
    meas_config.slots.apFlux = None
    meas_config.slots.gaussianFlux = None
    meas_config.slots.calibFlux = None
    meas_config.slots.modelFlux = None
    meas_task = SingleFrameMeasurementTask(config=meas_config, schema=schema)

    detection_config = SourceDetectionConfig()
    detection_config.reEstimateBackground = False
    detection_config.doTempLocalBackground=True
    if args.get('threshold'):
        detection_config.thresholdValue = args['threshold']
    else:
        detection_config.thresholdValue = 5
    detection_task = SourceDetectionTask(config=detection_config)

    deblend_config = SourceDeblendConfig()
    deblend_task = SourceDeblendTask(config=deblend_config, schema=schema)
    orig_noise_sigma = np.sqrt(1./result[ref_band][0].weight[0, 0])
    # Detect objects
    if args.get('pre_det_noise'):
        original_pixels = exp.image.array.copy()
        new_pixels = exp.image.array
        new_pixels.array[:] += np.random.normal(size=exp.image.array.shape)*args['pre_det_noise']

    table = afwTable.SourceTable.make(schema)
    det_result = detection_task.run(table, exp)

    if args.get('pre_det_noise'):
        new_pixels = original_pixels

    if args.get('post_det_noise'):
        exp.image.array[:] += np.random.normal(size=exp.image.array.shape)*args['post_det_noise']
        noise_sigma = np.sqrt(orig_noise_sigma**2 + args['post_det_noise']**2)
        exp.variance.array[:] = noise_sigma**2

    deblend_task.run(exp, det_result.sources)

    # Run on deblended images
    noise_replacer_config = NoiseReplacerConfig()
    footprints = {record.getId(): (record.getParent(), record.getFootprint())
                  for record in det_result.sources}

    # This constructor will replace all detected pixels with noise in the image
    replacer = NoiseReplacer(noise_replacer_config, exposure=exp,
                             footprints=footprints)
    sources = det_result.sources

    gals = sim._object_data
    ngal = len(gals)
    ref_band = 'i'
    ref_wcs = result[ref_band][0].wcs


    for src in sources:
        meas_task.callMeasure(src, exp)

        if src.get('deblend_nChild') != 0:
                continue

        replacer.insertSource(src.getId())

        peak = src.getFootprint().getPeaks()[0]
        x_peak, y_peak = peak.getIx() - 0.5, peak.getIy() - 0.5

        bbox = geom.Box2I(geom.Point2I(x_peak, y_peak), geom.Extent2I(1, 1))
        bbox.grow(args['stamp_size'] // 2)
        bbox.clip(exp.getBBox())

        sky = exp.getWcs().pixelToSky(geom.Point2D(x_peak, y_peak))
        uv_pos = (sky.getRa().asArcseconds(), sky.getDec().asArcseconds())
        xy_pos = (x_peak, y_peak)

        if args.get('post_blend_noise'):
            original_pixels = exp.image[bbox].array.copy()
            new_pixels = exp.image[bbox]
            new_pixels.array[:] += np.random.normal(size=exp.image[bbox].array.shape)*args['post_blend_noise']
            noise_sigma = np.sqrt(orig_noise_sigma**2 + args['post_blend_noise']**2)


        kgal = buildKGalaxyDM(exp, xy_pos, bbox, uv_pos, bfd_config, noise_sigma, weight, _id=src.getId())

        if args.get('post_blend_noise'):
            new_pixels.array[:] = original_pixels

        kc = dbfd.KColorGalaxy.KColorGalaxy(bfd_config, [kgal], nda=kgal.getNda())
        world = ref_wcs.toWorld(galsim.PositionD(x_peak, y_peak))
        dx, badcentering, msg = kc.recenter(2)

        outRecord = cat.addNew()
        if badcentering:
            outRecord.set(keys['flagKey'], 1)
            replacer.removeSource(src.getId())
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
        replacer.removeSource(src.getId())

    replacer.end()
    return sim, result, cat


def generate_blend_prior(args: dict):
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

    ref_band = 'i'
    image = result[ref_band][0]
    exp = convert_to_dm(result[ref_band][0])
    noise_sigma = np.sqrt(1./result[ref_band][0].weight[0, 0])

    schema = afwTable.SourceTable.makeMinimalSchema()

    # Setup algorithms to run
    meas_config = SingleFrameMeasurementConfig()
    meas_config.plugins.names = [
        "base_SdssCentroid",
        "base_SdssShape",
        "base_PsfFlux",
        "base_SkyCoord",
        "base_LocalBackground",
    ]
    # set these slots to none because we aren't running these algorithms
    meas_config.slots.apFlux = None
    meas_config.slots.gaussianFlux = None
    meas_config.slots.calibFlux = None
    meas_config.slots.modelFlux = None
    meas_task = SingleFrameMeasurementTask(config=meas_config, schema=schema)

    detection_config = SourceDetectionConfig()
    detection_config.reEstimateBackground = False
    detection_config.doTempLocalBackground=True
    if args.get('threshold'):
        detection_config.thresholdValue = args['threshold']
    else:
        detection_config.thresholdValue = 5
    detection_task = SourceDetectionTask(config=detection_config)

    deblend_config = SourceDeblendConfig()
    deblend_task = SourceDeblendTask(config=deblend_config, schema=schema)

    # Detect objects
    table = afwTable.SourceTable.make(schema)
    det_result = detection_task.run(table, exp)
    
    if args.get('add_prior_noise_det'):
        add_noise = np.sqrt(noise_sigma**2 + args['prior_noise']**2)
        exp_noise = afwImage.ExposureF(exp,deep=True)
        exp_noise.image.array += np.random.normal(size=exp_noise.image.array.shape)*add_noise

        det_noise_result = detection_task.run(table, exp_noise)

        for det in det_noise_result.sources:
            
            bbox = det.getFootprint().getBBox()
            
            for peak in det.getFootprint().getPeaks():
                px = peak.get('f_x')
                py = peak.get('f_y')
                for deep_det in det_result.sources:
                    deep_foot = deep_det.getFootprint()
                    if deep_foot.getBBox().contains(geom.Point2I(px,py)):
                        min_dist = 0
                        for deep_peak in deep_foot.getPeaks():
                            dpx = peak.get('f_x')
                            dpy = peak.get('f_y')
                            dist = np.sqrt((dpx-px)**2 + (dpy-py)**2)
                            if dist < min_dist:
                                min_dist = dist
                        if min_dist > 3:
                            deep_foot.addPeak(px, py, exp_noise.image.array[int(py), int(px)])
                    else:
                        rec = det_result.sources.addNew()
                        footprint = afwDet.Footprint(afwGeom.SpanSet(bbox), bbox)
                        footprint.addPeak(px, py, exp_noise.image.array[int(py), int(px)])            
                        rec.setFootprint(footprint)

    deblend_task.run(exp, det_result.sources)
    # Run on deblended images
    noise_replacer_config = NoiseReplacerConfig()
    footprints = {record.getId(): (record.getParent(), record.getFootprint())
                  for record in det_result.sources}

    # This constructor will replace all detected pixels with noise in the image
    replacer = NoiseReplacer(noise_replacer_config, exposure=exp,
                             footprints=footprints)
    sources = det_result.sources

    gals = sim._object_data

    ref_band = 'i'
    ref_wcs = result[ref_band][0].wcs
    atemplates = []
    ngals = 0
    for src in sources:
        meas_task.callMeasure(src, exp)

        if src.get('deblend_nChild') != 0:
                continue

        replacer.insertSource(src.getId())

        peak = src.getFootprint().getPeaks()[0]
        x_peak, y_peak = peak.getIx() - 0.5, peak.getIy() - 0.5

        bbox = geom.Box2I(geom.Point2I(x_peak, y_peak), geom.Extent2I(1, 1))
        bbox.grow(args['stamp_size'] // 2)
        bbox.clip(exp.getBBox())

        sky = exp.getWcs().pixelToSky(geom.Point2D(x_peak, y_peak))
        uv_pos = (sky.getRa().asArcseconds(), sky.getDec().asArcseconds())
        xy_pos = (x_peak, y_peak)

        kgal = buildKGalaxyDM(exp, xy_pos, bbox, uv_pos, bfd_config, noise_sigma, weight, _id=src.getId())

        kc = dbfd.KColorGalaxy.KColorGalaxy(bfd_config, [kgal], nda=kgal.getNda())
        world = ref_wcs.toWorld(galsim.PositionD(x_peak, y_peak))
        dx, badcentering, msg = kc.recenter(2)


        if badcentering:
            replacer.removeSource(src.getId())
            continue


        mom, cov = kc.get_moment(dx[0], dx[1], True)

        ltemplates = make_templates(rng, kc, sigma_xy, sn_min=minFlux/sigma_flux,
                                    sigma_flux=sigma_flux, sigma_step=priorSigmaStep,
                                    sigma_max=priorSigmaCutoff, xy_max=2, tid=src.getId(),  # fixme
                                    weight_sigma=args['weight_sigma'])
        atemplates.extend(ltemplates)

        if len(ltemplates) > 0:
            ngals += 1

        replacer.removeSource(src.getId())
    replacer.end()

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

