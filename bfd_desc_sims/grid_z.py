import galsim
import bfd
import os
from collections import defaultdict
import numpy as np
import glob

import lsst.afw.table as afwTable
import lsst.geom as geom
import lsst.daf.base as dafBase
import lsst.desc.bfd as dbfd
import lsst.afw.table as afwTable
import lsst.afw.image as afwImage
import lsst.afw.detection as afwDet
import lsst.afw.geom as afwGeom
import lsst.geom as geom
import lsst.daf.base as dafBase
from lsst.meas.algorithms import KernelPsf
from lsst.afw.math import FixedKernel
from lsst.meas.base import NoiseReplacerConfig, NoiseReplacer
from lsst.meas.algorithms import SourceDetectionTask, SourceDetectionConfig
from lsst.afw.geom import makeSkyWcs, makeCdMatrix
from lsst.meas.deblender import SourceDeblendTask, SourceDeblendConfig
from lsst.meas.algorithms import SingleGaussianPsf
from lsst.meas.base import SingleFrameMeasurementConfig, SingleFrameMeasurementTask
from lsst.meas.extensions.scarlet import ScarletDeblendTask, ScarletDeblendConfig

import lsst.desc.bfd as dbfd
from descwl_shear_sims.simple_sim import SimpleSim as Sim
from .utils import buildKGalaxy, define_cat_schema, define_prior_schema, make_templates, compress_cov, convert_to_dm, buildKGalaxyDM, getCovariances

class BobsEDist:
    """
    Sets up an ellipticity distribution: exp^(-e^2/2sig^2)*(1-e^2)^2
    Pass a UniformDeviate as ud, else one will be created and seeded at random.
    """
    def __init__(self, sigma, ud=None):
        self.ellip_sigma = sigma
        if ud is None:
            self.ud = galsim.UniformDeviate()
        else:
            self.ud = ud
        self.gd = galsim.GaussianDeviate(self.ud, sigma=self.ellip_sigma)

    def sample(self):
        if self.ellip_sigma==0.:
            e1,e2 = 0.,0.
        else:
            while True:
                while True:
                    e1 = self.gd()
                    e2 = self.gd()
                    esq = e1*e1 + e2*e2
                    if esq < 1.:
                        break
                if self.ud() < (1-esq)*(1-esq) :
                    break
        return e1,e2




def convert_dm(image, noise_sigma, scale, psf_image):
    image_sizex, image_sizey = image.array.shape
    # create the Masked Image
    masked_image = afwImage.MaskedImageF(image_sizex, image_sizey)
    masked_image.image.array[:] = image.array
    masked_image.variance.array[:] = noise_sigma**2
    masked_image.mask.array[:] = 0
    exp = afwImage.ExposureF(masked_image)

    # set PSF
    exp_psf = KernelPsf(FixedKernel(afwImage.ImageD(psf_image.array.astype(np.float))))
    exp.setPsf(exp_psf)

    # set WCS
    orientation = 0*geom.degrees
    cd_matrix = makeCdMatrix(
        scale=scale*geom.arcseconds, orientation=orientation)
    crpix = geom.Point2D(image_sizex/2, image_sizey/2)
    crval = geom.SpherePoint(20, 0, geom.degrees)
    wcs = makeSkyWcs(crpix=crpix, crval=crval, cdMatrix=cd_matrix)
    exp.setWcs(wcs)
    return exp


def generate_grid_catalog_z(args: dict):

    """Generate catalog with moment measurements"""
    bfd_config = dbfd.BFDConfig(use_mag=args['use_mag'], use_conc=args['use_conc'],
                                ncolors=args['ncolors'])
    weight = dbfd.KSigmaWeightF(args['weight_sigma'], args['weight_n'])

    keys, bfd_schema = define_cat_schema(bfd_config)
    keys['gal'] = bfd_schema.addField('gal', type=np.int32, doc="Which galaxy")
    keys['nblend'] = bfd_schema.addField('nblend', type=np.int32, doc="number of objects in blend")
    keys['true_x'] = bfd_schema.addField('true_x', type=float, doc="true x")
    keys['true_y'] = bfd_schema.addField('true_y', type=float, doc="true y")
    keys['g1'] = bfd_schema.addField('g1', type=float, doc="shear g1")
    keys['g2'] = bfd_schema.addField('g2', type=float, doc="shear g2")
    keys['redshift'] = bfd_schema.addField('redshift', type=float, doc="redshift")
    keys['redshift_bin'] = bfd_schema.addField('redshift_bin', type=float, doc="redshift binning")
    keys['flux'] = bfd_schema.addField('flux', type=float, doc="flux")
    keys['zbin'] = bfd_schema.addField('zbin', type=float, doc="redshift bin")
    cat = afwTable.SourceCatalog(bfd_schema)
    rng = galsim.UniformDeviate(args['seed'])
    zbins = args['zbins']
    e_sample = BobsEDist(args['sigma_e'], rng)

    sed_list = []
    if 'SED_list' in args:
        for i,sed in enumerate(args['SED_list']):
            sed = galsim.SED(sed, wave_type=args['sed_wave_unit'], flux_type='flambda')

            sed_list.append(sed)
            print(i,sed)
    else:
        sed_files = glob.glob(args['sed_path'])
        for i,sed in enumerate(sed_files):
            sed = galsim.SED(sed, wave_type=args['sed_wave_unit'], flux_type='flambda')

            sed_list.append(sed)
            #print(i,sed)

    filter_norm = args['filter_norm']
    filters_all = args['filters']
    filters = {}
    for filter_name in filters_all:
        filter_filename = os.path.join(args['filter_path'], f'LSST_{filter_name}.dat')
        filters[filter_name] = galsim.Bandpass(filter_filename, wave_type='nm')
        filters[filter_name] = filters[filter_name].thin(rel_err=1e-4)

    image_size = args['stamp_size'] + 2*args['border']
    stamp = args['stamp_size']


    galsim_origin = galsim.PositionD((image_size-1)/2, (image_size-1)/2)
    galsim_world_origin = galsim.CelestialCoord(20*galsim.degrees, 0*galsim.degrees)
    affine = galsim.AffineTransform(args['scale'], 0, 0, args['scale'], origin=galsim_origin)
    galsim_wcs = galsim.TanWCS(affine, world_origin=galsim_world_origin)

    psf = galsim.Moffat(beta=3.5, half_light_radius=args['sims_dict']['psf_kws']['half_light_radius'])
    psf_bounds = galsim.BoundsI(0, args['stamp_size']-1, 0, args['stamp_size']-1)
    psf_image = galsim.ImageF(psf_bounds, wcs=galsim_wcs)
    psf.drawImage(image=psf_image, wcs=galsim_wcs)


    nimages = len(filters)
    images = []
    timages = []

    tot_gal = args['n_grid']**2

    n_gal = args['n_blend']


    for igal in range(tot_gal):
        row = igal / tot_gal
        col = igal % tot_gal

        bounds = galsim.BoundsI(0, image_size-1, 0, image_size-1)

        fluxes = []
        mags = []
        hlrs = []
        redshifts = []
        redshift_bins = []
        shear_g1s = []
        shear_g2s = []
        nimages = len(filters)
        images = []

        for i in range(nimages):
            images.append(galsim.ImageF(bounds, wcs=galsim_wcs))
        
        cx = args['stamp_size']//2 + args['border']
        cy = args['stamp_size']//2 + args['border']

        xx = []
        yy = []

        
       
        pos = []
        for igal in range(n_gal):
            mag = np.random.uniform(args['mag_min'], args['mag_max'])
            flux = 10**(0.4 * (30 - mag))
            hlr = np.random.uniform(args['hlr_min'], args['hlr_max'])
            redshift = np.random.uniform(args['z_min'], args['z_max'])
            redshift_bin = np.digitize(redshift, zbins) - 1
            #shear_g1 = redshift/args['z_max']*args['g1']
            #shear_g2 = redshift/args['z_max']*args['g2']
            shear_g1 = args['g1_shear_bins'][redshift_bin]
            shear_g2 = args['g2_shear_bins'][redshift_bin]
            
            print('processing galaxy', igal)
            print('Flux, HLR, redshift: ', flux, hlr, redshift)
            #ised = sed_list[igal].withFluxDensity(target_flux_density=1.0, wavelength=500).atRedshift(redshift)
            ised = np.random.choice(sed_list).withFluxDensity(target_flux_density=1.0, wavelength=500).atRedshift(redshift)
            while True:
                if igal==0:
                    x = cx
                    y = cy
                    pos.append((x,y))
                    break
                x = rng()*image_size + args['border']
                y = rng()*image_size + args['border']

                dist = np.array([np.sqrt((a[0]-x)**2 + (a[1]-y)**2) for a in pos])

                if np.min(dist) > args['min_pix'] and np.max(dist) < args['max_pix']:
                    pos.append((x,y))
                    break

            xx.append(x)
            yy.append(y)
            mags.append(mag)
            shear_g1s.append(shear_g1)
            shear_g2s.append(shear_g2)
            fluxes.append(flux)
            redshifts.append(redshift)
            redshift_bins.append(redshift_bin)
            hlrs.append(hlr)

            sed = ised.withFlux(flux, filters[filter_norm])
            disk = galsim.Exponential(half_light_radius=hlr)
            mgal = disk*sed
            mgal = mgal.shear(g1=shear_g1, g2=shear_g2)
            cgal = galsim.Convolve([mgal, psf])

            for iim in range(nimages):
                tmp = cgal.drawImage(filters[filters_all[iim]], image=images[iim], center=(x,y), add_to_image=True )

        noise_sigma = args['noise_sigma']
        noise = galsim.GaussianNoise(rng, sigma=noise_sigma)
        for i,image in enumerate(images):
            image.addNoise(noise)
            #image.write(f'im1_{filters_all[i]}.fits')
        exposures=[]

        for i,image in enumerate(images):
            exp = convert_dm(image, args['noise_sigma'], args['scale'], psf_image)
            exposures.append(exp)
            if 'write_exp' in args:
                exp.writeFits(f'exp1_{filters_all[i]}.fits')


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
        #deblend_task = SourceDeblendTask(config=deblend_config, schema=schema)
        print('Simulate ',len(exposures))
        exposure = afwImage.MultibandExposure.fromExposures(filters_all, exposures)

        mdeblend_config = ScarletDeblendConfig()
        #mdeblend_config.sourceModel='double'
        mdeblend_task = ScarletDeblendTask(config=mdeblend_config, schema=schema)

        # Detect objects
        table = afwTable.SourceTable.make(schema)
        det_result = detection_task.run(table, exposure[filter_norm])

        try:
            result = mdeblend_task.run(exposure, det_result.sources)
        except Exception as e:
            print('problem with deblend:',e)
            continue

        sources = result[1][filters_all[0]].copy(deep=True)
        # Run on deblended images
        noise_replacer_config = NoiseReplacerConfig()
        footprints = {record.getId(): (record.getParent(), record.getFootprint())
                      for record in sources}

        exp = exposures[0]

        # This constructor will replace all detected pixels with noise in the image
        replacer = NoiseReplacer(noise_replacer_config, exposure=exp,
                                 footprints=footprints)

        bbox = geom.Box2I(geom.Point2I(0,0), geom.Extent2I(1, 1))
        bbox.grow(args['stamp_size'] // 2)

        parent_dict = defaultdict(int)
        for src in sources:
            if src.getParent()!=0:
                parent_dict[src.getParent()] += 1

        
        for src in sources:
            meas_task.callMeasure(src, exp)

            if src.get('deblend_nChild') != 0:
                    continue

            replacer.insertSource(src.getId())

            peak = src.getFootprint().getPeaks()[0]
            x_peak, y_peak = peak.getIx() - 0.5, peak.getIy() - 0.5

            dist = np.array([np.sqrt((a[0]-x_peak)**2 + (a[1]-y_peak)**2) for a in pos])
            index = np.argmin(dist)
            true_x = pos[index][0]
            true_y = pos[index][1]
            
            if args.get('use_footprint'):
                bbox = src.getFootprint().getBBox()
            else:
                bbox = geom.Box2I(geom.Point2I(x_peak, y_peak), geom.Extent2I(1, 1))
                bbox.grow(args['stamp_size'] // 2)
                bbox.clip(exp.getBBox())

            sky = exp.getWcs().pixelToSky(geom.Point2D(x_peak, y_peak))
            uv_pos = (sky.getRa().asArcseconds(), sky.getDec().asArcseconds())
            xy_pos = (x_peak, y_peak)
            try:
                kgals = []
                for iexp in exposures:
                    kgals.append(buildKGalaxyDM(iexp, xy_pos, bbox, uv_pos, bfd_config, noise_sigma, weight, _id=src.getId()))
            except Exception as e:
                replacer.removeSource(src.getId())
                print('problem buiding kgalaxy:',e)
                continue

            kc = dbfd.KColorGalaxy.KColorGalaxy(bfd_config, kgals, nda=kgals[0].getNda(), weights=args['bfd_weights'])
            kc.get_moment()
            dx, badcentering, msg = kc.recenter(2)

            if badcentering:
                outRecord.set(keys['flagKey'], 1)
                replacer.removeSource(src.getId())
                continue

            mom, cov = kc.get_moment(dx[0], dx[1], True)

            mom_even = mom.m
            mom_odd = mom.xy
            cov_even, cov_odd = compress_cov(cov)

            outRecord = cat.addNew()

            if src.getParent() == 0:
                outRecord.set(keys['nblend'], 1)
            else:
                outRecord.set(keys['nblend'], parent_dict[src.getParent()])
            outRecord.setParent(src.getParent())
            print('BFD mom',mom_even)
            outRecord.set(keys['gal'], index)
            outRecord.set(keys['true_x'], true_x)
            outRecord.set(keys['true_y'], true_y)
            outRecord.set(keys['redshift'], redshifts[index])
            outRecord.set(keys['redshift_bin'], redshift_bins[index])
            outRecord.set(keys['g1'], shear_g1s[index])
            outRecord.set(keys['g2'], shear_g2s[index])
            outRecord.set(keys['evenKey'], np.array(mom_even, dtype=np.float32))
            outRecord.set(keys['oddKey'], np.array(mom_odd, dtype=np.float32))
            outRecord.set(keys['cov_evenKey'], np.array(cov_even, dtype=np.float32))
            outRecord.set(keys['cov_oddKey'], np.array(cov_odd, dtype=np.float32))
            outRecord.set(keys['shiftKey'], np.array(
                [dx[0], dx[1]], dtype=np.float32))
            outRecord.set(keys['xKey'], x_peak+dx[0])
            outRecord.set(keys['yKey'], y_peak+dx[1])
            replacer.removeSource(src.getId())

        replacer.end()   

    return cat


def generate_grid_catalog_write_z(args: dict):
    """Generate catalog and wrie out result.  This is useful in multiprocessing"""
    sim, result, cat = generate_grid_catalog_z(args)
    cat.writeFits(f"{args['outdir']}/{args['galfile']}_{args['index']}.fits")
    if args.get('write_image'):
        for band,imlist in result.items():
            for ii,im in enumerate(imlist):
                im.image.write(f"{args['outdir']}/image_{args['index']}_{band}_{ii}.fits")


def generate_grid_prior_z(args: dict):
    """Generate prior catalog"""
    bfd_config = dbfd.BFDConfig(use_mag=args['use_mag'], use_conc=args['use_conc'],
                                ncolors=args['ncolors'])
    n_even = bfd_config.BFDConfig.MSIZE
    n_odd = bfd_config.BFDConfig.XYSIZE
    weight = dbfd.KSigmaWeightF(args['weight_sigma'], args['weight_n'])

    zbins = args['zbins']
    keys, bfd_schema = define_prior_schema(bfd_config)
    keys['zDataKey'] = bfd_schema.addField('zData',  doc="template redshift", type="ArrayF",
                                           size=len(zbins)-1)
    keys['zKey'] = bfd_schema.addField('z',  doc="template redshift", type=np.float)

    cat = afwTable.SourceCatalog(bfd_schema)
    rng2 = np.random.RandomState(args['seed'])
    rng = galsim.UniformDeviate(args['seed'])
    e_sample = BobsEDist(args['sigma_e'], rng)

    sed_list = []
    if 'SED_list' in args:
        for i,sed in enumerate(args['SED_list']):
            sed = galsim.SED(sed, wave_type=args['sed_wave_unit'], flux_type='flambda')

            sed_list.append(sed)
            print(i,sed)
    else:
        sed_files = glob.glob(args['sed_path'])
        for i,sed in enumerate(sed_files):
            sed = galsim.SED(sed, wave_type=args['sed_wave_unit'], flux_type='flambda')

            sed_list.append(sed)
            print(i,sed)


    filter_norm = args['filter_norm']
    filters_all = args['filters']
    filters = {}
    for filter_name in filters_all:
        filter_filename = os.path.join(args['filter_path'], f'LSST_{filter_name}.dat')
        filters[filter_name] = galsim.Bandpass(filter_filename, wave_type='nm')
        filters[filter_name] = filters[filter_name].thin(rel_err=1e-4)

    image_size = args['stamp_size'] + 2*args['border']
    stamp = args['stamp_size']

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

    galsim_origin = galsim.PositionD((image_size-1)/2, (image_size-1)/2)
    galsim_world_origin = galsim.CelestialCoord(20*galsim.degrees, 0*galsim.degrees)
    affine = galsim.AffineTransform(args['scale'], 0, 0, args['scale'], origin=galsim_origin)
    galsim_wcs = galsim.TanWCS(affine, world_origin=galsim_world_origin)

    psf = galsim.Moffat(beta=3.5, half_light_radius=args['sims_dict']['psf_kws']['half_light_radius'])
    psf_bounds = galsim.BoundsI(0, args['stamp_size']-1, 0, args['stamp_size']-1)
    psf_image = galsim.ImageF(psf_bounds, wcs=galsim_wcs)
    psf.drawImage(image=psf_image, wcs=galsim_wcs)


    nimages = len(filters)
    images = []
    timages = []

    tot_gal = args['n_grid_prior']**2

    n_gal = args['n_blend']

    for itotgal in range(tot_gal):
        row = itotgal / tot_gal
        col = itotgal % tot_gal

        bounds = galsim.BoundsI(0, image_size-1, 0, image_size-1)
        redshifts = []
        redshift_bins = []
        nimages = len(filters)
        images = []

        for i in range(nimages):
            images.append(galsim.ImageF(bounds, wcs=galsim_wcs))

        cx = args['stamp_size']//2 + args['border']
        cy = args['stamp_size']//2 + args['border']

        xx = []
        yy = []

        pos = []
        for igal in range(n_gal):
            print('processing galaxy', igal)

            mag = np.random.uniform(args['mag_min'], args['mag_max'])
            flux = 10**(0.4 * (30 - mag))
            hlr = np.random.uniform(args['hlr_min'], args['hlr_max'])
            redshift = np.random.uniform(args['z_min'], args['z_max'])
            #shear_g1 = redshift/args['z_max']*args['g1']
            #shear_g2 = redshift/args['z_max']*args['g2']
            print('Flux, HLR, redshift: ', flux, hlr, redshift)


            redshift_bin = np.digitize(redshift, zbins) - 1
            z_data = np.zeros(len(zbins)-1)
            z_data[redshift_bin] = 1
            print(redshift,redshift_bin)
            while True:
                if igal==0:
                    x = cx
                    y = cy
                    pos.append((x,y))
                    break
                x = rng()*image_size + args['border']
                y = rng()*image_size + args['border']

                dist = np.array([np.sqrt((a[0]-x)**2 + (a[1]-y)**2) for a in pos])

                if np.min(dist) > args['min_pix'] and np.max(dist) < args['max_pix']:
                    pos.append((x,y))
                    break

            xx.append(x)
            yy.append(y)
            redshifts.append(redshift)
            redshift_bins.append(redshift_bin)

            ised = np.random.choice(sed_list).withFluxDensity(target_flux_density=1.0, wavelength=500).atRedshift(redshift)
            #ised = sed_list[igal].withFluxDensity(target_flux_density=1.0, wavelength=500).atRedshift(redshift)
            sed = ised.withFlux(flux, filters[filter_norm])
            disk = galsim.Exponential(half_light_radius=hlr)
            mgal = disk*sed
            #mgal = mgal.shear(g1=shear_g1, g2=shear_g2)
            cgal = galsim.Convolve([mgal, psf])

            for iim in range(nimages):
                tmp = cgal.drawImage(filters[filters_all[iim]], image=images[iim], center=(x,y), add_to_image=True )

        noise_sigma = args['template_noise_sigma']
        noise = galsim.GaussianNoise(rng, sigma=noise_sigma)
        for i,image in enumerate(images):
            image.addNoise(noise)
            #image.write(f'im_{filters_all[i]}.fits')
        exposures=[]

        for i,image in enumerate(images):
            exp = convert_dm(image, args['template_noise_sigma'], args['scale'], psf_image)
            exposures.append(exp)

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
        #deblend_task = SourceDeblendTask(config=deblend_config, schema=schema)
        print('Simulate ',len(exposures), 'filters')
        exposure = afwImage.MultibandExposure.fromExposures(filters_all, exposures)

        mdeblend_config = ScarletDeblendConfig()
        mdeblend_task = ScarletDeblendTask(config=mdeblend_config, schema=schema)

        # Detect objects
        table = afwTable.SourceTable.make(schema)
        det_result = detection_task.run(table, exposure[filter_norm])

        try:
            result = mdeblend_task.run(exposure, det_result.sources)
        except Exception as e:
            print('problem with deblend:',e)
            continue

        sources = result[1][filters_all[0]].copy(deep=True)
        # Run on deblended images
        noise_replacer_config = NoiseReplacerConfig()
        footprints = {record.getId(): (record.getParent(), record.getFootprint())
                      for record in sources}

        exp = exposures[0]

        # This constructor will replace all detected pixels with noise in the image
        replacer = NoiseReplacer(noise_replacer_config, exposure=exp,
                                 footprints=footprints)

        bbox = geom.Box2I(geom.Point2I(0,0), geom.Extent2I(1, 1))
        bbox.grow(args['stamp_size'] // 2)

        parent_dict = defaultdict(int)

        for src in sources:
            if src.getParent()!=0:
                parent_dict[src.getParent()] += 1


        for src in sources:
            meas_task.callMeasure(src, exp)

            if src.get('deblend_nChild') != 0:
                    continue

            replacer.insertSource(src.getId())

            peak = src.getFootprint().getPeaks()[0]
            x_peak, y_peak = peak.getIx() - 0.5, peak.getIy() - 0.5

            dist = np.array([np.sqrt((a[0]-x_peak)**2 + (a[1]-y_peak)**2) for a in pos])
            index = np.argmin(dist)
            true_x = pos[index][0]
            true_y = pos[index][1]

            z = redshifts[index]

            if args.get('use_footprint'):
                bbox = src.getFootprint().getBBox()
            else:
                bbox = geom.Box2I(geom.Point2I(x_peak, y_peak), geom.Extent2I(1, 1))
                bbox.grow(args['stamp_size'] // 2)
                bbox.clip(exp.getBBox())

            sky = exp.getWcs().pixelToSky(geom.Point2D(x_peak, y_peak))
            uv_pos = (sky.getRa().asArcseconds(), sky.getDec().asArcseconds())
            xy_pos = (x_peak, y_peak)

            try:
                kgals = []
                for iexp in exposures:
                    kgals.append(buildKGalaxyDM(iexp, xy_pos, bbox, uv_pos, bfd_config, noise_sigma, weight, _id=src.getId()))
            except Exception as e:
                replacer.removeSource(src.getId())
                print('problem buiding kgalaxy:',e)
                continue

            kc = dbfd.KColorGalaxy.KColorGalaxy(bfd_config, kgals, nda=kgals[0].getNda(), weights=args['bfd_weights'])
            dx, badcentering, msg = kc.recenter(2)

            if badcentering:
                replacer.removeSource(src.getId())
                continue

            ltemplates = make_templates(rng2, kc, sigma_xy, sn_min=minFlux/sigma_flux,
                                        sigma_flux=sigma_flux, sigma_step=priorSigmaStep,
                                        sigma_max=priorSigmaCutoff, xy_max=2, tid=src.getId(),  # fixme
                                        weight_sigma=args['weight_sigma'], zbins=len(z_data))
            for temp in ltemplates:
                temp.setZData(z_data)

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
        rec.set(keys['zDataKey'], np.array(temp.zData).astype(np.float32))

    # build selection prior and compute
    momentPriorSel = bfd_config.Prior(minFlux, maxFlux, covMat, ud,
                                      True, noiseFactor, priorSigmaStep,
                                      priorSigmaCutoff,
                                      True, False)

    for temp in atemplates:
        momentPriorSel.addTemplate(temp)

    momentPriorSel.prepare()

    pqr, _, _ = momentPriorSel.getPqrZ(bfd_config.TargetGalaxy(), len(zbins)-1)
    pqr_all = []
    npqr_all = []
    for ipqr in pqr:
        ipqr /= ngals
        pqr_all.append(ipqr._pqr)
        npqr = ipqr._pqr*(-1.)
        npqr[0] += 1
        npqr_all.append(npqr)
        sel_pqr = np.array(pqr_all, dtype=float).flatten()
        no_sel_pqr = np.array(npqr_all, dtype=float).flatten()

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
    metadata.set('sel_pqr', sel_pqr)
    metadata.set('desel_pqr', no_sel_pqr)
    metadata.set('zBins', len(zbins)-1)
    cat.getTable().setMetadata(metadata)
    return cat, momentPrior, momentPriorSel
