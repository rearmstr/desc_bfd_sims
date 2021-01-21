import galsim
import bfd
import glob

import numpy as np
import lsst.afw.table as afwTable
import lsst.afw.image as afwImage
import lsst.geom as geom
import lsst.daf.base as dafBase
import lsst.desc.bfd as dbfd

from lsst.afw.geom import makeSkyWcs
from lsst.meas.algorithms import KernelPsf
from lsst.afw.math import FixedKernel

from descwl_shear_sims.simple_sim import SimpleSim as Sim
from typing import Any


def buildKGalaxy(weight: dbfd.KSigmaWeightF,
                 bfd_config: dbfd.BFDConfig,
                 xy_pos: tuple,
                 image: galsim.ImageF,
                 bbox: galsim.BoundsI,
                 psf_image: galsim.ImageF,
                 noise_sigma: float,
                 galsim_wcs: galsim.fitswcs.GSFitsWCS,
                 nda: float = 1.,
                 _id: int = 0):
    """Build KGalaxy from basic structures"""
    center = galsim.PositionD(xy_pos[0], xy_pos[1])
    jacobian = galsim_wcs.jacobian(center).getMatrix()
    uvref = galsim_wcs.toWorld(center)
    uv_pos = (np.rad2deg(uvref.ra.rad), np.rad2deg(uvref.dec.rad))

    # Need 0.5 pixel offset
    xy_local = (center.x - bbox.xmin + 0.5, center.y - bbox.ymin + 0.5)
    bfd_wcs = bfd.WCS(jacobian, xyref=xy_local, uvref=uv_pos)
    psf_image_arr = psf_image.array
    image_arr = image[bbox].array

    kdata = bfd.generalImage(
        image_arr, uv_pos, psf_image_arr, wcs=bfd_wcs, pixel_noise=noise_sigma)

    conjugate = set(np.where(kdata.conjugate.flatten()==False)[0])
    kgal = bfd_config.KGalaxy(weight, kdata.kval.flatten(), kdata.kx.flatten(), kdata.ky.flatten(),
                              kdata.kvar.flatten(), kdata.d2k, conjugate, xy_pos, _id, nda)
    return kgal


def buildKGalaxyDM(exp, xy_pos, bbox, uv_pos, bfd_config, noise_sigma, wt, nda=1, _id=0, 
                   size=None):

    local_psf = exp.getPsf().computeKernelImage(
        geom.Point2D(xy_pos[0], xy_pos[1]))
    affine_wcs = exp.getWcs().linearizePixelToSky(
        geom.Point2D(xy_pos[0], xy_pos[0]), geom.arcseconds)
    jacobian = affine_wcs.getLinear().getMatrix()
    xy_local = (xy_pos[0] - bbox.getMinX() + 1, xy_pos[1] - bbox.getMinY() + 1)

    image = exp[bbox].image.array
    psf_image = local_psf.array

    bfd_wcs = bfd.WCS(jacobian, xyref=xy_local, uvref=uv_pos)
    kdata = bfd.generalImage(image, uv_pos, psf_image,
                             wcs=bfd_wcs, pixel_noise=noise_sigma, size=size)
    conjugate = set(np.where(kdata.conjugate.flatten() == False)[0])
    kgal = bfd_config.KGalaxy(wt, kdata.kval.flatten(), kdata.kx.flatten(), kdata.ky.flatten(),
                              kdata.kvar.flatten(), kdata.d2k, conjugate, xy_pos, _id, nda)

    return kgal



def define_cat_schema(bfd_config: dbfd.BFDConfig):
    """Create schema for bfd catalogs"""
    n_even = bfd_config.BFDConfig.MSIZE
    n_odd = bfd_config.BFDConfig.XYSIZE

    bfd_schema = afwTable.SourceTable.makeMinimalSchema()
    keys = {}
    keys['evenKey'] = bfd_schema.addField('bfd_even', type="ArrayF",
                                          size=n_even, doc="Even Bfd moments")
    keys['oddKey'] = bfd_schema.addField('bfd_odd', type="ArrayF",
                                         size=n_odd, doc="odd moments")
    keys['shiftKey'] = bfd_schema.addField('bfd_shift', type="ArrayF",
                                           size=2, doc="amount shifted to null moments")
    keys['cov_evenKey'] = bfd_schema.addField('bfd_cov_even', type="ArrayF",
                                              size=n_even*(n_even + 1) // 2,
                                              doc="even moment covariance matrix")
    keys['cov_oddKey'] = bfd_schema.addField('bfd_cov_odd', type="ArrayF",
                                             size=n_odd*(n_odd + 1) // 2,
                                             doc="odd moment covariance matrix")
    keys['flagKey'] = bfd_schema.addField(
        'bfd_flag', type="Flag", doc="Set to 1 for any fatal failure")
    keys['xKey'] = bfd_schema.addField('x', type=float, doc="x position")
    keys['yKey'] = bfd_schema.addField('y', type=float, doc="x position")

    return keys, bfd_schema


def define_prior_schema(bfd_config: dbfd.BFDConfig):
    """Create schema for prior catalogs"""
    bfd_schema = afwTable.SourceTable.makeMinimalSchema()
    keys = {}
    keys['mKey'] = bfd_schema.addField("m", doc="template m", type="ArrayF",
                                       size=bfd_config.BFDConfig.MXYSIZE)
    keys['dmKey'] = bfd_schema.addField("dm", doc="template m", type="ArrayF",
                                        size=bfd_config.BFDConfig.MSIZE*bfd_config.BFDConfig.DSIZE)
    keys['dxyKey'] = bfd_schema.addField("dxy", doc="template m", type="ArrayF",
                                         size=bfd_config.BFDConfig.XYSIZE*bfd_config.BFDConfig.DSIZE)
    keys['ndaKey'] = bfd_schema.addField("nda", doc="nda", type=np.float)
    keys['idKey'] = bfd_schema.addField("bfd_id", doc="id", type=np.int64)

    return keys, bfd_schema


def define_pqr_schema(bfd_config: dbfd.BFDConfig):
    """Create schema for pqr catalogs"""
    n_even = bfd_config.BFDConfig.MSIZE

    schema = afwTable.SourceTable.makeMinimalSchema()
    keys = {}
    keys['pqrKey'] = schema.addField("pqr", doc="pqr", type="ArrayF",
                                     size=bfd_config.BFDConfig.DSIZE)
    keys['momKey'] = schema.addField("moment", doc="moment", type="ArrayF",
                                     size=n_even)
    keys['momCovKey'] = schema.addField("moment_cov", doc="moment", type="ArrayF",
                                        size=n_even*(n_even + 1) // 2)
    keys['numKey'] = schema.addField("n_templates", doc="number", type=np.int64)
    keys['uniqKey'] = schema.addField("n_unique", doc="unique", type=np.int32)
    keys['flagKey'] = schema.addField(
        'bfd_flag', type="Flag", doc="Set to 1 for any fatal failure")
    return keys, schema


def compress_cov(cov: Any):
    """Convert 2d symetric array to 1d for storage"""
    cov_even = cov.m
    cov_odd = cov.xy

    cov_even_save = []
    cov_odd_save = []
    for ii in range(cov_even.shape[0]):
        cov_even_save.extend(cov_even[ii][ii:])
    for ii in range(cov_odd.shape[0]):
        cov_odd_save.extend(cov_odd[ii][ii:])
    return np.array(cov_even_save), np.array(cov_odd_save)


def uncompress_cov(cov: Any, size: int):
    """Convert stored 1d array to full 2-d symmetric version"""
    full = np.zeros((size, size), dtype=np.float32)

    start = 0
    for i in range(size):
        full[i][i:] = cov[start:start + size - i]
        start += size - i

    for i in range(size):
        for j in range(i):
            full[i, j] = full[j, i]
    return full


def getCovariances(file: str, n_even: int, n_odd: int):
    """Read covariance information from first entry in moment catalog"""
    cat = afwTable.BaseCatalog.readFits(file)

    # Use the only first object in the catalog
    rec = cat[0]

    cov_even = rec.get('bfd_cov_even')
    cov_odd = rec.get('bfd_cov_odd')

    full_cov_even = uncompress_cov(cov_even, n_even)
    full_cov_odd = uncompress_cov(cov_odd, n_odd)

    return (full_cov_even, full_cov_odd)


def make_templates(rng, kc, sigma_xy, sigma_flux=1., sn_min=0., sigma_max=6.5, sigma_step=1.,
                   xy_max=2., tid=0, weight_sigma=4,
                   **kwargs):
    """Return a list of Template instances that move the object on a grid of
    coordinate origins that keep chisq contribution of flux and center below
    the allowed max.
    sigma_xy    Measurement error on target x & y moments (assumed equal, diagonal)
    sigma_flux  Measurement error on target flux moment
    sn_min      S/N for minimum flux cut applied to targets
    sigma_max   Maximum number of std deviations away from target that template will be used
    sigma_step  Max spacing between shifted templates, in units of measurement sigmas
    xy_max      Max allowed centroid shift, in sky units (prevents runaways)
    """
    dx, badcentering, msg = kc.recenter(2)  # fixme

    if badcentering:
        return []

    # Determine derivatives of 1st moments on 2 principal axes,
    # and steps will be taken along these grid axes.

    jacobian0 = kc.xy_jacobian(dx)
    eval, evec = np.linalg.eigh(jacobian0)

    if np.any(eval >= 0.):
        return []

    detj0 = np.linalg.det(jacobian0)  # Determinant of Jacobian
    xy_step = np.abs(sigma_step*sigma_xy / eval)

    da = xy_step[0]*xy_step[1]
    xy_offset = rng.uniform(0, 1, size=2) - 0.5

    # Now explore contiguous region of xy grid that yields useful templates.
    result = []
    grid_try = set(((0, 0),))  # Set of all grid points remaining to try
    grid_done = set()           # Grid points already investigated

    flux_min = sn_min*sigma_flux

    while len(grid_try) > 0:
        # Try a new grid point
        mn = grid_try.pop()

        grid_done.add(mn)  # Mark it as already attempted
        # Offset and scale
        xy = np.dot(evec, xy_step*(np.array(mn) + xy_offset))
        # Ignore if we have wandered too far

        if np.dot(xy, xy) > xy_max*xy_max:
            continue

        mom, cov = kc.get_moment(dx[0] + xy[0], dx[1] + xy[1])
        even = mom.m
        odd = mom.xy

        detj = 0.25 * (even[kc.bfd_config.MR]**2 -
                       even[kc.bfd_config.M1]**2 - even[kc.bfd_config.M2]**2)

        # Ignore if determinant of Jacobian has gone negative, meaning
        # we have crossed out of convex region for flux
        if detj <= 0.:
            continue

        # Accumulate chisq that this template would have for a target
        # First: any target will have zero MX, MY
        chisq = (odd[kc.bfd_config.MX]**2 +
                 odd[kc.bfd_config.MY]**2) / sigma_xy**2

        # Second: there is suppression by jacobian of determinant
        chisq += -2. * np.log(detj/detj0)

        # Third: target flux will never be below flux_min
        if (even[kc.bfd_config.MF] < flux_min):
            chisq += ((flux_min - even[kc.bfd_config.MF])/sigma_flux)**2

        if chisq <= sigma_max*sigma_max:
            # This is a useful template!  Add it to output list

            tmpl = kc.get_template(dx[0] + xy[0], dx[1] + xy[1])
            tmpl.nda = tmpl.nda * da
            tmpl.jSuppression = detj / detj0
            tmpl.id = tid
            result.append(tmpl)

            # Try all neighboring grid points not yet tried
            for mn_new in ((mn[0] + 1, mn[1]),
                           (mn[0] - 1, mn[1]),
                           (mn[0], mn[1] + 1),
                           (mn[0], mn[1] - 1)):
                if mn_new not in grid_done:
                    grid_try.add(mn_new)

    return result


def read_prior(filename: str, bfd_config: dbfd.BFDConfig, seed: int = None, max_files=-1):
    """Create prior from stored variables"""
    n_even = bfd_config.BFDConfig.MSIZE
    n_odd = bfd_config.BFDConfig.XYSIZE

    files = glob.glob(filename)
    if max_files > 0:
        files = files[:max_files]

    if seed:
        ud = dbfd.UniformDeviate(seed)
    else:
        ud = dbfd.UniformDeviate()

    init = False
    for ifile, file in enumerate(files):
        try:
            temp = afwTable.BaseCatalog.readFits(file)
        except:
            continue
        md = temp.getTable().getMetadata().toDict()
        print('importing File', file, ' with', len(temp))
        if init is False:

            cov_even = np.array(md['COV_EVEN'])
            cov_odd = np.array(md['COV_ODD'])
            covMat = bfd_config.MomentCov(cov_even.reshape(n_even, n_even),
                                          cov_odd.reshape(n_odd, n_odd))

            minFlux = md['FLUXMIN']
            maxFlux = md['FLUXMAX']
            nSample = md['NSAMPLE']
            selectionOnly = False
            noiseFactor = md['NOISEFACTOR']
            priorSigmaStep = md['PRIORSIGMASTEP']
            priorSigmaCutoff = md['PRIORSIGMACUTOFF']
            priorSigmaBuffer = md['PRIORSIGMABUFFER']
            invariantCovariance = md['INVARIANTCOVARIANCE']
            desel_pqr = np.array(md['DESEL_PQR'], dtype=np.float32)

            prior = bfd_config.KDTreePrior(minFlux, maxFlux, covMat, ud,
                                           nSample, selectionOnly,
                                           noiseFactor, priorSigmaStep, priorSigmaCutoff,
                                           priorSigmaBuffer, invariantCovariance)
            init = True
            base_md = md
        else:
            assert((md['COV_EVEN']==base_md['COV_EVEN']) &
                   (md['COV_ODD']==base_md['COV_ODD']) &
                   (md['FLUXMIN']==base_md['FLUXMIN']) &
                   (md['FLUXMAX']==base_md['FLUXMAX']) &
                   (md['NOISEFACTOR']==base_md['NOISEFACTOR']) &
                   (md['PRIORSIGMASTEP']==base_md['PRIORSIGMASTEP']) &
                   (md['PRIORSIGMACUTOFF']==base_md['PRIORSIGMACUTOFF']) &
                   (md['PRIORSIGMABUFFER']==base_md['PRIORSIGMABUFFER']))

        for s in temp:

            ti = bfd_config.TemplateInfo()
            ti.m = s.get('m')
            if np.any(np.isnan(ti.m)):
                continue
            ti.dm = s.get('dm').reshape(bfd_config.BFDConfig.MSIZE, bfd_config.BFDConfig.DSIZE)
            ti.dxy = s.get('dxy').reshape(bfd_config.BFDConfig.XYSIZE, bfd_config.BFDConfig.DSIZE)
            ti.nda = s.get('nda')
            ti.id = s.get('bfd_id')

            prior.addTemplateInfo(ti)

    print('Preparing prior')
    prior.prepare()

    return prior, md, desel_pqr


def generate_pqr(args: dict,
                 prior: Any,
                 file: str,
                 desel_pqr: Any):
    """From a catalog and prior compute the pqr values"""
    print('Reading ', file)
    cat = afwTable.BaseCatalog.readFits(file)
    bfd_config = dbfd.BFDConfig(use_mag=args['use_mag'], use_conc=args['use_conc'],
                                ncolors=args['ncolors'])
    n_even = bfd_config.BFDConfig.MSIZE
    n_odd = bfd_config.BFDConfig.XYSIZE

    tgs = []
    n_lost = 0

    full_cov_even = np.zeros((n_even, n_even), dtype=np.float32)
    full_cov_odd = np.zeros((n_odd, n_odd), dtype=np.float32)
    full_cov_even[:, :] = np.nan
    full_cov_odd[:, :] = np.nan
    if args.get('add_nblend_pqr'):
        nblend = {}
    for irec, rec in enumerate(cat):
        if 'diff_covariance' in args:
            cov_even = rec.get('bfd_cov_even')
            cov_odd = rec.get('bfd_cov_odd')

            full_cov_even = uncompress_cov(cov_even, n_even)
            full_cov_odd = uncompress_cov(cov_odd, n_odd)

            cov = bfd_config.MomentCov(full_cov_even, full_cov_odd)

        else:
            if np.any(np.isnan(full_cov_even)):
                cov_even = rec.get('bfd_cov_even')
                cov_odd = rec.get('bfd_cov_odd')

                full_cov_even = uncompress_cov(cov_even, n_even)
                full_cov_odd = uncompress_cov(cov_odd, n_odd)

                cov = bfd_config.MomentCov(full_cov_even, full_cov_odd)
            else:
                cov = bfd_config.MomentCov(full_cov_even, full_cov_odd)

        if rec['bfd_flag']:
            tg = bfd_config.TargetGalaxy()
            tgs.append(tg)
            n_lost += 1
            continue

        moment = bfd_config.Moment(rec.get('bfd_even'), rec.get('bfd_odd'))
        pos = np.array([rec.get('coord_ra').asDegrees(),
                        rec.get('coord_dec').asDegrees()])
        tid = rec.get('id')
        tg = bfd_config.TargetGalaxy(moment, cov, pos, tid)
        tgs.append(tg)
        if args.get('add_nblend_pqr'):
            nblend[tid] = rec['nblend']

    results = prior.getPqrCatalog(tgs, args['n_threads'], args['n_chunk'])
    keys, schema = define_pqr_schema(bfd_config)
    if args.get('add_nblend_pqr'):
        keys['nblend'] = schema.addField('nblend', type=np.int32, doc="number of objects in blend")

    outcat = afwTable.BaseCatalog(schema)
    pqr_sum = bfd_config.Pqr()
    bad = 0
    nl_desel_pqr = bfd_config.Pqr(desel_pqr).neglog()
    for ii,(r, tg) in enumerate(zip(results, tgs)):
        out = outcat.addNew()
        if np.isfinite(r[0]._pqr[0])==False or r[0]._pqr[0] <= 0:
            bad += 1
            out.set(keys['flagKey'], 1)
            out.set(keys['pqrKey'], nl_desel_pqr._pqr)
            continue

        pqr = bfd_config.Pqr(r[0]._pqr).neglog()
        out.set(keys['pqrKey'], pqr._pqr)
        out.set(keys['numKey'], r[1])
        out.set(keys['uniqKey'], r[2])
        out.set(keys['momKey'], tg.mom.m)
        cov_even, cov_odd = compress_cov(tg.cov)
        out.set(keys['momCovKey'], np.array(cov_even, dtype=np.float32))
        out.set('coord_ra', tg.position[0]*geom.radians)
        out.set('coord_dec', tg.position[1]*geom.radians)
        pqr_sum += pqr

        if args.get('add_nblend_pqr'):
            try:
                out.set(keys['nblend'], nblend[tg.id])
            except:
                pass
            #    import pdb;pdb.set_trace()

    return outcat, pqr_sum


def convert_to_dm(obj):
    image_sizex, image_sizey = obj.image.array.shape

    masked_image = afwImage.MaskedImageF(image_sizex, image_sizey)
    masked_image.image.array[:] = obj.image.array

    var = 1.0/obj.weight[0, 0]
    masked_image.variance.array[:] = var
    masked_image.mask.array[:] = 0
    exp = afwImage.ExposureF(masked_image)

    psf_image = obj.get_psf(0, 0)
    exp_psf = KernelPsf(FixedKernel(afwImage.ImageD(psf_image.array.astype(np.float))))
    exp.setPsf(exp_psf)

    # set WCS
    cd_matrix = obj.wcs.cd
    crpix = geom.Point2D(obj.wcs.crpix[0], obj.wcs.crpix[1])
    crval = geom.SpherePoint(obj.wcs.center.ra.deg, obj.wcs.center.dec.deg, geom.degrees)
    wcs = makeSkyWcs(crpix=crpix, crval=crval, cdMatrix=cd_matrix)
    exp.setWcs(wcs)
    return exp
