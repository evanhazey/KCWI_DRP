from keckdrpframework.primitives.base_img import BaseImg
from kcwidrp.primitives.kcwi_file_primitives import kcwi_fits_reader, \
    kcwi_fits_writer, strip_fname
#from kcwidrp.primitives.GetAtlasLines import gaus
from kcwidrp.core.kcwi_get_std import kcwi_get_std
from kcwidrp.core.bokeh_plotting import bokeh_plot
from bokeh.plotting import figure
from kcwidrp.core.kcwi_plotting import save_plot
#from kcwidrp.core.bspline import Bspline
from kcwidrp.core.kcwi_pkg_resources import get_resource_path
import yaml
from PIL import Image


import os
import time
import numpy as np
from astropy.io import fits

import zap 
from zap.zap import SKYSEG
import scipy.stats as scistats
from astropy.stats import sigma_clip
from astropy.modeling import models, fitting
from scipy.interpolate import interp1d
from regions import PixCoord, EllipsePixelRegion
from scipy.interpolate import UnivariateSpline, splrep, splev
import astropy.units as u
import time


class MakeMasterSky3D(BaseImg):
    """
    Make master sky cube using PCA.

    Uses Principal Component Analysis (PCA) via ZAP (Soto et al.,2016) generate a master sky cube
    for sky subtraction. Works on the *icube.fits files. This is an alternative to the 2D bspline method of sky subtraction, 
    and is particularly useful for red side data with strong sky lines that are changing rapidly in time across the field.

    This routine handles the file `kcwi.yaml`, which controls the master
    sky generation.  This file consists of one line per image, with the first
    column indicating the raw object icube image to be sky-subtracted.  The following
    columns can either indicate a separate image to use for sky subtraction, the
    filename of a mask fits image for masking object flux, or indicate that the
    object is a continuum source and either automatically find the object, or
    specify the location and width of the continuum source.  Below are example
    one-line entries and what they mean:

    1. Skip sky subtraction for this particular frame:

        * kr230925_00075: 
        *     skip: True

    2. Point to a different image for the sky. Turn off
    sky subtraction for the offsky frame to make sure it is
    not processed twice:

        * kr230925_00075:
        *    offsky: kr230925_00076.fits
        * kr230925_00076:
        *    skip: True
        
    2a.  Indicate that an off sky mask file should be used to mask object flux when
    deriving the sky model (see kcwi_maskskyzap_ds9.py). This mask should
    be in the reduction directory e.g, pathtodata/redux/:

        * kr230925_00075:
        *     offsky: kr230925_00076
        *     zap_offsky_mask: kr230925_00076_zapsmsk.fits
        * kr230925_00076:
        *    skip: True
            
    3. Indicate that a mask file should be used to mask object flux when
    deriving the sky model (see kcwi_maskskyzap_ds9.py). This mask should
    be in the reduction directory e.g, pathtodata/redux/:

        * kr230925_00075: 
        *     skymask: kr230925_00075_zapsmsk.fits

    4. Indicate that this is a bright continuum source and automatically mask
    the source from the sky model. This will save the mask as *zasmskauto.fits
    in the reduction directory.

        * kr230925_00075:
        *     zap_use_auto_cont: True

    5. Indicate that there is a faint continuum source and specify the location
    of the source (in pixels) to be masked. Supply the vertices of a rectangle 
    with the lower left coorindates x1,y1 then the upper right x2,y2 (see figure).
    Leave no spaces between the coordnates and separate the x and y with a comma: 
               --- .x2,y2
              |    | 
              |    |
        x1,y1 .---- 
    
        * kr230925_00075:
        *     zap_use_faint_cont: True
        *     zap_faint_cont_x1y1: 22,22
        *     zap_faint_cont_x2y2: 44,44

    If no `sky.yaml` file exists, or there is no entry for the input object
    frame, then the entire image is used to generate the sky model.

    It is good practice to run all the data through first, then inspect the
    sky subtraction and see which frames will benefit from masking or from a
    dedicated sky observation.
    
    If a sky model is generated, the routine will write out a \*_sky.fits image
    and add a sky entry in the proc table. A toggle can be made in the config file
    to add the sky model as an extension to the \*_icube.fits file instead of 
    writing out a separate \*_sky.fits file.

    Below is a full list of the possible entries in the `sky.yaml` file for ZAP sky subtraction:
    You do not need to include all of these entries for each file, only the ones relevant to the 
    frame and the features you want to use. However, if you want to add all parameters, be sure
    to use None and False values for the features that are not relevant to the frame 

    #Object name (do not include ".fits")
    krYYMMDD_XXXXX:
        # General sky subtraction instructions #
        skip: True or False
        offsky: krYYMMDD_XXXXX or None
        # ZAP sky subtraction instructions #
        zap_skymask: krYYMMDD_XXXXX_icube_zapskymask.fits or None
        zap_offsky_mask: krYYMMDD_XXXXX_icube_zapskymask.fits or None
        zap_use_auto_cont: True or False
        zap_use_faint_cont: True or False
        zap_faint_cont_x1y1: 5,5 or None
        zap_faint_cont_x2y2: 12,12 or None
        zap_skysegmentoption: 'single', 'Soto+2016', or 'custom'
        zap_customskysegments: [] or None
        zap_cfwidth: 300 or None        
    """

    def __init__(self, action, context):
        BaseImg.__init__(self, action, context)
        self.logger = context.pipeline_logger

    def skyyaml_parser(self, skyyamlfile):
        self.logger.info("Reading %s" % skyyamlfile)
        with open(skyyamlfile, 'r') as file:
            skyyaml = yaml.safe_load(file)
        ofn = (strip_fname(self.action.args.name))
        # is our file in the list?
        if skyyaml.get(ofn, None) is not None:
            reduxdir = self.config.instrument.output_directory
            #Does the user want to skip sky subtraction for this frame?
            if skyyaml[ofn].get('skip', None) == True:
                self.logger.info("Skipping sky subtraction for %s" %
                                        ofn)
                keycom = 'sky corrected?'
                self.action.args.ccddata.header['SKYCOR'] = (False,
                                                                keycom)
                return False

            #Does the user want auto continuum masking for ZAP?
            elif skyyaml[ofn].get('zap_use_auto_cont', None) == True:
                self.logger.info("Automatic continuum masking requested for"
                                    " %s" % ofn)
                self.action.args.zap_use_auto_cont = True

            #Does the user want faint continuum masking for ZAP?
            elif skyyaml[ofn].get('zap_use_faint_cont', None) == True:
                self.logger.info("Faint continuum masking requested")
                #Did the user supply the positions?
                if ((skyyaml[ofn].get('zap_faint_cont_x1y1', None) is not None) and (skyyaml[ofn].get('zap_faint_cont_x1y1', None) != 'None')) and ((skyyaml[ofn].get('zap_faint_cont_x2y2', None) is not None) and (skyyaml[ofn].get('zap_faint_cont_x2y2', None) != 'None')):
                    self.logger.info("ZAP Using faint continuum source for %s" % ofn)
                    self.action.args.zap_use_faint_cont = True
                    self.action.args.zap_faint_cont_x1y1 = tuple((int(skyyaml[ofn]['zap_faint_cont_x1y1'].split(',')[0]), int(skyyaml[ofn]['zap_faint_cont_x1y1'].split(',')[1])))
                    self.action.args.zap_faint_cont_x2y2 = tuple((int(skyyaml[ofn]['zap_faint_cont_x2y2'].split(',')[0]), int(skyyaml[ofn]['zap_faint_cont_x2y2'].split(',')[1])))
                    self.logger.info("Using input continuum position of "
                                        "lower left (x1,y1)==%s upper right (x2,y2)=%s" %
                                        (self.action.args.zap_faint_cont_x1y1,
                                        self.action.args.zap_faint_cont_x2y2))
                else:
                    self.logger.warning("Faint continuum masking requested but no positions supplied. Use zap_faint_cont_x1,y1 and zap_faint_cont_x2y2 so the program " \
                    "can find the faint continuum source. Moving forward with no faint continuum mask.")
                    self.action.args.zap_use_faint_cont = False

            # Do we have a science sky mask file?
            elif (skyyaml[ofn].get('zap_skymask', None) is not None) and (skyyaml[ofn].get('zap_skymask', None) != 'None'):
                zap_skymask = os.path.join(reduxdir, skyyaml[ofn]['zap_skymask'])
                self.logger.info("ZAP sky mask requested")
                if os.path.exists(zap_skymask):
                    self.logger.info("Using ZAP sky mask file: %s" % zap_skymask)
                    self.action.args.zap_skymask = zap_skymask
                else:
                    self.logger.warning("ZAP sky mask supplied but not found: %s. Proceeding with no ZAP sky mask." % zap_skymask)
            
            #Do we have an off sky frame?
            elif (skyyaml[ofn].get('offsky', None) is not None) and (skyyaml[ofn].get('offsky', None) != 'None'):
                offsky = os.path.join(reduxdir, skyyaml[ofn]['offsky']+'_icube.fits')
                self.logger.info("Offsky frame requested")
                if os.path.exists(offsky):
                    self.logger.info("Using off-sky frame: %s" % offsky)
                    self.action.args.offsky = offsky
                    #Does the sky frame have a ZAP sky mask file?
                    if (skyyaml[ofn].get('zap_offsky_mask', None) is not None) and (skyyaml[ofn].get('zap_offsky_mask', None) != 'None'):
                        zap_offsky_mask = os.path.join(reduxdir, skyyaml[ofn]['zap_offsky_mask'])
                        if os.path.exists(zap_offsky_mask):
                            self.logger.info("Using ZAP off-sky mask file: %s" % zap_offsky_mask)
                            self.action.args.zap_offsky_mask = zap_offsky_mask
                        else:
                            self.logger.warning("ZAP off-sky mask requested but not found: %s. Proceeding with no ZAP off-sky mask." % zap_offsky_mask)
                            self.action.args.zap_offsky_mask = None
                else:
                    self.logger.warning("Off-sky frame supplied but not found: %s. Proceeding with no off-sky frame." % offsky)
                    self.action.args.offsky = None
                    self.action.args.zap_offsky_mask = None
            
            #Do we have parameters to control ZAP?
            # Sky segment for this object 
            if (skyyaml[ofn].get('zap_skysegmentoption', None) is not None) and (skyyaml[ofn].get('zap_skysegmentoption', None) != 'None'):
                self.logger.info("Frame specific ZAP sky segment to be used")
                zap_skysegmentoption = skyyaml[ofn]['zap_skysegmentoption']
                #Unrecognized skysegment option
                if (zap_skysegmentoption.lower() != 'single') and (zap_skysegmentoption.lower() != 'soto+2016') and (zap_skysegmentoption.lower() != 'custom'):
                    self.logger.warning("Unknown sky segment option: %s. Proceeding with default sky segment option in kcwi.cfg." % zap_skysegmentoption)
                    self.action.args.zap_skysegmentoption = None
                #Custom sky segment option 
                elif (zap_skysegmentoption.lower() == 'custom'):
                    self.logger.info("User will supply custom sky segments")
                    if (skyyaml[ofn].get('zap_customskysegments', None) is not None) and (skyyaml[ofn].get('zap_customskysegments', None).lower() != 'none'):
                        zap_customskysegments = skyyaml[ofn]['zap_customskysegments']
                        self.action.args.zap_customskysegments = list(zap_customskysegments)
                        self.action.args.zap_skysegmentoption = zap_skysegmentoption
                        self.logger.info("Custom sky segments to be used: %s" % self.action.args.zap_customskysegments)
                    else:
                        self.logger.warning("Custom sky segments not found. Proceeding with default sky segment option in kcwi.cfg.")
                        self.action.args.zap_customskysegments = None
                        self.action.args.zap_skysegmentoption = None
                #Single or Soto+2016
                elif (zap_skysegmentoption.lower() == 'single') or (zap_skysegmentoption.lower() == 'soto+2016'):
                    self.action.args.zap_skysegmentoption = zap_skysegmentoption
                    self.logger.info("User requested %s: " % zap_skysegmentoption)


            # Continuum filter width for this object
            if (skyyaml[ofn].get('zap_cfwidth', None) is not None) and (skyyaml[ofn].get('zap_cfwidth', None) != 'None'):
                    zap_cfwidth = skyyaml[ofn]['zap_cfwidth']
                    self.action.args.zap_cfwidth = zap_cfwidth
                    self.logger.info("User supplied a cfwidth: %s " % zap_cfwidth)

            # Does the user want to user want interactive mode here?
            if (skyyaml[ofn].get('zap_interactive', None) is not None) and (skyyaml[ofn].get('zap_interactive', None) != 'None'):
                if (skyyaml[ofn].get('zap_interactive', None) is True) or (skyyaml[ofn].get('zap_interactive', None) == 'True'):
                    self.action.args.zap_interactive = True
                    self.logger.info("User requests interactive mode for this frame")
            else:
                self.action.args.zap_interactive = False


            #Return the arguments 
            return self.action.args
            
        #No YAML entry for this file
        else:
            self.logger.info("No sky yaml entry for %s, using entire image to generate sky model" % ofn)
            return self.action.args

    def _pre_condition(self):
        """
        Checks if we can create a master sky
        """
        self.logger.info("Checking precondition for MakeMasterSky3D")

        if self.config.instrument.skipsky:
            self.logger.warning("Sky subtraction turned off, "
                                "skipping MakeMasterSky3D")
            return False

        #Check if user wants to run ZAP
        if (self.config.instrument.skysubmethod != '3D-PCA') and (self.config.instrument.skysubmethod != '2D-BSPLINE+3D-PCA'):
            self.logger.warning("User does not want sky subtraction using ZAP, "
                                "skipping MakeMasterSky3D")
            return False

        #suffix = 'sky'  # self.action.args.new_type.lower()
        ofn = self.action.args.name
        rdir = self.config.instrument.output_directory

        # Are we a standard star?
        stdfile = None
        stdname = None
        if 'object' in self.action.args.imtype.lower():
            self.logger.info("Checking OBJECT keyword")
            stdfile, stdname = kcwi_get_std(
                self.action.args.ccddata.header['OBJECT'], self.logger)
            if not stdfile:
                self.logger.info("Checking TARGNAME keyword")
                stdfile, stdname = kcwi_get_std(
                    self.action.args.ccddata.header['TARGNAME'], self.logger)
        else:
            self.logger.warning("Not object type: %s" %
                                self.action.args.imtype)
        self.action.args.stdfile = stdfile
        self.action.args.stdname = stdname

        
        # Parse through sky subtraction instructions
        self.action.args.skyyaml = None
        self.action.args.offsky = None
        self.action.args.zap_skymask = None
        self.action.args.zap_offsky_mask = None
        self.action.args.zap_use_auto_cont = False
        self.action.args.zap_use_faint_cont = False
        self.action.args.zap_faint_cont_x1y1 = None
        self.action.args.zap_faint_cont_x2y2 = None
        self.action.args.zap_skysegmentoption = None
        self.action.args.zap_customskysegments = None
        self.action.args.zap_cfwidth = None
        self.action.args.zap_interactive = None
        
        #Check if there is a YAML sky file, parse it if so
        if os.path.exists('sky.yaml'):
            self.action.args.skyyaml = True
            self.skyyaml_parser('sky.yaml')

        return True


    def _perform(self):
        self.logger.info("Performing Sky Subtraction using 3D-PCA method via ZAP on *icube.fits")
        log_string = MakeMasterSky3D.__module__
        
        def crop_cube(hdu):
            """Crop the HDUList to only consist of the region"""
            
            hdr = hdu[0].header
            wave = (np.arange(hdr['NAXIS3']) + 1 - hdr['CRPIX3']) * hdr['CD3_3'] + hdr['CRVAL3']
            wavegood_idx = np.where((wave >= hdr['WAVGOOD0']) & (wave <= hdr['WAVGOOD1']))[0]
            
            for h in hdu:
                h.data = h.data[wavegood_idx]
            hdu[0].header['CRVAL3'] = wave[wavegood_idx][0]
            return hdu

        def collapse_header(hdr):
            """
            Quick wrapper to collapse a 3-D header into a 2-D one.
            Copied from KCWIKit, then copied from KSkyWizard
            """

            hdr_img=hdr.copy()
            hdr_img['NAXIS']=2
            del hdr_img['NAXIS3']
            del hdr_img['CD3_3']
            del hdr_img['CTYPE3']
            del hdr_img['CUNIT3']
            del hdr_img['CNAME3']
            del hdr_img['CRVAL3']
            del hdr_img['CRPIX3']

            return hdr_img

        def scale_extinct_sky(hdr_sky, hdr_sci):
            """Atmospheric extinction correction from official KCWI DRP"""

            # get airmass
            air_sky = hdr_sky['AIRMASS']
            air_sci = hdr_sci['AIRMASS']
            # read extinction data
            path = 'data/extin/snfext.fits'
            package = __name__.split('.')[0]
            full_path = get_resource_path(package, path)
            if os.path.exists(full_path):
                hdul = fits.open(full_path)
                exwl = hdul[1].data['LAMBDA']
                exma = hdul[1].data['EXT']
                # get object wavelengths
                dw = hdr_sky['CD3_3']
                w0 = hdr_sky['CRVAL3']
                owls = np.arange(hdr_sky['NAXIS3']) * dw + w0
                # linear interpolation
                exint = interp1d(exwl, exma, kind='cubic', bounds_error=False,
                                fill_value='extrapolate')
                # resample extinction curve
                oexma = exint(owls)
                # convert to flux ratio
                flxr_sky = 10.**(oexma * air_sky * 0.4)
                flxr_sci = 10.**(oexma * air_sci * 0.4)
                    
                return flxr_sky / flxr_sci

        def trim_skysegments(skyseglist, wavearr):
            #remove the extra sky segement falls outside of the spectral region in case ZAP runs into problem
            skyseg0 = np.array(skyseglist)
            idx_remove_low = np.where(skyseglist < wavearr[0])[0]
            idx_remove_up = np.where(skyseglist > wavearr[-1])[0]
            idx_remove = np.concatenate((idx_remove_low[1:], idx_remove_up[1:]))
            trimmed_skyseg = np.delete(skyseglist, idx_remove) #the first index is zero; need to keep
            return trimmed_skyseg.tolist() 
                   
        def plot_skystats(self, wavee, cleancubee, noskysubb, skyfnamm, zap_skysegg):
            #Setting plotting parameters
            alphan=0.75
            c_drp, c_zap1, c_zapN, c_ex, c_bspline = 'darkorange', 'deepskyblue', 'royalblue', 'black', 'grey'
            c_drp_hist, c_zap1_hist, c_zapN_hist, c_ex_hist, c_noskysub = (1.0, 0.549, 0.0, 0.5), (0.0, 0.749, 1.0, 0.5), (65/255, 105/255, 225/255, 0.5) , 'black', 'darkgrey' #(R, G, B, Alpha)
            axis_stats = np.array(['NOSKYSUB','BSPLINE', 'ZAP'])
            positions_stats  = np.arange(len(axis_stats)) #np.flip(np.arange(len(axis_stats)))
            labels_stats = (axis_stats.tolist())
            marker_avg, marker_med, marker_mod, marker_std, marker_rms = 'x', 'D', '*', 's', 'P'
            alpha_hist = 0.5
            hatch_noskysub, hatch_drp, hatch_zap1 = '|','\\', '/',
            label_noskysub, label_drp, label_zap1 = 'NOSKYSUB','BSPLINE', 'ZAP'
            # Plot the statistical plots
            mask='' # Locate or construct quick sky mask
            if self.action.args.zap_skymask is not None: #science sky mask?
                mask = fits.getdata(self.action.args.zap_skymask)
            else: #make a quick sky mask using the entire cube
                mask_shape = np.sum(scihdu[0].data, axis = 0).shape
                mask = np.zeros(mask_shape, dtype=int)
                mask[:, :1], mask[:, -1:] = 1, 1 #x mask the edges to avoid edge effects in the sky model
                mask[:2, :], mask[-2:, :] = 1, 1 #y mask the edges to avoid edge effects in the sky model
            allsky = (mask == 0)
            #Load in and sigma clip the cubes
            zap1 = sigma_clip(cleancubee, sigma=3).data
            bsplinepre = sigma_clip(noskysubb, sigma=3).data
            noskysub = bsplinepre
            flux_zap1_sky, flux_bsplinepre, flux_bsplinepre = '','',''
            if (self.action.args.zap_skymask is not None) or (self.action.args.zap_skymask is not None):
                #Extract mean sky spectrum for each source
                flux_zap1_sky = np.nanmean(zap1[:,allsky],axis = 1)
                flux_nosub_sky = np.nanmean(noskysub[:,allsky],axis = 1)
                flux_bsplinepre = np.nanmean(bsplinepre[:,allsky],axis = 1)
            else: 
                #Extract median sky spectrum for each source (since no sky msk was supplied)
                flux_zap1_sky = np.nanmean(zap1[:,allsky],axis = 1)
                flux_nosub_sky = np.nanmean(noskysub[:,allsky],axis = 1)
                flux_bsplinepre = np.nanmean(bsplinepre[:,allsky],axis = 1)
            # Create sky model based on 2D bspline (pseudo DRP) #
            x0, y0 = wavee, flux_nosub_sky
            numsteps = int(1.25*(len(wavee)))
            x_new = np.linspace(np.min(wavee), np.max(wavee), numsteps)
            y_new = np.interp(x_new, x0, y0)
            tck = splrep(x_new, y_new)
            flux_drp_sky = flux_bsplinepre - splev(wavee, tck)
            ## Calculate statistical measures of the sky spectra ##
            #DRP SKY
            flux_drp_sky_avg = np.average(flux_drp_sky)
            flux_drp_sky_med = np.median(flux_drp_sky)
            flux_drp_sky_std = np.std(flux_drp_sky)
            flux_drp_sky_rms = np.sqrt(np.mean(flux_drp_sky*flux_drp_sky))
            flux_drp_sky_mod = scistats.mode(flux_drp_sky, axis=None).mode
            #NOSUB SKY
            flux_nosub_sky_avg = np.average(flux_nosub_sky)
            flux_nosub_sky_med = np.median(flux_nosub_sky)
            flux_nosub_sky_std = np.std(flux_nosub_sky)
            flux_nosub_sky_rms = np.sqrt(np.mean(flux_nosub_sky*flux_nosub_sky))
            flux_nosub_sky_mod = scistats.mode(flux_nosub_sky, axis=None).mode
            #ZAP1 SKY
            flux_zap1_sky_avg = np.average(flux_zap1_sky)
            flux_zap1_sky_med = np.median(flux_zap1_sky)
            flux_zap1_sky_std = np.std(flux_zap1_sky)
            flux_zap1_sky_rms = np.sqrt(np.mean(flux_zap1_sky*flux_zap1_sky))
            flux_zap1_sky_mod = scistats.mode(flux_zap1_sky, axis=None).mode
            flux_zap1_sky_med = np.median(flux_zap1_sky)

            ### DEFINE PLOTTING ###
            import matplotlib.pyplot as plt
            plt.close()
            f, ax  = plt.subplots(figsize=(15,8), ncols=2, nrows=2)

            ### PLOT THE SPECTRA ### 
            #Sky Spectrum
            ax[0,0].set_title('Mean Sky Spectrum')
            ax2 = ax[0,0].twinx()
            ax2.step(wavee, flux_nosub_sky, c=c_noskysub, alpha=alphan, label='NoSkySub')
            ax2.set_ylabel(r'Original Sky Flux $[e^{-1}]$')
            ax[0,0].step([], [], c=c_noskysub, alpha=alphan, label='NoSkySub')
            ax[0,0].step(wavee, flux_drp_sky, c=c_drp, alpha=alphan, label='BSPLINE')
            ax[0,0].step(wavee, flux_zap1_sky, c=c_zapN, alpha=alphan, label='ZAP')
            ax[0,0].set_xlabel(r'Observed Wavelength $[\AA]$')
            ax[0,0].set_ylabel(r'Flux $[e^{-1}]$')
            ax[0,0].legend(loc='upper left')

            ### PLOT HISTOGRAMS ###
            #Sky
            ax[0,1].set_title('Sky Histogram')
            ax[0,1].hist(flux_nosub_sky, edgecolor=c_noskysub, color='white',  hatch=hatch_noskysub, alpha=alpha_hist, label='NOSKYSUB')
            ax[0,1].hist(flux_drp_sky, edgecolor=c_drp, color='white',  hatch=hatch_drp, alpha=alpha_hist, label='BSPLINE')
            ax[0,1].hist(flux_zap1_sky, edgecolor=c_zap1, color='white',  hatch=hatch_zap1, alpha=alpha_hist, label='ZAP')
            ax[0,1].set_xlabel(r'Sky Flux $[e^{-}]$')
            ax[0,1].set_xlim(flux_drp_sky.min(), flux_drp_sky.max())
            ax[0,1].set_ylabel(r'Number')
            ax[0,1].set_yscale('log')
            ax[0,1].legend()

            ### PLOT STATISTICAL METRICS ###
            # STATS PER WAVELENGTH OF SKY SEGMENT SLICE #
            #Standard Deviation
            for i in range(len(zap_skysegg)-1):
                idx = (obswave >= zap_skysegg[i]) & (obswave <= zap_skysegg[i+1])
                len_idx = len(wavee[idx])
                #NOSKYSUB
                ax[1,0].plot(wavee[idx], np.ones(len_idx)*np.std(flux_nosub_sky[idx]), c=c_noskysub, alpha=alphan,marker=marker_std)
                ax[1,0].text(np.median(wavee[idx]), np.std(flux_nosub_sky[idx]), '{}'.format(int(np.std(flux_nosub_sky[idx]))))
                #BSPLINE
                ax[1,0].plot(wavee[idx], np.ones(len_idx)*np.std(flux_drp_sky[idx]), c=c_drp, alpha=alphan, marker=marker_std)
                ax[1,0].text(np.median(wavee[idx]), np.std(flux_drp_sky[idx]), '{:.2f}'.format(np.std(flux_drp_sky[idx])))
                #ZAP
                ax[1,0].plot(wavee[idx], np.ones(len_idx)*np.std(flux_zap1_sky[idx]), c=c_zap1, alpha=alphan, marker=marker_std)
                ax[1,0].text(np.median(wavee[idx]), np.std(flux_zap1_sky[idx]), '{:.2f}'.format(np.std(flux_zap1_sky[idx])))
            ax[1,0].step([],[],c=c_noskysub,label=label_noskysub)
            ax[1,0].step([],[],c=c_drp,label=label_drp)
            ax[1,0].step([],[],c=c_zap1,label='ZAP')
            ax[1,0].set_xlabel(r'Observed Wavelength $[\AA]$')
            ax[1,0].set_ylabel(r'Standard Deviation [$e^-$]')
            ax[1,0].set_yscale('log')
            ax[1,0].legend()


            #Sky Spectra
            #NO SUB SKY
            ax[1,1].set_title('Sky Statistical Metrics')
            ax[1,1].set_xticks(positions_stats)
            ax[1,1].set_xticklabels(labels_stats)
            ax[0,1].patch.set_facecolor('none')
            #NoSkySub
            ax[1,1].scatter(positions_stats[0], flux_nosub_sky_std,c=c_noskysub, marker=marker_std, alpha=alphan)
            ax[1,1].scatter(positions_stats[0], flux_nosub_sky_rms,c=c_noskysub, marker=marker_rms, alpha=alphan)
            ax[1,1].scatter(positions_stats[0], flux_nosub_sky_mod,c=c_noskysub, marker=marker_mod, alpha=alphan)
            ax[1,1].scatter(positions_stats[0], flux_nosub_sky_med,c=c_noskysub, marker=marker_med, alpha=alphan)
            ax[1,1].scatter(positions_stats[0], flux_nosub_sky_avg,c=c_noskysub, marker=marker_avg, alpha=alphan)
            #DRP SKY
            ax[1,1].scatter(positions_stats[1], flux_drp_sky_std, c=c_drp, marker=marker_std, alpha=alphan)
            ax[1,1].scatter(positions_stats[1], flux_drp_sky_rms, c=c_drp, marker=marker_rms, alpha=alphan)
            ax[1,1].scatter(positions_stats[1], flux_drp_sky_avg, c=c_drp, marker=marker_avg, alpha=alphan)
            ax[1,1].scatter(positions_stats[1], flux_drp_sky_med, c=c_drp, marker=marker_med, alpha=alphan)
            ax[1,1].scatter(positions_stats[1], flux_drp_sky_mod, c=c_drp, marker=marker_mod, alpha=alphan)
            #ZAP1 SKY
            ax[1,1].scatter(positions_stats[2], flux_zap1_sky_std, c=c_zap1, marker=marker_std, alpha=alphan)
            ax[1,1].scatter(positions_stats[2], flux_zap1_sky_rms, c=c_zap1, marker=marker_rms, alpha=alphan)
            ax[1,1].scatter(positions_stats[2], flux_zap1_sky_avg, c=c_zap1, marker=marker_avg, alpha=alphan)
            ax[1,1].scatter(positions_stats[2], flux_zap1_sky_med, c=c_zap1, marker=marker_med, alpha=alphan)
            ax[1,1].scatter(positions_stats[2], flux_zap1_sky_mod, c=c_zap1, marker=marker_mod, alpha=alphan)
            #Tidy up
            ax[1,1].scatter([],[],marker=marker_std, c=c_ex, label='StanDev')
            ax[1,1].scatter([],[],marker=marker_rms, c=c_ex, label='RMS')
            ax[1,1].scatter([],[],marker=marker_avg, c=c_ex, label='Average')
            ax[1,1].scatter([],[],marker=marker_med, c=c_ex, label='Median')
            ax[1,1].scatter([],[],marker=marker_mod, c=c_ex, label='Mode')
            ax[1,1].set_xlabel('')
            ax[1,1].set_ylabel(r'Sky Statistics $[e^{-}]$')
            ax[1,1].set_yscale('log')
            ax[1,1].legend()

            #Fix up spacing and save
            pngpathh = skyfnamm+'.png'
            #f.tight_layout()
            f.savefig(pngpathh, dpi=300, bbox_inches='tight')
            return pngpathh, flux_zap1_sky_std
        
        def plotvarcurves(zobjs,skyvarfnn):
            import matplotlib.pyplot as plt
            nseg = len(zobjs.models)
            fig, axes = plt.subplots(nseg, 3, figsize=(16, nseg * 2),
                                 tight_layout=True)
            if nseg==1:
                i=0
                var = zobjs.models[i].explained_variance_
                #compute derivative function
                arr, nsigma = var, 5
                npix = int(0.25 * arr.shape[0])
                deriv = np.diff(arr[:npix])
                ind = int(.15 * deriv.size)
                mn1 = deriv[ind:].mean()
                std1 = deriv[ind:].std() * nsigma
                #Variance
                ax1, ax2, ax3 = axes
                ax1.plot(var, linewidth=3)
                ax1.plot([zobjs.nevals[i], zobjs.nevals[i]], [min(var), max(var)])
                ax1.set_ylabel('Variance')
                #dVariance/dn
                ax2.plot(np.arange(deriv.size), deriv)
                ax2.hlines([mn1, mn1 - std1], 0, len(deriv), colors=('k', '0.5'))
                ax2.plot([zobjs.nevals[i] - 1, zobjs.nevals[i] - 1],
                        [min(deriv), max(deriv)])
                ax2.set_ylabel('d/dn Var')
                #d2Var/dn2
                deriv2 = np.diff(deriv)
                ax3.plot(np.arange(deriv2.size), np.abs(deriv2))
                ax3.plot([zobjs.nevals[i] - 2, zobjs.nevals[i] - 2],
                        [min(deriv2), max(deriv2)])
                ax3.set_ylabel('(d^2/dn^2) Var')
                # ax3.set_xlabel('Number of Components')
                ax1.set_title('Segment {0}, {1} - {2} Angstroms'.format(
                    i, zobjs.lranges[i][0], zobjs.lranges[i][1]))
            else:
                for i in range(nseg):
                    var = zobjs.models[i].explained_variance_
                    #compute derivative function
                    arr, nsigma = var, 5
                    npix = int(0.25 * arr.shape[0])
                    deriv = np.diff(arr[:npix])
                    ind = int(.15 * deriv.size)
                    mn1 = deriv[ind:].mean()
                    std1 = deriv[ind:].std() * nsigma
                    #Variance
                    ax1, ax2, ax3 = axes[i]
                    ax1.plot(var, linewidth=3)
                    ax1.plot([zobjs.nevals[i], zobjs.nevals[i]], [min(var), max(var)])
                    ax1.set_ylabel('Variance')
                    #dVariance/dn
                    ax2.plot(np.arange(deriv.size), deriv)
                    ax2.hlines([mn1, mn1 - std1], 0, len(deriv), colors=('k', '0.5'))
                    ax2.plot([zobjs.nevals[i] - 1, zobjs.nevals[i] - 1],
                            [min(deriv), max(deriv)])
                    ax2.set_ylabel('d/dn Var')
                    #d2Var/dn2
                    deriv2 = np.diff(deriv)
                    ax3.plot(np.arange(deriv2.size), np.abs(deriv2))
                    ax3.plot([zobjs.nevals[i] - 2, zobjs.nevals[i] - 2],
                            [min(deriv2), max(deriv2)])
                    ax3.set_ylabel('(d^2/dn^2) Var')
                    # ax3.set_xlabel('Number of Components')
                    ax1.set_title('Segment {0}, {1} - {2} Angstroms'.format(
                        i, zobjs.lranges[i][0], zobjs.lranges[i][1]))
            fig.tight_layout()
            pngvarpath = skyvarfnn+'.png'
            fig.savefig(pngvarpath, dpi=300, bbox_inches='tight')
            return pngvarpath

        ### LOAD *icube.fits FILE, CROP IT SPECTRALLY, REPLACE NANS, SAVE AS *icube_cropped.fits ###
        ofn_full = self.action.args.name
        rdir = self.config.instrument.output_directory
        fn = os.path.join(rdir, strip_fname(ofn_full) + '_icube.fits')
        scihdu_notrim = fits.open(fn) #should have already passed the pre_condition check above to be a *icube.fits file
        scihdu = crop_cube(scihdu_notrim) #Reduce zaxis to WAVGOOD0/1
        scihdr = scihdu[0].header
        obswave = (np.arange(scihdr['NAXIS3']) + 1 - scihdr['CRPIX3']) * scihdr['CD3_3'] + scihdr['CRVAL3']
        # Save the non-sky subtracted cube
        if getattr(self.action.args.ccddata, "noskysub", None) is None:
            noskysub = scihdu[0].data
            self.action.args.ccddata.noskysub = scihdu[0].data # store NON-SKY subtracted image
            kcwi_fits_writer(self.action.args.ccddata,
                table=self.action.args.table,
                output_file=self.action.args.name,
                output_dir=self.config.instrument.output_directory,
                suffix="icube")
        else:
            noskysubhdr = scihdu['NOSKYSUB'].data

        #replace the edge pixels with NaNs
        badpix = np.where(np.mean(scihdu['FLAGS'].data, axis = 0) > 100)
        scihdu[0].data[:, badpix[0], badpix[1]] = np.nan
        scihdu['UNCERT'].data[:, badpix[0], badpix[1]] = np.nan
        maskflags = np.mean(scihdu['FLAGS'].data, axis = 0)
        hdr2d = collapse_header(scihdu[0].header)
        mhdu = fits.ImageHDU(maskflags, header = hdr2d)

        # MAKE WHITELIGHT IMAGE #
        if (self.config.instrument.zap_interactive == True) or (self.action.args.zap_interactive == True):
            #This below is used to make the white light images that ZAP then uses as a preliminary mask 
            wlimg_wave_range_red = [6380, 7200] #pick this region to generate the white-lighted image because sky lines are much stronger elsewhere. 
            wlimg_wave_range_blue = [3600, 5500] # TODO: Can also make it as an input or variable parameter in the GUI
            if obswave[-1] > wlimg_wave_range_red[0]: #Update wl image bounds to ensure its creation
                wlimg_wave_range = wlimg_wave_range_red
            else:
                wlimg_wave_range = wlimg_wave_range_blue
            #For other gratings i.e., RM or RH:
            windex = (obswave > wlimg_wave_range[0]) & (obswave < wlimg_wave_range[1])
            if np.sum(windex) ==0:
                wlimg_wave_range = [scihdr['WAVGOOD0'], scihdr['WAVGOOD1']]
            wlimg_index = np.where((obswave >= wlimg_wave_range[0]) & (obswave <= wlimg_wave_range[1]))[0]
            wlimg = np.sum(scihdu[0].data[wlimg_index], axis = 0)
            hdr2d = collapse_header(scihdu[0].header)
            wlhdu = fits.PrimaryHDU(wlimg, header = hdr2d)
            hdulist = fits.HDUList([wlhdu, mhdu])
            hdulist.writeto(os.path.join(rdir, strip_fname(ofn_full) + '_icube_zapwlimg.fits'), overwrite = True)
        
        
        ### HANDLE SKY MASKS AND OFF-SKY FRAMES ##
        #Does the user have a skyfile to specify sky subtraction parameters?
        if (self.action.args.skyyaml is not None) or (self.action.args.stdfile is not None):
            #Is there a science sky mask available?
            if self.action.args.zap_skymask is not None:
                scihdu[0].header['ZAPSKYMASK'] = self.action.args.zap_skymask
                #Trim the edges of the cube to get an even better sky model
                zsm = fits.open(self.action.args.zap_skymask)
                zsm.data[:, :1], zsm.data[:, -1:] = 1, 1 #x mask the edges to avoid edge effects in the sky model
                zsm.data[:2, :], zsm.data[-2:, :] = 1, 1 #y mask the edges to avoid edge effects in the sky model
                zsm.writeto(self.action.args.zap_skymask, overwrite = True)

            #Is automated continuum masking being requested?
            elif (self.action.args.zap_use_auto_cont == True) or (self.action.args.stdfile is not None):
                if self.action.args.stdfile is not None:
                    self.logger.info("Processing standard star, finding bright continuum source automatically")
                else:
                    self.logger.info("Finding bright continuum source automatically")
                scihdu[0].header['ZAPAUTOMASK'] = True
                fnautomask = os.path.join(rdir, strip_fname(ofn_full) + '_icube_zapsmskauto.fits')
                self.action.args.zap_skymask = fnautomask
                #Generate initial guesses for the 2D Gaussian fit to find the continuum source and generate the ZAP sky mask
                wl = np.sum(scihdu[0].data, axis = 0) #make whitelight image 
                wl_clip = sigma_clip(wl, sigma=3).data #try and remove leftover cosmic rays
                wl_clip[:, :2], wl_clip[:, -2:] = 0, 0 #x trim the edges of the cube to avoid edge effects in the fit
                wl_clip[:3, :], wl_clip[-3:, :] = 0, 0 #y trim the edges of the cube to avoid edge effects in the fit
                amp_og = np.max(wl_clip) #Initial amplitude guess
                idx_pk = np.where(wl_clip == np.max(wl_clip)) #flux peak of the whitelight image
                x_og, y_og  = idx_pk[1][0], idx_pk[0][0] #x peak, y peak for initial guess
                xsigma_og, ysigma_og = 4 , 2.5 #sigma gueses for the 2D Gaussian fit
                theta_og = 0 #shoud be close to zero for the angle of the 2D Gaussian fit
                y, x = np.mgrid[:wl_clip.shape[0], :wl_clip.shape[1]]
                # 2D gaussian fit to peak
                fitter = fitting.LevMarLSQFitter()
                mod_og = models.Gaussian2D(amplitude=amp_og, x_mean=x_og, y_mean=y_og, x_stddev=xsigma_og, y_stddev=ysigma_og, theta=theta_og)
                mod_fit = fitter(mod_og,  x, y, wl_clip)
                #Take best fit parameters and turn into elliptical mask
                growthfactor = 1.25 #=2.355 to convert sigma to fwhm, can use smaller factor too
                x, y, xwidth, ywidth = mod_fit.x_mean.value, mod_fit.y_mean.value, mod_fit.x_stddev.value*growthfactor, mod_fit.y_stddev.value*growthfactor
                center = center = PixCoord(x, y)
                ellipse_reg = EllipsePixelRegion(center=center, width=xwidth, height=ywidth, angle=mod_fit.theta.value*u.radian)
                mask = ellipse_reg.to_mask(mode='center').to_image(shape=wl_clip.shape)
                zapskymask = fits.PrimaryHDU(mask)
                zapskymask.writeto(fnautomask, overwrite = True)
                self.action.args.zap_skymask = fnautomask
                self.logger.info("Continuum source xmean and ymean at (%.2f , %.2f),"
                                "x_width= %.2f, y_width=%.2f, and theta=%.2f degrees"
                                % (x, y, xwidth, ywidth, np.degrees(mod_fit.theta.value)))

            #Are we masking a faint source and being given its box vertices positions?
            elif self.action.args.zap_use_faint_cont==True:
                x1, x2 = self.action.args.zap_faint_cont_x1y1[0], self.action.args.zap_faint_cont_x2y2[0]
                y1, y2 = self.action.args.zap_faint_cont_x1y1[1], self.action.args.zap_faint_cont_x2y2[1]
                mask_shape = np.sum(scihdu[0].data, axis = 0).shape
                fnfaintskymaskzap = os.path.join(rdir, strip_fname(ofn_full) + '_icube_zapsmskfaint.fits')
                faintskymaskzaparr = np.zeros(mask_shape, dtype=int)
                faintskymaskzaparr[y1:y2, x1:x2] = 1
                faintskymaskzaparr[:, :1], faintskymaskzaparr[:, -1:] = 1, 1 #x mask the edges to avoid edge effects in the sky model
                faintskymaskzaparr[:2, :], faintskymaskzaparr[-2:, :] = 1, 1 #y mask the edges to avoid edge effects in the sky model
                faintskymaskzap = fits.PrimaryHDU(faintskymaskzaparr, header = hdr2d)
                faintskymaskzap.writeto(fnfaintskymaskzap, overwrite = True)
                scihdu[0].header['ZAPFAINTMASK'] = True
                scihdu[0].header['ZAPFAINTX1Y1'] = self.action.args.zap_faint_cont_x1y1
                scihdu[0].header['ZAPFAINTX2Y2'] = self.action.args.zap_faint_cont_x2y2
                self.action.args.zap_skymask = fnfaintskymaskzap

            #Using an off field sky frame
            elif self.action.args.offsky is not None:
                scihdu[0].header['OFFSKY'] = self.action.args.offsky
                offskyhdu = fits.open(self.action.args.offsky)
                offskyhdu = crop_cube(offskyhdu) #crop the data cube to good wavelength region
                scale_factor = scale_extinct_sky(offskyhdu[0].header, scihdr) #scale the sky to the same airmass as sci
                offskyhdu[0].data *= scale_factor[:, np.newaxis, np.newaxis]
                
                #scale the sky spectrum if the exposure time between the science and sky doesn't match.
                scaling_factor = scihdr['XPOSURE'] / offskyhdu[0].header['XPOSURE'] #ensure sky and science exposures match 
                offskyhdu[0].data *= scaling_factor
                offskyhdu[0].header['XPOSURE'] = scihdr['XPOSURE']
                skyhdr = offskyhdu[0].header

                #replace the edge pixels with NaNs
                skybadpix = np.where(np.mean(offskyhdu['FLAGS'].data, axis = 0) > 100)
                offskyhdu[0].data[:, skybadpix[0], skybadpix[1]] = np.nan

                #Clunky but write out the file
                offskyhdu[0].header['ZAPPROCESSED'] = True
                offskyhdu[0].header['ZAPOFFSKY'] = True
                offskyhdu.writeto(self.action.args.offsky, overwrite = True)

                if self.action.args.zap_offsky_mask is not None:
                    scihdu[0].header['ZAPOFFSKYMASK'] = self.action.args.zap_offsky_mask

                #Is interactive mode set?
                if (self.config.instrument.zap_interactive == True) or (self.action.args.zap_interactive == True):
                    offskywave = (np.arange(skyhdr['NAXIS3']) + 1 - skyhdr['CRPIX3']) * skyhdr['CD3_3'] + skyhdr['CRVAL3']
                    if offskywave[-1] > wlimg_wave_range_red[0]:
                        wlimg_wave_range = wlimg_wave_range_red
                    else:
                        wlimg_wave_range = wlimg_wave_range_blue
                    # in case we are using RM or RH:
                    offskywindex = (offskywave > wlimg_wave_range[0]) & (offskywave < wlimg_wave_range[1])
                    if np.sum(offskywindex) ==0:
                        wlimg_wave_range = [skyhdr['WAVGOOD0'], skyhdr['WAVGOOD1']]

                    #Create white-light image
                    offskywlimg_index = np.where((offskywave >= wlimg_wave_range[0]) & (offskywave <= wlimg_wave_range[1]))[0]
                    offskywlimg = np.sum(offskyhdu[0].data[offskywlimg_index], axis = 0)
                    offskyhdr2d = collapse_header(offskyhdu[0].header)
                    offskywlhdu = fits.PrimaryHDU(offskywlimg, header = offskyhdr2d)
                    offskymask = np.mean(offskyhdu['FLAGS'].data, axis = 0)
                    offskymhdu = fits.ImageHDU(offskymask, header = offskyhdr2d)
                    offskyhdulist = fits.HDUList([offskywlhdu, offskymhdu])
                    offskyhdulist.writeto(os.path.join(rdir, strip_fname(self.action.args.offsky) + '_zapwlimg.fits'), overwrite = True)


        ### ESTABLISH SKY SEGMENTS FOR ZAP ###
        skyseg0 = []
        zap_skysegmentoption = self.config.instrument.zap_skysegmentoption #Default to system config
        #Did the user define a sky segment option in the sky.yaml file?
        if self.action.args.zap_skysegmentoption is not None:
            zap_skysegmentoption = self.action.args.zap_skysegmentoption
            self.logger.info("# Frame specific skysegment option chosen #")
        ## Setting sky segment based on provided info ##
        # Single segment using "WAVEGOOD" bounds of the cube
        if zap_skysegmentoption.lower() == 'single': 
            skyseg0 = [obswave[0], obswave[-1]]
            self.logger.info("# Using single sky segment. #")
        #Using Sky Segments from Table 1 in Soto+16 (see paper for details) 
        elif zap_skysegmentoption.lower() == 'soto+2016':
            skyseg0 = [0, 5400, 5850, 6440, 6750, 7200, 7700, 8265, 8602, 8731, 9275, 10000]
            self.logger.info("# Using sky segments defined by Soto+2016. #")
        #User defined sky segments
        elif zap_skysegmentoption.lower() == 'custom':
            skyseg0 = self.config.instrument.zap_customskysegments #Default to system configuration
            if self.action.args.zap_customskysegments is not None: #Did the user define custom sky segments 
                skyseg0 = self.action.args.zap_customskysegments
                self.logger.info("# Frame specific custom segments supplied #")
            if not isinstance(skyseg0, (list, np.ndarray)):
                self.logger.warning("# Custom segment supplied not a list or array: %s  #" % skyseg0)
                skyseg0 = [obswave[0], obswave[-1]]
                self.logger.warning("# Using single sky segment instead. #")
            else:
                self.logger.info("# Using custom sky segments #")
        else: #Unknown option given
            skyseg0 = [obswave[0], obswave[-1]] #uses the "WAVEGOOD" bounds of the cube 
            self.logger.info("# Unknown option given for skysegment. Using a single sky segment. #")
        zap_skyseg = trim_skysegments(skyseg0, obswave) #remove sky segements that fall outside of the spectral region in case ZAP runs into problem
        #continuum filter width 
        zap_cfwidth = self.config.instrument.zap_cfwidth # 300 is default
        if self.action.args.zap_cfwidth is not None: #If user specified a width for this file
            zap_cfwidth = int(self.action.args.zap_cfwidth)
            self.logger.info("# Frame specific cfwidth supplied #")
        if (zap_cfwidth is not None) and (str(zap_cfwidth).lower() != 'none'): #Maing sure the user did not specify a frame specific cfwidth
            if (zap_cfwidth > 0.5*(obswave.max() - obswave.min())): #Is the width more than half the wavelength range?
                zap_cfwidth = int(0.5*(obswave.max() - obswave.min())) #if so, set it to be half
                self.logger.warning("Default or chosen width is larger than the wavelength range. It is now set to 1/2 of Deltalambda: %s. To change this setting, specify cfwidth in the sky.yaml file for this object" % zap_cfwidth)
        else:
            zap_cfwidth = 300
            self.logger.info("# Using default cfwidth as none was supplied")
        self.logger.info("# Using cfwidth= %s and sky segments at %s Angstroms #" % (zap_cfwidth, zap_skyseg))

        #Clunky but write out the file
        self.action.args.ccddata.data = scihdu[0].data
        self.action.args.ccddata.header = scihdu[0].header
        self.action.args.ccddata.uncertainty = scihdu['UNCERT'].data
        self.action.args.ccddata.mask = scihdu['MASK'].data
        self.action.args.ccddata.flags = scihdu['FLAGS'].data
        if getattr(self.action.args.ccddata, "prezap", None) is None:
            self.action.args.ccddata.prezap = scihdu[0].data # store cube to be used by ZAP
        kcwi_fits_writer(self.action.args.ccddata,
            table=self.action.args.table,
            output_file=self.action.args.name,
            output_dir=self.config.instrument.output_directory,
            suffix="icube")

        ### RUN ZAP ###
        scihdu.close()
        #Set number of CPUS
        ncpus = self.config.instrument.NCPUS #Fine if this is none, Python's multithreading processor will take care of setting this
        #Set the sky segments in ZAP
        SKYSEG[:] = zap_skyseg
        # Are we running ZAP on science frame or sky frame?
        zap_time_start = time.perf_counter()
        zobj = ''
        if self.action.args.offsky is not None: #Run ZAP using using seperate sky frame to generate sky model
            self.logger.info("-----##### RUNNING ZAP USING OFF FIELD SKY #####-----")
            icube_forzap = os.path.join(rdir, strip_fname(ofn_full) + '_icube.fits')
            off_skymask_forzap = self.action.args.zap_offsky_mask
            extSVD = zap.SVDoutput(self.action.args.offsky, mask = off_skymask_forzap, ncpu=ncpus, zlevel = 'median')
            zobj = zap.process(icube_forzap, interactive = True, cfwidthSP = zap_cfwidth, cfwidthSVD = zap_cfwidth, ncpu=ncpus, extSVD=extSVD)
        else: #Run on the single science frame 
            self.logger.info("-----##### RUNNING ZAP USING IN FIELD SKY #####-----")
            icube_forzap = os.path.join(rdir, strip_fname(ofn_full) + '_icube.fits')
            skymask_forzap = self.action.args.zap_skymask
            zobj = zap.process(icube_forzap, mask = skymask_forzap, interactive = True, cfwidthSP = zap_cfwidth, cfwidthSVD = zap_cfwidth, ncpu=ncpus, zlevel = 'median')
        zap_time_end = time.perf_counter()
        self.logger.info("-----##### ZAP complete after {:.2f} seconds #####-----".format(zap_time_end - zap_time_start))

        ### SIGMA CLIP SKY CUBE ###
        nsig = 3
        skycube0 = zobj.cube - zobj.cleancube #residuls between the inpout cube and the ZAP cleaned cube.
        skycube = skycube0.copy()
        mask = maskflags #get the mask to avoid the edge pixel. Should be similar if using the off-field sky
        use = np.abs(mask - 1) > 1e-6 #mask = 1 for edge mask
        skycube[:,~use] = np.nan
        skycube_clipped = sigma_clip(skycube, sigma = nsig, axis = (1,2))
        median_sky = np.ma.median(skycube_clipped, axis = (1,2)).data
        median_cube = median_sky[:, np.newaxis, np.newaxis] * np.ones((1, np.shape(skycube)[1], np.shape(skycube)[2]))
        skycube[skycube_clipped.mask] = median_cube[skycube_clipped.mask]
        skycube[:, ~use] = skycube0[:, ~use]
        cleancube = zobj.cube - skycube

        ### PLOT RESULTS ###
        if self.config.instrument.plot_level >= 1:
            skyfnam = "plots/zapsky_%05d_%s_%s_%s" % \
                     (self.action.args.ccddata.header['FRAMENO'],
                      self.action.args.illum, self.action.args.grating,
                      self.action.args.ifuname)
            p = figure(x_range=(0, 1), y_range=(0, 1),
                plot_width=self.config.instrument.plot_width,
                plot_height=self.config.instrument.plot_height)
            pngpath, standev = plot_skystats(self, wavee=obswave, cleancubee=cleancube, noskysubb=noskysub, skyfnamm=skyfnam, zap_skysegg=zap_skyseg)
            full_pngpath = os.path.join(os.getcwd(), pngpath)
            print(full_pngpath)
            p.image_url(url=[pngpath], x=0, y=1) #,w=1, h=1, anchor="bottom_left")
            bokeh_plot(p, self.context.bokeh_session)
            if self.config.instrument.plot_level >= 2:
                input("Next? <cr>: ")
            else:
                time.sleep(self.config.instrument.plot_pause)

        ### INTERACTIVELY RUN ZAP REQUESTED ###
        if (self.config.instrument.zap_interactive == True) or (self.action.args.zap_interactive == True):
            iteration=0
            done = False
            rerunzap=False
            self.logger.info('Iteratively Running ZAP:')
            n, tmp_stats, tmp_neigenvals, tmp_skysegs, tmp_skymask, tmp_cfwidth = 0, [], [], [], [], []
            while not done:
                #Plotting the sky diagnostics plot
                skyfnam = "plots/zapsky_iteration-%s_%05d_%s_%s_%s" % \
                     (iteration,
                      self.action.args.ccddata.header['FRAMENO'],
                      self.action.args.illum, self.action.args.grating,
                      self.action.args.ifuname)
                pngpath, standev = plot_skystats(self, wavee=obswave, cleancubee=cleancube, noskysubb=noskysub, skyfnamm=skyfnam, zap_skysegg=zap_skyseg)
                p.image_url(url=[pngpath], x=0, y=0, w=1, h=1, anchor="bottom_left")
                bokeh_plot(p, self.context.bokeh_session)
                self.logger.info('Diagnostic plots generated and saved at: %s' % pngpath)
                input("Next? <cr>: ")

                #Plotting the variance curves used to determine number of eigenspectra to be used
                skyfnamvar = "plots/zapsky_variancecurves_iteration-%s_%05d_%s_%s_%s" % \
                     (iteration,
                      self.action.args.ccddata.header['FRAMENO'],
                      self.action.args.illum, self.action.args.grating,
                      self.action.args.ifuname)
                pvarpath = plotvarcurves(zobj, skyfnamvar)
                p.image_url(url=[pvarpath], x=0, y=0, w=1, h=1, anchor="bottom_left")
                bokeh_plot(p, self.context.bokeh_session)
                self.logger.info('Variance diagnostic plots generated and saved at: %s' % pvarpath)
                input("Next? <cr>: ")

                # Collect then print current/previous iterations statistics
                tmp_neigenvals.append(zobj.nevals)
                tmp_stats.append(standev)
                tmp_cfwidth.append(zap_cfwidth)
                if self.action.args.zap_offsky_mask is not None:
                    tmp_skymask.append(self.action.args.zap_offsky_mask)
                elif self.action.args.zap_skymask is not None:
                    tmp_skymask.append(self.action.args.zap_skymask)
                else:
                    tmp_skymask.append(None)
                curr_skyseg = []
                nseg = len(zobj.models)
                for y in range(nseg):
                    curr_skyseg.append(zobj.lranges[y][0])
                    curr_skyseg.append(zobj.lranges[y][1])
                tmp_skysegs.append([curr_skyseg])


                #Plot standard deviation for all runs up to this point
                #Now ask the user what/if they want to change anything
                for z in range(len(tmp_stats)):
                    self.logger.info("Iteration= %s, standev of sky spec: %s, cfwidth: %s, skymask: %s, neigenvals: %s, skysegments: %s" % (z, tmp_stats[z], tmp_cfwidth[z], tmp_skymask[z], tmp_neigenvals[z], tmp_skysegs[z]))
                stage = input("What would you like to modify? (skysegments, cfwidth, skymask, Neigenvals, or none/Enter):")

                #The ueser does not want to change anything anymore, end the loop
                if (len(stage) <=0) or ('none' in stage.lower()):
                    self.logger.info('User does not want to make a change. Moving on')
                    done = True
                
                # User wants to change the number of eigenvalues used
                elif 'neigenval' in stage.lower():
                    self.logger.info('Review the variance diagnostic plots to determine number of eigenvalues to use at: %s' % pvarpath)
                    self.logger.info('Current number of eigenvalues used per sky segment: %s' % zobj.nevals)
                    nunevals = list(input("Please enter a list of the number of eigenspectra values that you would like to use (seperate by commas i.e., 1,2,3,4,5): ").split(','))
                    nunevals = [int(x) for x in nunevals]
                    self.logger.info("# Reprocessing file with new number of eigenvalues: %s #" % nunevals)
                    zobj.reprocess(nevals=nunevals)
                    zap_time_start = time.perf_counter()
                    rerunzap = True
                    #Rerun ZAP with new nevals 
                    if self.action.args.offsky is not None: #Run ZAP using using seperate sky frame to generate sky model
                        self.logger.info("-----##### RUNNING ZAP USING OFF FIELD SKY W/ NEW NEIGENVALUES #####-----")
                        icube_forzap = os.path.join(rdir, strip_fname(ofn_full) + '_icube.fits')
                        off_skymask_forzap = self.action.args.zap_offsky_mask
                        extSVD = zap.SVDoutput(self.action.args.offsky, mask = off_skymask_forzap, ncpu=ncpus, zlevel = 'median')
                        zobj = zap.process(icube_forzap, nevals=nunevals, interactive = True, cfwidthSP = zap_cfwidth, cfwidthSVD = zap_cfwidth, ncpu=ncpus, extSVD=extSVD)
                    else: #Run on the single science frame 
                        self.logger.info("-----##### RUNNING ZAP USING IN FIELD SKY W/ NEW NEIGENVALUES #####-----")
                        icube_forzap = os.path.join(rdir, strip_fname(ofn_full) + '_icube.fits')
                        skymask_forzap = self.action.args.zap_skymask
                        zobj = zap.process(icube_forzap, nevals=nunevals, mask = skymask_forzap, interactive = True, cfwidthSP = zap_cfwidth, cfwidthSVD = zap_cfwidth, ncpu=ncpus, zlevel = 'median')
                    zap_time_end = time.perf_counter()
                    self.logger.info("-----##### ZAP complete after {:.2f} seconds #####-----".format(zap_time_end - zap_time_start))
                
                #User wants to change the sky segments
                elif 'skysegment' in stage.lower():
                    rerunzap = False
                    tmp_skyseg0 = ''
                    segtype = input("Please enter the sky segment you want to use? (single, custom, or soto+2016): ")
                    #Single component
                    if 'single' in segtype.lower():
                        self.logger.info('User will use a single sky segment')
                        tmp_skyseg0 = [obswave[0], obswave[-1]]
                        rerunzap = True
                    # Soto+2016
                    elif 'soto+2016' in segtype.lower():
                        self.logger.info('User requested sky segments from Soto+2016')
                        tmp_skyseg0 = [0, 5400, 5850, 6440, 6750, 7200, 7700, 8265, 8602, 8731, 9275, 10000]
                        rerunzap = True
                    #Custom segments 
                    elif 'custom' in segtype.lower():
                        self.logger.info('User will supply custom sky segments')
                        tmp_skyseg0input = list(input("Enter the sky segments as a comma seperate list with no quotes (0,5400,5850): ").split(','))
                        tmp_skyseg0 = [int(x) for x in tmp_skyseg0input]
                        if isinstance(tmp_skyseg0, list):
                            self.logger.info('Supplied custom segments: %s' % tmp_skyseg0)
                            rerunzap = True
                        else:
                            self.logger.warning('Supplied custom segments not a list. please try again a comma seperate list with no quotes (0,5400,5850)')
                    else:
                        self.logger.warning('Sky segment option not recognized. Try again (single, soto+2016, custom). Rerunning with same sky:')
                    #Should we rerun ZAP?
                    if rerunzap:
                        tmp_skyseg = trim_skysegments(tmp_skyseg0, obswave)
                        SKYSEG[:] = tmp_skyseg
                        zap_skyseg = tmp_skyseg
                        self.logger.info("# Reprocessing file with new sky segment. #")
                        #Rerun ZAP with new sky segments
                        zap_time_start = time.perf_counter()
                        if self.action.args.offsky is not None: #Run ZAP using using seperate sky frame to generate sky model
                            self.logger.info("-----##### RUNNING ZAP USING OFF FIELD SKY W/ NEW SKY SEGMENT(S) #####-----")
                            icube_forzap = os.path.join(rdir, strip_fname(ofn_full) + '_icube.fits')
                            off_skymask_forzap = self.action.args.zap_offsky_mask
                            extSVD = zap.SVDoutput(self.action.args.offsky, mask = off_skymask_forzap, ncpu=ncpus, zlevel = 'median')
                            zobj = zap.process(icube_forzap, interactive = True, cfwidthSP = zap_cfwidth, cfwidthSVD = zap_cfwidth, ncpu=ncpus, extSVD=extSVD)
                        else: #Run on the single science frame 
                            self.logger.info("-----##### RUNNING ZAP USING IN FIELD SKY W/ NEW SKY SEGMENT(S) #####-----")
                            icube_forzap = os.path.join(rdir, strip_fname(ofn_full) + '_icube.fits')
                            skymask_forzap = self.action.args.zap_skymask
                            zobj = zap.process(icube_forzap, mask = skymask_forzap, interactive = True, cfwidthSP = zap_cfwidth, cfwidthSVD = zap_cfwidth, ncpu=ncpus, zlevel = 'median')
                        zap_time_end = time.perf_counter()
                        self.logger.info("-----##### ZAP complete after {:.2f} seconds #####-----".format(zap_time_end - zap_time_start))
                    else:
                        self.logger.warning("No useable sky segments supplied, will not run ZAP. Try again")

                #User wants to supply a mask
                elif 'skymask' in stage.lower():
                    rereunzap=False
                    if self.action.args.offsky is not None:
                        pathmask = input('Please enter path to *_zapsmsk.fits file for off sky frame %s: ' % self.action.args.offsky)
                    else:   
                        pathmask = input('Please enter path to a *_zapsmsk.fits file: ')
                    if os.path.exists(pathmask):
                        self.logger.info("# Will reprocess frame using supplied skymask: %s " % pathmask)
                        rereunzap=True
                    else:
                        self.logger.warning("Supplied mask does not exits: %s" % pathmask)
                    if rereunzap:
                        #Rerun ZAP with new nevals
                        new_skymask = pathmask
                        #Run ZAP with newly supplied mask
                        zap_time_start = time.perf_counter()
                        if self.action.args.offsky is not None: #Run ZAP using using seperate sky frame to generate sky model
                            self.logger.info("!!! Applying supplied mask to the OFFSKY FRAME. To apply this mask to the current frame for in-field sky, remove the 'offsky' keyword in the sky.yaml. !!!")
                            self.logger.info("-----##### RUNNING ZAP USING OFF FIELD SKY W/ NEW MASK #####-----")
                            self.action.args.zap_offsky_mask = new_skymask
                            icube_forzap = os.path.join(rdir, strip_fname(ofn_full) + '_icube.fits')
                            extSVD = zap.SVDoutput(self.action.args.offsky, mask = new_skymask, ncpu=ncpus, zlevel = 'median')
                            zobj = zap.process(icube_forzap, interactive = True, cfwidthSP = zap_cfwidth, cfwidthSVD = zap_cfwidth, ncpu=ncpus, extSVD=extSVD)
                        else: #Run on the single science frame 
                            self.logger.info("-----##### RUNNING ZAP USING IN FIELD SKY W/ NEW MASK #####-----")
                            self.action.args.zap_skymask = new_skymask
                            icube_forzap = os.path.join(rdir, strip_fname(ofn_full) + '_icube.fits')
                            zobj = zap.process(icube_forzap, mask = new_skymask, interactive = True, cfwidthSP = zap_cfwidth, cfwidthSVD = zap_cfwidth, ncpu=ncpus, zlevel = 'median')
                        zap_time_end = time.perf_counter()
                        self.logger.info("-----##### ZAP complete after {:.2f} seconds #####-----".format(zap_time_end - zap_time_start))
                
                #User wants to change the cfwidth
                elif 'cfwidth' in stage.lower():
                    self.logger.info('Current cfwidth: %s' % zap_cfwidth)
                    zap_cfwidth = int(input("Please enter a new cfwidth (integer): "))
                    self.logger.info("# Reprocessing frame with new cfwidth: %s #" % zap_cfwidth)
                    zap_time_start = time.perf_counter()
                    rerunzap = True
                    #Rerun ZAP with new nevals 
                    if self.action.args.offsky is not None: #Run ZAP using using seperate sky frame to generate sky model
                        self.logger.info("-----##### RUNNING ZAP USING OFF FIELD SKY W/ NEW CFWIDTH #####-----")
                        icube_forzap = os.path.join(rdir, strip_fname(ofn_full) + '_icube.fits')
                        off_skymask_forzap = self.action.args.zap_offsky_mask
                        extSVD = zap.SVDoutput(self.action.args.offsky, mask = off_skymask_forzap, ncpu=ncpus, zlevel = 'median')
                        zobj = zap.process(icube_forzap, interactive = True, cfwidthSP = zap_cfwidth, cfwidthSVD = zap_cfwidth, ncpu=ncpus, extSVD=extSVD)
                    else: #Run on the single science frame 
                        self.logger.info("-----##### RUNNING ZAP USING IN FIELD SKY W/ NEW CFWIDTH #####-----")
                        icube_forzap = os.path.join(rdir, strip_fname(ofn_full) + '_icube.fits')
                        skymask_forzap = self.action.args.zap_skymask
                        zobj = zap.process(icube_forzap, mask = skymask_forzap, interactive = True, cfwidthSP = zap_cfwidth, cfwidthSVD = zap_cfwidth, ncpu=ncpus, zlevel = 'median')
                    zap_time_end = time.perf_counter()
                    self.logger.info("-----##### ZAP complete after {:.2f} seconds #####-----".format(zap_time_end - zap_time_start))

                #Check to see if the cube was ZAPPED again then remake the sky spectrum 
                if rerunzap:
                    nsig = 3
                    skycube0 = zobj.cube - zobj.cleancube #residuls between the inpout cube and the ZAP cleaned cube.
                    skycube = skycube0.copy()
                    mask = maskflags #get the mask to avoid the edge pixel. Should be similar if using the off-field sky
                    use = np.abs(mask - 1) > 1e-6 #mask = 1 for edge mask
                    skycube[:,~use] = np.nan
                    skycube_clipped = sigma_clip(skycube, sigma = nsig, axis = (1,2))
                    median_sky = np.ma.median(skycube_clipped, axis = (1,2)).data
                    median_cube = median_sky[:, np.newaxis, np.newaxis] * np.ones((1, np.shape(skycube)[1], np.shape(skycube)[2]))
                    skycube[skycube_clipped.mask] = median_cube[skycube_clipped.mask]
                    skycube[:, ~use] = skycube0[:, ~use]
                    cleancube = zobj.cube - skycube
                iteration+=1
                #Back to top of the while loop
            #End interactive loop


        ### UPDATE SKY SEGMENT HEADERS ### 
        if self.config.instrument.offsky is not None:
            scihdu[0].header['ZAPOFFSKY'] = strip_fname(self.config.instrument.offsky)
        scihdu[0].header['ZAPSEGMODE'] = self.config.instrument.zap_skysegmentoption
        scihdu[0].header['ZAPCWITH'] = zap_cfwidth
        scihdu[0].header['ZAPNSEG'] = len(SKYSEG)-1
        nskyseg = np.arange(len(SKYSEG))
        for k in range(len(SKYSEG)):
            segstr = 'ZAPSEG{}'.format(k)
            scihdu[0].header[segstr] = SKYSEG[k]

        ### SAVE THE FINAL CUBE ###
        cleanhdu = scihdu.copy()
        cleanhdu[0].data = cleancube

        #If user asked to append the sky model as an extension
        if self.config.instrument.zap_append_sky == True:
            cleanhdu.append(scihdu[0])
            cleanhdu[-1].name = 'PREZAP'
            skyhdu = fits.ImageHDU(data=skycube, header=scihdu[0].header)
            skyhdu.name = 'ZAPSKYMODEL'
            cleanhdu.append(skyhdu)
        
        #If user asked to save the intermediate products for inspection, save the sky model as seperate file 
        if self.config.instrument.zap_interactive == True:
            #Save the sky model as a seperate FITS file for easy inspection
            skyhdu.writeto(os.path.join(rdir, strip_fname(ofn_full) + '_icube_zapsky.fits'), overwrite = True)
        
        #Write out the ZAPPED datacube and requested extensions
        self.action.args.ccddata.data = cleanhdu[0].data
        self.action.args.ccddata.header = cleanhdu[0].header
        self.action.args.ccddata.uncertainty = cleanhdu['UNCERT'].data
        self.action.args.ccddata.mask = cleanhdu['MASK'].data
        self.action.args.ccddata.flags = cleanhdu['FLAGS'].data
        if self.config.instrument.zap_append_sky:
            self.action.args.ccddata.zapskymodel = cleanhdu['ZAPSKYMODEL'].data
            if getattr(self.action.args.ccddata, "prezap", None) is None: #make sure not to overwrite the original input cube
                self.action.args.ccddata.prezap = cleanhdu['PREZAP'].data
        #attrname = getattr(self.action.args.ccddata, "UNCERT", None)
        #print('Attribute Name FLAG: {}'.format(attrname))
        #print(cleanhdu.info())
        kcwi_fits_writer(self.action.args.ccddata,
            table=self.action.args.table,
            output_file=self.action.args.name,
            output_dir=self.config.instrument.output_directory,
            suffix="icube")

        #Update proc table
        #Show that the file has been processed with ZAP
        self.context.proctab.update_proctab(frame=self.action.args.ccddata,
                                    suffix='icube',
                                    newtype="ZSKY",
                                    filename=self.action.args.name)
        self.context.proctab.write_proctab(tfil=self.config.instrument.procfile)
        #General update that this file has been processed
        self.context.proctab.update_proctab(frame=self.action.args.ccddata,
                            suffix='icube',
                            newtype="OBJECT",
                            filename=self.action.args.name)
        self.context.proctab.write_proctab(tfil=self.config.instrument.procfile)

        #Update logger info 
        log_string = MakeMasterSky3D.__module__
        self.logger.info(log_string)

    # END: class MakeMasterSky3D()
