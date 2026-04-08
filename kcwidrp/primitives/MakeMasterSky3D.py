from keckdrpframework.primitives.base_img import BaseImg
from kcwidrp.primitives.kcwi_file_primitives import kcwi_fits_reader, \
    kcwi_fits_writer, strip_fname
from kcwidrp.primitives.GetAtlasLines import gaus
from kcwidrp.core.kcwi_get_std import kcwi_get_std
from kcwidrp.core.bokeh_plotting import bokeh_plot
from kcwidrp.core.kcwi_plotting import save_plot
from kcwidrp.core.bspline import Bspline
from bokeh.plotting import figure
from kcwidrp.core.kcwi_pkg_resources import get_resource_path
import yaml

import os
import time
import numpy as np
from scipy.optimize import curve_fit
from astropy.io import fits

import zap 
from zap.zap import SKYSEG
from astropy.stats import sigma_clip
from scipy.interpolate import interp1d
from regions import Regions
import pkg_resources
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

    1. Skip sky subtraction for this particular object image:

        * kr230925_00075.fits skip

    2. Point to a different image for the sky (this assumes the \*_sky.fits
    image has already been generated previously:

        * kr230925_00075.fits zap kr230925_00076.fits

    3. Indicate that a mask file should be used to mask object flux when
    deriving the sky model (see kcwi_maskskyzap_ds9.py):

        * kr230925_00075.fits zap kr230925_00075.fits kr230925_00075_smskzap.fits

    4. Indicate that this is a bright continuum source and automatically mask
    the continuum source from the sky model.

        * kr230925_00075.fits contzap

    5. Indicate that this is a faint continuum source and specify the location
    of the continuum source (in pixels). Supply the vertices for a rectangle 
    with the lower left coorindates x1,y1 followed by the upper right x2,y2, 
    leave no spaces between the coordnates and separate the x and y with a comma: 
             --- .x2,y2
            |    | 
            |    |
      x1,y1 .---- 
    
        * kr230925_00075.fits contzap 22,22 66,66

    If no `kcwi.sky` file exists, or there is no entry for the input object
    frame, then the entire image is used to generate the sky model.

    It is good practice to run all the data through first, then inspect the
    sky subtraction and see which frames will benefit from masking or from a
    dedicated sky observation.
    
    If a sky model is generated, the routine will write out a \*_sky.fits image
    and add a sky entry in the proc table. A toggle can be made in the config file
    to add the sky model as an extension to the \*_icube.fits file instead of 
    writing out a separate \*_sky.fits file.

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
                self.action.args.ccddata.header['ZAPAUTOMASK'] = True

            #Does the user want faint continuum masking for ZAP?
            elif skyyaml[ofn].get('zap_use_faint_cont', None) == True:
                self.logger.info("Faint continuum masking requested")
                #Did the user supply the positions?
                if ((skyyaml[ofn].get('zap_faint_cont_x1y1', None) is not None) and (skyyaml[ofn].get('zap_faint_cont_x1y1', None) != 'None')) and ((skyyaml[ofn].get('zap_faint_cont_x2y2', None) is not None) and (skyyaml[ofn].get('zap_faint_cont_x2y2', None) != 'None')):
                    self.logger.info("ZAP Using faint continuum source for %s" % ofn)
                    self.action.args.zap_use_faint_cont = True
                    self.action.args.zap_faint_cont_x1y1 = tuple((skyyaml[ofn]['zap_faint_cont_x1y1'].split(',')[0], skyyaml[ofn]['zap_faint_cont_x1y1'].split(',')[1]))
                    self.action.args.zap_faint_cont_x2y2 = tuple((skyyaml[ofn]['zap_faint_cont_x2y2'].split(',')[0], skyyaml[ofn]['zap_faint_cont_x2y2'].split(',')[1]))
                    self.action.args.ccddata.header['ZAPCONTMASK'] = True
                    self.action.args.ccddata.header['ZAPCONTX1Y1'] = self.action.args.zap_faint_cont_x1y1
                    self.action.args.ccddata.header['ZAPCONTX2Y2'] = self.action.args.zap_faint_cont_x2y2
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
                    self.action.args.ccddata.header['ZAPSKYMASK'] = zap_skymask
                else:
                    self.logger.warning("ZAP sky mask supplied but not found: %s. Proceeding with no ZAP sky mask." % zap_skymask)
            
            #Do we have an off sky frame?
            elif (skyyaml[ofn].get('offsky', None) is not None) and (skyyaml[ofn].get('offsky', None) != 'None'):
                offsky = os.path.join(reduxdir, skyyaml[ofn]['offsky']+'_icube.fits')
                self.logger.info("Offsky frame requested")
                if os.path.exists(offsky):
                    self.logger.info("Using off-sky frame: %s" % offsky)
                    self.action.args.offsky = offsky
                    self.action.args.ccddata.header['ZAPOFFSKY'] = offsky
                    #Does the sky frame have a ZAP sky mask file?
                    if (skyyaml[ofn].get('zap_offsky_mask', None) is not None) and (skyyaml[ofn].get('zap_offsky_mask', None) != 'None'):
                        zap_offsky_mask = os.path.join(reduxdir, skyyaml[ofn]['zap_offsky_mask'])
                        if os.path.exists(zap_offsky_mask):
                            self.logger.info("Using ZAP off-sky mask file: %s" % zap_offsky_mask)
                            self.action.args.zap_offsky_mask = zap_offsky_mask
                            self.action.args.ccddata.header['ZAPOFFSKYMASK'] = zap_offsky_mask
                        else:
                            self.logger.warning("ZAP off-sky mask supplied but not found: %s. Proceeding with no ZAP off-sky mask." % zap_offsky_mask)
                            self.action.args.zap_offsky_mask = None
                else:
                    self.logger.warning("Off-sky frame supplied but not found: %s. Proceeding with no off-sky frame." % offsky)
                    self.action.args.offsky = None
                    self.action.args.zap_offsky_mask = None
            
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
                                "skipping MakeMasterSky")
            return False

        #Check if user wants to run ZAP
        if (self.config.instrument.skysubmethod != '3D-PCA') and (self.config.instrument.skysubmethod != '2D-bspline+3D-PCA'):
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
        #Check if there is a YAML 
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


        ### LOAD *icube.fits FILE, CROP IT SPECTRALLY, REPLACE NANS, SAVE AS *icube_cropped.fits ###
        ofn_full = self.action.args.name
        rdir = self.config.instrument.output_directory
        fn = os.path.join(rdir, strip_fname(ofn_full) + '_icube.fits')
        scihdu_notrim = fits.open(fn) #should have already passed the pre_condition check above to be a *icube.fits file
        scihdu = crop_cube(scihdu_notrim) #Reduce zaxis to WAVGOOD0/1
        scihdr = scihdu[0].header
        obswave = (np.arange(scihdr['NAXIS3']) + 1 - scihdr['CRPIX3']) * scihdr['CD3_3'] + scihdr['CRVAL3']

        #replace the edge pixels with NaNs
        badpix = np.where(np.mean(scihdu['FLAGS'].data, axis = 0) > 100)
        scihdu[0].data[:, badpix[0], badpix[1]] = np.nan
        scihdu['UNCERT'].data[:, badpix[0], badpix[1]] = np.nan
        maskflags = np.mean(scihdu['FLAGS'].data, axis = 0)
        hdr2d = collapse_header(scihdu[0].header)
        mhdu = fits.ImageHDU(maskflags, header = hdr2d)

        # MAKE WHITELIGHT IMAGE #
        if self.config.instrument.zap_interactive == True:
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
            hdulist.writeto(os.path.join(rdir, strip_fname(ofn_full) + '_wlimg.fits'), overwrite = True)
        

        ### HANDLE SKY MASKS AND OFF-SKY FRAMES ##
        zap_using_offsky = False
        #Does the user have a skyfile to specify sky subtraction parameters?
        if self.action.args.skyyaml is not None:
            #Is there a science sky mask available?
            #if self.action.args.zap_skymask is not None:
            #    self.action.args.zap_skymask = os.path.join(rdir, self.action.args.zap_skymask)
            
            #Is automated continuum masking being requested?
            if self.action.args.zap_use_auto_cont:
                #Is automated continuum masking being requested?
                self.logger.info("Finding bright continuum source automatically")
                yplt = self.action.args.ccddata.data.flat
                sig = float(np.nanstd(yplt))
                con_sl_sig_max = sig
                con_sl_max = si
                con_sl_max_flx_data = yplt.copy()
                ipk = np.argmax(con_sl_max_flx_data)
                fpk = con_sl_max_flx_data[ipk]

                # gaussian fit to max slice
                res, _ = curve_fit(gaus, con_sl_max_pos_data,
                                con_sl_max_flx_data, p0=[fpk, ppk, 1.])
                self.logger.info("Continuum source max at %.2f in "
                                "slice %d with width %.2f px"
                                % (res[1], con_sl_max, res[2]))

                # First define source extent
                con_pos_mask_0 = res[1] - 7. * res[2]
                con_pos_mask_1 = res[1] + 7. * res[2]

                auto_cont_pos = res[1]
                auto_cont_width = 7. * res[2]

                # Next define lower and upper windows
                con_pos_mask_lo_0 = con_pos_mask_0 - \
                    14 / self.action.args.xbinsize
                con_pos_mask_up_1 = con_pos_mask_1 + \
                    14 / self.action.args.xbinsize

            #Are we masking a faint source and being given its box vertices positions?
            elif self.action.args.zap_use_faint_cont==True:
                x1, x2 = self.action.args.zap_faint_cont_x1y1[0], self.action.args.zap_faint_cont_x2y2[0]
                y1, y2 = self.action.args.zap_faint_cont_x1y1[1], self.action.args.zap_faint_cont_x2y2[1]
                skymaskzaparr = np.zeros_like(wlhdu[0].data.shape, dtype=int)
                skymaskzaparr[y2:y1, x2:x1] = 1
                skymaskzap = fits.PrimaryHDU(skymaskzaparr, header = hdr2d)
                skymaskzap.close()


            #Using an off field sky frame
            elif self.action.args.offsky is not None:
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

                #Is interactive mode set?
                if self.config.instrument.zap_interactive == True:
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
                    offskyhdulist.writeto(os.path.join(rdir, strip_fname(self.action.args.offsky) + '_wlimg.fits'), overwrite = True)

        ### ESTABLISH SKY SEGMENTS FOR ZAP ###
        skyseg0 = []
        if self.config.instrument.zap_skysegmentoption == 'single': #Single segment using "WAVEGOOD" bounds of the cube
            skyseg0 = [obswave[0], obswave[-1]]
            self.logger.info("# Using single sky segment. #")
        elif self.config.instrument.zap_skysegmentoption == 'Soto+2016': #Use the sky segment from the old version of MUSE. See Table 1 in Soto+16 for details
            skyseg0 = [0, 5400, 5850, 6440, 6750, 7200, 7700, 8265, 8602, 8731, 9275, 10000] 
            self.logger.info("# Using sky segments defined by Soto+2016. #")
        elif self.config.instrument.zap_skysegmentoption == 'custom': #User defined sky segments
            skyseg0 = self.config.instrument.zap_customskysegments  
            self.logger.info("# Using custom sky segments. #")
        else: #Unknown option given
            skyseg0 = [obswave[0], obswave[-1]] #uses the "WAVEGOOD" bounds of the cube 
            self.logger.info("# Unknown option given for skysegment. Using a single sky segment. #")
        zap_skyseg = trim_skysegments(skyseg0, obswave) #remove sky segements that fall outside of the spectral region in case ZAP runs into problem
        zap_cfwidth = self.config.instrument.zap_cfwidth #cfwidth = 300, default setting of ZAP
        if zap_cfwidth is None:
            zap_cfwidth = 300
            self.config.instrument.zap_cfwidth = zap_cfwidth
        self.logger.info("# Using cfwidth= %s and sky segments at %s Angstroms #" % (zap_cfwidth, zap_skyseg))

        #Clunky but write out the file
        self.action.args.ccddata.data = scihdu[0].data
        self.action.args.ccddata.header = scihdu[0].header
        self.action.args.ccddata.uncertainty = scihdu['UNCERT'].data
        self.action.args.ccddata.mask = scihdu['MASK'].data
        self.action.args.ccddata.flags = scihdu['FLAGS'].data
        if self.action.args.ccddata.noskysub is not None:
            self.action.args.ccddata.noskysub = scihdu['NOSKYSUB'].data
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


        ### UPDATE SKY SEGMENT HEADERS ### 
        if self.config.instrument.offsky is not None:
            scihdu[0].header['ZAPSKYFRAME'] = strip_fname(self.config.instrument.offsky)
        scihdu[0].header['ZAPSEGMODE'] = self.config.instrument.zap_skysegmentoption
        scihdu[0].header['ZAPCWITH'] = zap_cfwidth
        scihdu[0].header['ZAPNSEG'] = len(SKYSEG)-1
        nskyseg = np.arange(len(SKYSEG))
        for k in range(len(SKYSEG)):
            segstr = 'ZAPSKYSEG{}'.format(k)
            scihdu[0].header[segstr] = SKYSEG[k]

        ### SAVE THE FINAL CUBE ###
        cleanhdu = scihdu.copy()
        cleanhdu[0].data = cleancube

        #If user asked to append the sky model as an extension
        if self.config.instrument.zap_append_sky == True:
            cleanhdu.append(scihdu[0])
            cleanhdu[-1].name = 'UNZAPPED'
            skyhdu = fits.ImageHDU(data=skycube, header=scihdu[0].header)
            skyhdu.name = 'ZAPSKYMODEL'
            cleanhdu.append(skyhdu)
        
        #If user asked to save the intermediate products for inspection, save the sky model as seperate file 
        if self.config.instrument.zap_interactive == True:
            #Save the sky model as a seperate FITS file for easy inspection
            skyhdu.writeto(os.path.join(rdir, strip_fname(ofn_full) + '_icube_zapsky.fits'), overwrite = True)
        
        #Write out the ZAPPED datacube and requested extensions
        ofn_full = self.action.args.name
        ofn = os.path.basename(ofn_full)

        #Finally update the current frame
        self.action.args.ccddata.data = cleanhdu[0].data
        self.action.args.ccddata.header = cleanhdu[0].header
        self.action.args.ccddata.uncertainty = cleanhdu['UNCERT'].data
        self.action.args.ccddata.mask = cleanhdu['MASK'].data
        self.action.args.ccddata.flags = cleanhdu['FLAGS'].data
        if self.action.args.ccddata.noskysub is not None:
            self.action.args.ccddata.noskysub = cleanhdu['NOSKYSUB'].data
        if self.config.instrument.zap_append_sky:
            self.action.args.ccddata.unzap = cleanhdu['UNZAPPED'].data
            self.action.args.ccddata.zapsky = cleanhdu['ZAPSKYMODEL'].data
        #attrname = getattr(self.action.args.ccddata, "UNCERT", None)
        #print('Attribute Name FLAG: {}'.format(attrname))
        #print(cleanhdu.info())
        kcwi_fits_writer(self.action.args.ccddata,
            table=self.action.args.table,
            output_file=self.action.args.name,
            output_dir=self.config.instrument.output_directory,
            suffix="icube")
        
        #Update logger info 
        log_string = MakeMasterSky3D.__module__
        self.logger.info(log_string)

        return self.action.args

    # END: class MakeMasterSky3D()
