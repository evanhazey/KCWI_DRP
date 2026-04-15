from astropy.io import fits as pf
from regions import Regions

import numpy as np
import sys
import os


def main():
    """Creates mask image from ds9 region file.

    To use this routine, process your data with default sky subtraction. Then
    display the target \*_icube.fits file in ds9. Use region shapes to indicate
    non-sky pixels in image (box, circle, etc.). Write out ds9 region file
    (\*.reg). Ensure the region file is in physical coordinates Then run this routine:

        * ``python ~/kderp/devel/kcwi_maskskycube_ds9.py kb180101_00111*_wlimg.fits ds9.reg``

    (replace paths/filenames with your local paths/filenames)

    This should create kb180101_00111_zapsmsk.fits, which will be used when you
    re-run the pipeline.

    Args:
        imagename (string): The name of a \*_wlimg.fits image
        regionname (string): The name of a ds9 region file

    Returns:
        None

    """
    # check args
    narg = len(sys.argv)

    # should be three (including routine name)
    if narg != 3:
        print("Usage: kcwi_maskskycube_ds9 <imagename> <regionname>")
        print("imagename : used for array dimensions and filename purposes, ")
        print("            must be an _wlimg image.")
        print("regionname: name of region file containing ds9 mask regions")
        print("            (typically a .reg)")
        exit()

    # read arg values
    imfname = sys.argv[1]
    regfname = sys.argv[2]

    # make sure it's an _wlimg image
    if 'wlimg.fits' not in imfname:
        print("image name must have *_wlimg.fits in name")
        exit()

    # do inputs exist?
    if not os.path.exists(imfname):
        print("Image file does not exist: "+imfname)
        exit()

    if not os.path.exists(regfname):
        print("Region file does not exist: "+regfname)
        exit()

    # create output mask image name
    outfile = imfname.replace("_wlimg.fits", "_zapsmsk.fits")
    print("Creating: "+outfile)

    # load the header from the pointed-to image.
    hdu_list = pf.open(imfname)
    header = hdu_list[0].header
    data = hdu_list[0].data

    # load the region file
    with open(regfname, 'r') as f:
        # Read it out as a string
        regstr = f.read()
        
        # Check if the region file is in physical coordinates
        if 'physical' in regstr:
            regstr = regstr.replace('physical', 'image')
        r = Regions.parse(regstr, format='ds9')
        region_mask = None
        for region in r.regions:
            if region_mask is None:
                region_mask = region.to_mask().to_image(data.shape).astype(bool)
            else:
                region_mask = region_mask | region.to_mask().to_image(data.shape).astype(bool)

    # write out the mask
    hdu = pf.PrimaryHDU(np.uint8(region_mask))
    hdu.writeto(outfile, overwrite=True)

    print("Done.")


if __name__ == "__main__":
    main()
