#!/usr/bin/env python
#
# Python 2 and 3 compatible as long as astropy remains 2/3 compatible
#
# Version history:
# 

from __future__ import absolute_import, division, print_function


__author__ = "Hadrien A.R. Devillepoix"
__copyright__ = "Copyright 2015-2017, Desert Fireball Network"
__license__ = "MIT"
__version__ = "1.0"


import os
import re
import glob
import itertools
import subprocess
import warnings

from astropy.time import Time, TimeDelta


# default file names and extensions
rawExtensionDefault = "NEF"
fitsExtension = "fits"
tifExtension = "tiff"
jpgExtension = "jpeg"
stdcfgfilename = "dfnstation.cfg"

RAW_EXTENSIONS = {'NEF' : 'nikon',
                 'CR2' : 'canon',
                 'ARW' : 'sony'}


PROCESSING_FILTER_BAND = {'RED' : 'R',
                          'GREEN 1' : 'V',
                          'GREEN 2' : 'V',
                          'BLUE' : 'B',
                          'GREEN_INTERPOL' : 'V',
                          'RAW' : 'panchromatic',
                          'GREEN_2X2' : 'G',
                          'RGB_2X2' : 'panchromatic'}


    
    
def time_factory(itime):
    '''
    Determine input time format and return Time object
    raises TypeError if cannot figure out format
    '''
    
    # already a Time object: return input
    if isinstance(itime, Time):
        t = itime
    # assume JD or UNIX if number of integer digits > 7
    elif isinstance(itime, float) or (isinstance(itime, str) and re.match("^\d+?\.\d+?$", itime)):
        f_itime = float(itime)
        if len(str(int(f_itime))) > 7:
            t = Time(f_itime, format='unix', scale='utc')
        else:
            t = Time(f_itime, format='jd', scale='utc')
    # assume ISOT / autodetect
    elif isinstance(itime, str):
        t = Time(itime, scale='utc')
    else:
        raise TypeError(str(itime) + 'is NOT a valid TIME input')
    
    return t



def round_to_nearest_n_seconds(itime, n):
    '''
    Returns the closest time that is the top of the minute or half minute
    Paramters:
        itime: some sort of time
        n: rounding, integer [1,60], cannot be prime with 60
    Returns:
        Corrected time stamp (astropy Time object in UTC scale)
    '''
    
    if not isinstance(n, int) or (n*int(60.0/n) != 60):
        raise TypeError("Can only round timestamp to n: integer [1,60], cannot be prime with 60")
    
    t = time_factory(itime)
    
    # 2880 = 24 * 3600 / 30
    integer_day_factor = 24 * 3600 / n
    td = TimeDelta(t.jd2 * integer_day_factor - round(t.jd2 * integer_day_factor), format='sec')
    
    return t - td * n

def resolve_glob(extension, directory=".", prefix="", suffix=""):
    '''
    list files with certain pattern, bash style (directory/prefix*.extension)
    extension: simple extension (ex: "NEF"), or list of extensions
    '''
    
    # check if extension parameter is a simple extension or a list of extensions
    if isinstance(extension, str):
        reslist = sorted(glob.glob(os.path.join(directory, prefix + "*" + suffix + "*." + extension)))
    else:
        reslist = sorted(list(itertools.chain.from_iterable([glob.glob(os.path.join(directory, prefix + "*" + suffix + "*." + e)) for e in extension])))
    return reslist


def getDfnstationConfigFile(directory="."):
    possibleFiles = resolve_glob(extension="cfg", directory=directory, suffix="dfnstation")
    if len(possibleFiles) > 1:
        warnings.warn("Several camera config files found in the directory.")
    if os.path.join(directory, stdcfgfilename) in possibleFiles:
        return stdcfgfilename
    elif len(possibleFiles) >= 1:
        return possibleFiles[0]
    else:
        return ""



