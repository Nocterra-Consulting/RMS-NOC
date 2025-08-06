""" Module which provides a common functional interface for loading video frames/images from different
    input data formats. """

from __future__ import print_function, division, absolute_import

import os
import sys
import copy
import datetime


# Rawpy for DFN images
try:
    import rawpy
except ImportError:
    pass


import cv2
import numpy as np
from astropy.io import fits

from RMS.Astrometry.Conversions import unixTime2Date, datetime2UnixTime
from RMS.Formats.FFfile import read as readFF
from RMS.Formats.FFfile import reconstructFrame as reconstructFrameFF
from RMS.Formats.FFfile import validFFName, filenameToDatetime
from RMS.Formats.FFfile import getMiddleTimeFF, selectFFFrames
from RMS.Formats.FRbin import read as readFR, validFRName
from RMS.Formats.Vid import readFrame as readVidFrame
from RMS.Formats.Vid import VidStruct
from RMS.GeoidHeightEGM96 import wgs84toMSLHeight
from RMS.Routines import Image
from RMS.Routines.GstreamerCapture import GstVideoFile

from RMS.RAW2FITS import fitsConversion

# If there is not display, messagebox will simply print to the console
if os.environ.get('DISPLAY') is None:
    messagebox = lambda title, message: print(title + ': ' + message)

else:

    # Try importing a Qt message box if available
    try:
        from RMS.Routines.CustomPyqtgraphClasses import qmessagebox as messagebox
    except:

        # Otherwise import a tk message box
        # tkinter import that works on both Python 2 and 3
        try:
            from tkinter import messagebox
        except:
            import tkMessageBox as messagebox


GST_IMPORTED = False
try:
    import gi
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst
    GST_IMPORTED = True
except:
    pass

# Import cython functions
import pyximport
pyximport.install(setup_args={'include_dirs': [np.get_include()]})
from RMS.Routines.DynamicFTPCompressionCy import FFMimickInterface


# ConstantsO
UWO_MAGICK_CAMO = 1144018537
UWO_MAGICK_EMCCD = 1141003881
UWO_MAGICK_ASGARD = 38037846


def getCacheID(first_frame, size):
    """ Get the frame chunk ID. """

    return "first:{:d},size:{:d}".format(int(first_frame), int(size))


def computeFramesToRead(read_nframes, total_frames, chunk_frames, first_frame):
    ### Compute the number of frames to read

    if read_nframes == -1:
        frames_to_read = total_frames

    else:

        # If the number of frames to read was not given, use the default value
        if read_nframes is None:
            frames_to_read = chunk_frames

        else:
            frames_to_read = read_nframes

        # Make sure not to try to read more frames than there's available
        if first_frame + frames_to_read > total_frames:
            frames_to_read = total_frames - first_frame

    return int(frames_to_read)


class InputType(object):
    def __init__(self):
        """ Template class for all input types. """

        self.current_frame = 0
        self.total_frames = 1

        # Only used for image mode
        self.single_image_mode = False

    def nextChunk(self):
        pass

    def prevChunk(self):
        pass

    def loadChunk(self, first_frame=None, read_nframes=None):
        pass

    def name(self, beginning=False):
        pass

    def currentTime(self, dt_obj=False):
        pass

    def nextFrame(self):
        """ Increment the current frame. """

        self.current_frame = (self.current_frame + 1)%self.total_frames

    def prevFrame(self):
        """ Decrement the current frame. """

        self.current_frame = (self.current_frame - 1)%self.total_frames

    def setFrame(self, fr_num):
        """ Set the current frame.

        Arguments:
            fr_num: [float] Frame number to set.
        """

        self.current_frame = fr_num%self.total_frames

    def loadFrame(self, avepixel=False):
        pass

    def getSequenceNumber(self):
        """ Returns the frame sequence number for the current frame.

        Return:
            [int] Frame sequence number.
        """

        return self.current_frame

    def currentFrameTime(self, frame_no=None, dt_obj=False):
        pass


class InputTypeRaw(InputType):
    def __init__(self, file_path, config, beginning_time=None, fps=None):
        """ Input file type handle for a single raw image.

        Arguments:
            dir_path: [str] Path to the vid file.
            config: [ConfigStruct object]

        Keyword arguments:
            beginning_time: [datetime] datetime of the beginning of the video. Optional, None by default.
            fps: [float] Known FPS of the images. None by default, in which case it will be read from the
                config file.
            detection: [bool] Indicates that the input is used for detection. False by default. This will
                control whether the binning is applied or not.

        """
        self.input_type = 'images'

        self.dir_path, self.image_file = os.path.split(file_path)
        self.config = config

        self.byteswap = False

        # Set the frames to a global shutter, so no correction is applied
        self.config.deinterlace_order = -2

        # This type of input probably won't have any calstars files
        self.require_calstars = False

        if 'rawpy' in sys.modules:
            ### Find images in the given folder ###
            img_types = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.fits', '.nef', '.cr2', '.cr3', '.dng']
        else:
            img_types = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.fits']

        self.beginning_datetime = beginning_time

        # Check if the file ends with support file extensions
        if self.beginning_datetime is None and \
                any([self.image_file.lower().endswith(fextens) for fextens in img_types]):
            try:
                
                # Extract the DFN timestamp from the file name
                image_filename_split = self.image_file.split("_")
                date_str = image_filename_split[1]
                time_str = image_filename_split[2]
                datetime_str = date_str + "_" + time_str
                
                beginning_datetime = datetime.datetime.strptime(
                    datetime_str,
                    "%Y-%m-%d_%H%M%S")

                self.beginning_datetime = beginning_datetime

            except:
                messagebox(title='Input error', \
                    message="Can't parse given DFN file name!")
                sys.exit()

        print('Using folder:', self.dir_path)


        # DFN frames start at 100 to accommodate picking previous frames, and 1024 picks total are allowed
        self.current_frame = 1
        self.total_frames = 1

        self.ff = None

        # Load the first image
        img = self.loadImage()

        # Get the image size (the binning correction doesn't have to be applied because the image is already
        #   binned)
        self.nrows = img.shape[0]
        self.ncols = img.shape[1]
        self.img_dtype = img.dtype

        if self.nrows > self.ncols:
            temp = self.nrows
            self.nrows = self.ncols
            self.ncols = temp
            img = np.rot90(img)

        self.ff = FFMimickInterface(self.nrows, self.ncols, self.img_dtype)
        self.ff.addFrame(img.astype(np.uint16))
        self.ff.finish()

        # If FPS is not given, use one from the config file
        if fps is None:
            self.fps = self.config.fps
            print('Using FPS from config file: ', self.fps)

        else:
            self.fps = fps
            print('Using FPS:', self.fps)


    def loadImage(self):
        
        ifile = os.path.join(self.dir_path, self.image_file)
            
        raw_exts = ['nef', 'cr2', 'cr3', 'dng']
        fits_exts = ['fits', 'fit']
        if self.image_file.lower().split('.')[-1] in raw_exts:
            raw_hdu = fitsConversion.raw2fits(ifile)
            print('loaded ' + self.image_file + ' using fitsConversion')
        elif self.image_file.lower().split('.')[-1] in fits_exts:
            raw_hdu = fits.open(f)[0]
            print('loaded ' + self.image_file + ' as raw FITS array')
        else:
            raise IndexError('supported file types are only: ' + raw_exts + fits_exts)
        
        debayered_hdu = fitsConversion.de_Bayer(raw_hdu, color_code='green2x2')
        print('De-Bayered the array by binning green pixels together')
        
        frame = debayered_hdu.data
        
        print('loaded image size: ' + str(np.shape(frame)))
            
        return frame

    def loadChunk(self, first_frame=None, read_nframes=None):
        """ Load the frame chunk file.

        Keyword arguments:
            first_frame: [int] First frame to read.
            read_nframes: [int] Number of frames to read. If not given (None), self.chunk_frames frames will be
                read. If -1, all frames will be read in.
        """
        return self.ff

    def name(self, beginning=False):
        return self.image_file

    def currentTime(self, dt_obj=False, beginning=False):
        """ Return the mean time of the current image.

        Keyword arguments:
            dt_obj: [bool] If True, a datetime object will be returned instead of a tuple.
            beginning: [bool] If True, the beginning time of the file will be returned instead of the middle
                time of the chunk.
        """

        if beginning:
            delta_t = 0
        else:
            delta_t = datetime.timedelta(seconds=self.total_frames/self.fps/2)

        # Compute the datetime of the current frame
        dt = self.beginning_datetime + delta_t

        if dt_obj:
            return dt

        else:
            return dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, dt.microsecond/1000

    def currentFrameTime(self, frame_no=None, dt_obj=False):
        if frame_no is None:
            frame_no = self.current_frame

        # Compute the datetime of the current frame
        dt = self.beginning_datetime + datetime.timedelta(seconds=frame_no/self.fps)

        if dt_obj:
            return dt

        else:
            return dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, dt.microsecond/1000


def detectInputType(input_path, config, beginning_time=None, fps=None, skip_ff_dir=False, detection=False,
    use_fr_files=False, preload_video=False, flipud=False, chunk_frames=None):
    """ Given the folder of a file, detect the input format.

    Arguments:
        input_path: [str] Input directory path (e.g. dir with FF files or path to a video file).
        config: [Config Struct]

    Keyword arguments:
        beginning_time: [datetime] Datetime of the video beginning. Optional, only can be given for
            video input formats.
        fps: [float] Frames per second, used only when images in a folder are used. If it's not given,
            it will be read from the config file.
        skip_ff_dir: [bool] Skip the input type where there are multiple FFs in the same directory. False
            by default. This is only used for ManualReduction.
        detection: [bool] Indicates that the input is used for detection. False by default. This will
                control whether the binning is applied or not. No effect on FF image handle.
        use_fr_files: [bool] Include FR files together with FF files. False by default, only used in SkyFit.
        preload_video: [bool] Preload the video file. False by default. This is only used for video files.
            Uses a lot of memory, so only use for small videos.
        flipud: [bool] Flip the image vertically. False by default.
        chunk_frames: [int] Number of frames to read in a chunk. None by default, in which case the defaults
            will be used specified for each input type.

    """
    

    if os.path.isdir(input_path):

        # Detect input type if a directory is given
        img_handle = detectInputTypeFolder(input_path, config, beginning_time=beginning_time, fps=fps,
            skip_ff_dir=skip_ff_dir, detection=detection, use_fr_files=use_fr_files, flipud=flipud,
            chunk_frames=chunk_frames)
        
    else:
        # Detect input type if a path to a file is given
        img_handle = detectInputTypeFile(input_path, config, beginning_time=beginning_time, fps=fps,
            detection=detection, preload_video=preload_video, flipud=flipud,
            chunk_frames=chunk_frames)

    return img_handle


def detectInputTypeFolder(input_dir, config, beginning_time=None, fps=None, skip_ff_dir=False, \
    detection=False, use_fr_files=False, flipud=False, chunk_frames=None):
    """ Given the folder of a file, detect the input format.

    Arguments:
        input_path: [str] Input directory path (e.g. dir with FF files).
        config: [Config Struct]

    Keyword arguments:
        beginning_time: [datetime] Datetime of the video beginning. Optional, only can be given for
            when images in a directory as used.
        fps: [float] Frames per second, used only when images in a folder are used. If it's not given,
            it will be read from the config file.
        skip_ff_dir: [bool] Skip the input type where there are multiple FFs in the same directory. False
            by default. This is only used for ManualReduction.
        detection: [bool] Indicates that the input is used for detection. False by default. This will
                control whether the binning is applied or not. No effect on FF image handle.
        use_fr_files: [bool] Include FR files together with FF files. False by default, only used in SkyFit.
        flipud: [bool] Flip the image vertically. False by default.
        chunk_frames: [int] Number of frames to read in a chunk. None by default, in which case the defaults
            will be used specified for each input type.

    """

    ### Find images in the given folder ###
    img_types = ['.png', '.jpg', '.jpeg', '.bmp', '.fit', '.tif', '.fits']

    if 'rawpy' in sys.modules:
        img_types += ['.nef', '.cr2','.cr3', '.dng']
        

    img_handle = None
    
    # If the given dir path is a directory, search for FF files or individual images
    if not os.path.isdir(input_dir):
        return None


    # Check if there are valid FF names in the directory
    if any([validFFName(ff_file) or validFRName(ff_file) for ff_file in os.listdir(input_dir)]):


        # If FR files are not used, only check for FF files
        if not use_fr_files:
            if not any([validFFName(ff_file) for ff_file in os.listdir(input_dir)]):
                print("No FF files found in directory!")
                return None


        if skip_ff_dir:
            messagebox(title='FF directory',
            message='ManualReduction only works on individual FF files, and not directories with FF files!')
            return None

        else:
            # Init the image handle for FF files in a directory
            img_handle = InputTypeFRFF(input_dir, config, use_fr_files=use_fr_files)
            img_handle.ncols = config.width
            img_handle.nrows = config.height

    elif any([any(file.lower().endswith(x) for x in img_types) for file in os.listdir(input_dir)]) and \
            config.width != 4912 and config.width != 7360:
        img_handle = InputTypeImages(input_dir, config, beginning_time=beginning_time, fps=fps,
                                     detection=detection, flipud=flipud, chunk_frames=chunk_frames)

    return img_handle


def checkIfVideoFile(file_name):
    """ Check if the given file is a supported video file format. 

    Arguments:
        file_name: [str] File name to check.

    Return:
        [bool] True if the file is a video file, False otherwise.

    """

    if file_name.lower().endswith('.mp4') or file_name.lower().endswith('.avi') \
        or file_name.lower().endswith('.mkv') or file_name.lower().endswith('.wmv') \
        or file_name.lower().endswith('.mov'):

        return True
    
    return False


def detectInputTypeFile(input_file, config, beginning_time=None, fps=None, detection=False, 
                        preload_video=False, flipud=False, chunk_frames=None):
    """ Given a file, detect the input format.

    Arguments:
        input_path: [str] Input file path (e.g. path to a video file).
        config: [Config Struct]

    Keyword arguments:
        beginning_time: [datetime] Datetime of the video beginning. Optional, only can be given for
            video input formats.
        fps: [float] Frames per second, used only for a DFN image. If it's not given, it will be read from 
            the config file.
        detection: [bool] Indicates that the input is used for detection. False by default. This will
                control whether the binning is applied or not. No effect on FF image handle.
        preload_video: [bool] Preload the video file. False by default. This is only used for video files.
        flipud: [bool] Flip the image vertically. False by default.
        chunk_frames: [int] Number of frames to read in a chunk. None by default, in which case the defaults
            will be used specified for each input type.

    """

    # If the given path is a file, look for a single FF file, video files, or vid files
    dir_path, file_name = os.path.split(input_file)
    
    img_handle = InputTypeRaw(input_file, config, beginning_time=beginning_time, fps=fps)

    return img_handle


if __name__ == "__main__":

    import argparse

    import matplotlib.pyplot as plt

    import RMS.ConfigReader as cr

    ### Functions for testing

    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="""Test.""", formatter_class=argparse.RawTextHelpFormatter)

    arg_parser.add_argument('dir_path', metavar='DIRPATH', type=str, nargs=1, \
                            help='Path to data.')

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################

    # Load the configuration file
    config = cr.parse(".config")

    # Test creating a fake FF
    nframes = 64
    img_h = 20
    img_w = 20

    ff = FFMimickInterface(img_h, img_w, np.uint16)

    frames = np.random.normal(10000, 500, size=(nframes, img_h, img_w)).astype(np.uint16)

    for frame in frames:
        ff.addFrame(frame.astype(np.uint16))

    ff.finish()

    # Compute real values
    avepixel = np.mean(frames, axis=0)
    stdpixel = np.std(frames, axis=0)

    print('Std mean ff:', np.mean(ff.stdpixel))
    print('Std mean:', np.mean(stdpixel))
    print('Mean diff:', np.mean(stdpixel - ff.stdpixel))
    plt.imshow(stdpixel - ff.stdpixel)
    plt.show()

    print('ave mean ff:', np.mean(ff.avepixel))
    print('ave mean:', np.mean(avepixel))
    print('Mean diff:', np.mean(avepixel - ff.avepixel))
    plt.imshow(avepixel - ff.avepixel)
    plt.show()

    # # Load the appropriate files
    # img_handle = detectInputType(cml_args.dir_path[0], config)

    # chunk_size = 64

    # for i in range(img_handle.total_frames//chunk_size + 1):

    #     first_frame = i*chunk_size

    #     # Load a chunk of frames
    #     ff = img_handle.loadChunk(first_frame=first_frame, read_nframes=chunk_size)

    #     print(first_frame, first_frame + chunk_size)
    #     plt.imshow(ff.maxpixel - ff.avepixel, cmap='gray')
    #     plt.show()

    #     # Show stdpixel
    #     plt.imshow(ff.stdpixel, cmap='gray')
    #     plt.show()

    #     # Show thresholded image
    #     thresh_img = (ff.maxpixel - ff.avepixel) > (1.0*ff.stdpixel + 30)
    #     plt.imshow(thresh_img, cmap='gray')
    #     plt.show()
