#!/usr/bin/env python
# coding: utf-8

import dicom_contour.contour as dcm
import os
import glob
import pydicom as dicom
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as nd
from skimage.transform import resize

def get_roi_contour_ds(file, index):
    ROI = file.ROIContourSequence[index]
    contours = [contour for contour in ROI.ContourSequence]
    return contours

def contour2poly(one_slice, path):
    """
    Given a contour dataset (a DICOM class) and path that has .dcm files of
    corresponding images return polygon coordinates for the contours.

    Inputs
        one_slice (dicom.dataset.Dataset) : DICOM dataset class that is identified as
                         (3006, 0016)  Contour Image Sequence
        path (str): path of directory containing DICOM images

    Return:
        pixel_coords (list): list of tuples having pixel coordinates
        img_ID (id): DICOM image id which maps input contour dataset
        img_shape (tuple): DICOM image shape - height, width
    """

    contour_coord = one_slice.ContourData
    # x, y, z coordinates of the contour in mm
    coord = []
    for i in range(0, len(contour_coord), 3):
        coord.append((contour_coord[i], contour_coord[i + 1], contour_coord[i + 2]))

    # extract the image id corresponding to given countour
    # read that dicom file
    img_ID = one_slice.ContourImageSequence[0].ReferencedSOPInstanceUID
    img = dicom.read_file(os.path.join(path, 'CT.'+ img_ID + '.dcm'))
    img_arr = img.pixel_array
    img_shape = img_arr.shape

    # physical distance between the center of each pixel
    x_spacing, y_spacing = float(img.PixelSpacing[0]), float(img.PixelSpacing[1])

    # this is the center of the upper left voxel
    origin_x, origin_y, _ = img.ImagePositionPatient

    # y, x is how it's mapped
    pixel_coords = [(np.ceil((x - origin_x) / x_spacing), np.ceil((y - origin_y) / y_spacing)) for x, y, _ in coord]
    return pixel_coords, img_ID, img_shape


def poly_to_mask(polygon, width, height):
    from PIL import Image, ImageDraw

    """Convert polygon to mask
    :param polygon: list of pairs of x, y coords [(x1, y1), (x2, y2), ...]
     in units of pixels
    :param width: scalar image width
    :param height: scalar image height
    :return: Boolean mask of shape (height, width)
    """

    # http://stackoverflow.com/a/3732128/1410871
    img = Image.new(mode='L', size=(width, height), color=0)
    ImageDraw.Draw(img).polygon(xy=polygon, outline=0, fill=1)
    mask = np.array(img)
    return mask


def get_mask_dict(contour_datasets, path):
    """
    Inputs:
        contour_datasets (list): list of dicom.dataset.Dataset for contours
        path (str): path of directory with images

    Return:
        img_contours_dict (dict): img_id : contour array pairs
    """

    from collections import defaultdict

    # create empty dict for
    img_contours_dict = defaultdict(int)

    for cdataset in contour_datasets:
        coords, img_id, shape = contour2poly(cdataset, path)
        mask = poly_to_mask(coords, *shape)
        img_contours_dict[img_id] += mask

    return img_contours_dict

def parse_dicom_file(filename):
    """Parse the given DICOM filename
    :param filename: filepath to the DICOM file to parse
    :return: dictionary with DICOM image data
    """

    try:
        dcm = dicom.read_file(filename)
        dcm_image = dcm.pixel_array

        try:
            intercept = dcm.RescaleIntercept
        except AttributeError:
            intercept = 0.0
        try:
            slope = dcm.RescaleSlope
        except AttributeError:
            slope = 0.0

        if intercept != 0.0 and slope != 0.0:
            dcm_image = dcm_image*slope + intercept
        return dcm_image
    except InvalidDicomError:
        return None

def get_img_mask_voxel(slice_orders, mask_dict, path):
    """
    Construct image and mask voxels

    Inputs:
        slice_orders (list): list of tuples of ordered img_id and z-coordinate position
        mask_dict (dict): dictionary having img_id : contour array pairs
        dir (str): directory path containing DICOM image files
    Return:
        img_voxel: ordered image voxel for CT/MR
        mask_voxel: ordered mask voxel for CT/MR
    """

    img_voxel = []
    mask_voxel = []
    for img_id, _ in slice_orders:
        img_array = parse_dicom_file(os.path.join(path, 'CT.' + img_id + '.dcm'))
        if img_id in mask_dict:
            mask_array = mask_dict[img_id]
        else:
            mask_array = np.zeros_like(img_array)
        img_voxel.append(img_array)
        mask_voxel.append(mask_array)
    return img_voxel, mask_voxel

def draw_oneslice(img_arr, msk_arr, alpha=0.35, sz=7, cmap='inferno',
                           save_path=None):

    """
    Show original image and masked on top of image
    next to each other in desired size
    Inputs:
        img_arr (np.array): array of the image
        msk_arr (np.array): array of the mask
        alpha (float): a number between 0 and 1 for mask transparency
        sz (int): figure size for display
        save_path (str): path to save the figure
    """

    msk_arr = np.ma.masked_where(msk_arr == 0, msk_arr)
    plt.figure(figsize=(sz, sz))
    plt.subplot(1, 2, 1)
    plt.imshow(img_arr, cmap='gray')
    plt.imshow(msk_arr, cmap=cmap, alpha=alpha)
    plt.subplot(1, 2, 2)
    plt.imshow(img_arr, cmap='gray')
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
        plt.close()


def resize_image(input, newshape, isResize=False, isMask=False):
    oldshape=input.shape
    dsfactor = [w / float(f) for w, f in zip(newshape, oldshape)]
    if isResize:
        output = resize(input, newshape)
    else:
        output = nd.interpolation.zoom(input, zoom=dsfactor)
    if isMask:
        output[np.abs(output)<1e-3]=0
        output[np.abs(output)>1e-3]=1

    return output, dsfactor

def slices_to_3d(slices):
    output = np.array(slices)
    output = np.reshape(output, output.shape + (1,))
    return output


dir="/home/hyao/Research/RUP-Net/Data/Livers"
struct="liver"
newshape=(64,64,64,1)
savefile="Data/data.npz"


inputs=[]
labels=[]
for onedir, subdir, _ in os.walk(dir):
    if len(subdir) == 0 :
        rsfiles=os.path.join(onedir, "RS.*.dcm")
        rsfiles=glob.glob(rsfiles)
        for i in rsfiles:
            print("Processing {} ...".format(os.path.dirname(i)))
            dcmfile=dicom.read_file(i)
            roinames=dcm.get_roi_names(dcmfile)
            id = [id for id, name in enumerate(roinames) if re.match(struct,name, re.IGNORECASE) is not None]
            if len(id)>1:
                print("More than one {} found. Please choose one (only index):".format(struct))
                for j in id:
                    print("{}: {}".format(roinames[j],j))
                #id=input("Your choice is :")
                id=id[0]
            elif len(id)==0:
                print("No {} found in file {}.".format(struct, i))
                #DEBUG: skip to next file
            else:
                id=id[0]

            contour_ds=get_roi_contour_ds(dcmfile, id)
            mask_dict = get_mask_dict(contour_ds, onedir)
            slice_orders = dcm.slice_order(onedir)
            #0: inferior -1: last one superior
            x, y = get_img_mask_voxel(slice_orders, mask_dict, onedir)
            x = slices_to_3d(x)
            y = slices_to_3d(y)
            x, dsfactor = resize_image(x, newshape=newshape, isMask=False, isResize=True)
            y, dsfactor  = resize_image(y, newshape=newshape, isMask=True, isResize=True)
            inputs.append(x)
            labels.append(y)

x=np.array(inputs)
y=np.array(labels)
np.savez_compressed(savefile,x=x,y=y)
#l=np.load(savefile)
#x=l['x']
#y=l['y']
#x.shape
#plt.imshow(x[1,35,:,:,0], cmap='gray')
#plt.imshow(y[1,35,:,:,0], cmap='gray')
#plt.imshow(labels[1][35,:,:,0], cmap='gray')











import os
import argparse
import sys
import shutil
if os.path.basename(os.getcwd())=="NH-AutoSeg":
    from Preprocessing.dicomUtility import *
else:
    from dicomUtility import *

#import time


args = None

def main(args):
    if args.debug:
        FORMAT = "%(levelname)s:%(lineno)d:%(message)s"
        logging.basicConfig(level=logging.DEBUG, format=FORMAT)
    ############patient list
    plist=[]
    for root, dirs, files in os.walk(args.indir):
        if len(dirs)>0:
            plist=dirs
    if args.debug:
        logging.debug("Patient List plist is {}".format(plist))

    ##############patient directory list
    pdirlist=[]
    for i in plist:
        thispd=args.indir+i #this patient dir
        pdirlist.append(thispd)
        
    if args.debug:
        logging.debug("Patient directory List pdirlist is {}".format(pdirlist))
        
    newshape=(args.numofslices, args.pixelsize, args.pixelsize, args.colors)

    ###############create hdf5 file for neural network
    import h5py
    #output dataset
    trainlen=int(len(plist)*args.ratio)
    testlen=len(plist)-trainlen

    trainshape=(trainlen,)+newshape
    oTrainSet = h5py.File(args.trainfile, "w")
    oTrainSet.create_dataset('X', trainshape, dtype='f')
    oTrainSet.create_dataset('Y', trainshape[0:4], dtype='i')
    #dt=h5py.string_dtype()
    #oTrainSet.create_dataset('ID', (fileshape[0],), dtype=dt)
    #oTrainSet.create_dataset('FACTOR', (fileshape[0],3), dtype='f')
    testshape=(testlen,)+newshape
    oTestSet = h5py.File(args.testfile, "w")
    oTestSet.create_dataset('X', testshape, dtype='f')
    oTestSet.create_dataset('Y', testshape[0:4], dtype='i')


    src_move_dir=os.path.dirname(args.trainfile)+"/src_for_"+os.path.basename(args.trainfile).replace(".h5","")
    if not os.path.exists(src_move_dir):
        os.makedirs(src_move_dir)

    for i in range(trainlen):
        thispd=args.indir+plist[i] #this patient dir
        print(thispd)
        X, Y, image_dsfactor, mask_dsfactor=my_readOnePatient(pdir=thispd,stName="Liver", newshape=(args.numofslices, args.pixelsize, args.pixelsize, args.colors), isdebug=False)
        X = my_HUthreshold(X, args.HUmin, args.HUmax)
        X=X.astype('float32')
        #my_showImageGrid(X, everyn=5)
        #X=X.reshape(newshape)
        #Y=Y.reshape(newshape)
        oTrainSet['X'][i]=X
        oTrainSet['Y'][i]=Y
        #oTrainSet['ID'][i]=plist[i]
        #oTrainSet['FACTOR'][i]=np.asarray(mask_dsfactor)

        shutil.move(thispd, src_move_dir)
    oTrainSet.close()

    src_move_dir=os.path.dirname(args.trainfile)+"/src_for_"+os.path.basename(args.testfile).replace(".h5","")
    if not os.path.exists(src_move_dir):
        os.makedirs(src_move_dir)
    for i in range(trainlen,len(plist)):
        thispd=args.indir+plist[i] #this patient dir
        print(thispd)
        X, Y, image_dsfactor, mask_dsfactor=my_readOnePatient(pdir=thispd,stName="Liver", newshape=(args.numofslices, args.pixelsize, args.pixelsize, args.colors), isdebug=False)
        X = my_HUthreshold(X, args.HUmin, args.HUmax)
        X=X.astype('float32')
        #my_showImageGrid(X, everyn=5)
        #X=X.reshape(newshape)
        #Y=Y.reshape(newshape)
        oTestSet['X'][i-trainlen]=X
        oTestSet['Y'][i-trainlen]=Y

        shutil.move(thispd, src_move_dir)
    oTestSet.close()

    return 0




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Northwell Health AutoSeg",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-b', '--debug', action="store_true", help="output debug info")
    parser.add_argument('-t', '--trainfile', type=str, default="../Dataset/DSForNN/train.h5", help="output train dataset with hdf5 extension")
    parser.add_argument('-e', '--testfile', type=str, default="../Dataset/DSForNN/test.h5", help="output test dataset with hdf5 extension")
    parser.add_argument('-r', '--ratio', type=float, default=0.8, help="ratio of input files as training set. 1-ratio for test set")
    parser.add_argument('-p', '--pixelsize', type=int, default=64, help="resample original image to pixelsize X pixelsize to increase training speed")
    parser.add_argument('-n', '--numofslices', type=int, default=64, help="resample original image to number of slices to uniform the input and increase training speed")
    parser.add_argument('-c', '--colors', type=int, default=1, help="number of color channels, -1 means use original SamplesperPixel")
    parser.add_argument('-s', '--struct', type=str, default="Liver", help="main structure")
    parser.add_argument('-i', '--HUmin', type=int, default=20,
                        help="min HU for struct, default: 20 for liver")
    parser.add_argument('-x', '--HUmax', type=int, default=100,
                        help="max HU for struct, default: 100 for liver")
    parser.add_argument('indir', type=str, help="input dir, next level of folder will be patient ID folder name")
    if sys.__stdin__.isatty():
        args = parser.parse_args()
    else:
        args = parser.parse_args(['-b','../Dataset/LiverDICOM/'])

    if os.path.basename(os.getcwd()) == "NH-AutoSeg":
        args.indir=args.indir.replace("..",".")
        args.trainfile=args.trainfile.replace("..",".")
        args.testfile = args.testfile.replace("..", ".")
    print(args)
    main(args)
