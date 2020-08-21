"""
Prepare randomly generated dataset for training.
functions: readNumTxt(), identicalRandCrop(), and mpRandCrop().

identicalRandCrop() is used within the function mpRandCrop(), 
thus both should be imported in cases when mpRandCrop() is used. 
"""



from PIL import Image
import os
import numpy as np
import torchvision.transforms as TR
import torchvision.transforms.functional as TF
from torchvision.utils import save_image
from tabulate import tabulate

def readNumTxt(txtPath, imgPath):
    """
    Read the textfile that each line is in form of 
    'filename of fluorescent image;number of cropped images containing MP;number of cropped images not containing MP'
    together with the image path.
    Get corresponding information as a dictionary.

    txtPath: path name of the textfile.
    imgPath: path name that contains both fluorescent and masked images
    """

    Info = dict()

    with open(txtPath, "r") as numtxt:
        lines = numtxt.readlines()
        for line in lines:
            filename, MP, nonMP = line.strip().split(';')
            MP, nonMP = int(MP), int(nonMP)
            Info[filename.split('_')[0]] = [filename, 0, (MP, nonMP)]

    masked_list = [img for img in os.listdir(imgPath) if os.path.isfile(os.path.join(imgPath, img)) and ('_mask' in img)]

    for m_img in masked_list:
        maskname = m_img.split('_')[0]
        assert maskname in Info, f'{maskname} not in Info'
        Info[maskname][1] = m_img

    return Info


def identicalRandCrop(img1, img2, size):
    """
    Apply identical random cropping on two images, return them as tensor object.
    img1, img2: given as PIL.Image object.
    size: size of the resulting cropped image.
    """
    if isinstance(size, int):
        size = (size, size)
    i, j, h, w = TR.RandomCrop.get_params(img1, output_size = size)
    img1, img2 = TF.crop(img1, i, j, h, w), TF.crop(img2, i, j, h, w)

    trans = TR.ToTensor()
    img1, img2 = trans(img1), trans(img2)

    return img1, img2


def mpRandCrop(imgPath, info, size, savePath=None, infoPath=None):
    """
    Generate randomly cropped images of given dataset. 
    According to the presence of MP, the cropped images are saved in two different folders. 

    imgPath: Location of the original images.
    info: A dictionary that has keys of individual number of images, items of list with file names of fluorescent and masked images 
          and the number of cropped images needed to be generated.
    size: Size of the resulting cropped image.
    ratio: The ratio of MP images:Non-MP images that should be given in form of tupel (e.g. (no. of MP images, no. of non-MP images))
    savePath: Location where cropped images will be saved.
              The function will automatically create directories in the given location.
    """
    
    # Create directories in savePath.
    MP_savePath = os.path.join(savePath, "MPimg")
    nonMP_savePath = os.path.join(savePath, "nonMPimg")
    MP_fl_savePath = os.path.join(MP_savePath, "rawimg")
    MP_m_savePath = os.path.join(MP_savePath, "labels")
    nonMP_fl_savePath = os.path.join(nonMP_savePath, "rawimg")
    nonMP_m_savePath = os.path.join(nonMP_savePath, "labels")
    for pathname in [MP_savePath, nonMP_savePath, MP_fl_savePath, MP_m_savePath, nonMP_fl_savePath, nonMP_m_savePath]:
        if not os.path.exists(pathname):
            os.mkdir(pathname)
    
    MP_all, nonMP_all = 0, 0

    # Generate cropped images.
    IDnum = info.keys()
    for ID in IDnum:
        totImg, MP, nonMP = 0, 0, 0
        fl_name, m_name, ratio = info[ID]
        fl_path, m_path = os.path.join(imgPath, fl_name), os.path.join(imgPath, m_name)
        
        max_MP, max_nonMP = ratio[0], ratio[1]
        max_tot = max_MP + max_nonMP

        assert os.path.isfile(fl_path), f'The fluorescent image {fl_name} does not exist.'
        assert os.path.isfile(m_path), f'The masked image {m_name} does not exist.'

        img_fl, img_m = Image.open(fl_path), Image.open(m_path).convert('RGB')
        
        # Use identical random cropping for raw and masked images, until they reach the pre-defined number of cropped images.
        while totImg < max_tot:
            print(f'{ID}th image - Cropping #: ', totImg)
            crop_fl, crop_m = identicalRandCrop(img_fl, img_m, size)
 
            presence = 0 in crop_m
            if presence and (MP < max_MP):
                totImg += 1
                MP += 1
                save_image(crop_fl, os.path.join(MP_fl_savePath, ID + "_{:05d}.png".format(totImg)))
                save_image(crop_m, os.path.join(MP_m_savePath, ID + "_{:05d}.png".format(totImg)))
            elif (not presence) and (nonMP < max_nonMP):
                totImg += 1
                nonMP += 1
                save_image(crop_fl, os.path.join(nonMP_fl_savePath, ID + "_{:05d}.png".format(totImg)))
                save_image(crop_m, os.path.join(nonMP_m_savePath, ID + "_{:05d}.png".format(totImg)))
            else:
                AssertionError(f"Something's wrong! \n Presence of MP is {presence}, no.MP = {MP}, no.nonMP = {nonMP}, no.tot = {totImg}")
        
        MP_all += MP
        nonMP_all += nonMP

        # Generate textfile in infoPath for double-check the number of cropped files.
        with open(infoPath, 'a') as infotxt:
            infotxt.write(fl_name + ";" + str(MP) + ";" + str(nonMP) + "\n")

    total_images = MP_all + nonMP_all

    return total_images, MP_all, nonMP_all


            

import time

if __name__ == '__main__':

    # testing readNumTxt.
    txt = '/home/jiyeonb/number_of_cropped_images.txt'
    img = '/home/jiyeonb/MP_dataset'
    Info = readNumTxt(txt, img)
    keys = Info.keys()
    Info_table = list()
    for key in keys:
        line = Info[key]
        Info_table.append((key, line[0], line[1], line[2]))
    with open('/home/jiyeonb/MP_data_preparation/Information_Dictionary.txt', 'w') as info_preparation:
        info_preparation.write(tabulate(Info_table, headers=['No.', 'fl_img', 'm_img', '(MP, NonMP)']))
    
    # testing mpRandCrop.
    information = '/home/jiyeonb/MP_data_preparation/doublecheck.txt'
    savepath = '/home/jiyeonb/MP_data_preparation'
    startpoint = time.time()
    total_imgs, MPall, nonMPall = mpRandCrop(imgPath=img, info=Info, size=256, savePath=savepath, infoPath=information)
    endpoint = time.time()

    print('\nData preparation completed!\n' + f'Total processed time:  {endpoint - startpoint}\n' )

    orig_img = len(Info)
    print('[Result]\n' + f'From {orig_img} pairs of MP images, {total_imgs} pairs of cropped images are randomly generated.\n')
    print(f'Among them, {MPall} pairs ({(MPall / total_imgs):.2f}%) are containing MP, \nand {nonMPall} pairs ({(nonMPall / total_imgs):.2f}%) are not containing MP.')
    
    with open('/home/jiyeonb/MP_data_preparation/final_result.txt', 'w') as result:
        result.write('Data preparation completed!\n' + f'Total processed time:\t{endpoint - startpoint} seconds. \n')
        result.write('[Result]\n' + f'From {orig_img} pairs of MP images, {total_imgs} pairs of cropped images are randomly generated.\n')
        result.write(f'Among them, {MPall} pairs ({(MPall / total_imgs):.2f}%) are containing MP, \nand {nonMPall} pairs ({(nonMPall / total_imgs):.2f}%) are not containing MP.')