from PIL import Image
import os
import numpy as np
import torchvision.transforms as TR
import torchvision.transforms.functional as TF
from torchvision.utils import save_image
from tabulate import tabulate
import code

from data_preparation import readNumTxt
import shutil
import random

txt = '/home/jiyeonb/number_of_cropped_images.txt'
img = '/home/jiyeonb/MP_dataset'
Info = readNumTxt(txt, img)

# Divide groups of images & check the total number of images.
with open('/home/jiyeonb/finallog.txt') as l:
	total = 0
	grps = []
	grp = []
	line = l.readline()
	while True:
		line = l.readline()
		if '>' in line:
			grps.append(grp)
			grp = []
			continue
		elif not line:
			break
		line = line.strip()
		grp.append(line)
		total += sum(Info[line][2])
	assert total == 361482, f'{total} does not match expected total 361482.'
	grps.append(grp)
print('\n', grps)
print('Total number of sets: ', len(grps), '\n')


# Iterate over the cropped dataset, get all the file names. 
MPpath = "/home/jiyeonb/MP_data_preparation/MPimg/"
nonMPpath = "/home/jiyeonb/MP_data_preparation/nonMPimg/"
savePath = "/home/jiyeonb/MP_dataset/"
if not os.path.exists(savePath):
	os.mkdir(pathname) 

# Find file names only from rawimg directory because corresponding masked image shares the same name. 
MPlist = [img for img in os.listdir(os.path.join(MPpath, "rawimg"))]
nonMPlist = [img for img in os.listdir(os.path.join(nonMPpath, "rawimg"))]

MPlist, nonMPlist = sorted(MPlist), sorted(nonMPlist)
print(len(MPlist), len(nonMPlist))
# print(MPlist)
# print('---------------')
# print(nonMPlist)


# Split images into 5 sets. 
MPsets, nonMPsets = [], []
for _ in range(5):
	MPsets.append([])
	nonMPsets.append([])

for img in MPlist:
	for grpnum in range(5):
		if img.split('_')[0] in grps[grpnum]:
			MPsets[grpnum].append(img)

for img in nonMPlist:
	for grpnum in range(5):
		if img.split('_')[0] in grps[grpnum]:
			nonMPsets[grpnum].append(img)


# print('----------------------------')
# print(MPsets)
# print(nonMPsets)


# Split images into five directories, named dataset_n.
# Each directories has two subdirectories "rawimg" and "labels"
checknum = [70870,70930,70997,72501,76184]
for cnt in range(5):

	# Check the number of splitted images.
	totimg = len(MPsets[cnt]) + len(nonMPsets[cnt])
	print(len(MPsets[cnt]), len(nonMPsets[cnt]))
	assert totimg == checknum[cnt], f'{totimg} does not match expected number of images {checknum[cnt]}'

	MPset = random.shuffle(MPsets[cnt])
	nonMPset = random.shuffle(nonMPset[cnt])
	for idx in range(70500):

		rawpath = os.path.join(savePath, f"dataset_{cnt}", "rawimg")
		labpath = os.path.join(savePath, f"dataset_{cnt}", "labels")
		for pathname in [rawpath, labpath]:
			if not os.path.exists(pathname):
				os.mkdir(pathname)

		MPname = MPset[idx]
	 	shutil.copyfile(os.path.join(MPpath, "rawimg", MPname), os.path.join(rawpath, MPname))
	 	shutil.copyfile(os.path.join(MPpath, "labels", MPname), os.path.join(labpath, MPname))

	 	








# # ERROR !!!!!
# labelpath = os.path.join(MPpath, "labels")
# errMP = "/home/jiyeonb/train_unet/error_sample/"
# # random.shuffle(MPlist)
# for cnt in range(10):
# 	img_raw = MPlist[cnt]
# 	shutil.copy(img_raw, os.path.join(errMP, "rawimg"))
# 	img_lab = os.path.join(labelpath, img.split('/')[-1])
# 	shutil.copy(img_lab, os.path.join(errMP, "labels"))


