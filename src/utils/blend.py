# Created by: Kaede Shiohara
# Yamasaki Lab at The University of Tokyo
# shiohara@cvm.t.u-tokyo.ac.jp
# Copyright (c) 2021
# 3rd party softwares' licenses are noticed at https://github.com/mapooon/SelfBlendedImages/blob/master/LICENSE

import cv2
import numpy as np
import scipy as sp
from skimage.measure import label, regionprops
import random
from PIL import Image
import sys
import os
import cv2

COUNTER = 0


def alpha_blend(source,target,mask):
	mask_blured = get_blend_mask(mask)
	img_blended=(mask_blured * source + (1 - mask_blured) * target)
	return img_blended,mask_blured

def poisson_blend_cv2(source, target, mask):

	src_bgr = cv2.cvtColor(source, cv2.COLOR_RGB2BGR)
	dst_bgr = cv2.cvtColor(target, cv2.COLOR_RGB2BGR)

	# Choose center (middle of image)
	center = (target.shape[1] // 2, target.shape[0] // 2)

	binary_mask = (mask > 0).astype(np.uint8) * 255
	# Seamless clone
	br = cv2.boundingRect(binary_mask) # bounding rect (x,y,width,height)
	centerOfBR = (br[0] + br[2] // 2, br[1] + br[3] // 2)
	output_bgr = cv2.seamlessClone(src_bgr, dst_bgr, binary_mask, centerOfBR, cv2.NORMAL_CLONE)
	output_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)
	output_rgb_float = output_rgb.astype(np.float32) / 255.0

	return output_rgb_float
def apply_blend(source, target, mask, poisson_prob, poisson):
	mask_blured = get_blend_mask(mask)
	blend_list=[0.25,0.5,0.75,1,1,1]
	blend_ratio = blend_list[np.random.randint(len(blend_list))]
	mask_blured*=blend_ratio	
	if poisson:
		if random.random() < poisson_prob:
			img_blended = poisson_blend_cv2(source, target, mask_blured)
		else: 
			img_blended=(mask_blured * source + (1 - mask_blured) * target)
	else: 
		img_blended=(mask_blured * source + (1 - mask_blured) * target)
	return img_blended, mask_blured

def get_blend_mask(mask):
	H,W=mask.shape
	size_h=np.random.randint(192,257)
	size_w=np.random.randint(192,257)
	# size_h = np.random.randint(H*0.8, H/0.8)
	# size_w = np.random.randint(W*0.8, W/0.8)
	mask=cv2.resize(mask,(size_w,size_h))
	kernel_1=random.randrange(5,26,2)
	kernel_1=(kernel_1,kernel_1)
	kernel_2=random.randrange(5,26,2)
	kernel_2=(kernel_2,kernel_2)
	
	mask_blured = cv2.GaussianBlur(mask, kernel_1, 0)
	mask_blured = mask_blured/(mask_blured.max())
	mask_blured[mask_blured<1]=0
	
	mask_blured = cv2.GaussianBlur(mask_blured, kernel_2, np.random.randint(5,46))
	mask_blured = mask_blured/(mask_blured.max())
	mask_blured = cv2.resize(mask_blured,(W,H))
	return mask_blured.reshape((mask_blured.shape+(1,)))


def get_alpha_blend_mask(mask):
	kernel_list=[(11,11),(9,9),(7,7),(5,5),(3,3)]
	blend_list=[0.25,0.5,0.75]
	kernel_idxs=random.choices(range(len(kernel_list)), k=2)
	blend_ratio = blend_list[random.sample(range(len(blend_list)), 1)[0]]
	mask_blured = cv2.GaussianBlur(mask, kernel_list[0], 0)
	# print(mask_blured.max())
	mask_blured[mask_blured<mask_blured.max()]=0
	mask_blured[mask_blured>0]=1
	# mask_blured = mask
	mask_blured = cv2.GaussianBlur(mask_blured, kernel_list[kernel_idxs[1]], 0)
	mask_blured = mask_blured/(mask_blured.max())
	return mask_blured.reshape((mask_blured.shape+(1,)))

