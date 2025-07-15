# -*- coding: utf-8 -*-
import numpy as np
import cv2
import torch

from utils import utils_image as util

import random
from scipy import ndimage
import scipy
import scipy.stats as ss
from scipy.interpolate import interp2d
from scipy.linalg import orth
from PIL import Image, ImageEnhance
import string
from funcs import IoUfrom2bboxes, crop_face
from sbi import SBI_Dataset


"""
# --------------------------------------------
# Super-Resolution
# --------------------------------------------
#
# Kai Zhang (cskaizhang@gmail.com)
# https://github.com/cszn
# From 2019/03--2021/08
# --------------------------------------------
"""

def modcrop_np(img, sf):
    '''
    Args:
        img: numpy image, WxH or WxHxC
        sf: scale factor

    Return:
        cropped image
    '''
    w, h = img.shape[:2]
    im = np.copy(img)
    return im[:w - w % sf, :h - h % sf, ...]


"""
# --------------------------------------------
# anisotropic Gaussian kernels
# --------------------------------------------
"""
# def analytic_kernel(k):
#     """Calculate the X4 kernel from the X2 kernel (for proof see appendix in paper)"""
#     k_size = k.shape[0]
#     # Calculate the big kernels size
#     big_k = np.zeros((3 * k_size - 2, 3 * k_size - 2))
#     # Loop over the small kernel to fill the big one
#     for r in range(k_size):
#         for c in range(k_size):
#             big_k[2 * r:2 * r + k_size, 2 * c:2 * c + k_size] += k[r, c] * k
#     # Crop the edges of the big kernel to ignore very small values and increase run time of SR
#     crop = k_size // 2
#     cropped_big_k = big_k[crop:-crop, crop:-crop]
#     # Normalize to 1
#     return cropped_big_k / cropped_big_k.sum()


def anisotropic_Gaussian(ksize=15, theta=np.pi, l1=6, l2=6):
    """ generate an anisotropic Gaussian kernel
    Args:
        ksize : e.g., 15, kernel size
        theta : [0,  pi], rotation angle range
        l1    : [0.1,50], scaling of eigenvalues
        l2    : [0.1,l1], scaling of eigenvalues
        If l1 = l2, will get an isotropic Gaussian kernel.

    Returns:
        k     : kernel
    """

    v = np.dot(np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]), np.array([1., 0.]))
    V = np.array([[v[0], v[1]], [v[1], -v[0]]])
    D = np.array([[l1, 0], [0, l2]])
    Sigma = np.dot(np.dot(V, D), np.linalg.inv(V))
    k = gm_blur_kernel(mean=[0, 0], cov=Sigma, size=ksize)

    return k


def gm_blur_kernel(mean, cov, size=15):
    center = size / 2.0 + 0.5
    k = np.zeros([size, size])
    for y in range(size):
        for x in range(size):
            cy = y - center + 1
            cx = x - center + 1
            k[y, x] = ss.multivariate_normal.pdf([cx, cy], mean=mean, cov=cov)

    k = k / np.sum(k)
    return k


# def shift_pixel(x, sf, upper_left=True):
#     """shift pixel for super-resolution with different scale factors
#     Args:
#         x: WxHxC or WxH
#         sf: scale factor
#         upper_left: shift direction
#     """
#     h, w = x.shape[:2]
#     shift = (sf-1)*0.5
#     xv, yv = np.arange(0, w, 1.0), np.arange(0, h, 1.0)
#     if upper_left:
#         x1 = xv + shift
#         y1 = yv + shift
#     else:
#         x1 = xv - shift
#         y1 = yv - shift

#     x1 = np.clip(x1, 0, w-1)
#     y1 = np.clip(y1, 0, h-1)

#     if x.ndim == 2:
#         x = interp2d(xv, yv, x)(x1, y1)
#     if x.ndim == 3:
#         for i in range(x.shape[-1]):
#             x[:, :, i] = interp2d(xv, yv, x[:, :, i])(x1, y1)

#     return x


def blur(x, k):
    '''
    x: image, NxcxHxW
    k: kernel, Nx1xhxw
    '''
    n, c = x.shape[:2]
    p1, p2 = (k.shape[-2]-1)//2, (k.shape[-1]-1)//2
    x = torch.nn.functional.pad(x, pad=(p1, p2, p1, p2), mode='replicate')
    k = k.repeat(1,c,1,1)
    k = k.view(-1, 1, k.shape[2], k.shape[3])
    x = x.view(1, -1, x.shape[2], x.shape[3])
    x = torch.nn.functional.conv2d(x, k, bias=None, stride=1, padding=0, groups=n*c)
    x = x.view(n, c, x.shape[2], x.shape[3])

    return x



# def gen_kernel(k_size=np.array([15, 15]), scale_factor=np.array([4, 4]), min_var=0.6, max_var=10., noise_level=0):
#     """"
#     # modified version of https://github.com/assafshocher/BlindSR_dataset_generator
#     # Kai Zhang
#     # min_var = 0.175 * sf  # variance of the gaussian kernel will be sampled between min_var and max_var
#     # max_var = 2.5 * sf
#     """
#     # Set random eigen-vals (lambdas) and angle (theta) for COV matrix
#     lambda_1 = min_var + np.random.rand() * (max_var - min_var)
#     lambda_2 = min_var + np.random.rand() * (max_var - min_var)
#     theta = np.random.rand() * np.pi  # random theta
#     noise = -noise_level + np.random.rand(*k_size) * noise_level * 2

#     # Set COV matrix using Lambdas and Theta
#     LAMBDA = np.diag([lambda_1, lambda_2])
#     Q = np.array([[np.cos(theta), -np.sin(theta)],
#                   [np.sin(theta), np.cos(theta)]])
#     SIGMA = Q @ LAMBDA @ Q.T
#     INV_SIGMA = np.linalg.inv(SIGMA)[None, None, :, :]

#     # Set expectation position (shifting kernel for aligned image)
#     MU = k_size // 2 - 0.5*(scale_factor - 1) # - 0.5 * (scale_factor - k_size % 2)
#     MU = MU[None, None, :, None]

#     # Create meshgrid for Gaussian
#     [X,Y] = np.meshgrid(range(k_size[0]), range(k_size[1]))
#     Z = np.stack([X, Y], 2)[:, :, :, None]

#     # Calcualte Gaussian for every pixel of the kernel
#     ZZ = Z-MU
#     ZZ_t = ZZ.transpose(0,1,3,2)
#     raw_kernel = np.exp(-0.5 * np.squeeze(ZZ_t @ INV_SIGMA @ ZZ)) * (1 + noise)

#     # shift the kernel so it will be centered
#     #raw_kernel_centered = kernel_shift(raw_kernel, scale_factor)

#     # Normalize the kernel and return
#     #kernel = raw_kernel_centered / np.sum(raw_kernel_centered)
#     kernel = raw_kernel / np.sum(raw_kernel)
#     return kernel


def fspecial_gaussian(hsize, sigma):
    hsize = [hsize, hsize]
    siz = [(hsize[0]-1.0)/2.0, (hsize[1]-1.0)/2.0]
    std = sigma
    [x, y] = np.meshgrid(np.arange(-siz[1], siz[1]+1), np.arange(-siz[0], siz[0]+1))
    arg = -(x*x + y*y)/(2*std*std)
    h = np.exp(arg)
    h[h < scipy.finfo(float).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h = h/sumh
    return h


# def fspecial_laplacian(alpha):
#     alpha = max([0, min([alpha,1])])
#     h1 = alpha/(alpha+1)
#     h2 = (1-alpha)/(alpha+1)
#     h = [[h1, h2, h1], [h2, -4/(alpha+1), h2], [h1, h2, h1]]
#     h = np.array(h)
#     return h


def fspecial(filter_type, *args, **kwargs):
    '''
    python code from:
    https://github.com/ronaldosena/imagens-medicas-2/blob/40171a6c259edec7827a6693a93955de2bd39e76/Aulas/aula_2_-_uniform_filter/matlab_fspecial.py
    '''
    if filter_type == 'gaussian':
        return fspecial_gaussian(*args, **kwargs)
    # if filter_type == 'laplacian':
    #     return fspecial_laplacian(*args, **kwargs)

"""
# --------------------------------------------
# degradation models
# --------------------------------------------
"""


# def bicubic_degradation(x, sf=3):
#     '''
#     Args:
#         x: HxWxC image, [0, 1]
#         sf: down-scale factor

#     Return:
#         bicubicly downsampled LR image
#     '''
#     x = util.imresize_np(x, scale=1/sf)
#     return x


# def srmd_degradation(x, k, sf=3):
#     ''' blur + bicubic downsampling

#     Args:
#         x: HxWxC image, [0, 1]
#         k: hxw, double
#         sf: down-scale factor

#     Return:
#         downsampled LR image

#     Reference:
#         @inproceedings{zhang2018learning,
#           title={Learning a single convolutional super-resolution network for multiple degradations},
#           author={Zhang, Kai and Zuo, Wangmeng and Zhang, Lei},
#           booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
#           pages={3262--3271},
#           year={2018}
#         }
#     '''
#     x = ndimage.filters.convolve(x, np.expand_dims(k, axis=2), mode='wrap')  # 'nearest' | 'mirror'
#     x = bicubic_degradation(x, sf=sf)
    # return x


# def dpsr_degradation(x, k, sf=3):

#     ''' bicubic downsampling + blur

#     Args:
#         x: HxWxC image, [0, 1]
#         k: hxw, double
#         sf: down-scale factor

#     Return:
#         downsampled LR image

#     Reference:
#         @inproceedings{zhang2019deep,
#           title={Deep Plug-and-Play Super-Resolution for Arbitrary Blur Kernels},
#           author={Zhang, Kai and Zuo, Wangmeng and Zhang, Lei},
#           booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
#           pages={1671--1681},
#           year={2019}
#         }
#     '''
#     x = bicubic_degradation(x, sf=sf)
#     x = ndimage.filters.convolve(x, np.expand_dims(k, axis=2), mode='wrap')
#     return x


# def classical_degradation(x, k, sf=3):
#     ''' blur + downsampling

#     Args:
#         x: HxWxC image, [0, 1]/[0, 255]
#         k: hxw, double
#         sf: down-scale factor

#     Return:
#         downsampled LR image
#     '''
#     x = ndimage.filters.convolve(x, np.expand_dims(k, axis=2), mode='wrap')
#     #x = filters.correlate(x, np.expand_dims(np.flip(k), axis=2))
#     st = 0
#     return x[st::sf, st::sf, ...]


# def add_sharpening(img, weight=0.5, radius=50, threshold=10):
#     """USM sharpening. borrowed from real-ESRGAN
#     Input image: I; Blurry image: B.
#     1. K = I + weight * (I - B)
#     2. Mask = 1 if abs(I - B) > threshold, else: 0
#     3. Blur mask:
#     4. Out = Mask * K + (1 - Mask) * I
#     Args:
#         img (Numpy array): Input image, HWC, BGR; float32, [0, 1].
#         weight (float): Sharp weight. Default: 1.
#         radius (float): Kernel size of Gaussian blur. Default: 50.
#         threshold (int):
#     """
#     if radius % 2 == 0:
#         radius += 1
#     blur = cv2.GaussianBlur(img, (radius, radius), 0)
#     residual = img - blur
#     mask = np.abs(residual) * 255 > threshold
#     mask = mask.astype('float32')
#     soft_mask = cv2.GaussianBlur(mask, (radius, radius), 0)

#     K = img + weight * residual
#     K = np.clip(K, 0, 1)
#     return soft_mask * K + (1 - soft_mask) * img

def uniform_Gaussian(size):
    return np.full((size, size), 1 / size**2)

def add_blur(img, s):
    choice = random.choice([0, 1, 2])
    if choice == 0:
        wd = 30 * s
        l1 = wd*random.random()
        l2 = wd*random.random()
        k = anisotropic_Gaussian(ksize=2*random.randint(2,11)+3, theta=random.random()*np.pi, l1=l1, l2=l2)
    elif choice == 1:
        wd = 30 * s
        k = fspecial('gaussian', 2*random.randint(2,11)+3, wd*random.random())
    elif choice == 2:
        w = np.random.randint(3, 30* s + 1)
        k = uniform_Gaussian(w)
    img = ndimage.filters.convolve(img, np.expand_dims(k, axis=2), mode='mirror')
    return img


def add_resize(img, s):
    rnum = np.random.rand()
    if rnum > 0.8:  # up
        sf1 = random.uniform(1, 2)
    elif rnum < 0.7:  # down
        sf1 = random.uniform(0.25/s, 1)
    else:
        sf1 = 1.0
    img = cv2.resize(img, (int(sf1*img.shape[1]), int(sf1*img.shape[0])), interpolation=random.choice([1, 2, 3]))
    img = np.clip(img, 0.0, 1.0)

    return img


def add_Gaussian_noise(img, noise_level1=2, noise_level2=25):
    noise_level = random.randint(noise_level1, noise_level2)
    rnum = np.random.rand()
    if rnum > 0.6:   # add color Gaussian noise
        img += np.random.normal(0, noise_level/255.0, img.shape).astype(np.float32)
    elif rnum < 0.4: # add grayscale Gaussian noise
        img += np.random.normal(0, noise_level/255.0, (*img.shape[:2], 1)).astype(np.float32)
    else:            # add  noise
        L = noise_level2/255.
        D = np.diag(np.random.rand(3))
        U = orth(np.random.rand(3,3))
        conv = np.dot(np.dot(np.transpose(U), D), U)
        img += np.random.multivariate_normal([0,0,0], np.abs(L**2*conv), img.shape[:2]).astype(np.float32)
    img = np.clip(img, 0.0, 1.0)
    return img

#TODO figure out what to do with this
def add_speckle_noise(img, noise_level1=2, noise_level2=25):
    noise_level = random.randint(noise_level1, noise_level2)
    img = np.clip(img, 0.0, 1.0)
    rnum = random.random()
    if rnum > 0.6:
        img += img*np.random.normal(0, noise_level/255.0, img.shape).astype(np.float32)
    elif rnum < 0.4:
        img += img*np.random.normal(0, noise_level/255.0, (*img.shape[:2], 1)).astype(np.float32)
    else:
        L = noise_level2/255.
        D = np.diag(np.random.rand(3))
        U = orth(np.random.rand(3,3))
        conv = np.dot(np.dot(np.transpose(U), D), U)
        img += img*np.random.multivariate_normal([0,0,0], np.abs(L**2*conv), img.shape[:2]).astype(np.float32)
    img = np.clip(img, 0.0, 1.0)
    return img


def add_Poisson_noise(img):
    img = np.clip((img * 255.0).round(), 0, 255) / 255.
    vals = 10**(2*random.random()+2.0)  # [2, 4]
    if random.random() < 0.5:
        img = np.random.poisson(img * vals).astype(np.float32) / vals
    else:
        img_gray = np.dot(img[...,:3], [0.299, 0.587, 0.114])
        img_gray = np.clip((img_gray * 255.0).round(), 0, 255) / 255.
        noise_gray = np.random.poisson(img_gray * vals).astype(np.float32) / vals - img_gray
        img += noise_gray[:, :, np.newaxis]
    img = np.clip(img, 0.0, 1.0)
    return img


def add_JPEG_noise(img):
    quality_factor = random.randint(10, 95)
    img = cv2.cvtColor(util.single2uint(img), cv2.COLOR_RGB2BGR)
    result, encimg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor])
    img = cv2.imdecode(encimg, 1)
    img = cv2.cvtColor(util.uint2single(img), cv2.COLOR_BGR2RGB)
    return img

def enhance(img):
    # Convert to PIL image
    img_pil = Image.fromarray((img * 255).astype(np.uint8))

    # Choose enhancement type
    if random.random() < 0.5:
        enhancer = ImageEnhance.Brightness(img_pil)
    else:
        enhancer = ImageEnhance.Contrast(img_pil)

    # Sample factor from Uniform(0.5, 1.5)
    factor = random.uniform(0.5, 1.5)

    # Enhance
    img_enhanced_pil = enhancer.enhance(factor)

    # Convert back to NumPy array and normalize to [0, 1]
    img_enhanced_np = np.asarray(img_enhanced_pil).astype(np.float32) / 255.0

    return img_enhanced_np

def choose_image(image_list, path_lm):
    filename=random.choice(image_list)
    img=np.array(Image.open(filename))
	#Get dlib landmarks
    landmark=np.load(filename.replace('.png','.npy').replace('/frames/', path_lm))[0]
	#Get dlib bounding landmarks
    bbox_lm=np.array([landmark[:,0].min(),landmark[:,1].min(),landmark[:,0].max(),landmark[:,1].max()])
	#Get to two first bounding boxes detected by retina
    bboxes=np.load(filename.replace('.png','.npy').replace('/frames/','/retina/'))[:2]
				#Finding face with highest iou
    iou_max=-1
    for i in range(len(bboxes)):
        iou=IoUfrom2bboxes(bbox_lm,bboxes[i].flatten())
        if iou_max<iou:
            bbox=bboxes[i]
            iou_max=iou
	#Reorder landmark
    landmark=SBI_Dataset.reorder_landmark(landmark)

    img,_,__,___,y0_new,y1_new,x0_new,x1_new=crop_face(img,landmark,bbox,margin=False,crop_by_bbox=True,abs_coord=True,phase='train')
    return img

def add_distractors(img, image_list, p_d):
    n = 0
    while random.choice() < p_d and n < 10:
        if random.choice() < 0.5:
            img = add_text(img)
        else:
            img_to_add = choose_image(image_list)
            img = add_image(img, img_to_add)
        n += 1
    return img

def add_text(img):
    n_chars = random.randint(0, 10)
    text = ''.join(random.sample(string.printable, n_chars))
    height, width = img.shape[: 2]
    x = random.randint(-100, width)
    y = random.randint(0, height + 100)
    font = random.randint(0, 7)
    font_scale = 8 * random.random()
    color = (random.randints(0, 255), random.randint(0, 255), random.randint(0, 255))
    thickness = random.randint(1, 8)
    line_type = random.randint(0, 2)
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness, line_type)

def add_image(img, img_to_add):
    x = random.randint(20, 100)
    y = int(random.uniform(0.8 * x, 1.2 * x))
    
    img_to_add_resized = cv2.resize(img_to_add (x, y))

    height, width = img.shape[: 2]

    X = random.randint(0, max(0, width - x))
    Y = random.randint(0, max(0, height - y))

    img[Y:Y + y, X:X + x] = img_to_add_resized 
    return img


# def random_crop(lq, hq, sf=4, lq_patchsize=64):
#     h, w = lq.shape[:2]
#     rnd_h = random.randint(0, h-lq_patchsize)
#     rnd_w = random.randint(0, w-lq_patchsize)
#     lq = lq[rnd_h:rnd_h + lq_patchsize, rnd_w:rnd_w + lq_patchsize, :]

#     rnd_h_H, rnd_w_H = int(rnd_h * sf), int(rnd_w * sf)
#     hq = hq[rnd_h_H:rnd_h_H + lq_patchsize*sf, rnd_w_H:rnd_w_H + lq_patchsize*sf, :]
#     return lq, hq


# def degradation_bsrgan(img, sf=4, lq_patchsize=72, isp_model=None):
#     """
#     This is the degradation model of BSRGAN from the paper
#     "Designing a Practical Degradation Model for Deep Blind Image Super-Resolution"
#     ----------
#     img: HXWXC, [0, 1], its size should be large than (lq_patchsizexsf)x(lq_patchsizexsf)
#     sf: scale factor
#     isp_model: camera ISP model

#     Returns
#     -------
#     img: low-quality patch, size: lq_patchsizeXlq_patchsizeXC, range: [0, 1]
#     hq: corresponding high-quality patch, size: (lq_patchsizexsf)X(lq_patchsizexsf)XC, range: [0, 1]
#     """
#     isp_prob, jpeg_prob, scale2_prob = 0.25, 0.9, 0.25
#     sf_ori = sf

#     h1, w1 = img.shape[:2]
#     img = img.copy()[:h1 - h1 % sf, :w1 - w1 % sf, ...]  # mod crop
#     h, w = img.shape[:2]

#     if h < lq_patchsize*sf or w < lq_patchsize*sf:
#         raise ValueError(f'img size ({h1}X{w1}) is too small!')

#     hq = img.copy()

#     if sf == 4 and random.random() < scale2_prob:   # downsample1
#         if np.random.rand() < 0.5:
#             img = cv2.resize(img, (int(1/2*img.shape[1]), int(1/2*img.shape[0])), interpolation=random.choice([1,2,3]))
#         else:
#             img = util.imresize_np(img, 1/2, True)
#         img = np.clip(img, 0.0, 1.0)
#         sf = 2

#     shuffle_order = random.sample(range(7), 7)
#     idx1, idx2 = shuffle_order.index(2), shuffle_order.index(3)
#     if idx1 > idx2:  # keep downsample3 last
#         shuffle_order[idx1], shuffle_order[idx2] = shuffle_order[idx2], shuffle_order[idx1]

#     for i in shuffle_order:

#         if i == 0:
#             img = add_blur(img, sf=sf)

#         elif i == 1:
#             img = add_blur(img, sf=sf)

#         elif i == 2:
#             a, b = img.shape[1], img.shape[0]
#             # downsample2
#             if random.random() < 0.75:
#                 sf1 = random.uniform(1,2*sf)
#                 img = cv2.resize(img, (int(1/sf1*img.shape[1]), int(1/sf1*img.shape[0])), interpolation=random.choice([1,2,3]))
#             else:
#                 k = fspecial('gaussian', 25, random.uniform(0.1, 0.6*sf))
#                 k_shifted = shift_pixel(k, sf)
#                 k_shifted = k_shifted/k_shifted.sum()  # blur with shifted kernel
#                 img = ndimage.filters.convolve(img, np.expand_dims(k_shifted, axis=2), mode='mirror')
#                 img = img[0::sf, 0::sf, ...]  # nearest downsampling
#             img = np.clip(img, 0.0, 1.0)

#         elif i == 3:
#             # downsample3
#             img = cv2.resize(img, (int(1/sf*a), int(1/sf*b)), interpolation=random.choice([1,2,3]))
#             img = np.clip(img, 0.0, 1.0)

#         elif i == 4:
#             # add Gaussian noise
#             img = add_Gaussian_noise(img, noise_level1=2, noise_level2=25)

#         elif i == 5:
#             # add JPEG noise
#             if random.random() < jpeg_prob:
#                 img = add_JPEG_noise(img)

#         elif i == 6:
#             # add processed camera sensor noise
#             if random.random() < isp_prob and isp_model is not None:
#                 with torch.no_grad():
#                     img, hq = isp_model.forward(img.copy(), hq)

#     # add final JPEG compression noise
#     img = add_JPEG_noise(img)

#     # random crop
#     img, hq = random_crop(img, hq, sf_ori, lq_patchsize)

#     return img, hq

def degradation(img, image_list, path_lm, sf=4, lq_patchsize=64):
    """
    This is an extended degradation model by combining
    the degradation models of BSRGAN and Real-ESRGAN
    ----------
    img: HXWXC, [0, 1], its size should be large than (lq_patchsizexsf)x(lq_patchsizexsf)
    sf: scale factor
    use_shuffle: the degradation shuffle
    use_sharp: sharpening the img

    Returns
    -------
    img: low-quality patch, size: lq_patchsizeXlq_patchsizeXC, range: [0, 1]
    hq: corresponding high-quality patch, size: (lq_patchsizexsf)X(lq_patchsizexsf)XC, range: [0, 1]
    """

    h1, w1 = img.shape[:2]
    img = img.copy()[:h1 - h1 % sf, :w1 - w1 % sf, ...]  # mod crop
    h, w = img.shape[:2]

    if h < lq_patchsize*sf or w < lq_patchsize*sf:
        raise ValueError(f'img size ({h1}X{w1}) is too small!')

    hq = img.copy()

    shuffle_order = random.sample(range(7), 7)

    p = 0.5
    p_d = 0.2
    s = 0.5
    for i in shuffle_order:
        if i == 0 and random.random() < p:
            img = add_blur(img, s)
        elif i == 1 and random.random() < p:
            img = add_resize(img, s)
        elif i == 2 and random.random() < p/2:
            l1 = 2, l2 = 100
            img = add_Gaussian_noise(img, noise_level1=l1 * s, noise_level2=l2 * s)
        elif i == 3 and random.random() < p/2:
            l1 = 80, l2 = 100
            img = add_Gaussian_noise(img, noise_level1=l1 * s, noise_level2=l2 * s)
        elif i == 4 and random.random() < p:
            if random.random() < 0.5:
                img = add_Poisson_noise(img)
            else:
                img = add_speckle_noise(img)
        elif i == 5 and random.random() < p:
            img = add_JPEG_noise(img)
        elif i == 6 and random.random() < p:
            img = enhance(img)
        elif i == 7:
            img = add_distractors(img, image_list, path_lm, p_d)
        else:
            print('check the shuffle!')

    # resize to desired size
    img = cv2.resize(img, (int(1/sf*hq.shape[1]), int(1/sf*hq.shape[0])), interpolation=random.choice([1, 2, 3]))

    # add final JPEG compression noise
    #img = add_JPEG_noise(img)

    # random crop
    #img, hq = random_crop(img, hq, sf, lq_patchsize)

    return img, hq



if __name__ == '__main__':
    img = util.imread_uint('utils/test.png', 3)
    img = util.uint2single(img)
    sf = 4
    
    for i in range(20):
        #img_lq, img_hq = degradation_bsrgan(img, sf=sf, lq_patchsize=72)
        print(i)
        lq_nearest =  cv2.resize(util.single2uint(img_lq), (int(sf*img_lq.shape[1]), int(sf*img_lq.shape[0])), interpolation=0)
        img_concat = np.concatenate([lq_nearest, util.single2uint(img_hq)], axis=1)
        util.imsave(img_concat, str(i)+'.png')

#    for i in range(10):
#        img_lq, img_hq = degradation_bsrgan_plus(img, sf=sf, shuffle_prob=0.1, use_sharp=True, lq_patchsize=64)
#        print(i)
#        lq_nearest =  cv2.resize(util.single2uint(img_lq), (int(sf*img_lq.shape[1]), int(sf*img_lq.shape[0])), interpolation=0)
#        img_concat = np.concatenate([lq_nearest, util.single2uint(img_hq)], axis=1)
#        util.imsave(img_concat, str(i)+'.png')

#    run utils/utils_blindsr.py