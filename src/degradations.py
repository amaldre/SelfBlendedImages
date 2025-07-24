# -*- coding: utf-8 -*-
import numpy as np
import cv2
import torch

import random
from scipy import ndimage
import scipy
import scipy.stats as ss
from scipy.interpolate import interp2d
from scipy.linalg import orth
from PIL import Image, ImageEnhance
import string
from utils.funcs import IoUfrom2bboxes, crop_face
from tqdm import tqdm
from torchvision.utils import save_image
import os

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



def fspecial(filter_type, *args, **kwargs):
    '''
    python code from:
    https://github.com/ronaldosena/imagens-medicas-2/blob/40171a6c259edec7827a6693a93955de2bd39e76/Aulas/aula_2_-_uniform_filter/matlab_fspecial.py
    '''
    if filter_type == 'gaussian':
        return fspecial_gaussian(*args, **kwargs)
"""
# --------------------------------------------
# degradation models
# --------------------------------------------
"""

def uniform_Gaussian(size):
    return np.full((size, size), 1 / size**2)

def add_blur(img, s):
    choice = random.choice([0, 1, 2])
    base = 13
    if choice == 0:
        wd = int(base * s)
        l1 = wd*random.random()
        l2 = wd*random.random()
        k = anisotropic_Gaussian(ksize=2*random.randint(2,11)+3, theta=random.random()*np.pi, l1=l1, l2=l2)
    elif choice == 1:
        wd = int(base * s)
        k = fspecial('gaussian', 2*random.randint(2,11)+3, wd*random.random())
    elif choice == 2:
        w = np.random.randint(3, int(base* s + 1))
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
    # Convert image to float32 in [0, 1] range if not already
    img = img.astype(np.float32)

    noise_level = random.randint(int(noise_level1), int(noise_level2))
    rnum = np.random.rand()

    if rnum > 0.6:  # add color Gaussian noise
        noise = np.random.normal(0, noise_level / 255.0, img.shape).astype(np.float32)

    elif rnum < 0.4:  # add grayscale Gaussian noise
        noise_gray = np.random.normal(0, noise_level / 255.0, img.shape[:2])
        noise = np.repeat(noise_gray[:, :, np.newaxis], 3, axis=2).astype(np.float32)

    else:  # add multivariate Gaussian noise
        L = noise_level2 / 255.0
        D = np.diag(np.random.rand(3))
        U = orth(np.random.rand(3, 3))
        cov = np.dot(np.dot(U.T, D), U)
        noise = np.random.multivariate_normal([0, 0, 0], np.abs(L**2 * cov), img.shape[:2])
        noise = noise.reshape(img.shape).astype(np.float32)

    # Add noise
    img = img + noise

    # Clip to valid range [0, 1]
    img = np.clip(img, 0.0, 1.0)

    return img

#TODO figure out what to do with this
def add_speckle_noise(img, noise_level1=2, noise_level2=25):
    noise_level = random.randint(int(noise_level1), int(noise_level2))
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
    quality_factor = random.randint(80, 95)
    img = cv2.cvtColor(np.uint8((img.clip(0, 1)*255.).round()), cv2.COLOR_RGB2BGR)
    result, encimg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor])
    img = cv2.imdecode(encimg, 1)
    img = cv2.cvtColor(np.float32(img/255.), cv2.COLOR_BGR2RGB)
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
    from utils.sbi import SBI_Dataset
    found = False
    while not found:
        filename = random.choice(image_list)
        try:
            img = np.array(Image.open(filename))
            found = True
        except : 
            continue
        
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

def add_distractors(img, image_list, path_lm, p_d):
    n = 0
    while random.random() < p_d and n < 10:
        if random.random() < 0.5:
            img = add_text(img)
        else:
            img_to_add = choose_image(image_list, path_lm)
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
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    thickness = random.randint(1, 8)
    line_type = random.randint(0, 2)
    img = cv2.cvtColor(np.uint8((img.clip(0, 1)*255.).round()), cv2.COLOR_RGB2BGR)
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness, line_type)
    img = cv2.cvtColor(np.float32(img/255.), cv2.COLOR_BGR2RGB)
    return img 

def add_image(img, img_to_add):
    x = random.randint(20, 100)
    y = int(random.uniform(0.8 * x, 1.2 * x))
    
    img_to_add_resized = cv2.resize(img_to_add, (x, y))/255

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

def degradation(img, image_list, path_lm):
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

    # h1, w1 = img.shape[:2]
    # img = img.copy()[:h1 - h1 % sf, :w1 - w1 % sf, ...]  # mod crop
    # h, w = img.shape[:2]

    # if h < lq_patchsize*sf or w < lq_patchsize*sf:
    #     raise ValueError(f'img size ({h1}X{w1}) is too small!')

    hq = img.copy()

    shuffle_order = random.sample(range(8), 8)

    p = 0.25
    p_d = 0.15
    s = 1/3

    for i in shuffle_order:
        if i == 0 and random.random() < p:
            img = add_blur(img, s)
        elif i == 1 and random.random() < p:
            img = add_resize(img, s)
        elif i == 2 and random.random() < p/2:
            l1 = 2
            l2 = 50
            img = add_Gaussian_noise(img, noise_level1=l1 * s, noise_level2=l2 * s)
        elif i == 3 and random.random() < p/2:
            l1 = 25
            l2 = 50
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
        elif i > 7:
            print('check the shuffle!')

    # resize to desired size
    img = cv2.resize(img, (int(hq.shape[1]), int(hq.shape[0])), interpolation=random.choice([1, 2, 3]))

    # add final JPEG compression noise
    #img = add_JPEG_noise(img)

    # random crop
    #img, hq = random_crop(img, hq, sf, lq_patchsize)

    return img



if __name__ == '__main__':
    # img = cv2.imread('/home/alicia/dataShareID/Celeb-Df-v2_crop/test/frames/Celeb-real/id0_0001/136.png')/255
    # img_blur = add_blur(img, 0.5)
    # img_resize = add_resize(img, 0.5)
    # l1 = 2
    # l2 = 100
    # img_gaussian_1 = add_Gaussian_noise(img, noise_level1=l1 * 0.5, noise_level2=l2 * 0.5)
    # l1 = 80
    # l2 = 100
    # img_gaussian_2 = add_Gaussian_noise(img, noise_level1=l1 * 0.5, noise_level2=l2 * 0.5)
    # img_poisson = add_Poisson_noise(img)
    # img_speckle = add_speckle_noise(img)
    # img_jpeg = add_JPEG_noise(img)
    # img_enhance = enhance(img)

# Crée un dossier pour les images (ne plante pas si déjà là)
    save_dir = "debug_degraded"
    # os.makedirs(save_dir, exist_ok=True)

    # device = torch.device('cuda')
    # val_dataset = SBI_Dataset(phase = 'val', image_size = 380, poisson = True, random_mask = True)
    # val_loader=torch.utils.data.DataLoader(val_dataset,
    #                     batch_size=1,
    #                     shuffle=True,
    #                     collate_fn=val_dataset.collate_fn,
    #                     num_workers=4,
    #                     pin_memory=True,
    #                     worker_init_fn=val_dataset.worker_init_fn
    #                     )
    # for step, data in enumerate(tqdm(val_loader)):
    #     img = data['img'].to(device, non_blocking=True).float()
    #     degraded_list = []
    #     for i in range(img.size(0)):
    #         single_img_tensor = img[i]             # (C, H, W)
    
    #         single_img_np = single_img_tensor.cpu().numpy()  # (C, H, W)
    #         single_img_np = np.transpose(single_img_np, (1, 2, 0))  # (H, W, C) si nécessaire
    #         degraded_img_np = degradation(single_img_np, val_dataset.image_list, val_dataset.path_lm)
    
    #         # Reconvertir en tensor (C, H, W)
    #         degraded_img_np = np.transpose(degraded_img_np, (2, 0, 1))  # (C, H, W)
    #         degraded_img_tensor = torch.from_numpy(degraded_img_np).to(device).float()

    #         degraded_list.append(degraded_img_tensor)
    #         save_path = os.path.join(save_dir, f"step{step}_img{i}.png")
    #         save_image(degraded_img_tensor.clamp(0, 1), save_path)

    #     # Empiler pour obtenir un batch final
    #     degraded_img_batch = torch.stack(degraded_list, dim=0) 
#     img = util.imread_uint('utils/test.png', 3)
#     img = util.uint2single(img)
#     sf = 4
    
#     for i in range(20):
#         #img_lq, img_hq = degradation_bsrgan(img, sf=sf, lq_patchsize=72)
#         print(i)
#         lq_nearest =  cv2.resize(util.single2uint(img_lq), (int(sf*img_lq.shape[1]), int(sf*img_lq.shape[0])), interpolation=0)
#         img_concat = np.concatenate([lq_nearest, util.single2uint(img_hq)], axis=1)
#         util.imsave(img_concat, str(i)+'.png')

# #    for i in range(10):
# #        img_lq, img_hq = degradation_bsrgan_plus(img, sf=sf, shuffle_prob=0.1, use_sharp=True, lq_patchsize=64)
# #        print(i)
# #        lq_nearest =  cv2.resize(util.single2uint(img_lq), (int(sf*img_lq.shape[1]), int(sf*img_lq.shape[0])), interpolation=0)
# #        img_concat = np.concatenate([lq_nearest, util.single2uint(img_hq)], axis=1)
# #        util.imsave(img_concat, str(i)+'.png')

# #    run utils/utils_blindsr.py