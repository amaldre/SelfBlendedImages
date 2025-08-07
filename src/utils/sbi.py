# Created by: Kaede Shiohara
# Yamasaki Lab at The University of Tokyo
# shiohara@cvm.t.u-tokyo.ac.jp
# Copyright (c) 2021
# 3rd party softwares' licenses are noticed at https://github.com/mapooon/SelfBlendedImages/blob/master/LICENSE

import torch
from torchvision import datasets,transforms,utils
from torch.utils.data import Dataset,IterableDataset, ConcatDataset
from diffusers import StableDiffusionImg2ImgPipeline
from glob import glob
import os
import numpy as np
from PIL import Image
import random
import cv2
from torch import nn
import sys
import albumentations as alb
import warnings
warnings.filterwarnings('ignore')


import logging

script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the library directory
library_path = os.path.join(script_dir, 'library')

if os.path.isdir(library_path) and os.path.isfile(os.path.join(library_path, 'bi_online_generation.py')):
    sys.path.append(library_path)
    try:
        from bi_online_generation import random_get_hull
        exist_bi = True
    except ImportError as e:
        print(f"Error importing bi_online_generation: {e}")
        exist_bi = False
else:
    print('library or bi_online_generation.py does not exist at the expected path.')
    exist_bi = False


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from degradations import degradation
print(f"exist_bi: {exist_bi}")

class StableDiffusionSingleton:
    _instance = None

    @classmethod
    def get_pipeline(cls):
        if cls._instance is None:
            cls._instance = StableDiffusionImg2ImgPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16
            ).to("cuda")
        return cls._instance

# Constants for Stable Diffusion parameters
SD_PROMPT = "a realistic photo of a person"  # Generic prompt
SD_STRENGTH = 0.01  # Very low to avoid altering content
SD_GUIDANCE_SCALE = 1.0  # Low to limit changes


class SBI_Dataset(Dataset):
	def __init__(self,phase='train',image_size=224,n_frames=8, degradations = False, poisson = False, random_mask = False, pg= 0.0):
		
		assert phase in ['train','val','test']
		
		image_list,label_list=init_ff(phase,'frame',n_frames=n_frames)
		
		path_lm='/landmarks/' 

		label_list=[label_list[i] for i in range(len(image_list)) if os.path.isfile(image_list[i].replace('/frames/',path_lm).replace('.png','.npy')) and os.path.isfile(image_list[i].replace('/frames/','/retina/').replace('.png','.npy'))]
		image_list=[image_list[i] for i in range(len(image_list)) if os.path.isfile(image_list[i].replace('/frames/',path_lm).replace('.png','.npy')) and os.path.isfile(image_list[i].replace('/frames/','/retina/').replace('.png','.npy'))]
		self.path_lm=path_lm
		print(f'SBI({phase}): {len(image_list)}')
	

		self.image_list=image_list

		self.image_size=(image_size,image_size)
		self.phase=phase
		self.n_frames=n_frames

		self.transforms=self.get_transforms()
		self.source_transforms = self.get_source_transforms()
		self.degradations = degradations
		self.poisson = poisson
		self.random_mask = random_mask
		self.final_transforms = get_final_transforms()
		self.pg = pg


	def __len__(self):
		return len(self.image_list)

	def __getitem__(self,idx):
		flag=True
		while flag:
			try:
				#Initialization
				filename=self.image_list[idx]
				img=np.array(Image.open(filename))
				#Get dlib landmarks
				landmark=np.load(filename.replace('.png','.npy').replace('/frames/',self.path_lm))[0]
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
				landmark=self.reorder_landmark(landmark)

				#If training, random chance of flipping image
				if self.phase=='train':
					if np.random.rand()<0.5:
						#Function is same as LAA-Net
						img,_,landmark,bbox=self.hflip(img,None,landmark,bbox)

				#Get img landmarks and bbox for self-blending		
				img,landmark,bbox,__=crop_face(img,landmark,bbox,margin=True,crop_by_bbox=False)

				#Get self blending pristine and fake (change for stable and gan)
				#img_r,img_f,mask_f=self.self_blending(img.copy(),landmark.copy(), self.poisson, self.random_mask)
				img_r,img_f,mask_f=self.self_blending(img.copy(),landmark.copy(), self.poisson, self.random_mask, self.pg)

				#Augment during training
				if self.phase=='train' and not self.degradations:
					transformed=self.transforms(image=img_f.astype('uint8'),image1=img_r.astype('uint8'))
					img_f=transformed['image']
					img_r=transformed['image1']
					
				
				#Crop on fake image between 5 and 20% around face
				img_f,_,__,___,y0_new,y1_new,x0_new,x1_new=crop_face(img_f,landmark,bbox,margin=False,crop_by_bbox=True,abs_coord=True,phase=self.phase)
				
				#Apply same thing for pristine image
				img_r=img_r[y0_new:y1_new,x0_new:x1_new]
				
				#Resize to config size
				img_f=cv2.resize(img_f,self.image_size,interpolation=cv2.INTER_LINEAR).astype('float32')/255
				img_r=cv2.resize(img_r,self.image_size,interpolation=cv2.INTER_LINEAR).astype('float32')/255

				if (self.degradations):
					img_f = degradation(img_f, self.image_list, self.path_lm)
					img_r = degradation(img_r, self.image_list, self.path_lm)

				img_f = self.final_transforms(img_f)
				img_r = self.final_transforms(img_r)

				#Transpose
				img_f=img_f.transpose((2,0,1))
				img_r=img_r.transpose((2,0,1))

				flag=False
			except Exception as e:
				print(e)
				print(filename)
				idx=torch.randint(low=0,high=len(self),size=(1,)).item()
		
		return img_f,img_r
		
	def get_source_transforms(self):
		return alb.Compose([
				alb.Compose([
						alb.RGBShift((-20,20),(-20,20),(-20,20),p=0.3),
						alb.HueSaturationValue(hue_shift_limit=(-0.3,0.3), sat_shift_limit=(-0.3,0.3), val_shift_limit=(-0.3,0.3), p=1),
						alb.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1,0.1), p=1),
					],p=1),
	
				alb.OneOf([
					RandomDownScale(p=1),
					alb.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1),
				],p=1),
				
			], p=1.)

		
	def get_transforms(self):
		return alb.Compose([
			
			alb.RGBShift((-20,20),(-20,20),(-20,20),p=0.3),
			alb.HueSaturationValue(hue_shift_limit=(-0.3,0.3), sat_shift_limit=(-0.3,0.3), val_shift_limit=(-0.3,0.3), p=0.3),
			alb.RandomBrightnessContrast(brightness_limit=(-0.3,0.3), contrast_limit=(-0.3,0.3), p=0.3),
			alb.ImageCompression(quality_lower=40,quality_upper=100,p=0.5),
			
		], 
		additional_targets={f'image1': 'image'},
		p=1.)


	def randaffine(self,img,mask):
		f=alb.Affine(
				translate_percent={'x':(-0.03,0.03),'y':(-0.015,0.015)},
				scale=[0.95,1/0.95],
				fit_output=False,
				p=1)
			
		g=alb.ElasticTransform(
				alpha=50,
				sigma=7,
				alpha_affine=0,
				p=1,
			)

		transformed=f(image=img,mask=mask)
		img=transformed['image']
		
		mask=transformed['mask']
		transformed=g(image=img,mask=mask)
		mask=transformed['mask']
		return img,mask
	
	def apply_stable_diffusion(self, img):
		with torch.no_grad():
			result = StableDiffusionSingleton.get_pipeline()(
				prompt=SD_PROMPT,
				image=img,
				strength=SD_STRENGTH,
				guidance_scale=SD_GUIDANCE_SCALE,
				num_inference_steps=50
			).images[0]
		return result

	def apply_stylegan(self, img):
		return img

	def apply_stable_or_gan(self, img):
		if np.random.rand() < 0.5:
			return self.apply_stable_diffusion(img)
		else:
			return self.apply_stylegan(img)
		
	def self_blending(self,img,landmark, poisson, random_mask, pg):
		p_p = 0.5

		H,W=len(img),len(img[0])
		if np.random.rand()<0.25:
			landmark=landmark[:68]
		if exist_bi:
			logging.disable(logging.FATAL)
			mask, hull_type = random_get_hull(landmark, img, random_mask)
			mask = mask[:,:,0]
			logging.disable(logging.NOTSET)
		else:
			mask=np.zeros_like(img[:,:,0])
			cv2.fillConvexPoly(mask, cv2.convexHull(landmark), 1.)


		source = img.copy()
		if np.random.rand()<0.5:
			if np.random.rand()<pg:
				source = self.apply_stable_or_gan(source.copy())
			else:
				source = self.source_transforms(image=source.astype(np.uint8))['image']
		else:
			if np.random.rand()<pg:
				img = self.apply_stable_or_gan(img.copy())
			else:
				img = self.source_transforms(image=img.astype(np.uint8))['image']

		source, mask = self.randaffine(source,mask)

		img_blended,mask=B.apply_blend(source, img, mask, p_p, hull_type, poisson)
		img_blended = img_blended.astype(np.uint8)
		img = img.astype(np.uint8)

		return img,img_blended,mask
	
	@staticmethod
	def reorder_landmark(landmark):
		landmark_add=np.zeros((13,2))
		for idx,idx_l in enumerate([77,75,76,68,69,70,71,80,72,73,79,74,78]):
			landmark_add[idx]=landmark[idx_l]
		landmark[68:]=landmark_add
		return landmark

	def hflip(self,img,mask=None,landmark=None,bbox=None):
		H,W=img.shape[:2]
		landmark=landmark.copy()
		bbox=bbox.copy()

		if landmark is not None:
			landmark_new=np.zeros_like(landmark)

			
			landmark_new[:17]=landmark[:17][::-1]
			landmark_new[17:27]=landmark[17:27][::-1]

			landmark_new[27:31]=landmark[27:31]
			landmark_new[31:36]=landmark[31:36][::-1]

			landmark_new[36:40]=landmark[42:46][::-1]
			landmark_new[40:42]=landmark[46:48][::-1]

			landmark_new[42:46]=landmark[36:40][::-1]
			landmark_new[46:48]=landmark[40:42][::-1]

			landmark_new[48:55]=landmark[48:55][::-1]
			landmark_new[55:60]=landmark[55:60][::-1]

			landmark_new[60:65]=landmark[60:65][::-1]
			landmark_new[65:68]=landmark[65:68][::-1]
			if len(landmark)==68:
				pass
			elif len(landmark)==81:
				landmark_new[68:81]=landmark[68:81][::-1]
			else:
				raise NotImplementedError
			landmark_new[:,0]=W-landmark_new[:,0]
			
		else:
			landmark_new=None

		if bbox is not None:
			bbox_new=np.zeros_like(bbox)
			bbox_new[0,0]=bbox[1,0]
			bbox_new[1,0]=bbox[0,0]
			bbox_new[:,0]=W-bbox_new[:,0]
			bbox_new[:,1]=bbox[:,1].copy()
			if len(bbox)>2:
				bbox_new[2,0]=W-bbox[3,0]
				bbox_new[2,1]=bbox[3,1]
				bbox_new[3,0]=W-bbox[2,0]
				bbox_new[3,1]=bbox[2,1]
				bbox_new[4,0]=W-bbox[4,0]
				bbox_new[4,1]=bbox[4,1]
				bbox_new[5,0]=W-bbox[6,0]
				bbox_new[5,1]=bbox[6,1]
				bbox_new[6,0]=W-bbox[5,0]
				bbox_new[6,1]=bbox[5,1]
		else:
			bbox_new=None

		if mask is not None:
			mask=mask[:,::-1]
		else:
			mask=None
		img=img[:,::-1].copy()
		return img,mask,landmark_new,bbox_new
	
	def collate_fn(self,batch):
		img_f,img_r=zip(*batch)
		data={}
		data['img']=torch.cat([torch.tensor(img_r).float(),torch.tensor(img_f).float()],0)
		data['label']=torch.tensor([0]*len(img_r)+[1]*len(img_f))
		return data
		

	def worker_init_fn(self,worker_id):                                                          
		np.random.seed(np.random.get_state()[1][0] + worker_id)

def get_final_transforms():
	return transforms.Lambda(lambda img: img * 2.0 - 1.0)


class SBI_Custom_Dataset(SBI_Dataset):
	def init_datasets(self, phase, datasets, n_frames):
		self.image_list = []
		assert phase in ['train','val','test']
		if ('FF' in datasets):
			image_list_ff, _ = init_ff(phase,'frame',n_frames=n_frames)
			image_list_ff=[image_list_ff[i] for i in range(len(image_list_ff)) if os.path.isfile(image_list_ff[i].replace('/frames/', self.path_lm).replace('.png','.npy')) and os.path.isfile(image_list_ff[i].replace('/frames/',f'/{self.crop_mode}/').replace('.png','.npy'))]
			print(f'SBI_FF({phase}): {len(image_list_ff)}')
			self.image_list += image_list_ff
		if ('MSU-MFSD' in datasets):
			image_list_msu_mfsd, _ = init_MSU_MFD(phase, n_frames)
			print(f'SBI_MSU_MFSD({phase}): {len(image_list_msu_mfsd)}')
			self.image_list += image_list_msu_mfsd
		if ('REPLAY-ATTACK' in datasets):
			image_list_replay_attack, _ = init_replay_attack(phase, n_frames)
			print(f'SBI_REPLAY_ATTACK({phase}): {len(image_list_replay_attack)}')
			self.image_list += image_list_replay_attack
		if ('MOBIO' in datasets):
			image_list_mobio, _ = init_mobio(phase, n_frames)
			print(f'SBI_MOBIO({phase}): {len(image_list_mobio)}')
			self.image_list += image_list_mobio
		if ('SIM-MV2' in datasets):
			image_list_sim_mv2, _ = init_sim_mw2(phase, n_frames)
			print(f'SBI_SIM_MV2({phase}): {len(image_list_sim_mv2)}')
			self.image_list += image_list_sim_mv2

	def __init__(self, phase='train', datasets = ['FF'], image_size=224,n_frames=8, degradations = False, poisson = False, random_mask = False, crop_mode = 'retina', pg=0.0):
		path_lm='/landmarks/' 
		self.path_lm=path_lm
		self.crop_mode = crop_mode
		self.init_datasets(phase, datasets, n_frames)
		self.image_size=(image_size,image_size)
		self.phase=phase
		self.n_frames=n_frames
		self.transforms=self.get_transforms()
		self.source_transforms = self.get_source_transforms()
		self.degradations = degradations
		self.poisson = poisson
		self.random_mask = random_mask
		self.final_transforms = get_final_transforms()
		self.pg = pg
	def __getitem__(self,idx):
		flag=True
		while flag:
			try:
				#Initialization
				filename=self.image_list[idx]
				img=np.array(Image.open(filename))
				#Get dlib landmarks
				landmark=np.load(filename.replace('.png','.npy').replace('/frames/',self.path_lm))[0]
				#Get dlib bounding landmarks
				bbox_lm=np.array([landmark[:,0].min(),landmark[:,1].min(),landmark[:,0].max(),landmark[:,1].max()])
				#Get to two first bounding boxes detected by retina
				retina_path = filename.replace('.png','.npy').replace('/frames/','/retina/')
				yunet_path = filename.replace('.png','.npy').replace('/frames/','/yunet/')
				if os.path.exists(retina_path) and self.crop_mode == 'retina':
					bboxes=np.load(filename.replace('.png','.npy').replace('/frames/','/retina/'))[:2]
					#Finding face with highest iou
					iou_max=-1
					for i in range(len(bboxes)):
						iou=IoUfrom2bboxes(bbox_lm,bboxes[i].flatten())
						if iou_max<iou:
							bbox=bboxes[i]
							iou_max=iou
				elif os.path.exists(yunet_path):
					bbox = np.load(yunet_path)
				else: 
					#Shouldn't happen, as yunet boxes are already checked for during dataset initialization
					print(f"Can't find bounding boxes for {filename}")
					continue
					
				#Reorder landmark
				landmark=self.reorder_landmark(landmark)

				#If training, random chance of flipping image
				if self.phase=='train':
					if np.random.rand()<0.5:
						#Function is same as LAA-Net
						img,_,landmark,bbox=self.hflip(img,None,landmark,bbox)

				#Get img landmarks and bbox for self-blending		
				img,landmark,bbox,__=crop_face(img,landmark,bbox,margin=True,crop_by_bbox=False)

				#Get self blending pristine and fake 
				img_r,img_f,mask_f=self.self_blending(img.copy(),landmark.copy(), self.poisson, self.random_mask)
				
				#Augment during training
				if self.phase=='train' and not self.degradations:
					transformed=self.transforms(image=img_f.astype('uint8'),image1=img_r.astype('uint8'))
					img_f=transformed['image']
					img_r=transformed['image1']
					
				
				#Crop on fake image between 5 and 20% around face
				img_f,_,__,___,y0_new,y1_new,x0_new,x1_new=crop_face(img_f,landmark,bbox,margin=False,crop_by_bbox=True,abs_coord=True,phase=self.phase)
				
				#Apply same thing for pristine image
				img_r=img_r[y0_new:y1_new,x0_new:x1_new]
				
				#Resize to config size
				img_f=cv2.resize(img_f,self.image_size,interpolation=cv2.INTER_LINEAR).astype('float32')/255
				img_r=cv2.resize(img_r,self.image_size,interpolation=cv2.INTER_LINEAR).astype('float32')/255

				if (self.degradations):
					img_f = degradation(img_f, self.image_list, self.path_lm)
					img_r = degradation(img_r, self.image_list, self.path_lm)

				img_f = self.final_transforms(img_f)
				img_r = self.final_transforms(img_r)

				#Transpose
				img_f=img_f.transpose((2,0,1))
				img_r=img_r.transpose((2,0,1))

				flag=False
			except Exception as e:
				print(e)
				print(filename)
				idx=torch.randint(low=0,high=len(self),size=(1,)).item()
		
		return img_f,img_r
	
class SourceConcat(ConcatDataset):
	def __init__(self, datasets):
		super().__init__(datasets)
		self.datasets = datasets
		self.concat = ConcatDataset(datasets)
		self.source_ids = self._build_source_index()

	def _build_source_index(self):
		source_ids = []
		for i, dataset in enumerate(self.datasets):
			source_ids.extend([i] * len(dataset))
		return source_ids

	def __len__(self):
		return len(self.concat)

	def __getitem__(self, idx):
		sample = super().__getitem__(idx)
		source_id = self.source_ids[idx]
		if isinstance(sample, tuple):
			sample = {'img_f': sample[0], 'img_r': sample[1]}
		sample["source_id"] = source_id
		return sample
	
	def custom_collate_fn(self, batch):
		img_f = [torch.tensor(sample['img_f']) for sample in batch]
		img_r = [torch.tensor(sample['img_r']) for sample in batch]
		source_ids = [sample['source_id'] for sample in batch]
		data = {}
		data['img'] = torch.cat([torch.stack(img_r).float(), torch.stack(img_f).float()], 0)
		data['label'] = torch.tensor([0]*len(img_r) + [1]*len(img_f), dtype=torch.long)
		data['source_id'] = torch.tensor(source_ids * 2, dtype=torch.long)
		return data

if __name__=='__main__':
	import blend as B
	from initialize import *
	from funcs import IoUfrom2bboxes,crop_face,RandomDownScale
	if exist_bi:
		from library.bi_online_generation import random_get_hull
	seed=10
	random.seed(seed)
	torch.manual_seed(seed)
	np.random.seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	image_dataset=SBI_Dataset(phase='test',image_size=256)
	batch_size=64
	dataloader = torch.utils.data.DataLoader(image_dataset,
					batch_size=batch_size,
					shuffle=True,
					collate_fn=image_dataset.collate_fn,
					num_workers=0,
					worker_init_fn=image_dataset.worker_init_fn
					)
	data_iter=iter(dataloader)
	data=next(data_iter)
	img=data['img']
	img=img.view((-1,3,256,256))
	utils.save_image(img, 'loader.png', nrow=batch_size, normalize=False, range=(0, 1))
else:
	from utils import blend as B
	from .initialize import *
	from .funcs import IoUfrom2bboxes,crop_face,RandomDownScale
	if exist_bi:
		from utils.library.bi_online_generation import random_get_hull


