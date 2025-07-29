from glob import glob
import os
import sys
import json
import numpy as np
from PIL import Image
from glob import glob 
import os
import pandas as pd
import re

def init_ff(phase, level='frame',n_frames=8):
	dataset_path='/mnt/ssd_nvme2/datasets/FaceForensics++/sbi/frames/'
	

	image_list=[]
	label_list=[]

	
	
	folder_list = sorted(glob(dataset_path+'*'))
	filelist = []
	list_dict = json.load(open(f'/mnt/ssd_nvme2/datasets/FaceForensics++/original_download/splits/{phase}.json','r'))
	for i in list_dict:
		filelist+=i
	folder_list = [i for i in folder_list if os.path.basename(i)[:3] in filelist]

	if level =='video':
		label_list=[0]*len(folder_list)
		return folder_list,label_list
	for i in range(len(folder_list)):
		# images_temp=sorted([glob(folder_list[i]+'/*.png')[0]])
		images_temp=sorted(glob(folder_list[i]+'/*.png'))
		if n_frames<len(images_temp):
			images_temp=[images_temp[round(i)] for i in np.linspace(0,len(images_temp)-1,n_frames)]
		image_list+=images_temp
		label_list+=[0]*len(images_temp)

	return image_list,label_list


def get_frames (n_frames, images_temp):
	image_candidates = []
	#retina = [image.replace(".png", ".npy").replace("frames", 'retina') for image in images_temp]
	for image in images_temp:
		yunet_path = image.replace(".png", ".npy").replace("frames", "yunet")
		landmarks_path = image.replace(".png", ".npy").replace("frames", "landmarks")
		if os.path.exists(yunet_path) and os.path.exists(landmarks_path):
			image_candidates.append(image)
	if n_frames<len(image_candidates):
		image_candidates =[image_candidates[round(i)] for i in np.linspace(0,len(image_candidates)-1, n_frames)]
	return image_candidates

def init_MSU_MFD(phase, n_frames):
	assert phase in ["train", "val"]
	#Initial dataset info
	dataset_frames_path = '/home/alicia/dataShareID/temp_datasets/MSU-MFD Database/MSU-MFSD/frames'
	train_clients = ['001', '013', '014', '023', '024', '026', '028', '029', '030', '032', '033', '035', '036', '037', '039', '042', '048', '049', '050', '051']
	val_clients = ['002', '003', '005', '006', '007', '008', '009', '011', '012', '021', '022', '034', '053', '054', '055']
	reference_clients = train_clients if phase == "train" else val_clients
	folder_list = []
	image_list = []
	label_list = []
	#Contains all videos that were cut in 32 frames by dlib
	all_video_names = os.listdir(dataset_frames_path)

	for video in all_video_names:
		#Check if video is in correct phase
		is_in_reference = False
		match = re.search(r'real_client(\d{3})_', video)
		if match:
			client_id = match.group(1)
			if client_id in reference_clients:
				is_in_reference = True
		else:
			print(f"Check video name format in MSU MFD, ({video}) does not match")
		video_full_path = os.path.join(dataset_frames_path, video)
		#Check if dlib found landmark
		landmarks_exist = os.path.exists(video_full_path.replace('frames', 'landmarks'))
		#Check if yunet found face
		yunet_exist = os.path.exists(video_full_path.replace('frames', 'yunet'))

		#Build folder_list of folders containing frames to be used
		if is_in_reference and landmarks_exist and yunet_exist:
			folder_list.append(video_full_path)
		
	#For each folder, get all image paths
	for i in range(len(folder_list)):
		images_temp=sorted(glob(folder_list[i]+'/*.png'))
		image_list+=get_frames(n_frames, images_temp)
		label_list+=[0]*len(images_temp)
	return image_list, label_list

def init_replay_attack(phase, n_frames):
	assert phase in ["train", "val"]
	val_names = os.listdir('/home/alicia/dataShareID/temp_datasets/Replay-Attack dataset/train/real')
	train_names = os.listdir('/home/alicia/dataShareID/temp_datasets/Replay-Attack dataset/test/real')
	dataset_frames_path = '/home/alicia/dataShareID/temp_datasets/Replay-Attack dataset/frames'
	reference_names = train_names if phase == 'train' else val_names
	folder_list = []
	image_list = []
	label_list = []
	#Contains all videos that were cut in 32 frames by dlib
	all_video_names = os.listdir(dataset_frames_path)
	for video in all_video_names:
		#Check if video is in correct phase
		is_in_reference = False
		if video in reference_names or video + '.mov' in reference_names:
			is_in_reference = True
		video_full_path = os.path.join(dataset_frames_path, video)
		#Check if dlib found landmark
		landmarks_exist = os.path.exists(video_full_path.replace('frames', 'landmarks'))
		#Check if yunet found face
		yunet_exist = os.path.exists(video_full_path.replace('frames', 'yunet'))
		if is_in_reference and landmarks_exist and yunet_exist:
			folder_list.append(video_full_path)

		#For each folder, get all image paths
	for i in range(len(folder_list)):
		images_temp=sorted(glob(folder_list[i]+'/*.png'))
		image_list+=get_frames(n_frames, images_temp)
		label_list+=[0]*len(images_temp)
	return image_list, label_list

def init_mobio(phase, n_frames):
	f_split_threshold = 414
	m_split_threshold = 431.5
	dataset_frames_path = '/home/alicia/dataShareID/temp_datasets/PHASE2_ALL/frames'
	all_video_names = os.listdir(dataset_frames_path)

	videos = {}

	for name in all_video_names:
		match = re.match(r'([fm])(\d{3})_(\d{2})', name)
		if match:
			x = match.group(1)
			individual_ID = int(match.group(2))
			session = match.group(3)
			video_id = f"{x}{match.group(2)}_{session}"
        # Check train/test split
		if x == 'f':
			if (phase == "train" and individual_ID < f_split_threshold) or (phase != "train" and individual_ID >= f_split_threshold):
				videos.setdefault(video_id, []).append(name)
		elif x == 'm':
			if (phase == "train" and individual_ID < m_split_threshold) or (phase != "train" and individual_ID >= m_split_threshold):
				videos.setdefault(video_id, []).append(name)

	image_list = []
	label_list = []
	#Select 8 frames per session (gXXX_YY) with YY ID of videos taken within the same conditions 
	for session in videos.keys():
		frame_list = []
		for frame in videos[session]:
			frame_dir_path = os.path.join(dataset_frames_path, frame)
			frame_dir_path_content = os.listdir(frame_dir_path)
			#One frame per frame dir at most
			if len(frame_dir_path_content):
				frame_path = os.path.join(frame_dir_path, frame_dir_path_content[0])
				frame_list.append(frame_path)
		image_list+=get_frames(n_frames, frame_list)
		label_list+=[0]*len(image_list)
	return image_list, label_list

def init_sim_mw2(phase, n_frames):
	assert phase in ["train", "val"]
	# mov = [range(1, 104), range(610, 661)]
	# avi = [range(105, 505)]
	# mp4 = [range(661, 892)]
	train = [i for r in [range(1, 84), range(105, 425), range(610, 651), range(661, 846)] for i in r]
	dataset_frames_path = '/home/alicia/dataShareID/temp_datasets/SiW-Mv2/frames'
	folder_list = []
	image_list = []
	label_list = []
	all_video_names = os.listdir(dataset_frames_path)
	#/home/alicia/dataShareID/temp_datasets/SiW-Mv2/frames/Live_776/149.png
	for video in all_video_names:
		is_in_reference = False
		match = re.search(r'Live_(\d{1,3})', video)
		number = int(match.group(1)) if match else None
		if number is None:
			print(f"Check folder, filename unknown: {video}")
			continue
		if phase == "train": 
			is_in_reference = number in train
		else:
			is_in_reference = not number in train
		video_full_path = os.path.join(dataset_frames_path, video)
		#Check if dlib found landmark
		landmarks_exist = os.path.exists(video_full_path.replace('frames', 'landmarks'))
		#Check if yunet found face
		yunet_exist = os.path.exists(video_full_path.replace('frames', 'yunet'))
		if is_in_reference and landmarks_exist and yunet_exist:
			folder_list.append(video_full_path)
	#For each folder, get all image paths
	for i in range(len(folder_list)):
		images_temp=sorted(glob(folder_list[i]+'/*.png'))
		image_list+=get_frames(n_frames, images_temp)
		label_list+=[0]*len(images_temp)
	return image_list, label_list


