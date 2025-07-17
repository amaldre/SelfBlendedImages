from glob import glob
import os
import sys
import json
import numpy as np
from PIL import Image
from glob import glob 
import os
import pandas as pd

T7 = '/media/alicia/T7/ShareID/TestDataSets'
DATASHAREID = "/media/alicia/dataShareID"
CROP_DIR = 'crop_data'
VID_EXTENSIONS = {'.mp4', '.mov', '.avi'}

def init_dataset(dataset):
	if dataset == 'FFIW':
		video_list,target_list=init_ffiw()
	elif dataset == 'FF':
		video_list,target_list, video_root=init_ff()
	elif dataset == 'DFD':
		video_list,target_list=init_dfd()
	elif dataset == 'DFDC':
		video_list,target_list=init_dfdc()
	elif dataset == 'DFDCP':
		video_list,target_list=init_dfdcp()
	elif dataset == 'CDF':
		video_list,target_list, video_root=init_cdf()
	elif dataset.upper() == 'GITW':
		video_list, target_list, video_root = init_guy()
	elif dataset.upper() == 'AKOOL':
		video_list, target_list, video_root = init_akool()
	elif dataset.upper() == 'FFCM_SUBSET':
		video_list, target_list, video_root = init_ffcm_subset()
	elif dataset.upper() == 'IBETA':
		video_list, target_list, video_root = init_ibeta()
	elif dataset.upper() == 'VIDNOZ':
		video_list, target_list, video_root = init_vidnoz()
	elif dataset.upper() == 'ALEXANDRE_MASTER':
		video_list, target_list, video_root = init_alexandre_master()
	elif dataset.upper() == 'ALEXANDRE':
		video_list, target_list, video_root = init_alexandre()
	elif dataset.upper() == 'ALEXANDRE_PRISTINE':
		video_list, target_list, video_root = init_alexandre_pristine()
	elif dataset.upper() == 'TEAM':
		video_list, target_list, video_root = init_team() 
	elif dataset.upper() == 'FAKE_TEAM':
		video_list, target_list, video_root = init_fake_team()
	else:
		print(dataset)
		NotImplementedError
	return video_list, target_list, video_root


def init_ff(dataset='all',phase='test'):
	video_root = "FaceForensics++"
	assert dataset in ['all','Deepfakes','Face2Face','FaceSwap','NeuralTextures']
	original_path='/datasets/FaceForensics++/original_download/original_sequences/youtube/'
	folder_list = sorted(glob(original_path+'*'))

	list_dict = json.load(open(f'/datasets/FaceForensics++/original_download/splits/{phase}.json','r'))
	filelist=[]
	for i in list_dict:
		filelist+=i
	image_list = [i for i in folder_list if os.path.basename(i)[:3] in filelist]
	label_list=[0]*len(image_list)


	if dataset=='all':
		fakes=['Deepfakes','Face2Face','FaceSwap','NeuralTextures']
	else:
		fakes=[dataset]

	folder_list=[]
	for fake in fakes:
		fake_path=f'/datasets/FaceForensics++/original_download/manipulated_sequences/{fake}/'
		folder_list_all=sorted(glob(fake_path+'*'))
		folder_list+=[i for i in folder_list_all if os.path.basename(i)[:3] in filelist]
	label_list+=[1]*len(folder_list)
	image_list+=folder_list
	print(len(image_list))
	return image_list,label_list,video_root



def init_dfd():
	real_path='data/FaceForensics++/original_sequences/actors/raw/videos/*.mp4'
	real_videos=sorted(glob(real_path))
	fake_path='data/FaceForensics++/manipulated_sequences/DeepFakeDetection/raw/videos/*.mp4'
	fake_videos=sorted(glob(fake_path))

	label_list=[0]*len(real_videos)+[1]*len(fake_videos)

	image_list=real_videos+fake_videos

	return image_list,label_list


def init_dfdc():
		
	label=pd.read_csv('data/DFDC/labels.csv',delimiter=',')
	folder_list=[f'data/DFDC/videos/{i}' for i in label['filename'].tolist()]
	label_list=label['label'].tolist()
	
	return folder_list,label_list


def init_dfdcp(phase='test'):

	phase_integrated={'train':'train','val':'train','test':'test'}

	with open('data/DFDCP/dataset.json') as f:
		df=json.load(f)
	fol_lab_list_all=[[f"data/DFDCP/{k.split('/')[0]}/videos/{k.split('/')[-1]}",df[k]['label']=='fake'] for k in df if df[k]['set']==phase_integrated[phase]]
	name2lab={os.path.basename(fol_lab_list_all[i][0]):fol_lab_list_all[i][1] for i in range(len(fol_lab_list_all))}
	fol_list_all=[f[0] for f in fol_lab_list_all]
	fol_list_all=[os.path.basename(p)for p in fol_list_all]
	folder_list=glob('data/DFDCP/method_*/videos/*/*/*.mp4')+glob('data/DFDCP/original_videos/videos/*/*.mp4')
	folder_list=[p for p in folder_list if os.path.basename(p) in fol_list_all]
	label_list=[name2lab[os.path.basename(p)] for p in folder_list]
	

	return folder_list,label_list




def init_ffiw():
	# assert dataset in ['real','fake']
	path='data/FFIW/FFIW10K-v1-release/'
	folder_list=sorted(glob(path+'source/val/videos/*.mp4'))+sorted(glob(path+'target/val/videos/*.mp4'))
	label_list=[0]*250+[1]*250
	return folder_list,label_list



def init_cdf():

	label_list=[]

	video_list_txt=os.path.join(DATASHAREID, 'CelebDFv2/List_of_testing_videos.txt')
	video_root = 'CelebDFv2'
	with open(video_list_txt) as f:
		
		folder_list=[]
		for data in f:
			# print(data)
			line=data.split()
			# print(line)
			path=line[1].split('/')
			folder_list+=['/home/alicia/dataShareID/CelebDFv2/'+path[0]+'/'+path[1]]
			label_list+=[1-int(line[0])]
		return folder_list, label_list, video_root

def read_custom_data(dataset_name, txt_name = 'List_of_testing_videos.txt', base_data = DATASHAREID):
	folder_list = []
	label_list = []
	with open(os.path.join(base_data, dataset_name, txt_name)) as f:
		for data in f:
			line = data.split()
			path = line[1]
			folder_list += [os.path.join(base_data, dataset_name, path)]
			label_list += [int(line[0])]
	return folder_list, label_list

def init_alexandre_master():
	video_root = 'alexandre_master'
	folder_list, label_list = read_custom_data(video_root, base_data = T7)
	print(len(label_list))
	return folder_list, label_list, video_root
def init_guy():
	video_root = 'GitW'
	folder_list, label_list = read_custom_data(video_root)
	print(len(label_list))
	return folder_list, label_list, video_root

def init_akool():
	video_root = 'akool'
	folder_list, label_list = read_custom_data(video_root, 'List_of_testing_videos_akool.txt')
	print(len(label_list))
	return folder_list, label_list, video_root

def init_ffcm_subset():
	video_root = 'ffcm_subset_100'
	folder_list, label_list = read_custom_data(video_root)
	print(len(label_list))
	return folder_list, label_list, video_root

def init_ibeta():
	video_root = 'ibeta'
	folder_list, label_list = read_custom_data(video_root, "List_of_testing_videos_ibeta.txt")
	print(len(label_list))
	return folder_list, label_list, video_root

def init_vidnoz():
	video_root = 'vidnoz'
	folder_list, label_list = read_custom_data(video_root, "List_of_testing_videos_vidnoz.txt")
	print(len(label_list))
	return folder_list, label_list, video_root

def init_custom_folder(video_root, label):
	folder_path = os.path.join(DATASHAREID, video_root)
	folder_list = [os.path.join(folder_path, video) for video in os.listdir(folder_path) if os.path.splitext(video)[1].lower() in VID_EXTENSIONS]
	label_list = [label] * len(folder_list)
	return folder_list, label_list, video_root

def init_alexandre():
	video_root = 'ShareIdFake/output'
	return init_custom_folder(video_root, 1)

def init_alexandre_pristine():
	video_root = 'alexandre_pristine'
	return init_custom_folder(video_root, 0)

def init_team():
	video_root = 'ShareIDTeam'
	return init_custom_folder(video_root, 0)

def init_fake_team():
	video_root = 'ShareIDFake/output'
	return init_custom_folder(video_root, 1)
#91
#alexandre| AUC: N/A (only one label), Accuracy: 0.0909, Avg Precision:  N/A (only one label), Avg Recall:  N/A (only one label)