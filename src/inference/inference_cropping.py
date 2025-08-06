import os
import torch
import numpy as np
import cv2
import os
import random
import argparse
from tqdm import tqdm

from preprocess import extract_frames
from datasets import *
from datasets import CROP_DIR
import warnings
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.append(PROJECT_ROOT)
from src.preprocess.faceDetectionModule import FaceDetectionModule
from src.inference.preprocess import crop_face
warnings.filterwarnings('ignore')

import pickle

def crop_retina(video_list, data_path):  
    from retinaface.pre_trained_models import get_model
    face_detector = get_model("resnet50_2020-07-20", max_size=2048,device=device)
    face_detector.eval()
    video_data = {}
    for filename in tqdm(video_list):
        video_data[filename] = {}
        face_list,idx_list=extract_frames(filename,args.n_frames,face_detector)
        video_data[filename]['face_list'] = face_list
        video_data[filename]['idx_list'] = idx_list
        
    with open(data_path, 'wb') as f:
        pickle.dump(video_data, f)
    print(f"Saved faces and videos indexes at {data_path}")
        
def crop_yunet(video_list, data_path):
    model = FaceDetectionModule(
            '/home/alicia/SelfBlendedImages/src/preprocess/yunet_face/1/model.onnx',
            score_threshold=0.7,
        )
    video_data = {}
    for filename in tqdm(video_list):
        video_data[filename] = {}
        cap_org = cv2.VideoCapture(filename)
	
        if not cap_org.isOpened():
            print(f'Cannot open: {filename}')
            return []
        croppedfaces=[]
        idx_list=[]
        frame_count_org = int(cap_org.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idxs = np.linspace(0, frame_count_org - 1, args.n_frames, endpoint=True, dtype=int)
        for cnt_frame in range(frame_count_org): 
            ret_org, frame_org = cap_org.read()
            if not ret_org:
                tqdm.write('Frame read {} Error! : {}'.format(cnt_frame,os.path.basename(filename)))
                break
            height,width=frame_org.shape[:-1]
            if cnt_frame not in frame_idxs:
                continue
            
            faces, landmarks = model.detect_multiple(frame_org, scale = 1)
            frame_rgb = cv2.cvtColor(frame_org, cv2.COLOR_BGR2RGB)
            try:
                if faces is None:
                    tqdm.write('No faces in {}:{}'.format(cnt_frame,os.path.basename(filename)))
                    continue
                size_list=[]
                croppedfaces_temp=[]
                idx_list_temp=[]
                # size_list=[]
                # croppedfaces = []
                # idx_list=[]
                for box in faces:
                    x0, y0, x1, y1 = box
                    # x0,y0,x1,y1=faces
                    bbox=np.array([[x0,y0],[x1,y1]])
                    croppedfaces_temp.append(cv2.resize(crop_face(frame_rgb,None,bbox,False,crop_by_bbox=True,only_img=True,phase='test'),dsize=(380, 380)).transpose((2,0,1)))
                    idx_list_temp.append(cnt_frame)
                    size_list.append((x1-x0)*(y1-y0))
                max_size=max(size_list)
                croppedfaces_temp=[f for face_idx,f in enumerate(croppedfaces_temp) if size_list[face_idx]>=max_size/2]
                idx_list_temp=[f for face_idx,f in enumerate(idx_list_temp) if size_list[face_idx]>=max_size/2]
                croppedfaces+=croppedfaces_temp
                idx_list+=idx_list_temp	
            except Exception as e:
                print(f'error in {cnt_frame}:{filename}')
                print(e)
                continue
        cap_org.release()
        video_data[filename]['face_list'] = croppedfaces
        video_data[filename]['idx_list'] = idx_list
    with open(data_path, 'wb') as f:
        pickle.dump(video_data, f)
    print(f"Saved faces and videos indexes at {data_path}")



def main(args):

    video_list, _, video_root = init_dataset(args.dataset)
    data_dir = os.path.join(CROP_DIR, video_root)
    data_path = os.path.join(data_dir, 'video_data.pkl' if args.model == 'retina' else 'video_data_yunet.pkl')
    print("------Cropping mode------")
    if (os.path.exists(data_path)):
        print(f"a .pkl file already exists at {data_path}, please check if it was not intended")
    else:
        os.makedirs(data_dir, exist_ok = True)
        if args.model == 'retina':
            print("Using retina to crop faces")
            crop_retina(video_list, data_path)
        else:
            print("Using ShareID\'s yunet")
            crop_yunet(video_list, data_path)

if __name__=='__main__':

    seed=1
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda')

    parser=argparse.ArgumentParser()
    parser.add_argument('-d',dest='dataset',type=str)
    parser.add_argument('-n',dest='n_frames',default=32,type=int)
    parser.add_argument('-model', dest = 'model', default = 'retina')
    args=parser.parse_args()

    main(args)