import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import argparse
from tqdm import tqdm
from retinaface.pre_trained_models import get_model
from preprocess import extract_frames
from datasets import *
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, average_precision_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

import pickle

def main(args):
    if args.dataset == 'FFIW':
        video_list,target_list=init_ffiw()
    elif args.dataset == 'FF':
        video_list,target_list, video_root=init_ff()
    elif args.dataset == 'DFD':
        video_list,target_list=init_dfd()
    elif args.dataset == 'DFDC':
        video_list,target_list=init_dfdc()
    elif args.dataset == 'DFDCP':
        video_list,target_list=init_dfdcp()
    elif args.dataset == 'CDF':
        video_list,target_list, video_root=init_cdf()
    elif args.dataset.upper() == 'GITW':
        video_list, target_list, video_root = init_guy()
    else:
        NotImplementedError

    data_path = os.path.join(video_root, 'video_data.pkl')
    print("------Cropping mode------")
    if (os.path.exists(data_path)):
        print(f"a .pkl file already exists at {data_path}, please check if it was not intended")
    else: 
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
    args=parser.parse_args()

    main(args)