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
from datasets import CROP_DIR
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, average_precision_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

import pickle

def main(args):

    video_list, _, video_root = init_dataset(args.dataset)
    data_dir = os.path.join(CROP_DIR, video_root)
    data_path = os.path.join(data_dir, 'video_data.pkl')
    print("------Cropping mode------")
    if (os.path.exists(data_path)):
        print(f"a .pkl file already exists at {data_path}, please check if it was not intended")
    else:
        os.mkdir(data_dir) 
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