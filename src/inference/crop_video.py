import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms,models,utils
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from PIL import Image
import sys
import random
import shutil
from model import Detector
import argparse
from datetime import datetime
from tqdm import tqdm
from retinaface.pre_trained_models import get_model
from preprocess import extract_frames
import warnings
import cv2
warnings.filterwarnings('ignore')

def main(args):

    face_detector = get_model("resnet50_2020-07-20", max_size=2048,device=device)
    face_detector.eval()

    face_list,idx_list=extract_frames(args.input_video,args.n_frames,face_detector)

    img = face_list
    output = os.path.join("figures", "crops")
    os.makedirs(os.path.join(output, os.path.splitext(os.path.basename(args.input_video))[0]), exist_ok = True)
    for i in range(len(img)):
        path = os.path.join(output, os.path.splitext(os.path.basename(args.input_video))[0], str(idx_list[i]) + ".png")  
        print(path)
        cv2.imwrite(path, cv2.cvtColor(img[i].transpose(1, 2, 0), cv2.COLOR_RGB2BGR))







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
    parser.add_argument('-i',dest='input_video',type=str)
    parser.add_argument('-n',dest='n_frames',default=32,type=int)
    args=parser.parse_args()

    main(args)

