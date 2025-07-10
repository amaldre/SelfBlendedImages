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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.sbi import get_final_transforms

def main(args):

    final_transforms = get_final_transforms()

    model = Detector().to(device)
    cnn_sd = torch.load(args.weight_name)["model"]
    model.load_state_dict(cnn_sd)
    model.eval()

    face_detector = get_model("resnet50_2020-07-20", max_size=2048, device=device)
    face_detector.eval()

    # Already returns cropped face images and frame indices
    face_list, idx_list = extract_frames(args.input_video, args.n_frames, face_detector)

    # Infer
    with torch.no_grad():
        img_tensor = torch.tensor(face_list).to(device).float() / 255
        for i in range(img_tensor.shape[0]):
            img_tensor[i] = final_transforms(img_tensor[i])
        pred = model(img_tensor).softmax(1)[:, 1]  # Probability for class 1

    # Prepare output folder
    video_name = os.path.splitext(os.path.basename(args.input_video))[0]
    output_dir = os.path.join("figures", "crops", video_name)
    os.makedirs(output_dir, exist_ok=True)
    # Save each cropped face with prediction written
    for i in range(len(face_list)):
        img = face_list[i]  # Shape: (C, H, W)

        img_cv = img.transpose(1, 2, 0)  # (H, W, C)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

        # Prediction score
        score = pred[i].item()
        label = f"Pred: {score:.4f}"

        # Put text
        position = (10, 30)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = (0, 255, 0)
        thickness = 2

        cv2.putText(img_cv, label, position, font, font_scale, color, thickness, cv2.LINE_AA)

        # Save
        save_path = os.path.join(output_dir, f"{idx_list[i]}.png")
        cv2.imwrite(save_path, img_cv)
        print(f"Saved: {save_path}")

    # Compute mean score
    mean_pred = pred.mean().item()
    print(f"Video fakeness score (mean): {mean_pred:.4f}")






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
    parser.add_argument('-w',dest='weight_name',type=str)
    parser.add_argument('-i',dest='input_video',type=str)
    parser.add_argument('-n',dest='n_frames',default=32,type=int)
    parser.add_argument('-f', action = 'store_true', dest='save_figure', default = False)
    args=parser.parse_args()

    main(args)

