from faceDetectionModule import FaceDetectionModule
from glob import glob
import os
import cv2
from tqdm import tqdm
import numpy as np
import argparse
import torch

VIDEO_EXT = {".mp4", ".mov", ".avi"}

def find_movies(folder, list):
    for file in os.listdir(folder):
        continue 

def get_movies_path(dataset_name):
    #movies_path should contain videos
    if dataset_name =='Original':
        movies_path = '/mnt/ssd_nvme2/datasets/FaceForensics++/original_download/original_sequences'
        frames_path = '/mnt/ssd_nvme2/datasets/FaceForensics++/sbi/frames'
    elif dataset_name.upper() == 'MSU-MFD':
        movies_path = ''
        frames_path = '/home/alicia/SelfBlendedImages/msu-mfsd/frames'
    elif dataset_name.upper() == 'SIM-MW2':
        movies_path = ''
        frames_path = '/home/alicia/SelfBlendedImages/sim-mw2/frames'
    elif dataset_name.upper() == 'REPLAY-ATTACK':
        movies_path = ''
        frames_path = '/home/alicia/SelfBlendedImages/replay-attack/frames'
    elif dataset_name.upper() == 'MOBIO':
        movies_path = ''
        frames_path = '/home/alicia/SelfBlendedImages/mobio/frames'
    else:
        raise NotImplementedError
    return movies_path, frames_path

def main():
    counter = 0
    parser=argparse.ArgumentParser()
    parser.add_argument('-d',dest='dataset')
    parser.add_argument('-c',dest='comp',choices=['raw','c23','c40'],default='raw')
    parser.add_argument('-n',dest='num_frames',type=int,default=32)
    args=parser.parse_args()

    device = torch.device('cuda')

    model = FaceDetectionModule(
            '/home/alicia/SelfBlendedImages/src/preprocess/yunet_face/1/model.onnx',
            score_threshold=0.7,
        )
    #Goal open each video, save the bbox as .npy
    movies_path, frame_path = get_movies_path(args.dataset)

    video_names = os.listdir(frame_path)
    print(f"Found {len(video_names)} videos to process")
    empty = 0
    for i in tqdm(range(len(video_names))):
        if len(os.listdir(os.path.join(frame_path,video_names[i]))):
            counter += save_bbox(model, os.path.join(frame_path,video_names[i]))
        else:
            empty += 1

    print(f"{counter} frames unprocessed due to errors")
    print(f"{empty} empty folders.")

def save_bbox(model, frame_folder):
    counter = 0
    for frame in os.listdir(frame_folder):
        try:
            img_path = os.path.join(frame_folder, frame)
            img = cv2.imread(img_path)
            face_bbox_corners, landmarks  = model.detect(img, scale = 1)
            bbox_points_2d = np.array([
                [face_bbox_corners[0], face_bbox_corners[1]], # Top-left (x0, y0)
                [face_bbox_corners[2], face_bbox_corners[3]]  # Bottom-right (x1, y1)
            ], dtype=np.int32) # Ensure dtype matches your detect function's output
            landmarks = np.array(landmarks, dtype=np.int32).reshape((5, 2))  # ✅
            combined_face_data = np.vstack((bbox_points_2d, landmarks))
            yunet_directory = os.path.dirname(img_path.replace('frames', 'yunet'))
            img_name = os.path.basename(img_path)
            os.makedirs(yunet_directory, exist_ok= True)
            np.save(os.path.join(yunet_directory, os.path.splitext(img_name)[0] + '.npy'), combined_face_data)
        except Exception as e:
            counter = counter + 1
            print(e)
    return counter


if __name__ == '__main__':
    main()