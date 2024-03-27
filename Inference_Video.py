import os.path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Resize, ToTensor, Compose, Normalize
from torchvision.models import resnet34, ResNet34_Weights
from tqdm.autonotebook import tqdm
import argparse
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import cv2
import warnings
warnings.filterwarnings("ignore")


def get_args():
    parser = argparse.ArgumentParser(description="Train_modelCNN")
    parser.add_argument("--video_path", "-p", type=str, default="video_test.mp4")
    parser.add_argument("--output_path", "-o", type=str, default="out_video.mp4")
    parser.add_argument("--image_size", "-i", type=int, default=224)
    parser.add_argument("--checkpoint_path", "-c", type = str, default=r"D:\AI\Project\DL\trained_models\animals")
    args = parser.parse_args()
    
    return args 

def inference(args):
    classes = ["butterfly", "cat", "chicken", "cow", "dog", "elephant", "horse", "sheep", "spider", "squirrel"]
    device = torch.deviec("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet34()
    model.fc = nn.Linear(in_features= 512, out_features= len(classes))
    checkpoint = torch.load(os.path.join(args.checkpoint_path, "best.pt"))
    model.to(device)
    model.eval()
    cap = cv2.VideoCapture(args.video_path) # load video
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    out_video = cv2.VideoWriter(args.output_path, cv2.VideoWriter_fourcc(*"MJPG"), int(cap.get(cv2.CAP_PROP_FPS)), (width, height))
    
    softmax = nn.Softmax()
    while cap.isOpened():
        flag, frame =  cap.read()
        if not flag: 
            break
        image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        iamge = cv2.resize(args.image_size, args.image_size)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        #equivalent to Normalize()
        image = (image - mean/ std)
        image = np.transpose(iamge, (2, 0, 1))[None, :, :, :]
        image = torch.from_numpy(image).float().to(device)
        with torch.no_grad():
            output = model(image)
            prob = softmax(output)
            predicted_prob, predicted_class = torch.max(prob, dim=1)
            score = predicted_prob[0]*100
            image = cv2.putText(frame, "{}:{:0.2f}%".format(classes[predicted_class[0]], score), (50,50), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 0, 255), 2, cv2.LINE_AA)
            out_video.write(image)
    cap.release()
    out_video.release()


if __name__ == '__main__':
    args = get_args()
    inference(args)
