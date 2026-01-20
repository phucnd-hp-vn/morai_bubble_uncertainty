# loading in and transforming data
import os
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader,ConcatDataset
from torch.autograd import Variable

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

#from skimage import io, transform
from PIL import Image
import cv2

# visualizing data
import matplotlib.pyplot as plt
import numpy as np
import warnings

# load dataset information
import yaml

import json
# image writing
# import imageio
# from skimage import img_as_ubyte

# Clear GPU cache
torch.cuda.empty_cache()

import argparse
from pathlib import Path

parser = argparse.ArgumentParser("ESFPNet based model - Batch Processing")
parser.add_argument('--input_root', type=str, required=True,
        help='Root directory containing all the video frame folders')
parser.add_argument('--output_root', type=str, required=True,
        help='Root directory to save mask results with same structure')
parser.add_argument('--model_type', type=str, default='B4',
        help='Type of model (default B4)')
parser.add_argument('--init_trainsize', type=int, default=480,
        help='Size of image for training (default = 480)')
parser.add_argument('--saved_model', type=str, required=True,
        help='Load saved baseline model') 
parser.add_argument('--batch_size', type=int, default=40,
        help='Batch size for processing (default = 8)')

args = parser.parse_args()

class test_dataset:
    def __init__(self, image_root, testsize): #
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        name_ = self.images[self.index].split('/')[-1]
        self.index += 1
        return image, name_

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
        
from collections import OrderedDict
import copy

from Encoder import mit
from Decoder import mlp
from mmcv.cnn import ConvModule

class ESFPNetStructure(nn.Module):

    def __init__(self, embedding_dim = 160):
        super(ESFPNetStructure, self).__init__()

        # Backbone
        if args.model_type == 'B0':
            self.backbone = mit.mit_b0()
        if args.model_type == 'B1':
            self.backbone = mit.mit_b1()
        if args.model_type == 'B2':
            self.backbone = mit.mit_b2()
        if args.model_type == 'B3':
            self.backbone = mit.mit_b3()
        if args.model_type == 'B4':
            self.backbone = mit.mit_b4()
        if args.model_type == 'B5':
            self.backbone = mit.mit_b5()

        self._init_weights()  # load pretrain

        # LP Header
        self.LP_1 = mlp.LP(input_dim = self.backbone.embed_dims[0], embed_dim = self.backbone.embed_dims[0])
        self.LP_2 = mlp.LP(input_dim = self.backbone.embed_dims[1], embed_dim = self.backbone.embed_dims[1])
        self.LP_3 = mlp.LP(input_dim = self.backbone.embed_dims[2], embed_dim = self.backbone.embed_dims[2])
        self.LP_4 = mlp.LP(input_dim = self.backbone.embed_dims[3], embed_dim = self.backbone.embed_dims[3])

        # Linear Fuse
        self.linear_fuse34 = ConvModule(in_channels=(self.backbone.embed_dims[2] + self.backbone.embed_dims[3]), out_channels=self.backbone.embed_dims[2], kernel_size=1,norm_cfg=dict(type='BN', requires_grad=True))
        self.linear_fuse23 = ConvModule(in_channels=(self.backbone.embed_dims[1] + self.backbone.embed_dims[2]), out_channels=self.backbone.embed_dims[1], kernel_size=1,norm_cfg=dict(type='BN', requires_grad=True))
        self.linear_fuse12 = ConvModule(in_channels=(self.backbone.embed_dims[0] + self.backbone.embed_dims[1]), out_channels=self.backbone.embed_dims[0], kernel_size=1,norm_cfg=dict(type='BN', requires_grad=True))

        # Fused LP Header
        self.LP_12 = mlp.LP(input_dim = self.backbone.embed_dims[0], embed_dim = self.backbone.embed_dims[0])
        self.LP_23 = mlp.LP(input_dim = self.backbone.embed_dims[1], embed_dim = self.backbone.embed_dims[1])
        self.LP_34 = mlp.LP(input_dim = self.backbone.embed_dims[2], embed_dim = self.backbone.embed_dims[2])

        # Final Linear Prediction
        self.linear_pred = nn.Conv2d((self.backbone.embed_dims[0] + self.backbone.embed_dims[1] + self.backbone.embed_dims[2] + self.backbone.embed_dims[3]), 1, kernel_size=1)

        #classification layer
        self.norm1 = nn.BatchNorm2d(512, eps=1e-5)
        self.Relu = nn.ReLU(inplace=True)
        self.Dropout = nn.Dropout(p=0.3)
        self.conv1 = nn.Conv2d(512, 256, 1, stride=1, padding=0)
        self.norm2 = nn.BatchNorm2d(256, eps=1e-5)
        self.conv2 = nn.Conv2d(256, 3, 1, stride=1, padding=0, bias=True) # 9 = number of classes
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.softmax = nn.Softmax(dim = 1)

    def _init_weights(self):

        if args.model_type == 'B0':
            pretrained_dict = torch.load('./Pretrained/mit_b0.pth')
        if args.model_type == 'B1':
            pretrained_dict = torch.load('./Pretrained/mit_b1.pth')
        if args.model_type == 'B2':
            pretrained_dict = torch.load('./Pretrained/mit_b2.pth')
        if args.model_type == 'B3':
            pretrained_dict = torch.load('./Pretrained/mit_b3.pth')
        if args.model_type == 'B4':
            pretrained_dict = torch.load('./Pretrained/mit_b4.pth')
        if args.model_type == 'B5':
            pretrained_dict = torch.load('./Pretrained/mit_b5.pth')


        model_dict = self.backbone.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.backbone.load_state_dict(model_dict)
        print("successfully loaded!!!!")


    def forward(self, x):

        ##################  Go through backbone ###################

        B = x.shape[0]

        #stage 1
        out_1, H, W = self.backbone.patch_embed1(x)
        for i, blk in enumerate(self.backbone.block1):
            out_1 = blk(out_1, H, W)
        out_1 = self.backbone.norm1(out_1)
        out_1 = out_1.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  #(Batch_Size, self.backbone.embed_dims[0], 88, 88)

        # stage 2
        out_2, H, W = self.backbone.patch_embed2(out_1)
        for i, blk in enumerate(self.backbone.block2):
            out_2 = blk(out_2, H, W)
        out_2 = self.backbone.norm2(out_2)
        out_2 = out_2.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  #(Batch_Size, self.backbone.embed_dims[1], 44, 44)

        # stage 3
        out_3, H, W = self.backbone.patch_embed3(out_2)
        for i, blk in enumerate(self.backbone.block3):
            out_3 = blk(out_3, H, W)
        out_3 = self.backbone.norm3(out_3)
        out_3 = out_3.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  #(Batch_Size, self.backbone.embed_dims[2], 22, 22)

        # stage 4
        out_4, H, W = self.backbone.patch_embed4(out_3)
        for i, blk in enumerate(self.backbone.block4):
            out_4 = blk(out_4, H, W)
        out_4 = self.backbone.norm4(out_4)
        out_4 = out_4.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  #(Batch_Size, self.backbone.embed_dims[3], 11, 11)


        #segmentation
        # go through LP Header
        lp_1 = self.LP_1(out_1)
        lp_2 = self.LP_2(out_2)
        lp_3 = self.LP_3(out_3)
        lp_4 = self.LP_4(out_4)

        # linear fuse and go pass LP Header
        lp_34 = self.LP_34(self.linear_fuse34(torch.cat([lp_3, F.interpolate(lp_4,scale_factor=2,mode='bilinear', align_corners=False)], dim=1)))
        lp_23 = self.LP_23(self.linear_fuse23(torch.cat([lp_2, F.interpolate(lp_34,scale_factor=2,mode='bilinear', align_corners=False)], dim=1)))
        lp_12 = self.LP_12(self.linear_fuse12(torch.cat([lp_1, F.interpolate(lp_23,scale_factor=2,mode='bilinear', align_corners=False)], dim=1)))

        # get the final output
        lp4_resized = F.interpolate(lp_4,scale_factor=8,mode='bilinear', align_corners=False)
        lp3_resized = F.interpolate(lp_34,scale_factor=4,mode='bilinear', align_corners=False)
        lp2_resized = F.interpolate(lp_23,scale_factor=2,mode='bilinear', align_corners=False)
        lp1_resized = lp_12

        out1 = self.linear_pred(torch.cat([lp1_resized, lp2_resized, lp3_resized, lp4_resized], dim=1))
        
        # Classification
        out2 = self.global_avg_pool(out_4)
        out2 = self.norm1(out2)
        out2 = self.Relu(out2)
        out2 = self.Dropout(out2)
        out2 = self.conv1(out2)
        out2 = self.norm2(out2)
        out2 = self.Relu(out2)
        out2 = self.conv2(out2)
        out2 = out2.view(B, -1)  # Flatten to [B, 3]

        return out1, out2

def find_all_image_folders(root_dir):
    """Find all folders that contain image files"""
    image_folders = []
    
    for root, dirs, files in os.walk(root_dir):
        # Check if this directory contains image files
        image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if image_files:
            image_folders.append(root)
    
    return image_folders

def process_single_folder(input_folder, output_folder, model, device):
    """Process all images in a single folder using batch processing"""
    print(f"Processing folder: {input_folder}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all image files in the folder
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    image_files = sorted(image_files)
    
    if not image_files:
        print(f"No image files found in {input_folder}")
        return
    
    print(f"Found {len(image_files)} images in {input_folder}")

    # Check which images need processing by comparing with output folder
    existing_masks = set(os.listdir(output_folder))
    image_files_to_process = [f for f in image_files if f not in existing_masks]
    
    if not image_files_to_process:
        print(f"All images in {input_folder} have already been processed. Skipping...")
        return
    
    print(f"Skipping {len(image_files) - len(image_files_to_process)} already processed images")

    
    
    # Transform for preprocessing
    transform = transforms.Compose([
        transforms.Resize((args.init_trainsize, args.init_trainsize)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Process images in batches
    for i in range(0, len(image_files_to_process), args.batch_size):
        batch_files = image_files_to_process[i:i + args.batch_size]
        batch_images = []
        batch_names = []
        
        # Load and preprocess batch
        for image_file in batch_files:
            try:
                image_path = os.path.join(input_folder, image_file)
                image = Image.open(image_path).convert('RGB')
                image_tensor = transform(image)
                batch_images.append(image_tensor)
                batch_names.append(image_file)
            except Exception as e:
                print(f"Error loading {image_file}: {str(e)}")
                continue
        
        if not batch_images:
            continue
        
        # Stack images into batch tensor
        batch_tensor = torch.stack(batch_images).to(device)
        
        # Inference on batch
        with torch.no_grad():
            pred1, pred2 = model(batch_tensor)

        # print("pred2", pred2.shape, pred2)
        
        # Post-process predictions
        pred1 = F.interpolate(pred1, size=(args.init_trainsize, args.init_trainsize), mode='bilinear', align_corners=False)
        pred1 = (pred1.sigmoid() > 0.5).float().cpu().numpy()
        
        # Save each prediction in the batch
        for j, (pred_mask, pred_class, image_file) in enumerate(zip(pred1, pred2, batch_names)):
            try:
                # Normalize mask
                pred_mask_np = pred_mask.squeeze()
                pred_mask_np = (pred_mask_np - pred_mask_np.min()) / (pred_mask_np.max() - pred_mask_np.min() + 1e-8)

                # Save class
                class_save_path = os.path.join(output_folder, image_file.replace('.jpg', '.txt').replace('.png', '.txt'))
                np.savetxt(class_save_path, pred_class.cpu().numpy(), fmt="%.6f")
                
                # Save mask
                mask_save_path = os.path.join(output_folder, image_file)
                cv2.imwrite(mask_save_path, (pred_mask_np * 255).astype(np.uint8))
                
            except Exception as e:
                print(f"Error saving {image_file}: {str(e)}")
                continue
        
        # Progress update
        processed_count = min(i + args.batch_size, len(image_files))
        if processed_count % (args.batch_size * 5) == 0 or processed_count == len(image_files):
            print(f"Processed {processed_count}/{len(image_files)} images in {input_folder}")
    
    print(f"Completed processing {input_folder}")

def main():
    # Set up device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print('Models moved to GPU.')
    else:
        device = torch.device("cpu")
        print('Using CPU.')
    
    # Load model
    print("Loading model...")
    ESFPNet = torch.load(args.saved_model)
    ESFPNet.eval()
    ESFPNet = ESFPNet.to(device)
    print("Model loaded successfully!")
    
    # Find all folders with images
    print(f"Scanning for image folders in: {args.input_root}")
    image_folders = find_all_image_folders(args.input_root)
    print(f"Found {len(image_folders)} folders with images")
    
    # Process each folder
    for i, input_folder in enumerate(image_folders):
        # Create corresponding output folder path
        relative_path = os.path.relpath(input_folder, args.input_root)
        output_folder = os.path.join(args.output_root, relative_path)
        
        print(f"\n[{i+1}/{len(image_folders)}] Processing: {relative_path}")
        
        # Process the folder
        process_single_folder(input_folder, output_folder, ESFPNet, device)
    
    print(f"\nAll processing completed! Results saved to: {args.output_root}")

if __name__ == "__main__":
    main() 
