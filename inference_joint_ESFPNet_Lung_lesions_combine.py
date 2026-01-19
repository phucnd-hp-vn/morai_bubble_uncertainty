# loading in and transforming data
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader,ConcatDataset
from torch.autograd import Variable

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import csv
import cv2

#from skimage import io, transform
from PIL import Image

# visualizing data
import matplotlib.pyplot as plt
import numpy as np
import warnings

# load dataset information
import yaml
import pandas as pd

import json
# image writing
import imageio
from skimage import img_as_ubyte

# Clear GPU cache
torch.cuda.empty_cache()

import argparse
from pathlib import Path

parser = argparse.ArgumentParser("ESFPNet based model")
parser.add_argument('--label_json_path', type=str, required=True,
        help='Location of the data directory containing json labels file')
parser.add_argument('--dataset', type=str, required=True,
        help='Location of the dataset for each tasks')
parser.add_argument('--model_type', type=str, default='B4',
        help='Type of model (default B4)')
parser.add_argument('--init_trainsize', type=int, default=352,
        help='Size of image for training (default = 352)')
parser.add_argument('--saved_model', type=str, required=True,
        help='Load saved model')
parser.add_argument('--bubbles_model_path', type=str, required=True,
        help='Path to the trained Bubble segmentation model.')
# parser.add_argument('--log_dir', type=str, required=True,
#         help='Save inference outputs')
args = parser.parse_args()

test_images_path  = args.dataset + '/imgs' 
test_masks_path   = args.dataset + '/masks'

class test_dataset:
    def __init__(self, image_root, gt_root, label_root, testsize):
        self.testsize = testsize
        self.images = [os.path.join(image_root, f) for f in os.listdir(image_root) if f.endswith(('.jpg', '.png'))]
        self.gts = [os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith(('.tif', '.png', '.jpg'))]
        self.labels = label_root
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)

        gt = self.binary_loader(self.gts[self.index])

        file_name = os.path.splitext(os.path.basename(self.images[self.index]))[0]

        with open(self.labels, 'r') as f:
            data = json.load(f)

        label_list = ['Binh thuong', 'Lanh tinh', 'Ac tinh']
        label_name = [file['label_name'] for file in data if file['object_id'] == file_name]

        # Prioritized label selection
        if len(label_name) == 0:
            raise ValueError(f"No label found for image {file_name}")
        elif len(label_name) > 1:
            print(label_name)
            if 'Ac tinh' in label_name:
                selected_label = 'Ac tinh'
                print(f"Warning: Multiple labels found for {file_name}, prioritizing 'Ac tinh'")
            else:
                selected_label = label_name[0]
                print(f"Warning: Multiple labels found for {file_name}, using the first one")
        else:
            selected_label = label_name[0]

        # Convert to class index
        label_tensor = torch.tensor(label_list.index(selected_label))

        self.index += 1
        return image, gt, label_tensor, file_name

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
        # print(out.shape)

        #classification
        out2 = self.global_avg_pool(out_4)
        out2 = self.norm1(out2)
        out2 = self.Relu(out2)
        out2 = self.Dropout(out2)
        out2 = self.conv1(out2)
        out2 = self.norm2(out2)
        out2 = self.Relu(out2)
        out2 = self.conv2(out2)
        out2 = out2.view(B, -1)
        return out1, out2
    
class ESFPNetStructureBubble(nn.Module):

    def __init__(self, embedding_dim = 160):
        super(ESFPNetStructureBubble, self).__init__()

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
        self.conv2 = nn.Conv2d(256, 2, 1, stride=1, padding=0, bias=True) # 9 = number of classes
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
        
        return out1
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

from torchmetrics.classification import MultilabelF1Score, MultilabelRecall, MultilabelPrecision, MultilabelConfusionMatrix
from torchmetrics import ConfusionMatrix

def evaluate_with_bubble_and_metrics(
    test_images_path: str,
    test_masks_path: str,
    label_json_path: str = None,
    threshold_mask: float = 0.5,
    coverage_threshold_save: float = 0.10,   # chỉ để in ra; KHÔNG lưu file
    coverage_threshold_override: float = 0.3
):
    ESFPNet = torch.load(args.saved_model, map_location=device)
    ESFPNet.eval()

    bubbles = ESFPNetStructureBubble()
    bubbles.load_state_dict(torch.load(
        args.bubbles_model_path, map_location=device
    ))
    bubbles.eval()
    bubbles = bubbles.to(device)

    num_classes = 3
    smooth = 1e-4
    label_list = ['Binh thuong', 'Lanh tinh', 'Ac tinh']

    val_loader = test_dataset(
        test_images_path + '/',
        test_masks_path + '/',
        label_json_path if label_json_path is not None else args.label_json_path,
        args.init_trainsize
    )

    total = 0
    correct_predictions = 0
    val_dice = 0.0
    count_dice = 0

    from torchmetrics import ConfusionMatrix
    confusion_matrix_metric = ConfusionMatrix(num_classes=num_classes, task="multiclass").to(device)
    confusion_matrix_total = torch.zeros(num_classes, num_classes, device=device)

    per_image_results = []

    with torch.no_grad():
        for _ in range(val_loader.size):
            image, gt, label, file_name = val_loader.load_data()

            # --- Prepare GT ---
            gt = np.asarray(gt, np.float32)
            if gt.max() > 0:
                gt = gt / (gt.max() + 1e-8)
            gt_np = np.array(gt)

            image = image.to(device)
            label = label.to(device)

            # --- Forward main model ---
            pred_mask, pred_class_logits = ESFPNet(image)

            # --- Pred segmentation (bin) ---
            pred_mask = F.interpolate(pred_mask, size=gt_np.shape, mode='bilinear', align_corners=False)
            pred_mask = pred_mask.sigmoid()
            pred_mask_bin = (pred_mask > threshold_mask).float()

            # --- Bubble mask + coverage ---
            bubble_mask = bubbles(image)
            bubble_mask = F.interpolate(bubble_mask, size=gt_np.shape, mode='bilinear', align_corners=False)
            bubble_mask = bubble_mask.sigmoid()
            bubble_mask_bin = (bubble_mask > threshold_mask).float()

            pred1_bin = pred_mask_bin[0, 0].detach().cpu().numpy().astype(np.uint8)
            bubble_bin = bubble_mask_bin[0, 0].detach().cpu().numpy().astype(np.uint8)

            intersection = np.logical_and(bubble_bin > 0, pred1_bin > 0).sum()
            pred1_area = (pred1_bin > 0).sum()
            coverage = (intersection / (pred1_area + 1e-8))
            print(f"[{file_name}] Coverage of pred_mask by bubble_mask: {coverage*100:.2f}%")

            # --- Classification probs ---
            logits = pred_class_logits.view(1, -1)
            probs = F.softmax(logits, dim=-1)
            predicted_class = int(torch.argmax(logits, dim=1).item())
            probs_list = [float(p.item()) for p in probs[0]]

            overridden = False
            # ===== OVERRIDE RULE =====
            if coverage > coverage_threshold_override:
                # 1) Force class -> 'Binh thuong'
                predicted_class = 0
                probs_list = [1.0, 0.0, 0.0]
                # 2) *** Force mask to all black ***
                pred_mask_bin.zero_()
                overridden = True
                print("  -> Overridden prediction to 'Binh thuong' and mask -> all black "
                      f"(coverage > {int(coverage_threshold_override*100)}%)")

            # --- Segmentation metric (Dice) computed AFTER possible override ---
            pred_mask_np = pred_mask_bin[0, 0].detach().cpu().numpy().astype(np.float32)
            input_flat = pred_mask_np.flatten()
            target_flat = gt_np.flatten().astype(np.float32)
            intersection_dice = (input_flat * target_flat).sum()
            dice_score = (2 * intersection_dice + smooth) / (input_flat.sum() + target_flat.sum() + smooth)

            val_dice += float(dice_score)
            count_dice += 1
            total += 1

            # --- Classification metrics update ---
            correct_predictions += int(predicted_class == int(label.item()))
            pred_tensor = torch.tensor([predicted_class], device=device)
            tgt_tensor = label.view(1)
            confusion_matrix_total += confusion_matrix_metric(pred_tensor, tgt_tensor)

            # --- Per-image record ---
            per_image_results.append({
                "file_name": file_name,
                "coverage": coverage,
                "pred_probs": [round(x, 4) for x in probs_list],
                "pred_class": predicted_class,
                "pred_label": label_list[predicted_class],
                "gt_label": int(label.item()),
                "overridden": overridden
            })

    avg_dice = (val_dice / count_dice) if count_dice > 0 else 0.0
    accuracy = (correct_predictions / total) if total > 0 else 0.0

    TP = confusion_matrix_total.diag()
    FP = confusion_matrix_total.sum(0) - TP
    FN = confusion_matrix_total.sum(1) - TP

    precision = TP / (TP + FP + 1e-8)
    recall    = TP / (TP + FN + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    precision_macro = precision.mean().item()
    recall_macro    = recall.mean().item()
    f1_macro        = f1.mean().item()

    print("\n==== Segmentation ====")
    print(f"Dice Val Segmentation: {100.0 * avg_dice:.2f}%")

    print("\n==== Classification ====")
    print(f"Accuracy Val Classification: {100.0 * accuracy:.2f}%")
    print("Precision (per class):", [round(x, 4) for x in precision.tolist()])
    print("Recall (per class):   ", [round(x, 4) for x in recall.tolist()])
    print("F1 (per class):       ", [round(x, 4) for x in f1.tolist()])
    print(f"Precision (macro): {round(precision_macro, 4)}")
    print(f"Recall (macro):    {round(recall_macro, 4)}")
    print(f"F1 (macro):        {round(f1_macro, 4)}")

    return {
        "dice_val_seg": 100.0 * avg_dice,
        "acc_val_cls": 100.0 * accuracy,
        "precision_per_class": [float(x) for x in precision.tolist()],
        "recall_per_class": [float(x) for x in recall.tolist()],
        "f1_per_class": [float(x) for x in f1.tolist()],
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "per_image": per_image_results
    }

# Device check
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print('Models moved to GPU.')
else:
    device = torch.device("cpu")
    print('Only CPU available.')

import os

evaluate_with_bubble_and_metrics(test_images_path, test_masks_path, args.label_json_path)
