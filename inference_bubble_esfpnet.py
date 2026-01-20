import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader,ConcatDataset
from torch.autograd import Variable
import json
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

#from skimage import io, transform
from PIL import Image

# visualizing data
import matplotlib.pyplot as plt
import numpy as np
import warnings

# load dataset information~
import yaml

from sklearn.metrics import accuracy_score

# image writing
# import imageio
# from skimage import img_as_ubyte

from sklearn.model_selection import train_test_split

from scipy.io import loadmat
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

parser = argparse.ArgumentParser("ESFPNet based model")
parser.add_argument('--dataset', type=str, required=True,
        help='Location of the dataset')
parser.add_argument('--model_type', type=str, default='model_type',
        help='Type of model (default B4)')
parser.add_argument('--init_trainsize', type=int, default=352,
        help='Size of image for training (default = 352)')
parser.add_argument('--batch_size', type=int, default=8,
        help='Batch size for training (default = 8)')
parser.add_argument('--saved_model', type=str, required=True,
        help='Load saved baseline model') 
args = parser.parse_args()


# Clear GPU cache
torch.cuda.empty_cache()

val_images_path   = args.dataset + '/valid/imgs'
val_masks_path    = args.dataset + '/valid/masks'


class test_dataset:
    def __init__(self, image_root, gt_root, testsize): #
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png') or f.endswith('.jpg')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
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
        gt = self.binary_loader(self.gts[self.index])
        file_name = os.path.splitext(os.path.basename(self.images[self.index]))[0]

        self.index += 1
        return image, gt

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

def ange_structure_loss(pred, mask, smooth=1):
    
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=15, stride=1, padding=7) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + smooth)/(union - inter + smooth)
    
    return (wbce + wiou).mean()

def dice_loss_coff(pred, target, smooth = 0.0001):
    
    num = target.size(0)
    pred = pred.contiguous()
    target = target.contiguous()    

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)
    
    return loss.sum()/num

def loss_class(pred, target):
    return nn.BCEWithLogitsLoss()(pred, target)

def SaveResult():
    ESFPNet = ESFPNetStructure()
    ESFPNet.load_state_dict(torch.load(args.saved_model))
    ESFPNet.eval()
    ESFPNet = ESFPNet.to(device)


    val_loader = test_dataset(val_images_path + '/', val_masks_path + '/', args.init_trainsize)

    dice_sum = 0.0
    dice_count = 0
    smooth = 1e-4

    tp, fn, fp, tn = 0, 0, 0, 0

    for i in range(val_loader.size):
        image, gt = val_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)

        image = image.to(device)

        with torch.no_grad():
            pred = ESFPNet(image)
            pred = F.interpolate(pred, size=gt.shape, mode='bilinear', align_corners=False)
            pred = pred.sigmoid()
            pred = (pred > 0.5).float()

        # Dice score (only for bubble cases)
        pred_np = pred.data.cpu().numpy().squeeze()
        pred_np = (pred_np - pred_np.min()) / (pred_np.max() - pred_np.min() + 1e-8)
        input_flat = np.reshape(pred_np, (-1))
        target_flat = np.reshape(gt, (-1))
        intersection = (input_flat * target_flat)
        dice_score = (2 * intersection.sum() + smooth) / (input_flat.sum() + target_flat.sum() + smooth)

        has_gt = gt.sum() > 0
        has_pred = pred_np.sum() > 0

        if has_gt:  # Only count Dice on images with bubbles
            dice_sum += dice_score
            dice_count += 1

        if has_gt and has_pred:
            tp += 1
        elif has_gt and not has_pred:
            fn += 1
        elif not has_gt and has_pred:
            fp += 1
        else:
            tn += 1

    # Final metrics
    dice_avg = dice_sum / (dice_count + 1e-8)
    missing_rate = fn / (fn + tp + 1e-8)
    false_positive_rate = fp / (fp + tn + 1e-8)

    print(f"\n✅ Evaluation Summary:")
    print(f"Dice (on bubbles only): {100 * dice_avg:.2f}%")
    print(f"Missing rate:           {missing_rate:.4f}")
    print(f"False positive rate:    {false_positive_rate:.4f}")

    ESFPNet.train()
    return 100 * dice_avg  # for comparison if used in training loop

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print('Models moved to GPU.')
else:
    device = torch.device("cpu")
    print('Only CPU available.')

SaveResult()