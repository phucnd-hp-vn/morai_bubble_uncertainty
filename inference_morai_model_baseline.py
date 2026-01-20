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
import imageio
from skimage import img_as_ubyte

from sklearn.model_selection import train_test_split

from scipy.io import loadmat
import matplotlib.pyplot as plt
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
parser.add_argument('--if_renew', type=bool, default=False,
        help='Check if split data to train_val_test')
parser.add_argument('--saved_model', type=str, required=True,
        help='Load saved model') 
args = parser.parse_args()


# Clear GPU cache
torch.cuda.empty_cache()
test_images_path  = args.dataset + '/imgs' 
test_masks_path   = args.dataset + '/masks'


class PolypDataset(Dataset):
    def __init__(self, image_root, gt_root, label_root, trainsize, augmentations):
        self.trainsize = trainsize
        self.augmentations = augmentations
        self.images = [os.path.join(image_root, f) for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.labels = label_root

        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.filter_files()
        self.size = len(self.images)

        if self.augmentations:
            print('Using RandomRotation, RandomFlip')
            self.img_transform = transforms.Compose([
                transforms.RandomRotation(90),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0, hue=0),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])
            self.gt_transform = transforms.Compose([
                transforms.RandomRotation(90),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor()
            ])
        else:
            print('No augmentation')
            self.img_transform = transforms.Compose([
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])
            self.gt_transform = transforms.Compose([
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor()
            ])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        file_name = os.path.splitext(os.path.basename(self.images[index]))[0]

        # Load labels from JSON
        with open(self.labels, 'r') as f:
            data = json.load(f)

        label_list = ['Binh thuong', 'Lanh tinh', 'Ac tinh']
        label_name = [file['label_name'] for file in data if file['object_id'] == file_name]

        # Label validation
        # Prioritized label selection
        if len(label_name) == 0:
            raise ValueError(f"No label found for image {file_name}")
        elif len(label_name) > 1:
            if 'Ac tinh' in label_name:
                selected_label = 'Ac tinh'
            else:
                selected_label = label_name[0]
        else:
            selected_label = label_name[0]

        # Convert to class index
        label_tensor = torch.tensor(label_list.index(selected_label))


        # Sync augmentation for image and mask
        seed = np.random.randint(2147483647)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if self.img_transform:
            image = self.img_transform(image)

        torch.manual_seed(seed)
        np.random.seed(seed)
        if self.gt_transform:
            gt = self.gt_transform(gt)

        return image, gt, label_tensor

    def filter_files(self):
        print(f"Checking image/GT size consistency: {len(self.images)} images, {len(self.gts)} masks")
        assert len(self.images) == len(self.gts), "Mismatch between image and GT file count"
        images, gts = [], []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size

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

def loss_class_multilabel(pred, target):
    return nn.BCEWithLogitsLoss()(pred, target)

def loss_class_multiclass(pred, target):
    return nn.CrossEntropyLoss()(pred, target)

from torch.autograd import Variable
# from torchmetrics.classification import MultilabelConfusionMatrix, MultilabelF1Score, MultilabelRecall, MultilabelPrecision
from torchmetrics import ConfusionMatrix

# Assuming ESFPNet and args are already defined globally

def SaveResult():
    ESFPNet = torch.load(args.saved_model)
    ESFPNet.eval()

    total = 0
    num_classes = 3
    correct_predictions = 0
    val_dice = 0.0
    count_dice = 0
    smooth = 1e-4

    confusion_matrix_metric = ConfusionMatrix(num_classes=num_classes, task="multiclass").to(device)
    confusion_matrix_total = torch.zeros(num_classes, num_classes).to(device)

    label_list = ['Binh thuong', 'Lanh tinh', 'Ac tinh']

    val_loader = test_dataset(
        test_images_path + '/', test_masks_path + '/',
        args.label_json_path, args.init_trainsize
    )

    for i in range(val_loader.size):
        image, gt, label, file_name = val_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)

        image = image.to(device)
        label = label.to(device)

        with torch.no_grad():
            pred_mask, pred_class_logits = ESFPNet(image)

        # Segmentation evaluation
        pred_mask = F.interpolate(pred_mask, size=gt.shape, mode='bilinear', align_corners=False)
        pred_mask = pred_mask.sigmoid()
        pred_mask_bin = (pred_mask > 0.5).float()

        pred_mask_np = pred_mask_bin.cpu().numpy().squeeze()
        pred_mask_np = (pred_mask_np - pred_mask_np.min()) / (pred_mask_np.max() - pred_mask_np.min() + 1e-8)
        gt_np = np.array(gt)

        input_flat = pred_mask_np.flatten()
        target_flat = gt_np.flatten()
        intersection = (input_flat * target_flat).sum()
        dice_score = (2 * intersection + smooth) / (input_flat.sum() + target_flat.sum() + smooth)

        # Optionally skip specific class if needed
        # if label.item() != 1:
        val_dice += dice_score
        count_dice += 1

        total += 1

        # Classification evaluation
        pred_class_logits = pred_class_logits.view(1, -1)
        predicted_class = pred_class_logits.argmax(dim=1)
        correct_predictions += (predicted_class == label).sum().item()

        # Update confusion matrix
        confusion_matrix_total += confusion_matrix_metric(predicted_class, label.unsqueeze(0))

    # Dice score (segmentation)
    avg_dice = val_dice / count_dice if count_dice > 0 else 0
    print("Dice Val Segmentation:", 100 * avg_dice)

    # Classification metrics
    accuracy = correct_predictions / total if total > 0 else 0
    print("Accuracy Val Classification:", 100 * accuracy)

    TP = confusion_matrix_total.diag()
    FP = confusion_matrix_total.sum(0) - TP
    FN = confusion_matrix_total.sum(1) - TP

    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    precision_macro = precision.mean()
    recall_macro = recall.mean()
    f1_macro = f1.mean()

    print("Precision (per class):", precision)
    print("Recall (per class):", recall)
    print("F1 (per class):", f1)
    print("Precision (macro):", precision_macro.item())
    print("Recall (macro):", recall_macro.item())
    print("F1 (macro):", f1_macro.item())

    ESFPNet.train()

    return 100 * avg_dice, 100 * accuracy

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print('Models moved to GPU.')
else:
    device = torch.device("cpu")
    print('Only CPU available.')

SaveResult()