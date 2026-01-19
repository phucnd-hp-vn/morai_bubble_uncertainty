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

import wandb
wandb.login()
wandb.init(project="ESFPNet_multimodel_bubbles_segmentation")

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
        help='Location of the bubble dataset')
parser.add_argument('--model_type', type=str, default='B4',
        help='Type of model (default B4)')
parser.add_argument('--init_trainsize', type=int, default=352,
        help='Size of image for training (default = 352)')
parser.add_argument('--batch_size', type=int, default=8,
        help='Batch size for training (default = 8)')
parser.add_argument('--n_epochs', type=int, default=500,
        help='Number of epochs for training (default = 500)')
args = parser.parse_args()


# Clear GPU cache
torch.cuda.empty_cache()

train_images_path = args.dataset + '/train/imgs'
train_masks_path  = args.dataset + '/train/masks'
val_images_path   = args.dataset + '/valid/imgs'
val_masks_path    = args.dataset + '/valid/masks'


class PolypDataset(Dataset):
    
    def __init__(self, image_root, gt_root, trainsize, augmentations): #
        self.trainsize = trainsize
        self.augmentations = augmentations
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]

        self.gts = [gt_root + f for f in os.listdir(gt_root)  if f.endswith('.jpg') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.filter_files()
        self.size = len(self.images)
        if self.augmentations == True:
            print('Using RandomRotation, RandomFlip')
            self.img_transform = transforms.Compose([
                transforms.RandomRotation(90, resample=False, expand=False, center=None),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0, hue=0),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
            self.gt_transform = transforms.Compose([
                transforms.RandomRotation(90, resample=False, expand=False, center=None),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor()
                ])
            
        else:
            print('no augmentation')
            self.img_transform = transforms.Compose([
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
            
            self.gt_transform = transforms.Compose([
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor()])
            

    def __getitem__(self, index):
        
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        
        file_name = os.path.splitext(os.path.basename(self.images[index]))[0]
        
        seed = np.random.randint(2147483647) # make a seed with numpy generator 
        np.random.seed(seed) # apply this seed to img tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7
        if self.img_transform is not None:
            image = self.img_transform(image)
            
        np.random.seed(seed) # apply this seed to img tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7
        if self.gt_transform is not None:
            gt = self.gt_transform(gt)
        
        return image, gt

    def filter_files(self):
        print(len(self.images), len(self.gts))
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
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

from torch.autograd import Variable
from torchmetrics.classification import MultilabelConfusionMatrix, MultilabelF1Score, MultilabelRecall, MultilabelPrecision


def evaluate():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ESFPNet.eval()

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

    # Log to wandb
    wandb.log({
        "dice_val_segmentation": dice_avg,
        "missing_rate": missing_rate,
        "false_positive_rate": false_positive_rate
    })

    ESFPNet.train()
    return 100 * dice_avg  # for comparison if used in training loop

import scipy.sparse.csgraph._laplacian

def training_loop(n_epochs, ESFPNet_optimizer, numIters):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    trainDataset = PolypDataset(
        train_images_path + '/',
        train_masks_path + '/',  # Make sure this exists if needed by your loader
        trainsize=args.init_trainsize,
        augmentations=True
    )
    train_loader = DataLoader(dataset=trainDataset, batch_size=args.batch_size, shuffle=True)

    best_dice_score = 0.0
    save_model_path = './SaveModel/bubbles_segmentation/'
    os.makedirs(save_model_path, exist_ok=True)

    for epoch in range(n_epochs):
        ESFPNet.train()
        epoch_loss_all = 0.0

        # === Training loop ===
        tp, fn, fp, tn = 0, 0, 0, 0
        smooth = 1e-4
        dice_sum = 0.0
        dice_count = 0

        for data in train_loader:
            images, masks = data
            images = images.to(device)
            masks = masks.to(device)

            ESFPNet_optimizer.zero_grad()
            pred_masks = ESFPNet(images)
            pred_masks = F.interpolate(pred_masks, scale_factor=4, mode='bilinear', align_corners=False)
            loss = ange_structure_loss(pred_masks, masks)

            loss.backward()
            ESFPNet_optimizer.step()
            epoch_loss_all += loss.item()

            # === Metrics on train ===
            with torch.no_grad():
                pred_probs = pred_masks.sigmoid()
                preds = (pred_probs > 0.5).float()

                for i in range(images.size(0)):
                    pred_np = preds[i].detach().cpu().numpy().squeeze()
                    mask_np = masks[i].detach().cpu().numpy().squeeze()

                    has_pred = pred_np.sum() > 0
                    has_gt = mask_np.sum() > 0

                    # Dice (only for positive)
                    if has_gt:
                        pred_np = (pred_np - pred_np.min()) / (pred_np.max() - pred_np.min() + 1e-8)
                        input_flat = np.reshape(pred_np, (-1))
                        target_flat = np.reshape(mask_np, (-1))
                        intersection = (input_flat * target_flat).sum()
                        dice_score = (2 * intersection + smooth) / (input_flat.sum() + target_flat.sum() + smooth)
                        dice_sum += dice_score
                        dice_count += 1

                    # Binary stats
                    if has_gt and has_pred:
                        tp += 1
                    elif has_gt and not has_pred:
                        fn += 1
                    elif not has_gt and has_pred:
                        fp += 1
                    else:
                        tn += 1

        # === Log training metrics ===
        epoch_loss = epoch_loss_all / len(train_loader)
        train_dice = dice_sum / (dice_count + 1e-8)
        train_missing_rate = fn / (fn + tp + 1e-8)
        train_fp_rate = fp / (fp + tn + 1e-8)

        wandb.log({
            "Loss_train": epoch_loss,
            "Dice_train": train_dice,
            "Missing_rate_train": train_missing_rate,
            "False_positive_rate_train": train_fp_rate,
        })

        print(f"Epoch {epoch+1}/{n_epochs} - Loss: {epoch_loss:.4f} | Dice: {train_dice:.4f} | Miss: {train_missing_rate:.4f} | FP: {train_fp_rate:.4f}")

        # === Save checkpoint ===
        checkpoint_path = os.path.join(save_model_path, f"epoch.pt")
        torch.save(ESFPNet.state_dict(), checkpoint_path)
        print(f"Model saved to: {checkpoint_path}")

        # === Run validation ===
        dice_val = evaluate()  # This should log its own metrics

        # Save best model
        if dice_val > best_dice_score:
            best_dice_score = dice_val
            best_model_path = os.path.join(save_model_path, "best_model.pt")
            torch.save(ESFPNet.state_dict(), best_model_path)
            print(f"✅ Best model updated (Val Dice: {dice_val:.2f}%) → Saved to: {best_model_path}")

    print(f"\n🏁 Training complete. Best Val Dice score: {best_dice_score:.2f}%")

import torch.optim as optim

for i in range(1):
    # Clear GPU cache
    torch.cuda.empty_cache()
   
    ESFPNet = ESFPNetStructure()
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        ESFPNet.to(device)
        print('Models moved to GPU.')
    else:
        print('Only CPU available.')
    print('#####################################################################################')

    # hyperparams for Adam optimizer
    lr=0.0001 #0.0001

    ESFPNet_optimizer = optim.AdamW(ESFPNet.parameters(), lr=lr)

    training_loop(args.n_epochs, ESFPNet_optimizer, i+1, )

wandb.finish()