import os
import csv
import json
import pickle
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from PIL import Image

import torchvision.transforms as transforms
from sklearn.metrics import precision_recall_fscore_support

# ==== Encoder/Decoder của bạn ====
from Encoder import mit
from Decoder import mlp
from mmcv.cnn import ConvModule

# ===== Matplotlib (headless) =====
import matplotlib
matplotlib.use("Agg")  # để render không cần màn hình
import matplotlib.pyplot as plt

# ======================
# Parse args
# ======================
parser = argparse.ArgumentParser("ESFPNet threshold sweep (macro-F1)")
parser.add_argument('--label_json_path', type=str, required=True,
                    help='Path to the JSON label file (object_id must match image filename without extension).')
parser.add_argument('--dataset', type=str, required=True,
                    help='Dataset directory containing imgs/ and masks/.')
parser.add_argument('--model_type', type=str, default='B4',
                    help='MiT backbone type (B0–B5), default: B4.')
parser.add_argument('--init_trainsize', type=int, default=352,
                    help='Input image size for inference, default: 352.')
parser.add_argument('--saved_model', type=str, required=True,
                    help='Path to the trained ESFPNet model (saved via torch.save).')
parser.add_argument('--out_dir', type=str, default='./threshold_optimization_coverage_test',
                    help='Output directory for threshold sweep results and reports.')
parser.add_argument('--grid_step', type=float, default=0.01,
                    help='Threshold grid step in range [0, 1], default: 0.01.')
parser.add_argument('--bubbles_model_path', type=str, required=True,
                    help='Path to the trained Bubble segmentation model.')
args = parser.parse_args()


# ======================
# Device
# ======================
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print('Models moved to GPU.')
else:
    device = torch.device("cpu")
    print('Only CPU available.')

# ======================
# Đường dẫn test
# ======================
test_images_path = os.path.join(args.dataset, 'imgs')
test_masks_path  = os.path.join(args.dataset, 'masks')

# ======================
# Dataset dùng cho test/val
# ======================
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

        if len(label_name) == 0:
            raise ValueError(f"No label found for image {file_name}")
        elif len(label_name) > 1:
            if 'Ac tinh' in label_name:
                selected_label = 'Ac tinh'
            else:
                selected_label = label_name[0]
        else:
            selected_label = label_name[0]

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

# ======================
# Mô hình ESFPNet: Seg + Class
# ======================
class ESFPNetStructure(nn.Module):
    def __init__(self, embedding_dim=160):
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

        self._init_weights()

        # LP Header
        self.LP_1 = mlp.LP(input_dim=self.backbone.embed_dims[0], embed_dim=self.backbone.embed_dims[0])
        self.LP_2 = mlp.LP(input_dim=self.backbone.embed_dims[1], embed_dim=self.backbone.embed_dims[1])
        self.LP_3 = mlp.LP(input_dim=self.backbone.embed_dims[2], embed_dim=self.backbone.embed_dims[2])
        self.LP_4 = mlp.LP(input_dim=self.backbone.embed_dims[3], embed_dim=self.backbone.embed_dims[3])

        # Linear Fuse
        self.linear_fuse34 = ConvModule(in_channels=(self.backbone.embed_dims[2] + self.backbone.embed_dims[3]),
                                        out_channels=self.backbone.embed_dims[2], kernel_size=1,
                                        norm_cfg=dict(type='BN', requires_grad=True))
        self.linear_fuse23 = ConvModule(in_channels=(self.backbone.embed_dims[1] + self.backbone.embed_dims[2]),
                                        out_channels=self.backbone.embed_dims[1], kernel_size=1,
                                        norm_cfg=dict(type='BN', requires_grad=True))
        self.linear_fuse12 = ConvModule(in_channels=(self.backbone.embed_dims[0] + self.backbone.embed_dims[1]),
                                        out_channels=self.backbone.embed_dims[0], kernel_size=1,
                                        norm_cfg=dict(type='BN', requires_grad=True))

        # Fused LP Header
        self.LP_12 = mlp.LP(input_dim=self.backbone.embed_dims[0], embed_dim=self.backbone.embed_dims[0])
        self.LP_23 = mlp.LP(input_dim=self.backbone.embed_dims[1], embed_dim=self.backbone.embed_dims[1])
        self.LP_34 = mlp.LP(input_dim=self.backbone.embed_dims[2], embed_dim=self.backbone.embed_dims[2])

        # Final Linear Prediction (seg)
        self.linear_pred = nn.Conv2d((self.backbone.embed_dims[0] + self.backbone.embed_dims[1] +
                                      self.backbone.embed_dims[2] + self.backbone.embed_dims[3]),
                                     1, kernel_size=1)

        # Classification head (3 classes)
        self.norm1 = nn.BatchNorm2d(512, eps=1e-5)
        self.Relu = nn.ReLU(inplace=True)
        self.Dropout = nn.Dropout(p=0.3)
        self.conv1 = nn.Conv2d(512, 256, 1, stride=1, padding=0)
        self.norm2 = nn.BatchNorm2d(256, eps=1e-5)
        self.conv2 = nn.Conv2d(256, 3, 1, stride=1, padding=0, bias=True)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.softmax = nn.Softmax(dim=1)

    def _init_weights(self):
        if args.model_type == 'B0':
            pretrained_dict = torch.load('./Pretrained/mit_b0.pth', map_location='cpu')
        if args.model_type == 'B1':
            pretrained_dict = torch.load('./Pretrained/mit_b1.pth', map_location='cpu')
        if args.model_type == 'B2':
            pretrained_dict = torch.load('./Pretrained/mit_b2.pth', map_location='cpu')
        if args.model_type == 'B3':
            pretrained_dict = torch.load('./Pretrained/mit_b3.pth', map_location='cpu')
        if args.model_type == 'B4':
            pretrained_dict = torch.load('./Pretrained/mit_b4.pth', map_location='cpu')
        if args.model_type == 'B5':
            pretrained_dict = torch.load('./Pretrained/mit_b5.pth', map_location='cpu')

        model_dict = self.backbone.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.backbone.load_state_dict(model_dict)
        print("Backbone weights loaded.")

    def forward(self, x):
        B = x.shape[0]

        # stage 1
        out_1, H, W = self.backbone.patch_embed1(x)
        for blk in self.backbone.block1:
            out_1 = blk(out_1, H, W)
        out_1 = self.backbone.norm1(out_1)
        out_1 = out_1.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # stage 2
        out_2, H, W = self.backbone.patch_embed2(out_1)
        for blk in self.backbone.block2:
            out_2 = blk(out_2, H, W)
        out_2 = self.backbone.norm2(out_2)
        out_2 = out_2.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # stage 3
        out_3, H, W = self.backbone.patch_embed3(out_2)
        for blk in self.backbone.block3:
            out_3 = blk(out_3, H, W)
        out_3 = self.backbone.norm3(out_3)
        out_3 = out_3.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # stage 4
        out_4, H, W = self.backbone.patch_embed4(out_3)
        for blk in self.backbone.block4:
            out_4 = blk(out_4, H, W)
        out_4 = self.backbone.norm4(out_4)
        out_4 = out_4.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # Seg: LP headers
        lp_1 = self.LP_1(out_1)
        lp_2 = self.LP_2(out_2)
        lp_3 = self.LP_3(out_3)
        lp_4 = self.LP_4(out_4)

        lp_34 = self.LP_34(self.linear_fuse34(torch.cat([lp_3, F.interpolate(lp_4, scale_factor=2, mode='bilinear', align_corners=False)], dim=1)))
        lp_23 = self.LP_23(self.linear_fuse23(torch.cat([lp_2, F.interpolate(lp_34, scale_factor=2, mode='bilinear', align_corners=False)], dim=1)))
        lp_12 = self.LP_12(self.linear_fuse12(torch.cat([lp_1, F.interpolate(lp_23, scale_factor=2, mode='bilinear', align_corners=False)], dim=1)))

        lp4_resized = F.interpolate(lp_4, scale_factor=8, mode='bilinear', align_corners=False)
        lp3_resized = F.interpolate(lp_34, scale_factor=4, mode='bilinear', align_corners=False)
        lp2_resized = F.interpolate(lp_23, scale_factor=2, mode='bilinear', align_corners=False)
        lp1_resized = lp_12

        out1 = self.linear_pred(torch.cat([lp1_resized, lp2_resized, lp3_resized, lp4_resized], dim=1))  # seg logit

        # Class
        out2 = self.global_avg_pool(out_4)
        out2 = self.norm1(out2)
        out2 = self.Relu(out2)
        out2 = self.Dropout(out2)
        out2 = self.conv1(out2)
        out2 = self.norm2(out2)
        out2 = self.Relu(out2)
        out2 = self.conv2(out2)
        out2 = out2.view(B, -1)  # [B,3]

        return out1, out2

# ======================
# Mô hình Bubble: Seg 1 kênh
# ======================
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

        #classification layer (không dùng ở đây)
        self.norm1 = nn.BatchNorm2d(512, eps=1e-5)
        self.Relu = nn.ReLU(inplace=True)
        self.Dropout = nn.Dropout(p=0.3)
        self.conv1 = nn.Conv2d(512, 256, 1, stride=1, padding=0)
        self.norm2 = nn.BatchNorm2d(256, eps=1e-5)
        self.conv2 = nn.Conv2d(256, 2, 1, stride=1, padding=0, bias=True)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.softmax = nn.Softmax(dim = 1)

    def _init_weights(self):

        if args.model_type == 'B0':
            pretrained_dict = torch.load('./Pretrained/mit_b0.pth', map_location='cpu')
        if args.model_type == 'B1':
            pretrained_dict = torch.load('./Pretrained/mit_b1.pth', map_location='cpu')
        if args.model_type == 'B2':
            pretrained_dict = torch.load('./Pretrained/mit_b2.pth', map_location='cpu')
        if args.model_type == 'B3':
            pretrained_dict = torch.load('./Pretrained/mit_b3.pth', map_location='cpu')
        if args.model_type == 'B4':
            pretrained_dict = torch.load('./Pretrained/mit_b4.pth', map_location='cpu')
        if args.model_type == 'B5':
            pretrained_dict = torch.load('./Pretrained/mit_b5.pth', map_location='cpu')

        model_dict = self.backbone.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.backbone.load_state_dict(model_dict)
        print("successfully loaded!!!!")

    def forward(self, x):
        B = x.shape[0]

        #stage 1
        out_1, H, W = self.backbone.patch_embed1(x)
        for i, blk in enumerate(self.backbone.block1):
            out_1 = blk(out_1, H, W)
        out_1 = self.backbone.norm1(out_1)
        out_1 = out_1.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # stage 2
        out_2, H, W = self.backbone.patch_embed2(out_1)
        for i, blk in enumerate(self.backbone.block2):
            out_2 = blk(out_2, H, W)
        out_2 = self.backbone.norm2(out_2)
        out_2 = out_2.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # stage 3
        out_3, H, W = self.backbone.patch_embed3(out_2)
        for i, blk in enumerate(self.backbone.block3):
            out_3 = blk(out_3, H, W)
        out_3 = self.backbone.norm3(out_3)
        out_3 = out_3.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # stage 4
        out_4, H, W = self.backbone.patch_embed4(out_3)
        for i, blk in enumerate(self.backbone.block4):
            out_4 = blk(out_4, H, W)
        out_4 = self.backbone.norm4(out_4)
        out_4 = out_4.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        #segmentation
        lp_1 = self.LP_1(out_1)
        lp_2 = self.LP_2(out_2)
        lp_3 = self.LP_3(out_3)
        lp_4 = self.LP_4(out_4)

        lp_34 = self.LP_34(self.linear_fuse34(torch.cat([lp_3, F.interpolate(lp_4,scale_factor=2,mode='bilinear', align_corners=False)], dim=1)))
        lp_23 = self.LP_23(self.linear_fuse23(torch.cat([lp_2, F.interpolate(lp_34,scale_factor=2,mode='bilinear', align_corners=False)], dim=1)))
        lp_12 = self.LP_12(self.linear_fuse12(torch.cat([lp_1, F.interpolate(lp_23,scale_factor=2,mode='bilinear', align_corners=False)], dim=1)))

        lp4_resized = F.interpolate(lp_4,scale_factor=8,mode='bilinear', align_corners=False)
        lp3_resized = F.interpolate(lp_34,scale_factor=4,mode='bilinear', align_corners=False)
        lp2_resized = F.interpolate(lp_23,scale_factor=2,mode='bilinear', align_corners=False)
        lp1_resized = lp_12

        out1 = self.linear_pred(torch.cat([lp1_resized, lp2_resized, lp3_resized, lp4_resized], dim=1))

        return out1

# ======================
# Helpers vẽ biểu đồ
# ======================
def _save_line_plot(x, ys, labels, xlabel, ylabel, title, save_path, vline_x=None):
    plt.figure(figsize=(6,4), dpi=200)
    for y, lb in zip(ys, labels):
        plt.plot(x, y, label=lb, linewidth=1.8)
    if vline_x is not None:
        plt.axvline(vline_x, linestyle="--", linewidth=1.2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if len(labels) > 1:
        plt.legend(loc="best", frameon=False)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def _save_bar(values, labels, title, ylabel, save_path):
    plt.figure(figsize=(6,4), dpi=200)
    xs = range(len(values))
    plt.bar(xs, values)
    plt.xticks(xs, labels, rotation=0)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.ylim(0, 1.0)
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def _save_confusion_matrix(cm, class_names, title, save_path, normalize=False):
    cm_plot = cm.astype(np.float32)
    if normalize:
        row_sums = cm_plot.sum(axis=1, keepdims=True) + 1e-8
        cm_plot = cm_plot / row_sums

    plt.figure(figsize=(5,4), dpi=200)
    plt.imshow(cm_plot, interpolation='nearest', aspect='auto')
    plt.title(title)
    plt.colorbar(fraction=0.046, pad=0.04)
    tick_marks = range(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=0)
    plt.yticks(tick_marks, class_names)
    fmt = ".2f"
    thresh = cm_plot.max() / 2.0
    for i in range(cm_plot.shape[0]):
        for j in range(cm_plot.shape[1]):
            plt.text(j, i, format(cm_plot[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm_plot[i, j] > thresh else "black",
                     fontsize=9)
    plt.ylabel('Ground Truth')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def plot_sweep_curves_from_rows(grid_rows, save_dir, best_th):
    """
    grid_rows: list[dict] do eval_threshold_grid sinh ra
    """
    os.makedirs(save_dir, exist_ok=True)

    ths = [r["threshold"] for r in grid_rows]
    macro_f1 = [r["f1_macro"] for r in grid_rows]
    f1_cls0 = [r["f1_cls0"] for r in grid_rows]
    f1_cls1 = [r["f1_cls1"] for r in grid_rows]
    f1_cls2 = [r["f1_cls2"] for r in grid_rows]
    dice_avg = [r["dice_avg"] for r in grid_rows]

    # 1) Macro-F1 vs threshold
    _save_line_plot(
        x=ths,
        ys=[macro_f1],
        labels=["Macro-F1"],
        xlabel="Threshold (τ)",
        ylabel="Score",
        title="Macro-F1 vs. Threshold",
        save_path=os.path.join(save_dir, "plot_macroF1_vs_threshold.png"),
        vline_x=best_th
    )

    # 2) F1 per-class vs threshold
    _save_line_plot(
        x=ths,
        ys=[f1_cls0, f1_cls1, f1_cls2],
        labels=["F1 Normal (cls0)", "F1 Benign (cls1)", "F1 Malignant (cls2)"],
        xlabel="Threshold (τ)",
        ylabel="F1",
        title="Per-class F1 vs. Threshold",
        save_path=os.path.join(save_dir, "plot_F1_per_class_vs_threshold.png"),
        vline_x=best_th
    )

    # 3) Dice avg vs threshold
    _save_line_plot(
        x=ths,
        ys=[dice_avg],
        labels=["Dice (avg)"],
        xlabel="Threshold (τ)",
        ylabel="Dice",
        title="Average Dice vs. Threshold",
        save_path=os.path.join(save_dir, "plot_dice_vs_threshold.png"),
        vline_x=best_th
    )

def plot_bars_at_best(prec, rec, f1, save_dir):
    """
    Vẽ bar chart P/R/F1 theo lớp tại τ*
    """
    os.makedirs(save_dir, exist_ok=True)
    cls_names = ["Normal", "Benign", "Malignant"]

    _save_bar(
        values=list(prec),
        labels=cls_names,
        title="Precision per class (at τ*)",
        ylabel="Precision",
        save_path=os.path.join(save_dir, "bar_precision_at_best.png")
    )
    _save_bar(
        values=list(rec),
        labels=cls_names,
        title="Recall per class (at τ*)",
        ylabel="Recall",
        save_path=os.path.join(save_dir, "bar_recall_at_best.png")
    )
    _save_bar(
        values=list(f1),
        labels=cls_names,
        title="F1 per class (at τ*)",
        ylabel="F1",
        save_path=os.path.join(save_dir, "bar_f1_at_best.png")
    )

def plot_coverage_hist(records, th, save_dir):
    """
    Histogram coverage: bị override vs. không bị override tại τ*
    """
    os.makedirs(save_dir, exist_ok=True)

    cov_over = []
    cov_keep = []
    for r in records:
        cov = r["coverage"]
        if cov > th:
            cov_over.append(cov)
        else:
            cov_keep.append(cov)

    plt.figure(figsize=(6,4), dpi=200)
    bins = 30
    plt.hist(cov_over, bins=bins, alpha=0.7, label="Overridden (cov > τ*)")
    plt.hist(cov_keep, bins=bins, alpha=0.7, label="Kept (cov ≤ τ*)")
    plt.xlabel("Coverage")
    plt.ylabel("Count")
    plt.title("Coverage histogram at τ*")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "hist_coverage_at_best.png"), bbox_inches="tight")
    plt.close()

# ======================
# 1) Thu thập dự đoán một lần
# ======================
@torch.no_grad()
def collect_predictions_once(save_cache_path=None):
    ESFPNet = torch.load(args.saved_model, map_location=device).to(device).eval()

    bubbles = ESFPNetStructureBubble()
    bubbles.load_state_dict(torch.load(args.bubbles_model_path, map_location=device))
    bubbles.eval()
    bubbles = bubbles.to(device)

    val_loader = test_dataset(
        test_images_path + '/', test_masks_path + '/',
        args.label_json_path, args.init_trainsize
    )

    records = []
    for _ in range(val_loader.size):
        image, gt, label, file_name = val_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)

        image = image.to(device)

        # Main model
        pred_mask, pred_class_logits = ESFPNet(image)
        pred_mask = F.interpolate(pred_mask, size=gt.shape, mode='bilinear', align_corners=False)
        pred_mask = (pred_mask.sigmoid() > 0.5).float()
        pred_mask_np = pred_mask.cpu().numpy().squeeze().astype(np.uint8)

        pred_class = pred_class_logits.view(1, -1).argmax(dim=1).item()  # 0..2

        # Bubbles
        bubble_mask = bubbles(image)
        bubble_mask = F.interpolate(bubble_mask, size=gt.shape, mode='bilinear', align_corners=False)
        bubble_mask = (bubble_mask.sigmoid() > 0.5).float()
        bubble_mask_np = bubble_mask.cpu().numpy().squeeze().astype(np.uint8)

        # Coverage
        pred_area = pred_mask_np.sum()
        if pred_area == 0:
            coverage = 0.0
        else:
            inter = np.logical_and(bubble_mask_np > 0, pred_mask_np > 0).sum()
            coverage = float(inter) / float(pred_area)

        records.append({
            "file_name": file_name,
            "gt_label": int(label.item()),
            "gt_mask": gt.astype(np.float32),  # dùng cho Dice
            "pred_class": int(pred_class),
            "pred_mask_np": pred_mask_np,   # 0/1
            "coverage": coverage            # 0..1
        })

    if save_cache_path:
        with open(save_cache_path, "wb") as f:
            pickle.dump(records, f)

    return records

# ======================
# 2) Quét lưới threshold, chọn theo macro-F1
# ======================
def eval_threshold_grid(records, thresholds, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    grid_rows = []
    best = {"th": None, "f1_macro": -1.0, "row": None}

    for th in thresholds:
        y_true = []
        y_pred = []
        dice_sum = 0.0
        n = 0

        for r in records:
            gt_label = r["gt_label"]
            pred_cls = r["pred_class"]
            pred_mask_np = r["pred_mask_np"].copy()
            gt_mask = r["gt_mask"]
            cov = r["coverage"]

            # ==== OVERRIDE: nếu coverage > th, ép class=0 và mask=0-all-black ====
            if cov > th:
                pred_cls = 0
                pred_mask_np[:] = 0  # all black

            y_true.append(gt_label)
            y_pred.append(pred_cls)

            # Dice (không dùng để chọn F1_macro, chỉ để tham khảo)
            input_flat = pred_mask_np.astype(np.float32).flatten()
            target_flat = gt_mask.flatten()
            inter = (input_flat * target_flat).sum()
            dice = (2 * inter + 1e-4) / (input_flat.sum() + target_flat.sum() + 1e-4)
            dice_sum += float(dice)
            n += 1

        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=[0,1,2], average=None, zero_division=0
        )
        f1_macro = f1.mean()
        dice_avg = dice_sum / max(n, 1)

        row = {
            "threshold": round(th, 4),
            "precision_cls0": prec[0], "recall_cls0": rec[0], "f1_cls0": f1[0],
            "precision_cls1": prec[1], "recall_cls1": rec[1], "f1_cls1": f1[1],
            "precision_cls2": prec[2], "recall_cls2": rec[2], "f1_cls2": f1[2],
            "f1_macro": f1_macro,
            "dice_avg": dice_avg
        }
        grid_rows.append(row)

        if f1_macro > best["f1_macro"]:
            best = {"th": th, "f1_macro": f1_macro, "row": row}

    # save CSV
    csv_path = os.path.join(save_dir, "threshold_sweep_macroF1.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(grid_rows[0].keys()))
        w.writeheader()
        w.writerows(grid_rows)

    print(f"[GRID] Best threshold by F1_macro={best['f1_macro']:.6f} at th={best['th']:.6f}")

    # === Vẽ các đường cong theo threshold ===
    try:
        plot_sweep_curves_from_rows(grid_rows, save_dir, best["th"])
    except Exception as e:
        print(f"[WARN] Could not plot sweep curves: {e}")

    return best["th"], best["row"], csv_path

# ======================
# 3) Báo cáo chi tiết tại ngưỡng tối ưu
# ======================
def summarize_at_threshold(records, th, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    overridden_files = []
    y_true = []
    y_pred = []

    num_classes = 3
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    for r in records:
        gt = r["gt_label"]
        pred = r["pred_class"]
        cov = r["coverage"]
        fname = r["file_name"]

        # ==== OVERRIDE: ép class=0 và mask=0 ====
        if cov > th:
            pred = 0
            overridden_files.append(fname)

        y_true.append(gt)
        y_pred.append(pred)
        cm[gt, pred] += 1

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0,1,2], average=None, zero_division=0
    )
    macro_f1 = f1.mean()

    with open(os.path.join(save_dir, "overridden_cases.txt"), "w") as f:
        for n in overridden_files:
            f.write(n + "\n")

    with open(os.path.join(save_dir, "summary.txt"), "w") as f:
        f.write(f"Best threshold: {th:.6f}\n")
        f.write(f"Macro-F1: {macro_f1:.6f}\n")
        for c in range(3):
            f.write(f"Class {c}: P={prec[c]:.4f} R={rec[c]:.4f} F1={f1[c]:.4f}\n")
        f.write("Confusion matrix (rows=GT, cols=Pred):\n")
        for r_ in cm:
            f.write(" ".join(map(str, r_)) + "\n")

    # === Vẽ confusion matrix + bar chart tại τ* ===
    class_names = ["Normal", "Benign", "Malignant"]
    _save_confusion_matrix(
        cm=cm, class_names=class_names,
        title="Confusion Matrix at τ* (counts)",
        save_path=os.path.join(save_dir, "cm_counts_at_best.png"),
        normalize=False
    )
    _save_confusion_matrix(
        cm=cm, class_names=class_names,
        title="Confusion Matrix at τ* (row-normalized)",
        save_path=os.path.join(save_dir, "cm_rownorm_at_best.png"),
        normalize=True
    )

    plot_bars_at_best(prec, rec, f1, save_dir)

    print(f"[REPORT] Saved overridden list & summary at {save_dir}")

# ======================
# MAIN
# ======================
def main():
    base_save_dir = args.out_dir
    os.makedirs(base_save_dir, exist_ok=True)

    # 1) Thu thập 1 lần (có cache)
    cache_path = os.path.join(base_save_dir, "records_cache.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            records = pickle.load(f)
        print("[CACHE] Loaded cached records.")
    else:
        records = collect_predictions_once(save_cache_path=cache_path)
        print("[CACHE] Collected and cached records.")

    # 2) Lưới threshold
    step = max(1, int(round(args.grid_step * 100)))
    thresholds = [round(x * 0.01, 4) for x in range(0, 101, step)]

    # 3) Sweep chọn th tốt nhất theo macro-F1
    best_th, best_row, sweep_csv = eval_threshold_grid(records, thresholds, save_dir=base_save_dir)

    # 4) Báo cáo chi tiết tại th tối ưu
    best_dir = os.path.join(base_save_dir, f"best_threshold_{best_th:.4f}".replace(".", "_"))

    # Tuỳ chọn: trực quan coverage
    try:
        plot_coverage_hist(records, best_th, best_dir)
    except Exception as e:
        print(f"[WARN] Could not plot coverage histogram: {e}")

    summarize_at_threshold(records, best_th, best_dir)

if __name__ == "__main__":
    main()
