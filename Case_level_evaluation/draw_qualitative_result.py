
# case_level_evaluation.py
import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import numpy as np
import cv2

# ---------------- Regex ----------------
FRAME_RE = re.compile(r"(\d+)$", re.IGNORECASE)

# ---------------- Colors ----------------
TN_COLOR = "#33CC00"            # Normal
FP_COLOR = "#FF0000"            # Lesion (UC -> "unclean")
CORRECTED_FP_COLOR = "#FFD700"  # Corrected FP on B/C (A lesion, B normal) or (B lesion, C normal)
CAP_COLOR = "#222222"           # Black cap strip


# ---------------- Utils ----------------
def frame_num(frame_id: str) -> int:
    m = FRAME_RE.search(frame_id)
    if not m:
        raise ValueError(f"Cannot parse frame number from id='{frame_id}'")
    return int(m.group(1))


def norm_label(lbl: str) -> str:
    s = str(lbl).strip().lower()
    if s in {"", "none", "null"}:
        return "normal"
    if s in {"binh thuong", "normal", "bt", "negative"}:
        return "normal"
    return "lesion"


def load_preds(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    data = sorted(data, key=lambda x: frame_num(x["id"]))
    return [
        {"f": frame_num(d["id"]), "kind": norm_label(d.get("label", "Binh thuong"))}
        for d in data
    ]


def build_kind_map(preds):
    return {p["f"]: p["kind"] for p in preds}


def select_diff_keyframes(modelA, modelB, max_keyframes=8, min_gap=50):
    """Keyframes vẫn dựa trên khác biệt A vs B (như code gốc)."""
    mA, mB = build_kind_map(modelA), build_kind_map(modelB)
    frames_sorted = sorted(set(mA) | set(mB))
    diffs = [f for f in frames_sorted if mA.get(f, "normal") != mB.get(f, "normal")]

    picked, last = [], -10**9
    for f in diffs:
        if f - last >= min_gap:
            picked.append(f)
            last = f
        if len(picked) >= max_keyframes:
            break
    return picked


# ---------------- Drawing ----------------
def draw_keyframes_row(
    fig,
    gs_cell,
    keyframe_dir: Optional[Path],
    modelA_dir: Optional[Path],
    modelB_dir: Optional[Path],
    modelC_dir: Optional[Path],
    keyframes: List[int],
    modelA_preds: Dict[int, str],
    modelB_preds: Dict[int, str],
    modelC_preds: Dict[int, str],
):
    if not keyframes:
        return

    n = len(keyframes)
    inner = gs_cell.subgridspec(4, n, wspace=0.4, hspace=0.4)

    def load_mask_or_black(frame: int, folder: Optional[Path], label: str):
        if label == "normal":
            return np.zeros((256, 256), dtype=np.uint8)
        if folder:
            for ext in (".png", ".jpg", ".jpeg"):
                p = folder / f"frame_{frame:06d}{ext}"
                if p.exists():
                    mask = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        return (mask > 30).astype(np.uint8) * 255
        return np.zeros((256, 256), dtype=np.uint8)

    def load_keyframe(frame: int, folder: Optional[Path]):
        if not folder:
            return None
        for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
            p = folder / f"frame_{frame:06d}{ext}"
            if p.exists():
                return plt.imread(str(p))
        return None

    row_labels = ["P126", "C", "D", "D + B"]

    rows = [
        (keyframe_dir, None, "CH100"),
        (modelA_dir, modelA_preds, "UC"),
        (modelB_dir, modelB_preds, "UC + B"),
        (modelC_dir, modelC_preds, "UC + B + C"),
    ]

    for i, f in enumerate(keyframes):
        for row, (folder, preds, label) in enumerate(rows):
            ax = fig.add_subplot(inner[row, i])

            if row == 0:
                img = load_keyframe(f, folder)
            else:
                kind = preds.get(f, "normal") if preds else "normal"
                img = load_mask_or_black(f, folder, kind)

            if img is not None:
                ax.imshow(img if row == 0 else img, cmap=None if row == 0 else "gray")
                ax.set_aspect("equal")

                # Viền theo logic tương tác:
                # - A lesion & B normal  => A: đỏ, B: vàng
                # - B lesion & C normal  => B: đỏ, C: vàng
                kindA = modelA_preds.get(f, "normal")
                kindB = modelB_preds.get(f, "normal")
                kindC = modelC_preds.get(f, "normal")

                # A vs B
                if kindA == "lesion" and kindB == "normal":
                    if row in (1, 2):
                        ax.add_patch(
                            Rectangle(
                                (0, 0), 1, 1,
                                transform=ax.transAxes,
                                fill=False,
                                edgecolor=FP_COLOR if row == 1 else CORRECTED_FP_COLOR,
                                linewidth=1.2,
                                zorder=20,
                                clip_on=False,
                            )
                        )
                else:
                    if row in (1, 2):
                        ax.add_patch(
                            Rectangle(
                                (0, 0), 1, 1,
                                transform=ax.transAxes,
                                fill=False,
                                edgecolor=TN_COLOR if row == 1 else FP_COLOR,
                                linewidth=1.2,
                                zorder=20,
                                clip_on=False,
                                )
                        )

                # B vs C (logic mới)
                if kindB == "lesion" and kindC == "normal":
                    if row in (2, 3):
                        ax.add_patch(
                            Rectangle(
                                (0, 0), 1, 1,
                                transform=ax.transAxes,
                                fill=False,
                                edgecolor=FP_COLOR if row == 2 else CORRECTED_FP_COLOR,
                                linewidth=1.2,
                                zorder=20,
                                clip_on=False,
                            )
                        )
                else:
                    if row == 3:
                        ax.add_patch(
                            Rectangle(
                                (0, 0), 1, 1,
                                transform=ax.transAxes,
                                fill=False,
                                edgecolor=CORRECTED_FP_COLOR,
                                linewidth=1.2,
                                zorder=20,
                                clip_on=False,
                                )
                        )
            else:
                ax.set_facecolor("#111")
                ax.text(
                    0.5, 0.5, "no image",
                    color="#eee", ha="center", va="center",
                    fontsize=4, transform=ax.transAxes
                )

            if row == 0:
                ax.text(
                    0.5, -0.12, f"frame# {f}",
                    fontsize=6, ha="center", va="top", transform=ax.transAxes
                )

            if i == 0:
                ax.annotate(
                    row_labels[row],
                    xy=(-0.35, 0.5), xycoords="axes fraction",
                    va="center", ha="right",
                    fontsize=8, fontweight="bold",
                    rotation=90
                )

            ax.axis("off")


def plot_track(ax, frames, kinds, y0, h, label, xmin, span, label_size=7, z=2):
    """Vẽ thanh dự đoán của 1 mô hình (nền xanh/đỏ theo kind)."""
    kind_map = {f: k for f, k in zip(frames, kinds)}
    full_frames = list(range(xmin, xmin + span))
    full_kinds = [kind_map.get(f, "normal") for f in full_frames]

    segs, start, last_kind = [], full_frames[0], full_kinds[0]
    for i in range(1, len(full_frames)):
        f, k = full_frames[i], full_kinds[i]
        if k != last_kind:
            segs.append((start, f - start, last_kind))
            start, last_kind = f, k
    segs.append((start, full_frames[-1] - start + 1, last_kind))

    for s, w, kind in segs:
        color = TN_COLOR if kind == "normal" else FP_COLOR
        ax.add_patch(Rectangle((s, y0), w, h, color=color, linewidth=0, zorder=z))

    ax.text(
        xmin - span * 0.012,
        y0 + h / 2,
        label,
        va="center", ha="right",
        fontsize=label_size, fontweight="semibold", zorder=z + 3
    )


def add_top_cap(ax, xmin, span, y0, h, cap_h=0.1, z=10):
    """Vẽ dải đen mỏng nằm *trên cùng* của mỗi thanh (đè lên màu)."""
    ax.add_patch(
        Rectangle(
            (xmin, y0 + h - cap_h),
            span, cap_h,
            facecolor=CAP_COLOR,
            edgecolor=CAP_COLOR,
            linewidth=0,
            zorder=z
        )
    )


def autosize_width(span: int, dpi: int, px_per_1k: float) -> float:
    w = (span / 1000.0) * (px_per_1k / dpi)
    return max(8.5, min(w, 18.0))


# ---------------- Plotting ----------------
def plot_case_level(
    out_png: Path,
    modelA,
    modelB,
    modelC,
    img_dir: Optional[Path],
    modelA_img_dir: Optional[Path],
    modelB_img_dir: Optional[Path],
    modelC_img_dir: Optional[Path],
    auto_keyframes: bool,
    max_keyframes: int,
    min_gap: int,
    manual_keyframes: Optional[List[int]],
    dpi: int = 200,
    px_per_1k: float = 220.0
):
    framesA = [p["f"] for p in modelA[1]]
    framesB = [p["f"] for p in modelB[1]]
    framesC = [p["f"] for p in modelC[1]]

    all_frames = sorted(set(framesA) | set(framesB) | set(framesC))
    xmin, xmax = min(all_frames), max(all_frames)
    span = xmax - xmin + 1

    # Keyframes (vẫn theo khác biệt A vs B)
    keyframes = []
    if auto_keyframes:
        keyframes = select_diff_keyframes(modelA[1], modelB[1], max_keyframes, min_gap)
    elif manual_keyframes:
        keyframes = sorted(manual_keyframes)

    modelA_map = {p["f"]: p["kind"] for p in modelA[1]}
    modelB_map = {p["f"]: p["kind"] for p in modelB[1]}
    modelC_map = {p["f"]: p["kind"] for p in modelC[1]}

    # Frames "Unclean-only lesion": A=lesion, B=normal (giữ nguyên logic gốc)
    unclean_only_frames = [
        f for f in range(xmin, xmax + 1)
        if modelA_map.get(f, "normal") == "lesion"
        and modelB_map.get(f, "normal") == "normal"
    ]

    # Frames "B-only lesion": B=lesion, C=normal (logic mới)
    b_only_frames = [
        f for f in range(xmin, xmax + 1)
        if modelB_map.get(f, "normal") == "lesion"
        and modelC_map.get(f, "normal") == "normal"
    ]

    width_in = autosize_width(span, dpi, px_per_1k)
    height_in = 7.2 if keyframes else 5.0

    fig = plt.figure(figsize=(width_in, height_in), dpi=dpi, facecolor="white")
    gs = GridSpec(2, 1, height_ratios=[1.25, 1.25], hspace=0.08, figure=fig)

    if keyframes:
        draw_keyframes_row(
            fig, gs[0],
            img_dir, modelA_img_dir, modelB_img_dir, modelC_img_dir,
            keyframes, modelA_map, modelB_map, modelC_map
        )
        ax = fig.add_subplot(gs[1])
    else:
        ax = fig.add_subplot(gs[0])

    if keyframes:
        legend_lines = [
            Line2D([0], [0], color=FP_COLOR, lw=1.5, label="FP (False Positive)"),
            Line2D([0], [0], color=CORRECTED_FP_COLOR, lw=1.5, label="Corrected FP"),
        ]
        fig.legend(
            handles=legend_lines,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.49),
            ncol=2,
            frameon=False,
            fontsize=7
        )

    # Geometry
    h, g = 3.5, 3.5
    y_gt = 10 + (h + g) * 3
    yA = y_gt - (h + g)
    yB = yA - (h + g)
    yC = yB - (h + g)

    # Ground Truth (giữ như code gốc: toàn normal)
    ax.broken_barh([(xmin, span)], (y_gt, h), facecolors=[TN_COLOR], zorder=1)
    ax.add_patch(Rectangle((xmin, y_gt), span, h, fill=False, edgecolor="black", linewidth=0.3, zorder=12))
    add_top_cap(ax, xmin, span, y_gt, h, z=15)
    ax.text(xmin - span * 0.012, y_gt + h / 2, "Ground Truth", va="center", ha="right",
            fontsize=7, fontweight="semibold", zorder=16)

    ax.annotate(
        "",
        xy=(xmax, y_gt + h + 1.2), xycoords="data",
        xytext=(xmin, y_gt + h + 1.2),
        arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
        annotation_clip=False
    )
    ax.text(xmin, y_gt + h + 1.6, "start frame", ha="center", va="bottom", fontsize=6)
    ax.text(xmax, y_gt + h + 1.6, "end frame", ha="center", va="bottom", fontsize=6)

    # Model A
    plot_track(ax, framesA, [p["kind"] for p in modelA[1]], yA, h, modelA[0], xmin, span, z=2)
    ax.add_patch(Rectangle((xmin, yA), span, h, fill=False, edgecolor="black", linewidth=0.6, zorder=12))
    add_top_cap(ax, xmin, span, yA, h, z=15)

    # Model B
    plot_track(ax, framesB, [p["kind"] for p in modelB[1]], yB, h, modelB[0], xmin, span, z=2)
    ax.add_patch(Rectangle((xmin, yB), span, h, fill=False, edgecolor="black", linewidth=0.6, zorder=12))

    # Overlay Corrected FP (vệt vàng) lên Model B theo logic gốc: (A lesion & B normal)
    for f in unclean_only_frames:
        ax.add_patch(Rectangle((f, yB), 1, h, facecolor=CORRECTED_FP_COLOR,
                               edgecolor=CORRECTED_FP_COLOR, linewidth=0, zorder=9))
    add_top_cap(ax, xmin, span, yB, h, z=15)

    # Model C
    plot_track(ax, framesC, [p["kind"] for p in modelC[1]], yC, h, modelC[0], xmin, span, z=2)
    ax.add_patch(Rectangle((xmin, yC), span, h, fill=False, edgecolor="black", linewidth=0.6, zorder=12))

    # Overlay Corrected FP (vệt vàng) lên Model C theo logic mới: (B lesion & C normal)
    for f in b_only_frames:
        ax.add_patch(Rectangle((f, yC), 1, h, facecolor=CORRECTED_FP_COLOR,
                               edgecolor=CORRECTED_FP_COLOR, linewidth=0, zorder=9))
    add_top_cap(ax, xmin, span, yC, h, z=15)

    # Ticks
    desired = int(min(10, max(6, round(width_in))))
    step = max(1, int(span / desired))
    ticks = list(range(xmin, xmax + 1, step))

    for t in ticks:
        for y in (y_gt, yA, yB, yC):
            ax.plot([t, t], [y + h, y + h - 0.4], color="black", linewidth=0.4, zorder=16)
            ax.plot([t, t], [y, y + 0.4], color="black", linewidth=0.4, zorder=16)
        txt = str(t).lstrip("0") or "0"
        ax.text(t, yC - 0.6, txt, ha="center", va="top", fontsize=5, color="black", zorder=16)

    # Keyframes marker
    for f in keyframes:
        ax.plot([f, f], [y_gt, y_gt + 0.4], color="black", linewidth=0.6, zorder=16)
        txt = str(f).lstrip("0") or "0"
        ax.text(f, y_gt - 0.6, txt, ha="center", va="top", fontsize=5,
                color="black", fontweight="bold", zorder=16)

    # Legend
    legend = [
        Patch(facecolor=TN_COLOR, edgecolor="black", label="TN (predict normal frame as normal)"),
        Patch(facecolor=FP_COLOR, edgecolor="black", label="FP (predict normal frame as lesion)"),
        Patch(
            facecolor=CORRECTED_FP_COLOR,
            edgecolor="black",
            label="Corrected FP"
        ),
    ]
    ax.legend(
        handles=legend,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=3,
        frameon=False,
        fontsize=7
    )

    ax.set_xlim(xmin, xmax + 1)
    ax.set_ylim(5, y_gt + h + 6)
    ax.set_yticks([])
    ax.set_xticks([])

    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)

    plt.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    print(f"Saved to: {out_png} | figsize=({width_in:.2f},{height_in:.2f})")


# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser(description="Case-level timeline visualization (3 models)")
    ap.add_argument("modelA_json", type=str)
    ap.add_argument("modelB_json", type=str)
    ap.add_argument("modelC_json", type=str)
    ap.add_argument("out_png", type=str)

    ap.add_argument("--modelA-name", default="Curated")
    ap.add_argument("--modelB-name", default="Diverse")
    ap.add_argument("--modelC-name", default="Diverse + Bubble")

    ap.add_argument("--img-dir", type=str, default=None)
    ap.add_argument("--modelA-img-dir", type=str, default=None)
    ap.add_argument("--modelB-img-dir", type=str, default=None)
    ap.add_argument("--modelC-img-dir", type=str, default=None)

    ap.add_argument("--auto-keyframes", action="store_true")
    ap.add_argument("--max-keyframes", type=int, default=8)
    ap.add_argument("--min-gap", type=int, default=530)
    ap.add_argument("--key-frames", dest="key_frames", type=int, nargs="*", default=[])

    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--px-per-1k", type=float, default=220.0)

    args = ap.parse_args()

    modelA_preds = load_preds(Path(args.modelA_json))
    modelB_preds = load_preds(Path(args.modelB_json))
    modelC_preds = load_preds(Path(args.modelC_json))

    img_dir = Path(args.img_dir) if args.img_dir else None
    modelA_img_dir = Path(args.modelA_img_dir) if args.modelA_img_dir else None
    modelB_img_dir = Path(args.modelB_img_dir) if args.modelB_img_dir else None
    modelC_img_dir = Path(args.modelC_img_dir) if args.modelC_img_dir else None

    plot_case_level(
        out_png=Path(args.out_png),
        modelA=(args.modelA_name, modelA_preds),
        modelB=(args.modelB_name, modelB_preds),
        modelC=(args.modelC_name, modelC_preds),
        img_dir=img_dir,
        modelA_img_dir=modelA_img_dir,
        modelB_img_dir=modelB_img_dir,
        modelC_img_dir=modelC_img_dir,
        auto_keyframes=args.auto_keyframes,
        max_keyframes=args.max_keyframes,
        min_gap=args.min_gap,
        manual_keyframes=args.key_frames if args.key_frames else None,
        dpi=args.dpi,
        px_per_1k=args.px_per_1k,
    )


if __name__ == "__main__":
    main()
