import os
import json
import math
import numpy as np
import pandas as pd

# =========================
# CONFIG
# =========================
GT_DIR = "/home/ailab/Documents/Bubbles_Uncertainty/Case_level_evaluation/cancer_video_annotation/annotation_json_txt"
PRED_DIR = "/home/ailab/Documents/Bubbles_Uncertainty/Case_level_evaluation/lesion_output_verunclean_final/json_combine_cabenh"

POS_LABEL = "Ac tinh"          # malignant
NORMAL_LABEL = "Binh thuong"   # normal (bạn cần metric này)
BENIGN_LABEL = "Lanh tinh"     # optional (nếu muốn breakdown)

DELTA_RATIO = 0.10

USE_PROB_THRESHOLD = False
PROB_THRESHOLD = 0.5


# =========================
# HELPERS
# =========================
def read_gt_positive_frames(txt_path: str) -> set:
    if not os.path.exists(txt_path):
        return set()
    with open(txt_path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}

def read_pred_json(json_path: str):
    if not os.path.exists(json_path):
        return None, None, None

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"JSON root is not a list: {json_path}")

    frame_ids = []
    labels = []
    y_pred_pos = []

    for item in data:
        fid = item.get("id")
        lab = item.get("label")
        pr = item.get("prob")
        if not fid:
            continue

        frame_ids.append(fid)
        labels.append(lab if lab is not None else "")

        if USE_PROB_THRESHOLD and pr is not None:
            y_pred_pos.append(1 if float(pr) >= PROB_THRESHOLD else 0)
        else:
            y_pred_pos.append(1 if lab == POS_LABEL else 0)

    return frame_ids, np.array(labels, dtype=object), np.array(y_pred_pos, dtype=int)

def indices_to_segments(indices):
    if len(indices) == 0:
        return []
    segs = []
    s = indices[0]
    prev = indices[0]
    for x in indices[1:]:
        if x == prev + 1:
            prev = x
        else:
            segs.append((s, prev))
            s = x
            prev = x
    segs.append((s, prev))
    return segs

def expand_one_segment_by_ratio(s, e, n_frames, delta_ratio):
    length = e - s + 1
    delta = int(math.ceil(delta_ratio * length))
    s2 = max(0, s - delta)
    e2 = min(n_frames - 1, e + delta)
    return s2, e2

def segments_union_mask(segs, n_frames):
    mask = np.zeros(n_frames, dtype=bool)
    for s, e in segs:
        mask[s:e+1] = True
    return mask

def frame_confusion(y_true, y_pred):
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    return tp, fn, fp, tn

def safe_div(a, b):
    return float(a) / float(b) if b != 0 else np.nan


# =========================
# MAIN
# =========================
def main():
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.max_columns", 200)
    pd.set_option("display.width", 180)
    pd.set_option("display.float_format", lambda x: f"{x:.6f}" if isinstance(x, float) else str(x))

    case_ids = sorted([os.path.splitext(f)[0] for f in os.listdir(GT_DIR) if f.endswith(".txt")])

    per_case_rows = []
    missing_pred = []
    bad_json = []

    # Event micro totals
    total_events = 0
    detected_events = 0

    # For micro aggregation of "Pred_Normal_Rate_on_Neg"
    total_neg_frames = 0
    total_pred_normal_on_neg = 0

    # For micro aggregation of "Normal_Rate_in_GT_Window"
    total_gtw_frames = 0
    total_pred_normal_in_gtw = 0

    for cid in case_ids:
        gt_path = os.path.join(GT_DIR, f"{cid}.txt")
        pred_path = os.path.join(PRED_DIR, f"{cid}.json")

        gt_pos_set = read_gt_positive_frames(gt_path)
        try:
            frame_ids, pred_labels, y_pred_pos = read_pred_json(pred_path)
        except Exception as e:
            bad_json.append((cid, str(e)))
            continue

        if frame_ids is None:
            missing_pred.append(cid)
            continue

        n = len(frame_ids)
        pred_id_set = set(frame_ids)
        gt_missing = len(gt_pos_set - pred_id_set)

        # y_true per frame: 1 = GT malignant
        y_true = np.array([1 if fid in gt_pos_set else 0 for fid in frame_ids], dtype=int)

        # Frame-by-frame (binary malignant vs non-malignant)
        tp, fn, fp, tn = frame_confusion(y_true, y_pred_pos)
        acc = safe_div(tp + tn, n)
        prec_pos = safe_div(tp, tp + fp)
        rec_pos = safe_div(tp, tp + fn)
        f1_pos = safe_div(2 * prec_pos * rec_pos, (prec_pos + rec_pos)) if not (np.isnan(prec_pos) or np.isnan(rec_pos) or (prec_pos + rec_pos) == 0) else np.nan
        spec = safe_div(tn, tn + fp)
        fp_rate = safe_div(fp, n)

        # ===== Events from GT-positive frames =====
        gt_indices = np.where(y_true == 1)[0]
        gt_segs = indices_to_segments(gt_indices.tolist())
        num_events = len(gt_segs)

        expanded_segs = []
        event_detected_list = []
        for (s, e) in gt_segs:
            s2, e2 = expand_one_segment_by_ratio(s, e, n, DELTA_RATIO)
            expanded_segs.append((s2, e2))
            ed = int(np.any(y_pred_pos[s2:e2+1] == 1))
            event_detected_list.append(ed)

        num_events_detected = int(np.sum(event_detected_list)) if num_events > 0 else 0
        event_detected_rate_case = safe_div(num_events_detected, num_events) if num_events > 0 else np.nan

        total_events += num_events
        detected_events += num_events_detected

        # ===== Union window for case-level =====
        gt_mask = segments_union_mask(expanded_segs, n) if len(expanded_segs) > 0 else np.zeros(n, dtype=bool)

        if gt_mask.any():
            case_detected = int(np.any((y_pred_pos == 1) & gt_mask))
            coverage_malig = float(np.sum((y_pred_pos == 1) & gt_mask)) / float(np.sum(gt_mask))
        else:
            case_detected = np.nan
            coverage_malig = np.nan

        # =========================
        # NEW: metrics for NORMAL_LABEL ("Binh thuong")
        # =========================
        # A) On negative frames (outside GT malignant frames) within malignant videos:
        neg_mask = (y_true == 0)
        neg_count = int(np.sum(neg_mask))
        pred_normal_on_neg = int(np.sum((pred_labels == NORMAL_LABEL) & neg_mask))
        pred_normal_rate_on_neg = safe_div(pred_normal_on_neg, neg_count) if neg_count > 0 else np.nan

        total_neg_frames += neg_count
        total_pred_normal_on_neg += pred_normal_on_neg

        # B) Inside expanded GT window: how often model says "Binh thuong" (a miss mode)
        if gt_mask.any():
            gtw_count = int(np.sum(gt_mask))
            pred_normal_in_gtw = int(np.sum((pred_labels == NORMAL_LABEL) & gt_mask))
            normal_rate_in_gtw = safe_div(pred_normal_in_gtw, gtw_count)
        else:
            gtw_count = 0
            pred_normal_in_gtw = 0
            normal_rate_in_gtw = np.nan

        total_gtw_frames += gtw_count
        total_pred_normal_in_gtw += pred_normal_in_gtw

        # (Optional) if you want benign label rate too:
        if gt_mask.any():
            pred_benign_in_gtw = int(np.sum((pred_labels == BENIGN_LABEL) & gt_mask))
            benign_rate_in_gtw = safe_div(pred_benign_in_gtw, gtw_count)
        else:
            benign_rate_in_gtw = np.nan

        per_case_rows.append({
            "Case_ID": cid,
            "Total_Frames": n,
            "GT_Pos_Frames": int(np.sum(y_true)),
            "GT_Pos_Events": num_events,
            "GT_Missing_FrameIDs_NotFoundInPred": gt_missing,

            # Frame-by-frame binary
            "TP": tp, "FN": fn, "FP": fp, "TN": tn,
            "Accuracy": acc,
            "Precision_Pos": prec_pos,
            "Recall_Pos": rec_pos,
            "F1_Pos": f1_pos,
            "Specificity": spec,
            "FP_frame_rate": fp_rate,

            # Case-level
            "Delta_Ratio": DELTA_RATIO,
            "CaseDetected": case_detected,
            "Coverage_Malignant_in_GT_Window": coverage_malig,

            # Event-level
            "Events_Detected": num_events_detected,
            "EventDetectedRate_case": event_detected_rate_case,

            # NEW: Normal-related (proxy for reducing benign/normal noise)
            "Neg_Frames": neg_count,
            "Pred_Normal_on_Neg": pred_normal_on_neg,
            "Pred_Normal_Rate_on_Neg": pred_normal_rate_on_neg,

            "GT_Window_Frames": gtw_count,
            "Pred_Normal_in_GT_Window": pred_normal_in_gtw,
            "Normal_Rate_in_GT_Window": normal_rate_in_gtw,

            # Optional benign breakdown
            "Benign_Rate_in_GT_Window": benign_rate_in_gtw,
        })

    df = pd.DataFrame(per_case_rows).sort_values("Case_ID")
    df_pos = df[df["GT_Pos_Frames"] > 0].copy()

    # ===== Summary =====
    total_frames = int(df["Total_Frames"].sum())
    TP = int(df["TP"].sum()); FN = int(df["FN"].sum()); FP = int(df["FP"].sum()); TN = int(df["TN"].sum())

    acc_micro = safe_div(TP + TN, total_frames)
    prec_micro = safe_div(TP, TP + FP)
    rec_micro = safe_div(TP, TP + FN)
    f1_micro = safe_div(2 * prec_micro * rec_micro, (prec_micro + rec_micro)) if not (np.isnan(prec_micro) or np.isnan(rec_micro) or (prec_micro + rec_micro) == 0) else np.nan

    case_detection_rate = df_pos["CaseDetected"].mean() if len(df_pos) else np.nan
    cov_mean = df_pos["Coverage_Malignant_in_GT_Window"].mean() if len(df_pos) else np.nan
    cov_median = df_pos["Coverage_Malignant_in_GT_Window"].median() if len(df_pos) else np.nan

    event_micro = safe_div(detected_events, total_events) if total_events > 0 else np.nan
    event_macro = df_pos["EventDetectedRate_case"].mean() if len(df_pos) else np.nan

    # NEW: normal rates (micro + macro)
    normal_on_neg_micro = safe_div(total_pred_normal_on_neg, total_neg_frames) if total_neg_frames > 0 else np.nan
    normal_on_neg_macro = df_pos["Pred_Normal_Rate_on_Neg"].mean() if len(df_pos) else np.nan

    normal_in_gtw_micro = safe_div(total_pred_normal_in_gtw, total_gtw_frames) if total_gtw_frames > 0 else np.nan
    normal_in_gtw_macro = df_pos["Normal_Rate_in_GT_Window"].mean() if len(df_pos) else np.nan

    df_summary = pd.DataFrame([{
        "Num_Cases": len(df),
        "Num_Positive_Cases(GT>0)": len(df_pos),

        "Total_Frames": total_frames,
        "Accuracy_micro": acc_micro,
        "Precision_Pos_micro": prec_micro,
        "Recall_Pos_micro": rec_micro,
        "F1_Pos_micro": f1_micro,

        "CaseDetected_rate": case_detection_rate,
        "Coverage_mean": cov_mean,
        "Coverage_median": cov_median,

        "Total_GT_Events": total_events,
        "Detected_GT_Events": detected_events,
        "EventDetectedRate_micro": event_micro,
        "EventDetectedRate_macro": event_macro,

        # NEW
        "Pred_Normal_Rate_on_Neg_micro": normal_on_neg_micro,
        "Pred_Normal_Rate_on_Neg_macro": normal_on_neg_macro,
        "Normal_Rate_in_GT_Window_micro": normal_in_gtw_micro,
        "Normal_Rate_in_GT_Window_macro": normal_in_gtw_macro,

        "Delta_Ratio": DELTA_RATIO,
    }])

    # ===== PRINT =====
    def pct(x):
        return "nan" if (x is None or (isinstance(x, float) and np.isnan(x))) else f"{100.0*x:.2f}"

    s = df_summary.iloc[0].to_dict()
    print("\n===================== SUMMARY =====================")
    print(f"Num cases: {int(s['Num_Cases'])} | Positive cases (GT>0): {int(s['Num_Positive_Cases(GT>0)'])}")
    print(f"Frame-level micro: Acc={pct(s['Accuracy_micro'])}  Prec={pct(s['Precision_Pos_micro'])}  "
          f"Rec={pct(s['Recall_Pos_micro'])}  F1={pct(s['F1_Pos_micro'])}")

    print("\nCase-level (delta=10%):")
    print(f"  CaseDetected rate: {pct(s['CaseDetected_rate'])}")
    print(f"  Coverage (malignant in GT-window): mean={s['Coverage_mean']:.4f}  median={s['Coverage_median']:.4f}")

    print("\nEvent-level (delta=10%):")
    print(f"  Total GT events: {int(s['Total_GT_Events'])} | Detected: {int(s['Detected_GT_Events'])}")
    print(f"  EventDetectedRate micro: {pct(s['EventDetectedRate_micro'])}")
    print(f"  EventDetectedRate macro: {pct(s['EventDetectedRate_macro'])}")

    print("\nNormal-label analysis (proxy):")
    print(f"  Pred '{NORMAL_LABEL}' rate on NEG frames (micro): {pct(s['Pred_Normal_Rate_on_Neg_micro'])}")
    print(f"  Pred '{NORMAL_LABEL}' rate on NEG frames (macro): {pct(s['Pred_Normal_Rate_on_Neg_macro'])}")
    print(f"  Pred '{NORMAL_LABEL}' rate inside GT-window (micro): {pct(s['Normal_Rate_in_GT_Window_micro'])}")
    print(f"  Pred '{NORMAL_LABEL}' rate inside GT-window (macro): {pct(s['Normal_Rate_in_GT_Window_macro'])}")

    print("\n================== PER-CASE (compact) ==================")
    compact_cols = [
        "Case_ID",
        "GT_Pos_Events", "Events_Detected", "EventDetectedRate_case",
        "CaseDetected", "Coverage_Malignant_in_GT_Window",
        "Pred_Normal_Rate_on_Neg", "Normal_Rate_in_GT_Window",
        "FP_frame_rate"
    ]
    df_print = df[compact_cols].copy()
    for c in ["EventDetectedRate_case", "Coverage_Malignant_in_GT_Window", "Pred_Normal_Rate_on_Neg",
              "Normal_Rate_in_GT_Window", "FP_frame_rate"]:
        df_print[c] = df_print[c].apply(lambda v: np.nan if pd.isna(v) else 100.0*float(v))
    # print(df_print.to_string(index=False))

    if missing_pred:
        print("\n⚠️ Missing prediction JSON for cases:", missing_pred)
    if bad_json:
        print("\n⚠️ JSON read errors:")
        for cid, err in bad_json:
            print(f" - {cid}: {err}")

if __name__ == "__main__":
    main()
