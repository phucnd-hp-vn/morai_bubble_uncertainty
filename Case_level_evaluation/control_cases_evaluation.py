import os
import glob
import json
import pandas as pd
import numpy as np

# ====== Cấu hình ======
PRED_DIR = "/json_control_cases"   # folder chứa *.json của ca chứng
POS_LABEL = "Ac tinh"       # dự đoán ác tính => FP đối với ca chứng
OUT_CSV = "control_framebyframe_per_case.csv"

rows = []
bad_json = []

json_paths = sorted(glob.glob(os.path.join(PRED_DIR, "*.json")))
if len(json_paths) == 0:
    raise FileNotFoundError(f"No .json files found in: {PRED_DIR}")

for path in json_paths:
    case_id = os.path.splitext(os.path.basename(path))[0]
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("JSON root is not a list.")
    except Exception as e:
        bad_json.append((case_id, str(e)))
        continue

    n_frames = len(data)
    fp = sum(1 for x in data if x.get("label") == POS_LABEL)
    tn = n_frames - fp

    # Các metric cho 1 video control
    specificity = (tn / n_frames) if n_frames > 0 else np.nan
    fp_rate = (fp / n_frames) if n_frames > 0 else np.nan
    any_fp = int(fp > 0)

    rows.append({
        "Case_ID": case_id,
        "Total_Frames_in_JSON": n_frames,
        "FP_frames_(pred_Ac_tinh)": fp,
        "TN_frames_(pred_not_Ac_tinh)": tn,
        "Specificity": specificity,
        "FP_frame_rate": fp_rate,
        "Any_FP": any_fp
    })

df = pd.DataFrame(rows).sort_values("Case_ID")
df.to_csv(OUT_CSV, index=False)

# ====== Tổng hợp (Panel B) ======
total_frames = df["Total_Frames_in_JSON"].sum()
FP_total = df["FP_frames_(pred_Ac_tinh)"].sum()
TN_total = df["TN_frames_(pred_not_Ac_tinh)"].sum()

overall_specificity = (TN_total / total_frames) if total_frames > 0 else np.nan
overall_fp_rate = (FP_total / total_frames) if total_frames > 0 else np.nan

pct_videos_any_fp = df["Any_FP"].mean() * 100.0 if len(df) > 0 else np.nan
avg_fp_per_video = df["FP_frames_(pred_Ac_tinh)"].mean() if len(df) > 0 else np.nan
median_fp_per_video = df["FP_frames_(pred_Ac_tinh)"].median() if len(df) > 0 else np.nan

print("=== CONTROL (Benign) Frame-by-frame Summary ===")
print(f"#Videos: {len(df)}")
print(f"Total frames (from JSON): {total_frames}")
print(f"FP frames (pred '{POS_LABEL}'): {FP_total}")
print(f"Specificity (%): {overall_specificity*100:.4f}")
print(f"FP frame rate (%): {overall_fp_rate*100:.4f}")
print(f"%Videos with any FP: {pct_videos_any_fp:.2f}")
print(f"Avg FP frames/video: {avg_fp_per_video:.3f}")
print(f"Median FP frames/video: {median_fp_per_video:.3f}")
print(f"Per-case CSV saved to: {OUT_CSV}")

if bad_json:
    print("\n[Warning] Some files could not be read:")
    for cid, err in bad_json:
        print(f"  - {cid}: {err}")


