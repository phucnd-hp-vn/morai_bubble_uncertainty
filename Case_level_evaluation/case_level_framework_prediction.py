import os
import json
import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x))  # ổn định số học
    return e_x / e_x.sum()

# Gốc của tất cả các thư mục
base_dir = './baseline/baseline_original' # Folder contains original prediction txt and jpg files
output_dir = './json_baseline_original'  # Folder to save output JSONs
os.makedirs(output_dir, exist_ok=True)

# Danh sách nhãn
labels = ["Binh thuong", "Lanh tinh", "Ac tinh"]

# Duyệt qua từng thư mục P202 đến P300
for folder_name in sorted(os.listdir(base_dir)):
    folder_path = os.path.join(base_dir, folder_name)
    if os.path.isdir(folder_path):
        output_json = []

        # Duyệt từng file trong thư mục con
        for file in os.listdir(folder_path):
            if file.endswith(".txt"):
                txt_path = os.path.join(folder_path, file)
                with open(txt_path, 'r') as f:
                    logits = [float(line.strip()) for line in f if line.strip()]
                
                # Tính xác suất bằng softmax
                probs = softmax(np.array(logits))
                max_idx = int(np.argmax(probs))

                # Gán nhãn tương ứng
                output_json.append({
                    "id": file.replace(".txt", ""),  # ví dụ: frame_000000
                    "label": labels[max_idx],
                    "prob": round(float(probs[max_idx]), 5)
                })

        # Ghi JSON cho từng folder
        output_path = os.path.join(output_dir, f"{folder_name}.json")
        with open(output_path, "w") as f:
            json.dump(output_json, f, indent=4)
        print(f"✅ Saved: {output_path}")

print("🎉 All done!")


import os
import json
import numpy as np
import cv2

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# Paths
base_dir = './baseline/baseline_original' # Folder contains original prediction txt and jpg files
bubble_mask_dir = './bubbles_masks_output/output' # Folder contains bubble masks
output_dir = './json_combine' # Folder to save output JSONs with coverage check (Framework combine)
os.makedirs(output_dir, exist_ok=True)

labels = ["Binh thuong", "Lanh tinh", "Ac tinh"]
# labels = ["Normal", "Benign", "Malignant"]  # The same meaning in English

for folder_name in sorted(os.listdir(base_dir)):
    folder_path = os.path.join(base_dir, folder_name)
    bubble_folder = os.path.join(bubble_mask_dir, folder_name)
    if os.path.isdir(folder_path):
        output_json = []

        for file in os.listdir(folder_path):
            if file.endswith(".txt"):
                txt_path = os.path.join(folder_path, file)
                image_id = file.replace(".txt", "")  # ví dụ: frame_000000

                # Read logits
                with open(txt_path, 'r') as f:
                    logits = [float(line.strip()) for line in f if line.strip()]
                probs = softmax(np.array(logits))
                max_idx = int(np.argmax(probs))
                label = labels[max_idx]
                prob = round(float(probs[max_idx]), 5)

                # Load predicted mask (gốc) và mask bong bóng
                pred_mask_path = os.path.join(folder_path, image_id + '.jpg')
                bubble_mask_path = os.path.join(bubble_folder, image_id + '.jpg')

                if os.path.exists(pred_mask_path) and os.path.exists(bubble_mask_path):
                    pred1 = cv2.imread(pred_mask_path, cv2.IMREAD_GRAYSCALE)
                    bubble_mask = cv2.imread(bubble_mask_path, cv2.IMREAD_GRAYSCALE)
                    # print(bubble_mask_path)
                    # if bubble_mask is None:
                    #     raise ValueError("bubble_mask is None — check how it is created or loaded")

                    # Binarize
                    pred1_bin = (pred1 > 0).astype(np.uint8)
                    bubble_mask_bin = (bubble_mask > 0).astype(np.uint8)

                    # Coverage calculation
                    intersection = np.logical_and(bubble_mask_bin, pred1_bin).sum()
                    pred1_area = pred1_bin.sum()
                    coverage = intersection / (pred1_area + 1e-8)

                    if coverage > 0.3:
                        label = "Binh thuong"
                        prob = 1.0

                # Append result
                output_json.append({
                    "id": image_id,
                    "label": label,
                    "prob": prob
                })

        # Save per-folder JSON
        with open(os.path.join(output_dir, f"{folder_name}.json"), "w") as f:
            json.dump(output_json, f, indent=4)
        print(f"✅ Saved: {folder_name}.json")

print("🎉 All done with coverage check!")
