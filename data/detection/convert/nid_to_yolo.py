import json
import os
import shutil

yolo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root = os.path.join(os.path.dirname(yolo_root), "orig_datasets", "nid-dataset")


with open(os.path.join(root, "images.txt")) as f:
    images = [line.strip().split(" ", 1)[1] for line in f]

with open(os.path.join(root, "tr_ID.txt")) as f:
    split = [int(line.strip()) for line in f]

with open(os.path.join(root, "multi_boxes.json")) as f:
    boxes_data = {entry["image"]: entry["objects"] for entry in json.load(f)}

for img_rel, is_train in zip(images, split):
    subset = "train" if is_train == 1 else "val"

    img_src = os.path.join(root, "images", img_rel)
    # print(f"Processing {img_src} for subset {subset}...")
    img_name = os.path.basename(img_rel)

    img_dst = os.path.join(yolo_root, "datasets", "nid_det", "images", subset, img_name)
    lbl_dst = os.path.join(yolo_root, "datasets", "nid_det", "labels", subset, img_name.replace(".JPG", ".txt"))

    os.makedirs(os.path.dirname(img_dst), exist_ok=True)
    os.makedirs(os.path.dirname(lbl_dst), exist_ok=True)

    shutil.copy(img_src, img_dst)

    with open(lbl_dst, "w") as f:
        for obj in boxes_data.get(img_rel, []):
            x0, x1 = obj["x0"], obj["x1"]
            y0, y1 = obj["y0"], obj["y1"]

            xc = (x0 + x1) / 2
            yc = (y0 + y1) / 2
            w = x1 - x0
            h = y1 - y0

            f.write(f"0 {xc} {yc} {w} {h}\n")

print("Conversion finished.")
