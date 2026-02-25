import json
import shutil
from pathlib import Path

DETECTION_DIR = Path(__file__).resolve().parents[2] / "detection"
SOURCE_DIR = DETECTION_DIR / "nid" 
DEST_DIR = DETECTION_DIR / "nid_yolo"

with open(SOURCE_DIR / "images.txt") as f:
    images = [line.strip().split(" ", 1)[1] for line in f]

with open(SOURCE_DIR / "tr_ID.txt") as f:
    split = [int(line.strip()) for line in f]

with open(SOURCE_DIR / "multi_boxes.json") as f:
    boxes_data = {entry["image"]: entry["objects"] for entry in json.load(f)}

for img_rel, is_train in zip(images, split):
    subset = "train" if is_train == 1 else "val"

    img_src = SOURCE_DIR / "images" / img_rel
    img_name = Path(img_rel).name

    img_dst = DEST_DIR / "images" / subset / img_name
    lbl_dst = DEST_DIR / "labels" / subset / img_name.replace(".JPG", ".txt")

    img_dst.parent.mkdir(parents=True, exist_ok=True)
    lbl_dst.parent.mkdir(parents=True, exist_ok=True)

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