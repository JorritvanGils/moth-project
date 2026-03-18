# this is how the old script looked like:from ultralytics import YOLO
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
yolo_root = os.path.abspath(os.path.join(base_dir, ".."))

image_path = os.path.join(yolo_root, "assets", "2020_08_08_Lichtfang_Hahnengrund_6729.JPG")
# image_path = os.path.join(yolo_root, "assets", "car.jpg")

use_pretrained = True
conf = 0.1  # 0.01 , 0.1, 0.25, 0.5, 0.75

if use_pretrained:
    model_path = os.path.join(yolo_root, "yolov8n.pt")
    model_name = "yolov8n.pt"
    model = YOLO(model_path)
    save_name = "inf/pretrained"
else:
    trained_model_path = os.path.join(
        yolo_root,
        "runs/eu_cls/runs_cpu/test_run/weights/best.pt"
        # "runs/nid_det/runs_cpu/test_run/weights/best.pt"
    )
    model_name = os.path.basename(trained_model_path)
    model = YOLO(trained_model_path)
    save_name = "inf/finetuned"

results = model.predict(
    image_path,
    conf=conf,
    iou=0.3,     # increase results in more detections (boxes have to overlap more to be considered the same object)
    save=True,
    project=os.path.join(yolo_root, "runs"),
    name=save_name,
    exist_ok=True
)
