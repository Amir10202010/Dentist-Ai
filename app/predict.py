from flask import Blueprint, request, jsonify
from ultralytics import YOLO
import cv2
import os
from werkzeug.utils import secure_filename

predict_bp = Blueprint("predict", __name__)

MODEL_PATH = os.path.join(os.getcwd(), "model", "best.pt")
model = YOLO(MODEL_PATH)
CONF_THRESH = 0.25
DEVICE = "cpu"

@predict_bp.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"success": False, "message": "Файл не загружен"}), 400

    file = request.files["image"]
    filename = secure_filename(file.filename)
    upload_dir = os.path.join("static", "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    image_path = os.path.join(upload_dir, filename)
    file.save(image_path)

    results = model.predict(image_path, conf=CONF_THRESH, verbose=False, device=DEVICE)
    boxes = results[0].boxes
    names = model.names
    image = results[0].orig_img

    output_dir = "static/predicts"
    os.makedirs(output_dir, exist_ok=True)
    existing = [d for d in os.listdir(output_dir) if d.startswith("res")]
    idx = len(existing) + 1
    run_dir = os.path.join(output_dir, f"res{idx}")
    os.makedirs(run_dir, exist_ok=True)

    result_paths = []

    for cls_id, cls_name in names.items():
        img_copy = image.copy()
        class_boxes = [b for b in boxes if int(b.cls[0]) == cls_id]
        if not class_boxes:
            continue

        for box in class_boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (255, 255, 0), 2)

        output_path = os.path.join(run_dir, f"{cls_name}.jpg")
        cv2.imwrite(output_path, img_copy)

        result_paths.append("/" + output_path.replace("\\", "/"))

    if not result_paths:
        return jsonify({"success": False, "message": "Патологии не обнаружены"}), 200

    return jsonify({"success": True, "results": result_paths})
