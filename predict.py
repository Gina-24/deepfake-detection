import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from utils import extract_face, generate_gradcam_heatmap, generate_image_heatmap
from heatmap_utils import overlay_heatmap
from noise_heatmap_utils import generate_noise_heatmap

UPLOAD_FOLDER = 'static/uploads'
HEATMAP_FOLDER = 'static/heatmaps'
OVERLAY_FOLDER = 'static/overlays'

model = load_model("model/mobilenetv2_finetuned.keras")

# -------------------------------
# IMAGE PREDICTION
# -------------------------------
def predict_image(filepath):
    img = cv2.imread(filepath)
    if img is None:
        raise ValueError(f"Could not read image: {filepath}")
    
    original_height, original_width = img.shape[:2]

    # Extract face
    face_crop, face_box_coords = extract_face(img)
    if face_crop is None:
        print("[WARN] No face found, using whole image.")
        face_crop = img
        face_box_coords = None

    # Preprocess
    img_resized = cv2.resize(face_crop, (224, 224))
    img_input = preprocess_input(np.expand_dims(img_resized.astype("float32"), axis=0))

    # Predict
    preds = model.predict(img_input)
    predicted_label = 'FAKE' if preds[0][0] > 0.5 else 'REAL'
    confidence = float(preds[0][0]) if predicted_label == 'FAKE' else 1 - float(preds[0][0])
    confidence_percentage = confidence * 100

    # File info
    base = os.path.basename(filepath)
    base_name, _ = os.path.splitext(base)
    heatmap_filename = f"{base_name}_heatmap.jpg"
    overlay_filename = f"{base_name}_overlay.jpg"
    heatmap_path = os.path.join(HEATMAP_FOLDER, heatmap_filename)
    overlay_path = os.path.join(OVERLAY_FOLDER, overlay_filename)

    os.makedirs(HEATMAP_FOLDER, exist_ok=True)
    os.makedirs(OVERLAY_FOLDER, exist_ok=True)

    # Grad-CAM
    heatmap = generate_gradcam_heatmap(img_input, model, last_conv_layer_name='Conv_1')

    # Save heatmap (with grunge if desired)
    generate_image_heatmap(filepath, heatmap_path)

    # Save overlay (abstract colorful gradient)
    overlay_heatmap(heatmap, filepath, output_path=overlay_path)

    return (
        predicted_label,
        confidence_percentage,
        heatmap_filename,
        overlay_filename,
        face_box_coords,
        round(os.path.getsize(filepath) / 1024, 2),
        original_width,
        original_height
    )

# -------------------------------
# VIDEO PREDICTION
# -------------------------------
def preprocess_video_frame(frame):
    frame = cv2.resize(frame, (224, 224))
    frame = preprocess_input(frame.astype("float32"))
    return np.expand_dims(frame, axis=0)

def generate_video_heatmap(file_path, heatmap_folder, interval=10, top_k=6):
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {file_path}")

    os.makedirs(HEATMAP_FOLDER, exist_ok=True)
    os.makedirs(OVERLAY_FOLDER, exist_ok=True)

    predictions = []
    frame_data = []
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % interval == 0:
            original_frame = frame.copy()
            input_tensor = preprocess_video_frame(frame)

            try:
                pred = model.predict(input_tensor)[0][0]
            except Exception as e:
                print(f"[ERROR] Prediction failed at frame {frame_index}: {e}")
                continue

            label = "FAKE" if pred > 0.5 else "REAL"
            confidence = pred if label == "FAKE" else 1 - pred
            confidence = float(confidence)

            frame_data.append((frame_index, original_frame, input_tensor, confidence))
            predictions.append(confidence)

        frame_index += 1

    cap.release()

    # Sort by confidence descending (most suspicious first)
    top_data = sorted(frame_data, key=lambda x: x[3], reverse=True)[:top_k]

    heatmap_paths = []
    overlay_paths = []

    for i, (f_index, frame, tensor, confidence) in enumerate(top_data):
        # Grad-CAM heatmap
        heatmap = generate_gradcam_heatmap(tensor, model, last_conv_layer_name='Conv_1')
        heatmap_resized = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
        heatmap_resized = np.uint8(255 * heatmap_resized)

        heatmap_filename = f"frame_{f_index}_heatmap.jpg"
        heatmap_path = os.path.join(HEATMAP_FOLDER, heatmap_filename)
        cv2.imwrite(heatmap_path, cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_BONE))
        heatmap_paths.append(heatmap_path)

        # Overlay
        overlay_filename = f"frame_{f_index}_overlay.jpg"
        overlay_path = os.path.join(OVERLAY_FOLDER, overlay_filename)
        try:
            overlay_result = overlay_heatmap(heatmap, frame, overlay_path)
            if not overlay_result:
                print(f"[WARN] Overlay failed for frame {f_index}")
        except Exception as e:
            print(f"[ERROR] Overlay error for frame {f_index}: {e}")
        overlay_paths.append(overlay_path)

    avg_confidence = round(np.mean(predictions) * 100, 2)
    overall_label = "FAKE" if np.mean(predictions) > 0.5 else "REAL"

    return overall_label, avg_confidence, heatmap_paths, overlay_paths

# -------------------------------
# UNIVERSAL MEDIA PREDICTION
# -------------------------------
def predict_media(file_path):
    ext = file_path.lower()
    if ext.endswith(('.png', '.jpg', '.jpeg')):
        return predict_image(file_path)
    elif ext.endswith(('.mp4', '.avi', '.mov')):
        return generate_video_heatmap(file_path, HEATMAP_FOLDER)
    else:
        return "Unsupported format", 0.0, [], []

# -------------------------------
# MAIN TESTING
# -------------------------------
if __name__ == "__main__":
    label, confidence, heatmap_filename, overlay_filename, *_ = predict_image("static/uploads/sample.jpg")
    print(f"[Image] {label} ({confidence:.2f}%) → Heatmap: {heatmap_filename}, Overlay: {overlay_filename}")

    label, confidence, heatmaps, overlays = generate_video_heatmap("static/uploads/sample.mp4", HEATMAP_FOLDER)
    print(f"[Video] {label} ({confidence:.2f}%) — Heatmaps: {len(heatmaps)} frames")
