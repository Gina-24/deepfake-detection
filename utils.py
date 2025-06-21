import os
import cv2
import uuid
import logging
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import mediapipe as mp
from matplotlib import cm
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from heatmap_utils import overlay_heatmap
from noise_heatmap_utils import generate_noise_heatmap

# === Frame Preprocessing ===
def preprocess_frame(frame):
    frame = cv2.resize(frame, (224, 224))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.astype(np.float32) / 255.0
    return np.expand_dims(frame, axis=0)

def preprocess_video_frame(frame):
    frame = cv2.resize(frame, (224, 224))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = img_to_array(frame)
    frame = preprocess_input(frame)
    return np.expand_dims(frame, axis=0)

# === Grad-CAM Heatmap Generator with Abstract Gradient ===
def generate_gradcam_heatmap(image_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)

    # Apply smooth abstract color gradient (no rainbow)
    heatmap = heatmap.numpy()
    heatmap = cv2.resize(heatmap, (224, 224))
    colormap = cm.get_cmap("plasma")
    colored_heatmap = colormap(heatmap)[:, :, :3]
    colored_heatmap = (colored_heatmap * 255).astype(np.uint8)
    return colored_heatmap

# === Artistic Heatmap with Grunge Style ===
def generate_image_heatmap(image_path, output_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    heatmap = np.absolute(laplacian)
    heatmap = cv2.normalize(heatmap, None, 0, 1.0, cv2.NORM_MINMAX)

    # Apply grunge/vintage-style colormap
    colormaps = ['magma', 'cividis', 'inferno', 'twilight_shifted', 'cubehelix']
    colormap = cm.get_cmap(np.random.choice(colormaps))
    colored = colormap(heatmap)[:, :, :3]
    colored = (colored * 255).astype(np.uint8)

    # Glitch effect
    for _ in range(2):
        row = np.random.randint(0, colored.shape[0])
        shift = np.random.randint(5, 15)
        colored[row] = np.roll(colored[row], shift, axis=0)

    # Add grunge noise
    noise = np.random.normal(0, 10, colored.shape).astype(np.uint8)
    textured = cv2.addWeighted(colored, 0.9, noise, 0.1, 0)

    # Save
    cv2.imwrite(output_path, cv2.cvtColor(textured, cv2.COLOR_RGB2BGR))

# === Laplacian Heatmap for Full Image ===
def generate_laplacian_heatmap(filepath):
    try:
        image = cv2.imread(filepath)
        if image is None:
            print(f"[ERROR] Failed to load: {filepath}")
            return None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_abs = np.absolute(laplacian)
        heatmap = np.uint8(255 * laplacian_abs / np.max(laplacian_abs))
        return heatmap
    except Exception as e:
        print(f"[ERROR] Laplacian error: {e}")
        return None

# === Face Detection (MediaPipe) ===
def extract_face(frame):
    mp_face_detection = mp.solutions.face_detection
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as detector:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.process(rgb)
        if results.detections:
            bbox = results.detections[0].location_data.relative_bounding_box
            h, w, _ = frame.shape
            x_min = int(bbox.xmin * w)
            y_min = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)
            x_max = min(x_min + width, w)
            y_max = min(y_min + height, h)
            face_crop = frame[max(y_min, 0):y_max, max(x_min, 0):x_max]
            return face_crop, {"top": y_min, "left": x_min, "width": width, "height": height}
    return None, None

# === Haar Cascade Face Box (Backup) ===
def detect_face_box(filepath):
    if not os.path.exists(filepath):
        logging.warning(f"Missing file: {filepath}")
        return None
    image = cv2.imread(filepath)
    if image is None:
        logging.error(f"Failed to load: {filepath}")
        return None
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except cv2.error as e:
        logging.error(f"cv2 error: {e}")
        return None

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) > 0:
        x, y, w, h = faces[0]
        return {"top": y, "left": x, "width": w, "height": h}
    else:
        logging.warning(f"No face in: {filepath}")
        return None

# === Main Video Processor (Efficient & Clean) ===
def process_video(file_path, filename, model, output_dir, interval=5, top_k=6):
    cap = cv2.VideoCapture(file_path)
    frame_count = 0
    predictions = []
    frame_nums = []
    confidence_scores = []
    suspicious_frames = []

    os.makedirs(output_dir, exist_ok=True)
    suspicious_dir = os.path.join(output_dir, f"suspicious_{filename}")
    os.makedirs(suspicious_dir, exist_ok=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % interval == 0:
            input_tensor = preprocess_video_frame(frame)
            pred = model.predict(input_tensor, verbose=0)[0][0]
            predictions.append(pred)
            confidence_scores.append(pred * 100)
            frame_nums.append(frame_count)
            suspicious_frames.append((frame, pred))
        frame_count += 1

    cap.release()

    avg = sum(predictions) / len(predictions) if predictions else 0
    confidence = round(avg * 100, 2)
    label = "Fake" if avg > 0.5 else "Real"

    # Save Laplacian Heatmap
    heatmap = generate_laplacian_heatmap(file_path)
    heatmap_filename = f"heatmap_{filename}.jpg"
    heatmap_path = os.path.join(output_dir, heatmap_filename)
    if heatmap is not None:
        cv2.imwrite(heatmap_path, heatmap)

    # Save Confidence Chart
    chart_filename = f"chart_{filename}.png"
    chart_path = os.path.join(output_dir, chart_filename)
    plt.figure(figsize=(8, 4))
    plt.plot(frame_nums, confidence_scores, marker='o')
    plt.title("Confidence Over Time")
    plt.xlabel("Frame Number")
    plt.ylabel("Fake Confidence (%)")
    plt.ylim(0, 100)
    plt.grid(True)
    plt.savefig(chart_path)
    plt.close()

    # Save Top-K Suspicious Frames
    top_k_indices = sorted(range(len(predictions)), key=lambda i: predictions[i], reverse=True)[:top_k]
    top_k_paths = []
    for i in top_k_indices:
        frame, score = suspicious_frames[i]
        suspicious_filename = os.path.join(suspicious_dir, f"suspicious_{i}_score_{int(score * 100)}.jpg")
        cv2.imwrite(suspicious_filename, frame)
        top_k_paths.append(suspicious_filename)

    return label, confidence, heatmap_filename, chart_filename, frame_nums, confidence_scores, top_k_paths
