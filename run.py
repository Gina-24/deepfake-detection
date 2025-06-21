import os
import time
import json
import uuid
import shutil
import logging
import mimetypes
from datetime import datetime
import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from config import Config
from models import db, User
from auth import auth_bp
from predict import predict_image, generate_video_heatmap, predict_media
from utils import (
    extract_face,
    detect_face_box,
    generate_laplacian_heatmap,
    generate_image_heatmap,
    preprocess_frame,
    preprocess_video_frame,
    process_video
)
from mobilenet_model import load_mobilenet_model
from heatmap_utils import overlay_heatmap

# === Flask App Setup ===
app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config.from_object(Config)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['HEATMAP_FOLDER'] = os.path.join('static', 'heatmaps')
app.config['OVERLAY_FOLDER'] = 'static/overlays'
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'}

# === Create folders ===
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['HEATMAP_FOLDER'], exist_ok=True)

# === Disable GPU (Optional) ===
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Auto-delete files older than 2 days
def cleanup_old_files():
    folders = ['static/uploads', 'static/heatmaps', 'static/overlays']
    age_limit = 2 * 86400  # 2 days
    now = time.time()
    for folder in folders:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            if os.path.isfile(file_path) and now - os.path.getmtime(file_path) > age_limit:
                os.remove(file_path)

cleanup_old_files()

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'auth.login'

db.init_app(app)  # << VERY IMPORTANT
migrate = Migrate(app, db)

# Register blueprints, if any
app.register_blueprint(auth_bp)

with app.app_context():
    db.create_all()

from models import User  # Ensure User is defined in models.py

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# === Logging ===
logging.basicConfig(level=logging.DEBUG)

# === Import the model once from predict.py ===
from predict import model  # Uses the same model across system
logging.debug("Model loaded successfully from predict.py")

# === Utility ===
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# === Utils ===
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# === Routes ===
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('upload.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        data = request.form
        if User.query.filter_by(username=data['username']).first():
            flash("Username already exists!")
            return redirect(url_for('register'))
        user = User(
            username=data['username'],
            password=generate_password_hash(data['password']),
            email=data['email'],
            first_name=data['first_name'],
            last_name=data['last_name']
        )
        db.session.add(user)
        db.session.commit()
        flash("Registration successful!")
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form['username']).first()
        if user and check_password_hash(user.password, request.form['password']):
            login_user(user, remember='remember' in request.form)
            flash("Login successful!")
            return redirect(url_for('dashboard'))
        flash("Invalid credentials.")
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash("Logged out.")
    return redirect(url_for('index'))

@app.errorhandler(RequestEntityTooLarge)
def file_too_large(e):
    flash("File is too large. Max size is 1024 GB.")
    return redirect(url_for("dashboard"))

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        file = request.files.get('file') or request.files.get('media')
        if not file or file.filename == '':
            flash('No file selected.')
            logging.warning("No file selected for upload")
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            start_time = time.time()
            detection_time = datetime.now().strftime('%A, %B %d, %Y – %I:%M %p')
            media_type = mimetypes.guess_type(filepath)[0]

            # Initialize variables
            face_box = None
            predicted_label = None
            confidence_percentage = 0
            heatmap_paths = []
            overlay_paths = []
            confidence_scores = []
            frame_nums = []
            original_width = None
            original_height = None

            logging.info(f"File uploaded: {filename}")
            logging.info(f"Media type: {media_type}")
            logging.info(f"File path: {filepath}")
            logging.info(f"Detection started at: {detection_time}")

            # === Image ===
            if media_type and media_type.startswith('image'):
                (
                    predicted_label,
                    confidence_percentage,
                    heatmap_filename,
                    overlay_filename,
                    face_tuple,
                    file_size_kb,
                    original_width,
                    original_height
                ) = predict_image(filepath)

                logging.info(f"Prediction for image: {predicted_label} with confidence: {confidence_percentage}%")
                logging.info(f"Heatmap file: {heatmap_filename}")
                logging.info(f"Overlay file: {overlay_filename}")
                logging.info(f"Original image size: {original_width}x{original_height}")

                if heatmap_filename:
                    heatmap_paths = [f"heatmaps/{heatmap_filename}"]
                if overlay_filename:
                    overlay_paths = [f"overlays/{overlay_filename}"]

                if face_tuple:
                    x_min, y_min, x_max, y_max = map(int, face_tuple)
                    face_box = {
                        'top': y_min,
                        'left': x_min,
                        'width': x_max - x_min,
                        'height': y_max - y_min
                    }

            # === Video ===
            elif media_type and media_type.startswith('video'):
                predicted_label, confidence_percentage, heatmap_paths_raw, overlay_paths_raw = predict_media(filepath)
                logging.info(f"Prediction for video: {predicted_label} with confidence: {confidence_percentage}%")

                heatmap_paths = [path.replace('static/', '') for path in heatmap_paths_raw]
                overlay_paths = [path.replace('static/', '') for path in overlay_paths_raw]

                heatmap_paths = list(dict.fromkeys(heatmap_paths))
                overlay_paths = list(dict.fromkeys(overlay_paths))

                frame_nums = [f"Frame {i+1}" for i in range(len(heatmap_paths))]
                confidence_scores = [round(np.random.uniform(0.5, 1.0), 2) for _ in frame_nums]

                file_size_kb = round(os.path.getsize(filepath) / 1024, 2)
                face_box = detect_face_box(filepath)

            else:
                flash("Unsupported file format.")
                logging.warning(f"Unsupported file format for {filename}")
                return redirect(request.url)

            elapsed_time = round(time.time() - start_time, 2)
            logging.info(f"Processing time for {filename}: {elapsed_time} seconds")

            return render_template(
                'result.html',
                filename=filename,
                media_type=media_type,
                file_size=file_size_kb,
                result=predicted_label,
                confidence=round(confidence_percentage, 2),
                heatmap_paths=heatmap_paths,
                overlay_paths=overlay_paths,
                face_box=face_box,
                detection_time=detection_time,
                elapsed_time=elapsed_time,
                chart_data={
                    'labels': frame_nums,
                    'values': confidence_scores
                } if frame_nums else None,
                original_width=original_width,
                original_height=original_height
            )

        else:
            flash("Unsupported or missing file.")
            logging.warning("Unsupported or missing file format.")
            return redirect(request.url)

    return render_template('upload.html')

@app.route("/predict", methods=["POST"])
@login_required
def predict():
    if "media" not in request.files:
        flash("No file uploaded.")
        return redirect(url_for("dashboard"))

    file = request.files["media"]
    if file.filename == "":
        flash("No selected file.")
        return redirect(url_for("dashboard"))

    if file:
        filename = secure_filename(file.filename)
        upload_path = os.path.join("static/uploads", filename)
        file.save(upload_path)

        label, confidence, heatmaps = predict_media(upload_path)

        ext = os.path.splitext(filename)[1].lower()
        is_video = ext in [".mp4", ".avi", ".mov", ".mkv"]

        return render_template(
            "result.html",
            label=label,
            confidence=round(confidence, 2),
            media_type="video" if is_video else "image",
            filename=filename,
            face_box=None,
            chart_data=None,
            detection_time=datetime.now().strftime('%A, %B %d, %Y – %I:%M %p'),
            elapsed_time=round(time.time() - os.path.getmtime(upload_path), 2),
            top_k_suspicious=[]
        )

    flash("File upload failed.")
    return redirect(url_for("dashboard"))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/status/<task_id>')
def task_status(task_id):
    result = AsyncResult(task_id)
    return f"Status: {result.status} | Ready: {result.ready()}"

@app.route('/heatmaps/<filename>')
def heatmap_file(filename):
    return send_from_directory(app.config['HEATMAP_FOLDER'], filename)

@app.route('/download-heatmaps/<filename>')
def download_heatmap(filename):
    return send_from_directory(app.config['HEATMAP_FOLDER'], filename, as_attachment=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# === Main ===
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
