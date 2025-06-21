import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from mobilenet_model import load_mobilenet_model
from sklearn.utils import class_weight
import numpy as np

# === Prepare directories ===
train_dir = 'data/train'
val_dir = 'data/val'
os.makedirs('model', exist_ok=True)

# === Data Augmentation ===
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# === Load Data ===
train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

print("Class indices:", train_generator.class_indices)

val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(
    'data/val',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# ✅ Step 2: Compute class weights
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.array([0, 1]),
    y=train_generator.classes
)
class_weights = dict(enumerate(class_weights))

# === Load Model ===
model = load_mobilenet_model()

checkpoint = ModelCheckpoint(
    'model/mobilenetv2_finetuned.h5',  # Save in HDF5 format
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# === Train ===
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=30,
    callbacks=[checkpoint, early_stopping],
    class_weight=class_weights
)

# At the end of epoch 1 (or another custom point)
sample_batch, sample_labels = next(val_generator)
preds = model.predict(sample_batch)

import numpy as np
print("Predicted class:", np.argmax(preds, axis=1))
print("Actual class:", np.argmax(sample_labels, axis=1))

# === Final Save ===
model.save('model/mobilenetv2_finetuned.h5')  # Final save in HDF5 format
