# model_loader.py

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras import Input

def load_mobilenet_model(weights_path=None):
    # Input shape
    input_tensor = Input(shape=(224, 224, 3))
    
    # Base MobileNetV2
    base_model = MobileNetV2(include_top=False, input_tensor=input_tensor, weights='imagenet')
    
    # Custom head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)
    
    # Final model
    model = Model(inputs=input_tensor, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Optionally load custom fake/real weights
    if weights_path:
        model.load_weights(weights_path)
        print("✅ Custom weights loaded from:", weights_path)
    else:
        print("✅ Model loaded with ImageNet weights (transfer learning)")

    return model
