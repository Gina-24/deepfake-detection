from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the trained model
model = load_model('model/mobilenetv2_finetuned.keras')

# Prepare the validation data
val_dir = 'data/val'
val_datagen = ImageDataGenerator(rescale=1./255)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical'
)

# Evaluate the model
loss, accuracy = model.evaluate(val_generator)
print(f"✅ Validation Accuracy: {accuracy * 100:.2f}%")
print(f"❌ Validation Loss: {loss:.4f}")
