import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

def create_pothole_model():
    """Create pothole model architecture"""
    IMG_SIZE = 224
    
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(inputs, outputs)
    return model

def create_garbage_binary_model():
    """Create binary garbage classifier (garbage vs clean)"""
    IMG_SIZE = 224
    
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(inputs, outputs)
    return model

def load_models():
    """Load both models with their weights"""
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    
    print("="*50)
    print("🤖 Loading models...")
    print("="*50)
    
    # Load pothole model
    pothole_model = create_pothole_model()
    pothole_weights_path = os.path.join(models_dir, "pothole_weights.weights.h5")
    
    if os.path.exists(pothole_weights_path):
        pothole_model.load_weights(pothole_weights_path)
        print("✅ Pothole model loaded")
    else:
        print(f"❌ Pothole weights not found at: {pothole_weights_path}")
    
    # Load binary garbage model
    garbage_model = create_garbage_binary_model()
    garbage_weights_path = os.path.join(models_dir, "garbage_binary_weights.weights.h5")
    
    if not os.path.exists(garbage_weights_path):
        garbage_weights_path = os.path.join(models_dir, "garbage_binary_model.keras")
    
    if os.path.exists(garbage_weights_path):
        if garbage_weights_path.endswith('.keras'):
            garbage_model = tf.keras.models.load_model(garbage_weights_path)
            print("✅ Binary Garbage model loaded from .keras file")
        else:
            garbage_model.load_weights(garbage_weights_path)
            print("✅ Binary Garbage model loaded from weights")
    else:
        print(f"❌ Garbage model not found at: {garbage_weights_path}")
    
    # Return 3 values: pothole_model, garbage_model, garbage_classes
    garbage_classes = ["clean", "garbage"]  # Binary classes
    print(f"   Classes: {garbage_classes}")
    
    return pothole_model, garbage_model, garbage_classes