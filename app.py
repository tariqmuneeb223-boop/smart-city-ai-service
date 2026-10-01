from fastapi import FastAPI, File, UploadFile, Header, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
from ultralytics import YOLO

# ✅ Load environment variables
load_dotenv()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = FastAPI(title="Smart City Attock AI Service")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ AI Service API Key from environment
AI_SERVICE_KEY = os.getenv("AI_SERVICE_KEY")

# ✅ API Key verification function
def verify_api_key(x_api_key: str = Header(...)):
    if not AI_SERVICE_KEY:
        # If no key is set, allow requests (for development)
        return x_api_key
    if x_api_key != AI_SERVICE_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key

models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# ------------------------------
# 1. Issue classifier (garbage, other, pothole, streetlight)
# ------------------------------
issue_model_path = os.path.join(models_dir, "issue_classifier.keras")
issue_classes = ['garbage', 'other', 'pothole', 'streetlight']
issue_category_map = {
    'garbage': 'WASTE',
    'other': 'OTHER',
    'pothole': 'ROAD_DAMAGE',
    'streetlight': 'STREET_LIGHT'
}

print("=" * 50)
print("Loading issue classifier from:", issue_model_path)
print("File exists:", os.path.exists(issue_model_path))

issue_model = None

if os.path.exists(issue_model_path):
    issue_model = tf.keras.models.load_model(issue_model_path)
    print("✅ Issue classifier loaded successfully!")
    issue_model.summary()
else:
    print("❌ Issue classifier file not found!")

# ------------------------------
# 2. Garbage-type classifier
# ------------------------------
garbage_model_path = os.path.join(models_dir, "garbage_type_classifier.keras")
garbage_classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
garbage_category_map = {
    'cardboard': 'RECYCLABLE',
    'glass': 'RECYCLABLE',
    'metal': 'RECYCLABLE',
    'paper': 'RECYCLABLE',
    'plastic': 'RECYCLABLE',
    'trash': 'GENERAL'
}

print("\nLoading garbage-type classifier from:", garbage_model_path)
print("File exists:", os.path.exists(garbage_model_path))

garbage_model = None

if os.path.exists(garbage_model_path):
    garbage_model = tf.keras.models.load_model(garbage_model_path)
    print("✅ Garbage-type classifier loaded successfully!")
else:
    print("❌ Garbage-type classifier file not found!")

print(f"Issue classes: {issue_classes}")
print(f"Garbage classes: {garbage_classes}")
print("=" * 50)

# ------------------------------
# 3. YOLOv8 for vehicle counting
# ------------------------------
print("\nLoading YOLOv8 for vehicle counting...")
yolo_model = YOLO("yolov8n.pt")
VEHICLE_CLASSES = {"car", "bus", "truck", "motorcycle"}
VEHICLE_COUNT_THRESHOLD = 12
print("✅ YOLOv8 loaded successfully!")
print("=" * 50)

# ------------------------------
# Preprocessing
# ------------------------------
def load_image_array(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array.astype(np.float32)

def count_vehicles(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    results = yolo_model(image, verbose=False)[0]
    names = results.names
    count = 0
    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = names[cls_id]
        if label in VEHICLE_CLASSES:
            count += 1
    return count

def classify_garbage_type(img_array):
    if garbage_model is None:
        return None
    preds = garbage_model.predict(img_array, verbose=0)[0]
    idx = int(np.argmax(preds))
    confidence = float(preds[idx])
    predicted_class = garbage_classes[idx]
    return {
        "material_type": predicted_class,
        "material_category": garbage_category_map.get(predicted_class, "UNKNOWN"),
        "material_confidence": round(confidence * 100, 2),
        "material_all_scores": {
            name: round(float(preds[i]), 4)
            for i, name in enumerate(garbage_classes)
        }
    }

# ------------------------------
# Endpoints
# ------------------------------
@app.get("/")
async def root():
    return {
        "service": "Smart City Attock AI Service",
        "status": "running",
        "models": {
            "issue_classifier": issue_model is not None,
            "garbage_classifier": garbage_model is not None
        }
    }

@app.post("/classify-issue")
async def classify_issue(
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key)
):
    if issue_model is None:
        return {"success": False, "error": "Issue classifier not loaded"}

    try:
        contents = await file.read()

        # Step 1: check for traffic jam via vehicle counting (YOLO)
        vehicle_count = count_vehicles(contents)
        if vehicle_count >= VEHICLE_COUNT_THRESHOLD:
            return {
                "success": True,
                "class": "traffic",
                "category": "TRAFFIC_ISSUE",
                "confidence": None,
                "vehicle_count": vehicle_count,
                "message": f"Detected {vehicle_count} vehicles -- classified as traffic jam"
            }

        # Step 2: not a traffic jam -- run the CNN
        img_array = load_image_array(contents)
        preds = issue_model.predict(img_array, verbose=0)[0]
        idx = int(np.argmax(preds))
        confidence = float(preds[idx])
        predicted_class = issue_classes[idx]

        all_scores = {
            name: round(float(preds[i]), 4)
            for i, name in enumerate(issue_classes)
        }

        response = {
            "success": True,
            "class": predicted_class,
            "category": issue_category_map.get(predicted_class, "UNKNOWN"),
            "confidence": round(confidence * 100, 2),
            "vehicle_count": vehicle_count,
            "all_scores": all_scores
        }

        if predicted_class == 'garbage':
            material_result = classify_garbage_type(img_array)
            if material_result is not None:
                response.update(material_result)

        return response
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/classify-garbage")
async def classify_garbage(
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key)
):
    if garbage_model is None:
        return {"success": False, "error": "Garbage-type classifier not loaded"}
    try:
        contents = await file.read()
        img_array = load_image_array(contents)
        material_result = classify_garbage_type(img_array)

        return {
            "success": True,
            "class": material_result["material_type"],
            "category": material_result["material_category"],
            "confidence": material_result["material_confidence"],
            "all_scores": material_result["material_all_scores"]
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "issue_loaded": issue_model is not None,
        "garbage_loaded": garbage_model is not None,
        "tensorflow_version": tf.__version__,
        "api_key_configured": AI_SERVICE_KEY is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)