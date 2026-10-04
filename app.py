# ✅ Set env vars BEFORE importing TensorFlow (prevents warnings, saves RAM)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.makedirs('/tmp/matplotlib', exist_ok=True)

from fastapi import FastAPI, File, UploadFile, Header, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from dotenv import load_dotenv
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import cv2
import onnxruntime as ort

# ✅ Load environment variables
load_dotenv()

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
def verify_api_key(x_api_key: str = Header(None)):
    if not AI_SERVICE_KEY:
        return x_api_key or "no-key"
    if x_api_key != AI_SERVICE_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key

models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# ------------------------------
# 1. Issue classifier
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
# 3. YOLOv8 ONNX for vehicle counting
# ------------------------------
print("\nLoading YOLOv8 ONNX model for vehicle counting...")
yolo_session = None
try:
    onnx_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolov8n.onnx")
    yolo_session = ort.InferenceSession(onnx_path)
    print("✅ YOLOv8 ONNX loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load YOLOv8 ONNX: {e}")

# COCO class IDs for vehicles: 2=car, 3=motorcycle, 5=bus, 7=truck
VEHICLE_CLASSES = {2, 3, 5, 7}
VEHICLE_COUNT_THRESHOLD = 12

# ------------------------------
# Warm up CNN models once at startup
# ------------------------------
print("🔥 Warming up CNN models...")
try:
    _dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
    if issue_model is not None:
        issue_model.predict(_dummy, verbose=0)
    if garbage_model is not None:
        garbage_model.predict(_dummy, verbose=0)
    print("✅ CNN models warmed up")
except Exception as e:
    print(f"⚠️ Warmup failed (non-fatal): {e}")
print("=" * 50)

# ------------------------------
# Preprocessing helpers
# ------------------------------
def load_image_array(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array.astype(np.float32)


def preprocess_yolo(image_bytes):
    """Preprocess image for YOLOv8 ONNX model (letterbox + normalize)."""
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = np.array(img)

    shape = img.shape[:2]
    new_shape = (640, 640)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )

    img = np.ascontiguousarray(img.transpose(2, 0, 1), dtype=np.float32) / 255.0
    return img[None]


def count_vehicles(image_bytes):
    """Runs YOLO ONNX on the image and counts vehicles."""
    if yolo_session is None:
        return 0

    input_tensor = preprocess_yolo(image_bytes)
    input_name = yolo_session.get_inputs()[0].name
    outputs = yolo_session.run(None, {input_name: input_tensor})[0]

    # Output shape: (1, 84, 8400) — 4 bbox + 80 class scores
    predictions = outputs[0].T  # (8400, 84)
    count = 0
    for pred in predictions:
        class_scores = pred[4:]
        class_id = int(np.argmax(class_scores))
        confidence = float(class_scores[class_id])
        if confidence > 0.5 and class_id in VEHICLE_CLASSES:
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
            "garbage_classifier": garbage_model is not None,
            "yolo_onnx": yolo_session is not None
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "issue_loaded": issue_model is not None,
        "garbage_loaded": garbage_model is not None,
        "yolo_loaded": yolo_session is not None,
        "tensorflow_version": tf.__version__,
        "api_key_configured": AI_SERVICE_KEY is not None
    }

@app.head("/health")
async def health_head():
    """Supports UptimeRobot's HEAD request so it shows 'Up' instead of 405."""
    return Response(status_code=200)

@app.post("/classify-issue")
async def classify_issue(
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key)
):
    if issue_model is None:
        return {"success": False, "error": "Issue classifier not loaded"}

    try:
        contents = await file.read()

        # ✅ Run CNN FIRST — fast, decides 95% of cases
        img_array = load_image_array(contents)
        preds = issue_model.predict(img_array, verbose=0)[0]
        idx = int(np.argmax(preds))
        confidence = float(preds[idx])
        predicted_class = issue_classes[idx]

        all_scores = {
            name: round(float(preds[i]), 4)
            for i, name in enumerate(issue_classes)
        }

        # ✅ Only run YOLO when CNN is uncertain (saves 3-5s on most images)
        vehicle_count = None
        if confidence < 0.6 or predicted_class == 'other':
            vehicle_count = count_vehicles(contents)
            if vehicle_count >= VEHICLE_COUNT_THRESHOLD:
                return {
                    "success": True,
                    "class": "traffic",
                    "category": "TRAFFIC_ISSUE",
                    "confidence": None,
                    "vehicle_count": vehicle_count,
                    "message": f"Detected {vehicle_count} vehicles — classified as traffic jam"
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

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)