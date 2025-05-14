from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from PIL import Image
import io
from ultralytics import YOLO

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLOv8 model
MODEL_PATH = r"E:\Downloads\New folder (5)\corn.pt"  # Replace with your model path
model = YOLO(MODEL_PATH)

# Class ID to name mapping (update this as per your dataset)
CLASS_NAMES = {
    0: "Common Rust",
    1: "Gray Leaf Spot",
    2: "Leaf Blight",
}

class Prediction(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: list[float]

@app.post("/predict", response_model=list[Prediction])
async def predict(file: UploadFile = File(...)):
    try:
        # Read and preprocess image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Run inference
        results = model(image)

        # Parse results
        predictions = []
        for result in results[0].boxes.data:  # Access boxes directly
            x1, y1, x2, y2, conf, cls = result.tolist()

            cls_id = int(cls)
            predictions.append(Prediction(
                class_id=cls_id,
                class_name=CLASS_NAMES.get(cls_id, "Unknown"),
                confidence=round(float(conf), 3),
                bbox=[round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]
            ))

        return predictions
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        return []

@app.get("/")
def read_root():
    return {"status": "API is running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)