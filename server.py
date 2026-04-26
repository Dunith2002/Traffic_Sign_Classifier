from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from keras.models import load_model
from PIL import Image
import numpy as np
import io

app = FastAPI()

# Allow browser to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load Model Once ---
model = load_model('./training/TSR_Augmented.h5')

CLASSES = {
    0:'Speed limit (20km/h)', 1:'Speed limit (30km/h)', 2:'Speed limit (50km/h)',
    3:'Speed limit (60km/h)', 4:'Speed limit (70km/h)', 5:'Speed limit (80km/h)',
    6:'End of speed limit (80km/h)', 7:'Speed limit (100km/h)', 8:'Speed limit (120km/h)',
    9:'No passing', 10:'No passing veh over 3.5 tons', 11:'Right-of-way at intersection',
    12:'Priority road', 13:'Yield', 14:'Stop', 15:'No vehicles',
    16:'Veh > 3.5 tons prohibited', 17:'No entry', 18:'General caution',
    19:'Dangerous curve left', 20:'Dangerous curve right', 21:'Double curve',
    22:'Bumpy road', 23:'Slippery road', 24:'Road narrows on the right',
    25:'Road work', 26:'Traffic signals', 27:'Pedestrians', 28:'Children crossing',
    29:'Bicycles crossing', 30:'Beware of ice/snow', 31:'Wild animals crossing',
    32:'End speed + passing limits', 33:'Turn right ahead', 34:'Turn left ahead',
    35:'Ahead only', 36:'Go straight or right', 37:'Go straight or left',
    38:'Keep right', 39:'Keep left', 40:'Roundabout mandatory',
    41:'End of no passing', 42:'End no passing veh > 3.5 tons'
}

CONFIDENCE_THRESHOLD = 0.80

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image bytes from the browser
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")

    # --- Same preprocessing as your OpenCV app ---
    img_resized = img.resize((48, 48))
    img_array = np.array(img_resized).astype('float32') / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_batch, verbose=0)
    class_id = int(np.argmax(predictions, axis=-1)[0])
    confidence = float(np.max(predictions))

    # Return top-3 as well for display
    top3_ids = np.argsort(predictions[0])[-3:][::-1].tolist()
    top3 = [
        {"label": CLASSES[i], "confidence": round(float(predictions[0][i]) * 100, 2)}
        for i in top3_ids
    ]

    return {
        "detected": confidence >= CONFIDENCE_THRESHOLD,
        "label": CLASSES[class_id] if confidence >= CONFIDENCE_THRESHOLD else None,
        "confidence": round(confidence * 100, 2),
        "top3": top3
    }

# Serve the frontend HTML from /static
app.mount("/", StaticFiles(directory="static", html=True), name="static")