import cv2
import numpy as np
from keras.models import load_model
from PIL import Image

# 1. Load the trained model
model = load_model('./training/TSR_Augmented.h5')

# 2. Classes Dictionary
classes = { 
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

# 3. Initialize Webcam
cap = cv2.VideoCapture(0)
window_name = "Real-Time Traffic Sign Recognition"

# Variables for Optimization
frame_count = 0
last_class_id = -1
last_confidence = 0.0

print("App Started. Hold sign in green box. Press 'q' or click 'X' to exit.")

while True:
    success, frame = cap.read()
    if not success:
        break

    frame_count += 1
    
    # Define ROI box
    height, width, _ = frame.shape
    box_size = 220
    x1, y1 = (width // 2 - box_size // 2), (height // 2 - box_size // 2)
    x2, y2 = (width // 2 + box_size // 2), (height // 2 + box_size // 2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # --- PREPROCESSING & PREDICTION ---
    # Only run the heavy math every 10th frame
    if frame_count % 10 == 0:
        crop_img = frame[y1:y2, x1:x2]
        
        if crop_img.size > 0:
            # Prepare image for model
            img_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb).resize((48, 48))
            # img_pil = Image.fromarray(img_rgb).resize((48, 48))
            img_normalized = np.array(img_pil).astype('float32') / 255.0
            img_batch = np.expand_dims(img_normalized, axis=0)

            # Predict and store results
            predictions = model.predict(img_batch, verbose=0) # verbose=0 silences the terminal logs
            last_class_id = np.argmax(predictions, axis=-1)[0]
            last_confidence = np.max(predictions)

    # --- DISPLAY LOGIC ---
    # Use the results from the 'last' prediction to avoid flickering
    if last_confidence > 0.80:
        label_text = f"{classes[last_class_id]} ({round(last_confidence*100, 2)}%)"
        cv2.putText(frame, label_text, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow(window_name, frame)

    # --- EXIT CONDITIONS ---
    # 1. Press 'q' key
    # 2. Click the 'X' button on the window
    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()