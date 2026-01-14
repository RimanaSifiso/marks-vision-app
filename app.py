import streamlit as st
import cv2
import numpy as np
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

# --- PAGE CONFIG ---
st.set_page_config(page_title="Auto-Marker Prototype", page_icon="📝")

st.title("Marks Vision")
st.markdown("""
This app simulates an automatic marking system. 
1. Upload a photo of an assignment.
2. It detects **Red Pen** marks.
3. It uses **Microsoft TrOCR** to read the score.
""")

# --- MODEL LOADING (Cached) ---
# --- MODEL LOADING (Cached) ---
@st.cache_resource
def load_model():
    model_name = "microsoft/trocr-small-handwritten" 
    # ADD use_fast=False to avoid the tokenizer bug
    processor = TrOCRProcessor.from_pretrained(model_name, use_fast=False)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    return processor, model

with st.spinner("Loading AI Models... (this may take a minute)"):
    processor, model = load_model()

# --- HELPER FUNCTIONS ---

def extract_red_regions(image):
    """
    Converts PIL image to OpenCV, masks Red Ink, and finds bounding boxes.
    """
    # Convert PIL to OpenCV (RGB -> BGR)
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img_rgb = np.array(image) # Keep an RGB copy for cropping
    
    # Convert to HSV
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)

    # Red color range (Red wraps around 0/180)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    mask = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
    
    # Noise cleanup
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    results = []
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100: # Filter small noise
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Add padding
            pad = 10
            x_new = max(0, x - pad)
            y_new = max(0, y - pad)
            w_new = w + 2*pad
            h_new = h + 2*pad
            
            # Crop
            crop = img_rgb[y_new:y_new+h_new, x_new:x_new+w_new]
            pil_crop = Image.fromarray(crop)
            
            results.append({
                "bbox": (x, y, w, h),
                "crop": pil_crop
            })
            
    return results, img_cv

def predict_text(pil_image):
    pixel_values = processor(images=pil_image, return_tensors="pt").pixel_values
    with torch.no_grad():
        generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text

# --- MAIN APP LOGIC ---

uploaded_file = st.file_uploader("Upload an Exam Paper (Image)", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    # Display Original
    image = Image.open(uploaded_file).convert("RGB")
    
    # Process
    with st.spinner("Scanning for red marks..."):
        regions, img_cv_debug = extract_red_regions(image)

    if not regions:
        st.warning("No red marks detected! Try an image with clearer red ink.")
    else:
        st.success(f"Detected {len(regions)} red regions.")
        
        # Draw boxes on original image for visualization
        for region in regions:
            x, y, w, h = region['bbox']
            cv2.rectangle(img_cv_debug, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Convert back to RGB for Streamlit display
        st.image(cv2.cvtColor(img_cv_debug, cv2.COLOR_BGR2RGB), caption="Detected Regions", use_column_width=True)

        st.subheader("Recognized Marks:")
        
        # Display crops and predictions in columns
        for i, region in enumerate(regions):
            col1, col2 = st.columns([1, 3])
            
            with col1:
                st.image(region['crop'], width=100)
            
            with col2:
                # Run Inference
                prediction = predict_text(region['crop'])
                st.metric(label=f"Mark #{i+1}", value=prediction)