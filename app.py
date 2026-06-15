import streamlit as st
import cv2
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from shapely import wkt
from PIL import Image
import numpy as np
import pandas as pd
import io


st.set_page_config(page_title="Disaster Damage AI", layout="wide")
st.title("🚁 Post-Disaster Damage Assessment System")
st.markdown("Upload a post-disaster satellite image and its corresponding metadata to generate an instant AI damage report.")

@st.cache_resource # This stops the app from reloading the model every single time you click a button
def load_model():
    device = torch.device("cpu") # For a web app, CPU is usually safer unless deploying to a GPU server
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 2)
    # NOTE: You will need to download your .pth file from Google Drive and put it in the same folder as this script!
    # To get the .pth file you nned to run all the commands in the Google Colab, it gets saved in the Drive after training the Model
    model.load_state_dict(torch.load('resnet18_damage_assessment_v1.pth', map_location=device))
    model.eval()
    return model, device

model, device = load_model()

test_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

col1, col2 = st.columns(2)
with col1:
    uploaded_img = st.file_uploader("Upload Satellite Image (.png)", type=["png", "jpg", "jpeg"])
with col2:
    uploaded_json = st.file_uploader("Upload Metadata (.json)", type=["json"])

if uploaded_img is not None and uploaded_json is not None:
    st.info("Files uploaded successfully. Running AI Assessment...")
    
    # Read Image
    file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
    original_image = cv2.imdecode(file_bytes, 1)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    # Read JSON
    label_data = json.load(uploaded_json)
    buildings = label_data['features']['xy']
    
    damaged_count = 0
    safe_count = 0
    report_data = [] # To store data for our Excel/CSV export!

    progress_bar = st.progress(0)
    
    for i, building in enumerate(buildings):
        try:
            polygon_text = building['wkt']
            building_shape = wkt.loads(polygon_text)
            minx, miny, maxx, maxy = building_shape.bounds
            
            pad = 5
            crop_array = original_image[int(miny)-pad : int(maxy)+pad, int(minx)-pad : int(maxx)+pad]
            
            if crop_array.size == 0: continue
                
            crop_pil = Image.fromarray(crop_array)
            input_tensor = test_transforms(crop_pil).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(input_tensor)
                _, predicted = torch.max(output, 1)
                prediction = predicted.item()
                
            start_point = (int(minx)-pad, int(miny)-pad)
            end_point = (int(maxx)+pad, int(maxy)+pad)
            
            if prediction == 0: 
                color = (255, 0, 0) # Red
                damaged_count += 1
                status = "Damaged"
            else:               
                color = (0, 255, 0) # Green
                safe_count += 1
                status = "Safe"
                
            cv2.rectangle(original_image, start_point, end_point, color, 3)
            
            # Save data for the report
            report_data.append({"Building_ID": i, "Status": status, "X_Coord": minx, "Y_Coord": miny})
            
        except Exception as e:
            continue
            
        progress_bar.progress(min((i + 1) / len(buildings), 1.0))

    st.success("✅ Assessment Complete!")
    
    st.image(original_image, caption="AI Assessment Map (Red = Damaged, Green = Safe)", use_container_width=True)
    
    col3, col4, col5 = st.columns(3)
    col3.metric("Total Buildings", damaged_count + safe_count)
    col4.metric("Damaged", damaged_count)
    col5.metric("Safe", safe_count)

    df = pd.DataFrame(report_data)
    csv = df.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="📥 Download Damage Assessment Report (CSV)",
        data=csv,
        file_name='disaster_assessment_report.csv',
        mime='text/csv',
    )
