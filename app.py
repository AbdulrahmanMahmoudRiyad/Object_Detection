# import streamlit as st
# from ultralytics import YOLO
# from PIL import Image
# import torch
# import io

import streamlit as st
from ultralytics import YOLO
from PIL import Image

# Download YOLOv8 model weights if not already present
model = YOLO('yolov8n.pt')  # This will download the model if not already present

# Streamlit App
st.title("YOLOv8 Object Detection")
st.write("Upload an image and click 'Analyse Image' to see the detected objects.")

# Image Upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file)

    # Display the image
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")

    if st.button('Analyse Image'):
        st.write("Analysing...")

        # Perform object detection
        results = model(image)

        # Extract detected object names
        detected_classes = results[0].boxes.cls.cpu().numpy()  # Adjust this line based on the actual structure
        detected_names = [model.names[int(cls)] for cls in detected_classes]

        # Display results
        st.write("Detected Objects:")
        for name in detected_names:
            st.write(name)

        # Display annotated image
        annotated_image = results[0].plot()  # Render method to get annotated image
        st.image(annotated_image, caption='Annotated Image', use_column_width=True)
