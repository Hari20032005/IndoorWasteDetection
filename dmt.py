import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile

# Title and description for the app
st.title("Waste Object Detection with YOLOv8")
st.write("Upload an image to see the detection results.")

# File uploader for image input
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Button to trigger prediction
    if st.button("Predict"):
        # Save the uploaded image to a temporary file so that it can be read by the YOLO model
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
            image.save(tmp_file.name)

            # Load the YOLO model (adjust the model path as necessary)
            # If you have a saved model, use that. Otherwise, you can use a pretrained model.
            model = YOLO('waste_detection_best.pt')

            # Run the prediction. You can adjust the confidence threshold if needed.
            results = model.predict(source=tmp_file.name, conf=0.25, save=False)

        # Get the annotated image from the prediction result.
        # The `plot()` method returns the image with bounding boxes and labels.
        annotated_image = results[0].plot()

        # Display the annotated image
        st.image(annotated_image, caption="Prediction Output", use_column_width=True)

        # Optionally, display detection details (e.g., bounding boxes, class scores, etc.)
        st.write("Detection details:")
        st.write(results[0].boxes)
