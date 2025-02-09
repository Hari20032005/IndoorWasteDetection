# Install required packages
!pip install ultralytics
!pip install ray==2.0.0

# Import necessary libraries
from ultralytics import YOLO
import torch
import yaml
from pathlib import Path
import os

# Import ray and patch it to avoid the _get_session error in the callback
import ray
# Patch ray.train._internal.session if _get_session is missing
if not hasattr(ray.train._internal.session, "_get_session"):
    ray.train._internal.session._get_session = lambda: None

# Define absolute paths for Kaggle
DATASET_PATH = "/kaggle/input/wastet"

# Create data.yaml file with absolute paths
data_yaml = {
    'train': f"{DATASET_PATH}/train/images",
    'val': f"{DATASET_PATH}/valid/images",
    'test': f"{DATASET_PATH}/test/images",
    'nc': 4,
    'names': ['cap', 'paper', 'plastic', 'shell']
}

# Save the yaml file
with open('data.yaml', 'w') as f:
    yaml.dump(data_yaml, f)

# Verify the paths
print("Verifying paths:")
print(f"Train path exists: {os.path.exists(data_yaml['train'])}")
print(f"Validation path exists: {os.path.exists(data_yaml['val'])}")
print(f"Test path exists: {os.path.exists(data_yaml['test'])}")

# Print the content of data.yaml
print("\nContent of data.yaml:")
with open('data.yaml', 'r') as f:
    print(f.read())

# Verify GPU availability
print("\nGPU Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))

# Initialize YOLOv8 model
model = YOLO('yolov8n.pt')

# Training configuration (note: the unsupported 'raytune' parameter has been removed)
args = {
    'data': str(Path('data.yaml').absolute()),  # Use absolute path
    'epochs': 50,
    'imgsz': 640,
    'batch': 8,
    'patience': 20,
    'device': 0,
    'workers': 2,
    'save': True,
    'cache': True,
    'project': 'waste_detection',
    'name': 'training_run',
    'exist_ok': True
}

# Train the model using keyword argument unpacking
try:
    print("\nStarting training...")
    results = model.train(**args)
    print("Training completed successfully!")
except Exception as e:
    print(f"An error occurred during training: {str(e)}")
    import traceback
    print(traceback.format_exc())

# Save the trained model
try:
    model.save('waste_detection_best.pt')
    print("\nModel saved successfully!")
except Exception as e:
    print(f"Error saving model: {str(e)}")

# Optional: Test the model on a few images
print("\nRunning test predictions...")
test_images_path = f"{DATASET_PATH}/test/images"
# Using '*.jpg' to match jpg files
test_images = list(Path(test_images_path).glob('*.jpg'))[:3]
for img_path in test_images:
    try:
        results = model.predict(str(img_path), save=True, conf=0.25)
        print(f"Processed: {img_path.name}")
    except Exception as e:
        print(f"Error processing {img_path.name}: {str(e)}")
