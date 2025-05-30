from ultralytics import YOLO

# Input image URL
image_url = 'https://ultralytics.com/images/bus.jpg'

# --- Object Detection ---
# Load a pretrained YOLOv8 detection model
detection_model = YOLO('yolov8n.pt')

# Perform detection
print("Running object detection...")
detection_results = detection_model.predict(source=image_url, save=True, project="runs/detect_script", name="bus_detection_output")
print(f"Detection results saved to: {detection_results[0].save_dir}")
print(f"Detected image: {detection_results[0].path}") # Path to the saved image

# --- Image Segmentation ---
# Load a pretrained YOLOv8 segmentation model
segmentation_model = YOLO('yolov8n-seg.pt')

# Perform segmentation
print("\nRunning image segmentation...")
segmentation_results = segmentation_model.predict(source=image_url, save=True, project="runs/segment_script", name="bus_segmentation_output")
print(f"Segmentation results saved to: {segmentation_results[0].save_dir}")
print(f"Segmented image: {segmentation_results[0].path}") # Path to the saved image

print("\nDone! Check the 'runs' folder for output images.")