from ultralytics import YOLO
import os

# Load YOLO11n PyTorch model
model = YOLO("yolo11n.pt")

# Export to TensorRT (FP16)
engine_path = model.export(
    format="engine",
    half=True,
    imgsz=(640, 480)  # match dataset resolution
)

# Rename the exported engine file
new_engine_path = "yolo11n_float16.engine"
os.rename(engine_path, new_engine_path)

print(f"Exported TensorRT engine saved as: {new_engine_path}")

# Load the renamed TensorRT model
trt_model = YOLO(new_engine_path)

# Run inference
results = trt_model("https://ultralytics.com/images/bus.jpg")
results.show()

