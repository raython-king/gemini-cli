# Modern Foundation Models Guide

Comprehensive guide for using latest CV foundation models with the multi-agent system.

## Overview

This guide covers state-of-the-art foundation models:

1. **Depth Anything V3**: Monocular depth estimation with metric depth
2. **DINO v2**: Self-supervised vision features (1.1B parameters)
3. **SAM 2**: Video and image segmentation with temporal propagation
4. **Grounding DINO**: Open-vocabulary object detection with natural language

## Quick Start

### Installation

```bash
# Install core dependencies
pip install torch torchvision transformers

# Install model-specific packages
pip install depth-anything-v3  # Depth Anything V3
pip install git+https://github.com/facebookresearch/dinov2.git  # DINO v2
pip install git+https://github.com/facebookresearch/segment-anything-2.git  # SAM 2
pip install groundingdino-py supervision  # Grounding DINO
```

### Using the Skill

```bash
# CLI (interactive mode)
gemini skill run cv.modern_foundation_model --interactive

# CLI (direct)
gemini skill run cv.modern_foundation_model \
  --model depth_anything_v3 \
  --task inference \
  --model-size base

# Programmatic
const result = await skillManager.executeSkill('cv.modern_foundation_model', {
  model: 'dinov2',
  task: 'feature_extraction',
  modelSize: 'large',
  applicationContext: {
    domain: 'medical imaging',
    specificTask: 'Extract features for disease classification',
    inputType: 'CT scans',
    outputRequirements: 'High-dimensional feature vectors',
  },
});
```

## Model Guides

### 1. Depth Anything V3

**Best for**: 3D reconstruction, AR/VR, robotics, autonomous driving

#### Features
- **Metric Depth**: Provides absolute depth values in meters (not just relative depth)
- **Zero-shot**: Works on any image without fine-tuning
- **Real-time**: Up to 50 FPS on modern GPUs (small model)
- **Robust**: Handles diverse scenes, lighting conditions

#### Basic Usage

```python
from transformers import pipeline
from PIL import Image
import numpy as np

# Load model
depth_estimator = pipeline(
    "depth-estimation",
    model="depth-anything/Depth-Anything-V3-Base",
    device="cuda",
)

# Single image
image = Image.open("scene.jpg")
result = depth_estimator(image)
depth_map = np.array(result["depth"])

# Get metric depth
metric_depth = result["predicted_depth"]  # in meters
print(f"Depth range: {metric_depth.min():.2f}m to {metric_depth.max():.2f}m")
```

#### Applications

**1. 3D Scene Reconstruction**
```python
def depth_to_pointcloud(depth, intrinsics):
    """Convert depth map to 3D point cloud"""
    h, w = depth.shape
    i, j = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')

    # Project to 3D
    z = depth
    x = (i - intrinsics['cx']) * z / intrinsics['fx']
    y = (j - intrinsics['cy']) * z / intrinsics['fy']

    points = np.stack([x, y, z], axis=-1)
    return points.reshape(-1, 3)

# Use case
depth = depth_estimator(image)["predicted_depth"]
pointcloud = depth_to_pointcloud(depth, camera_intrinsics)
save_ply("scene.ply", pointcloud)
```

**2. AR Occlusion**
```python
# Real-time depth for AR
while camera.is_open():
    frame = camera.read()
    depth = depth_estimator(frame)["depth"]

    # Render virtual object with proper occlusion
    virtual_obj = render_3d_object(position, rotation)
    occluded_obj = apply_occlusion(virtual_obj, depth)

    display(occluded_obj)
```

**3. Depth-Based Segmentation**
```python
# Segment objects by depth ranges
depth = depth_estimator(image)["predicted_depth"]

foreground = depth < 2.0  # Objects within 2 meters
midground = (depth >= 2.0) & (depth < 5.0)
background = depth >= 5.0

# Apply effects
image_blur_bg = blur(image, background)
```

#### Performance Tips
- Use **small model** for real-time applications (50 FPS)
- Use **large model** for highest quality depth (20 FPS)
- Enable **FP16** precision for 2x speedup
- Use **temporal filtering** for video to reduce flicker

---

### 2. DINO v2

**Best for**: Feature extraction, image retrieval, few-shot learning, dense prediction

#### Features
- **Self-supervised**: Trained without labels on 142M curated images
- **High-quality features**: 78.9% linear probe on ImageNet (base), 83.5% (giant)
- **Versatile**: Works for classification, segmentation, detection, retrieval
- **Scale**: From 21M to 1.1B parameters

#### Basic Usage

```python
from transformers import AutoModel, AutoImageProcessor
import torch

# Load model
model = AutoModel.from_pretrained("facebook/dinov2-base")
processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
model.eval()
model.cuda()

# Extract features
image = Image.open("image.jpg")
inputs = processor(images=image, return_tensors="pt").cuda()

with torch.no_grad():
    outputs = model(**inputs)

# Global features (CLS token)
global_features = outputs.pooler_output  # [1, 768]

# Patch features (for dense tasks)
patch_features = outputs.last_hidden_state[:, 1:, :]  # [1, 196, 768]
```

#### Applications

**1. Image Retrieval**
```python
# Build image database
database_features = []
for image in tqdm(database_images):
    inputs = processor(images=image, return_tensors="pt").cuda()
    with torch.no_grad():
        features = model(**inputs).pooler_output
    database_features.append(features.cpu())

database_features = torch.cat(database_features)

# Query
query_inputs = processor(images=query_image, return_tensors="pt").cuda()
with torch.no_grad():
    query_features = model(**query_inputs).pooler_output.cpu()

# Compute similarity
similarities = torch.nn.functional.cosine_similarity(
    query_features, database_features
)
top_k = similarities.argsort(descending=True)[:5]
```

**2. Few-Shot Classification**
```python
# Linear probing (frozen features)
import torch.nn as nn

classifier = nn.Linear(768, num_classes).cuda()

# Freeze backbone
for param in model.parameters():
    param.requires_grad = False

# Train only classifier
optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)

for images, labels in train_loader:
    images, labels = images.cuda(), labels.cuda()

    with torch.no_grad():
        features = model(images).pooler_output

    logits = classifier(features)
    loss = F.cross_entropy(logits, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

**3. Dense Prediction (Semantic Segmentation)**
```python
# Use patch features for segmentation
def segment_image(image, model, processor):
    inputs = processor(images=image, return_tensors="pt").cuda()

    with torch.no_grad():
        outputs = model(**inputs)

    # Get patch features
    patch_features = outputs.last_hidden_state[:, 1:, :]  # Exclude CLS
    B, N, D = patch_features.shape
    H = W = int(N ** 0.5)

    # Reshape to spatial grid
    features = patch_features.reshape(B, H, W, D)

    # Add segmentation head
    seg_head = nn.Conv2d(D, num_classes, 1).cuda()
    logits = seg_head(features.permute(0, 3, 1, 2))

    return logits

# Fine-tune on segmentation task
logits = segment_image(image, model, processor)
loss = F.cross_entropy(logits, target_masks)
```

#### Performance Tips
- Use **base model** for balanced performance (80 images/sec on A100)
- Use **giant model** for best features (15 images/sec)
- Extract features **offline** for large datasets
- Use **PCA** to reduce feature dimensionality if needed

---

### 3. SAM 2

**Best for**: Video segmentation, interactive annotation, object tracking

#### Features
- **Promptable**: Segment with points, boxes, or masks
- **Video support**: Temporal propagation across frames
- **Interactive**: Real-time feedback for annotation
- **Efficient**: Streaming architecture for long videos

#### Basic Usage

**Image Segmentation**
```python
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import numpy as np

# Load model
predictor = SAM2ImagePredictor(
    build_sam2("sam2_hiera_b+.yaml", "sam2_hiera_base_plus.pt")
)

# Set image
image = cv2.imread("image.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
predictor.set_image(image_rgb)

# Prompt with point
point = np.array([[500, 375]])  # x, y
label = np.array([1])  # 1=foreground, 0=background

masks, scores, _ = predictor.predict(
    point_coords=point,
    point_labels=label,
    multimask_output=True,
)

# Use best mask
best_mask = masks[np.argmax(scores)]
```

**Video Segmentation**
```python
from sam2.sam2_video_predictor import SAM2VideoPredictor

# Load video predictor
video_predictor = SAM2VideoPredictor(
    build_sam2("sam2_hiera_b+.yaml", "sam2_hiera_base_plus.pt")
)

# Initialize with video
with video_predictor.init_state(video_path="video.mp4") as state:
    # Add prompt on first frame
    _, _, _ = video_predictor.add_new_points(
        inference_state=state,
        frame_idx=0,
        obj_id=0,
        points=point,
        labels=label,
    )

    # Propagate through video
    for frame_idx, obj_ids, mask_logits in video_predictor.propagate_in_video(state):
        mask = (mask_logits[0] > 0.0).cpu().numpy()

        # Process mask for current frame
        save_mask(frame_idx, mask)
```

#### Applications

**1. Video Object Tracking**
```python
# Track multiple objects
with video_predictor.init_state(video_path="sports.mp4") as state:
    # Add first object (player #1)
    video_predictor.add_new_points(state, frame_idx=0, obj_id=0, points=player1_point)

    # Add second object (player #2)
    video_predictor.add_new_points(state, frame_idx=0, obj_id=1, points=player2_point)

    # Track both
    for frame_idx, obj_ids, masks in video_predictor.propagate_in_video(state):
        player1_mask = masks[0]
        player2_mask = masks[1]

        # Extract bounding boxes
        bbox1 = mask_to_bbox(player1_mask)
        bbox2 = mask_to_bbox(player2_mask)
```

**2. Interactive Annotation**
```python
# Progressive refinement with user feedback
mask = sam_predictor.predict(point_coords=initial_points)[0]

# User adds correction points
while not user_satisfied:
    # Show current mask
    display(mask)

    # Get user feedback
    correction_point, correction_label = get_user_input()

    # Refine mask
    all_points = np.concatenate([all_points, [correction_point]])
    all_labels = np.concatenate([all_labels, [correction_label]])

    mask, _, _ = sam_predictor.predict(
        point_coords=all_points,
        point_labels=all_labels,
    )
```

**3. Automatic Video Annotation**
```python
# Combine with object detector for automatic tracking
from groundingdino.util.inference import predict as grounding_dino_predict

# Detect objects in first frame
boxes, _, phrases = grounding_dino_predict(detector, first_frame, "person . car")

# Track each detected object
with video_predictor.init_state(video_path) as state:
    for obj_id, box in enumerate(boxes):
        video_predictor.add_new_points(
            state, frame_idx=0, obj_id=obj_id, box=box
        )

    # Propagate all objects
    for frame_idx, obj_ids, masks in video_predictor.propagate_in_video(state):
        save_all_masks(frame_idx, masks, phrases)
```

#### Performance Tips
- Use **tiny model** for real-time interactive annotation
- Use **base/large** for highest quality masks
- **Limit memory bank size** for very long videos
- **Re-prompt** objects after long occlusions

---

### 4. Grounding DINO

**Best for**: Open-vocabulary detection, visual grounding, referring expressions

#### Features
- **Open-vocabulary**: Detect any object with text queries
- **Zero-shot**: No training needed for new categories
- **Flexible queries**: Support compositional phrases
- **Integration-ready**: Works with SAM for instance segmentation

#### Basic Usage

```python
from groundingdino.util.inference import load_model, load_image, predict
import supervision as sv

# Load model
model = load_model(
    "groundingdino/swint_ogc.py",
    "groundingdino_swint_ogc.pth"
)

# Load image
image_source, image = load_image("scene.jpg")

# Text prompt (open vocabulary!)
TEXT_PROMPT = "person . car . traffic light . bicycle"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

# Detect
boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_THRESHOLD,
    text_threshold=TEXT_THRESHOLD,
)

# Visualize
detections = sv.Detections(xyxy=boxes)
labels = [f"{phrase} {logit:.2f}" for phrase, logit in zip(phrases, logits)]

annotator = sv.BoxAnnotator()
annotated = annotator.annotate(image_source.copy(), detections, labels)
```

#### Applications

**1. Open-Vocabulary Detection**
```python
# Detect any objects with natural language
queries = [
    "red car",
    "person wearing blue shirt",
    "smartphone on table",
    "coffee cup",
]

for query in queries:
    boxes, scores, phrases = predict(
        model, image, caption=query, box_threshold=0.3
    )
    print(f"{query}: {len(boxes)} detections")
```

**2. Grounding DINO + SAM 2 Pipeline**
```python
# Complete instance segmentation pipeline

# Step 1: Detect objects with Grounding DINO
boxes, _, phrases = predict(
    grounding_model,
    image,
    caption="person . dog . cat",
    box_threshold=0.35,
)

# Step 2: Segment each detected object with SAM
sam_predictor.set_image(image)
masks = []

for box in boxes:
    mask, _, _ = sam_predictor.predict(
        box=box[None, :],
        multimask_output=False,
    )
    masks.append(mask[0])

# Step 3: Visualize
overlay = image.copy()
for i, (mask, phrase) in enumerate(zip(masks, phrases)):
    color = np.random.randint(0, 255, 3).tolist()
    overlay[mask] = overlay[mask] * 0.5 + np.array(color) * 0.5
    add_label(overlay, phrase, box[i])
```

**3. Referring Expression Grounding**
```python
# Ground complex referring expressions
referring_expressions = [
    "the person on the left wearing a red hat",
    "the largest car in the parking lot",
    "the dog sitting next to the person",
]

for expression in referring_expressions:
    boxes, scores, _ = predict(
        model,
        image,
        caption=expression,
        box_threshold=0.25,  # Lower threshold for specific queries
    )

    # Get most confident detection
    if len(boxes) > 0:
        best_box = boxes[scores.argmax()]
        print(f"'{expression}' grounded to: {best_box}")
```

**4. Zero-Shot Object Counting**
```python
# Count objects of specific type
def count_objects(image, object_type):
    boxes, _, _ = predict(
        model, image, caption=object_type, box_threshold=0.35
    )
    return len(boxes)

# Use case
num_people = count_objects(image, "person")
num_cars = count_objects(image, "car")
print(f"Scene contains {num_people} people and {num_cars} cars")
```

#### Performance Tips
- **Clear prompts**: Use specific, unambiguous text
- **Adjust thresholds**: Lower box_threshold for rare objects
- **Batch processing**: Process multiple images together
- **Fine-tune**: On domain data for specialized applications

---

## Integration Examples

### 1. Complete 3D Scene Understanding

Combine multiple models for comprehensive scene analysis:

```python
from modern_foundation_models import DepthAnythingV3, DINOv2, SAM2, GroundingDINO

# Load models
depth_model = DepthAnythingV3("base")
feature_model = DINOv2("base")
segment_model = SAM2("base")
detect_model = GroundingDINO("base")

# Process image
image = load_image("scene.jpg")

# 1. Detect objects
boxes, _, labels = detect_model.predict(image, "person . car . tree . building")

# 2. Estimate depth
depth = depth_model.estimate(image)

# 3. Segment each object
segments = []
for box in boxes:
    mask = segment_model.segment(image, box=box)
    segments.append(mask)

# 4. Extract features
features = feature_model.extract(image)

# 5. Build 3D representation
scene_3d = {
    'objects': [
        {
            'label': label,
            'box': box,
            'mask': mask,
            'depth': depth[mask].mean(),
            'features': features[mask].mean(axis=0),
        }
        for label, box, mask in zip(labels, boxes, segments)
    ],
    'depth_map': depth,
    'point_cloud': depth_to_pointcloud(depth, camera_intrinsics),
}

save_scene("scene_3d.pkl", scene_3d)
```

### 2. AR/VR Application

Real-time scene understanding for augmented reality:

```python
import cv2

# Initialize models
depth_estimator = DepthAnythingV3("small")  # Fast model
object_detector = GroundingDINO("tiny")  # Fast detection
segmenter = SAM2("tiny")

# AR application loop
camera = cv2.VideoCapture(0)

while True:
    # Capture frame
    ret, frame = camera.read()
    if not ret:
        break

    # Estimate depth (real-time)
    depth = depth_estimator.estimate(frame)

    # Detect surfaces for placing virtual objects
    boxes, _, labels = object_detector.predict(
        frame, "floor . table . wall", box_threshold=0.4
    )

    # Place virtual object
    if "table" in labels:
        table_idx = labels.index("table")
        table_box = boxes[table_idx]
        table_depth = depth[table_box].mean()

        # Render 3D object on table
        virtual_obj = render_3d_model(
            position=table_box.center,
            depth=table_depth,
            scale=compute_scale(table_box, depth),
        )

        # Apply occlusion using depth
        occluded = apply_depth_occlusion(virtual_obj, depth)

        # Composite
        frame = composite(frame, occluded)

    cv2.imshow("AR View", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
```

### 3. Video Understanding Pipeline

Comprehensive video analysis:

```python
# Load video
video_path = "input_video.mp4"
cap = cv2.VideoCapture(video_path)

# Initialize models
detector = GroundingDINO("base")
tracker = SAM2("base")
feature_extractor = DINOv2("base")

# Detect objects in first frame
ret, first_frame = cap.read()
boxes, _, labels = detector.predict(
    first_frame, "person . vehicle . animal", box_threshold=0.35
)

# Track objects through video
with tracker.init_video_state(video_path) as state:
    # Initialize tracking
    for obj_id, box in enumerate(boxes):
        tracker.add_object(state, frame_idx=0, obj_id=obj_id, box=box)

    # Process each frame
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Get masks for all tracked objects
        masks = tracker.get_masks(state, frame_idx)

        # Extract features for each object
        for obj_id, mask in enumerate(masks):
            # Crop object region
            obj_region = frame[mask]

            # Extract features
            features = feature_extractor.extract(obj_region)

            # Store for analysis
            save_features(frame_idx, obj_id, labels[obj_id], features)

        frame_idx += 1

cap.release()

# Analyze extracted data
analyze_video_features("video_features.pkl")
```

### 4. Medical Imaging Application

Specialized application for medical image analysis:

```python
# Load medical image (CT scan)
ct_scan = load_dicom("patient_scan.dcm")

# 1. Extract features with DINOv2 (pre-trained on natural images)
feature_model = DINOv2("large")
features = feature_model.extract(ct_scan)

# 2. Fine-tune on medical data
medical_classifier = nn.Linear(feature_model.hidden_size, num_diseases)

# Train classifier (frozen features)
for ct_images, diagnoses in medical_dataloader:
    with torch.no_grad():
        features = feature_model(ct_images).pooler_output

    logits = medical_classifier(features)
    loss = F.cross_entropy(logits, diagnoses)
    # ... training loop

# 3. Detect anomalies with Grounding DINO (fine-tuned)
grounding_model = GroundingDINO("base")
# Fine-tune on medical images with radiologist annotations

anomalies, scores, descriptions = grounding_model.predict(
    ct_scan,
    caption="tumor . lesion . fracture . inflammation",
    box_threshold=0.4,
)

# 4. Segment anomalies precisely with SAM
sam = SAM2("base")
sam.set_image(ct_scan)

for anomaly_box in anomalies:
    mask, _, _ = sam.predict(box=anomaly_box)
    # Measure size, shape, location
    analysis = analyze_anomaly(mask, ct_scan)
    report.add(analysis)

# 5. Generate report
generate_medical_report(anomalies, analyses, confidences)
```

## CLI Usage

### Modern Foundation Model Skill

```bash
# Depth Anything V3 - Depth estimation
gemini skill run cv.modern_foundation_model \
  --model depth_anything_v3 \
  --task inference \
  --model-size base \
  --application-domain "robotics" \
  --specific-task "Depth for obstacle avoidance" \
  --input-type "camera stream" \
  --output-requirements "Real-time depth maps"

# DINO v2 - Feature extraction
gemini skill run cv.modern_foundation_model \
  --model dinov2 \
  --task feature_extraction \
  --model-size large \
  --application-domain "e-commerce" \
  --specific-task "Product image similarity search" \
  --input-type "product images" \
  --output-requirements "Feature vectors for retrieval"

# SAM 2 - Video segmentation
gemini skill run cv.modern_foundation_model \
  --model sam2 \
  --task inference \
  --model-size base \
  --application-domain "video editing" \
  --specific-task "Background removal in videos" \
  --input-type "video files" \
  --output-requirements "Segmentation masks per frame"

# Grounding DINO - Object detection
gemini skill run cv.modern_foundation_model \
  --model grounding_dino \
  --task inference \
  --model-size base \
  --application-domain "surveillance" \
  --specific-task "Detect specific objects with text queries" \
  --input-type "security camera feeds" \
  --output-requirements "Bounding boxes with labels"

# Fine-tuning example
gemini skill run cv.modern_foundation_model \
  --model dinov2 \
  --task fine_tuning \
  --model-size base \
  --dataset-path "/data/medical_images" \
  --num-epochs 20 \
  --learning-rate 1e-5 \
  --application-domain "medical imaging" \
  --specific-task "Disease classification"
```

## Performance Benchmarks

### Depth Anything V3

| Model Size | Params | Speed (FPS) | Memory (GB) | AbsRel (NYU) |
|------------|--------|-------------|-------------|--------------|
| Small      | 24M    | ~50         | ~2          | 0.058        |
| Base       | 97M    | ~35         | ~4          | 0.055        |
| Large      | 335M   | ~20         | ~6          | 0.052        |

### DINO v2

| Model Size | Params | Speed (imgs/sec) | Memory (GB) | ImageNet Linear Probe |
|------------|--------|------------------|-------------|-----------------------|
| Small      | 21M    | ~120             | ~1.5        | 79.0%                 |
| Base       | 86M    | ~80              | ~3          | 78.9%                 |
| Large      | 300M   | ~40              | ~8          | 81.1%                 |
| Giant      | 1.1B   | ~15              | ~25         | 83.5%                 |

### SAM 2

| Model Size | Params | Speed (FPS) | Memory (GB) | J&F (DAVIS) |
|------------|--------|-------------|-------------|-------------|
| Tiny       | 38M    | ~35         | ~2          | 72.0        |
| Small      | 95M    | ~25         | ~4          | 74.5        |
| Base+      | 224M   | ~15         | ~8          | 76.2        |

### Grounding DINO

| Model Size | Params | Speed (FPS) | Memory (GB) | AP (COCO zero-shot) |
|------------|--------|-------------|-------------|---------------------|
| Tiny       | 28M    | ~18         | ~4          | 48.0                |
| Base       | 218M   | ~12         | ~8          | 52.5                |

## Best Practices

### 1. Model Selection

- **Depth Anything V3**: Choose based on speed/quality tradeoff
  - Small: Real-time applications, robotics
  - Base: Balanced (recommended for most uses)
  - Large: Highest quality depth maps

- **DINO v2**: Choose based on task requirements
  - Small/Base: Fast feature extraction
  - Large: Better transfer learning
  - Giant: Best features for critical applications

- **SAM 2**: Choose based on use case
  - Tiny: Interactive annotation, real-time
  - Small/Base: Video segmentation, tracking

- **Grounding DINO**: Choose based on vocabulary size
  - Tiny: Limited vocabulary, fast
  - Base: Open vocabulary, best accuracy

### 2. Optimization

**Memory Optimization**:
```python
# Gradient checkpointing
model.gradient_checkpointing_enable()

# Mixed precision
from torch.cuda.amp import autocast
with autocast():
    output = model(input)

# Batch size tuning
optimal_batch_size = find_optimal_batch_size(model, device)
```

**Speed Optimization**:
```python
# TorchScript compilation
model.eval()
traced_model = torch.jit.trace(model, example_input)

# ONNX export
torch.onnx.export(model, example_input, "model.onnx")

# TensorRT (NVIDIA)
import tensorrt as trt
# Convert to TensorRT for max speed
```

### 3. Deployment

**Edge Deployment**:
```python
# Quantize to INT8
from torch.quantization import quantize_dynamic
quantized_model = quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Mobile deployment
from torch.utils.mobile_optimizer import optimize_for_mobile
mobile_model = optimize_for_mobile(traced_model)
mobile_model._save_for_lite_interpreter("model_mobile.ptl")
```

**Cloud Deployment**:
```python
# FastAPI service
from fastapi import FastAPI, File, UploadFile
from PIL import Image

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file)
    result = model(image)
    return {"result": result}

# Docker container
# See Dockerfile example in repository
```

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```python
# Solution: Reduce batch size or use gradient accumulation
torch.cuda.empty_cache()
model.gradient_checkpointing_enable()
```

**2. Slow Inference**
```python
# Solution: Enable mixed precision and optimize model
model.half()  # Convert to FP16
model.eval()  # Disable dropout, etc.
torch.backends.cudnn.benchmark = True  # Optimize for fixed input sizes
```

**3. Poor Detection Quality (Grounding DINO)**
```python
# Solution: Adjust thresholds and refine prompts
boxes, logits, phrases = predict(
    model, image,
    caption="specific descriptive query",  # Be specific!
    box_threshold=0.25,  # Lower for rare objects
    text_threshold=0.20,  # Adjust text matching
)
```

**4. Video Segmentation Loses Track (SAM 2)**
```python
# Solution: Re-prompt after occlusions
if occlusion_detected:
    video_predictor.add_new_points(
        state, frame_idx=current_frame,
        obj_id=obj_id, points=new_point
    )
```

## Resources

### Official Repositories
- [Depth Anything V3](https://github.com/DepthAnything/Depth-Anything-V3)
- [DINO v2](https://github.com/facebookresearch/dinov2)
- [SAM 2](https://github.com/facebookresearch/segment-anything-2)
- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO)

### Papers
- **Depth Anything V3**: "Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data" (2024)
- **DINO v2**: "DINOv2: Learning Robust Visual Features without Supervision" (2023)
- **SAM 2**: "SAM 2: Segment Anything in Images and Videos" (2024)
- **Grounding DINO**: "Grounding DINO: Marrying DINO with Grounded Pre-Training" (2023)

### Community
- Discord: [Join our community](https://discord.gg/gemini-cli)
- GitHub Discussions: [Ask questions](https://github.com/raython-king/gemini-cli/discussions)
- Twitter: [@gemini_cli](https://twitter.com/gemini_cli)

---

**Version**: 1.0.0
**Last Updated**: 2025-11-18
**Models Supported**: Depth Anything V3, DINO v2, SAM 2, Grounding DINO
**Total Skills**: 7 (2 LLM + 4 CV + 1 Research)
