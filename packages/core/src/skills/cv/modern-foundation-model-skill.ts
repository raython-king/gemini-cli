/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { z } from 'zod';
import type { SkillDefinition, SkillContext, SkillResult } from '../types.js';

/**
 * Input schema for Modern Foundation Model Application Skill
 */
const ModernFoundationModelInputSchema = z.object({
  model: z.enum([
    'depth_anything_v3',
    'dinov2',
    'sam2',
    'grounding_dino',
  ]).describe('Modern foundation model to use'),

  task: z.enum([
    'inference',
    'fine_tuning',
    'feature_extraction',
    'pipeline_integration',
  ]).default('inference').describe('Task type'),

  modelSize: z.enum(['tiny', 'small', 'base', 'large', 'huge']).default('base')
    .describe('Model size variant'),

  applicationContext: z.object({
    domain: z.string().describe('Application domain (e.g., medical, robotics, AR/VR)'),
    specificTask: z.string().describe('Specific task description'),
    inputType: z.string().describe('Input data type (images, videos, point clouds, etc.)'),
    outputRequirements: z.string().describe('Required output format and characteristics'),
  }).describe('Application context'),

  computeResources: z.object({
    device: z.enum(['cpu', 'cuda', 'mps']).default('cuda'),
    batchSize: z.number().default(1),
    precision: z.enum(['fp32', 'fp16', 'int8']).default('fp16'),
  }).optional().describe('Compute resources'),

  fineTuningConfig: z.object({
    datasetPath: z.string().optional(),
    numEpochs: z.number().default(10),
    learningRate: z.number().default(1e-4),
    frozenLayers: z.array(z.string()).optional(),
  }).optional().describe('Fine-tuning configuration'),

  includeDeployment: z.boolean().default(true)
    .describe('Include deployment optimizations'),
});

export type ModernFoundationModelInput = z.infer<typeof ModernFoundationModelInputSchema>;

/**
 * Output schema for Modern Foundation Model Application Skill
 */
const ModernFoundationModelOutputSchema = z.object({
  modelInfo: z.object({
    name: z.string(),
    version: z.string(),
    description: z.string(),
    parameters: z.string(),
    capabilities: z.array(z.string()),
    limitations: z.array(z.string()),
  }),

  implementation: z.object({
    setup: z.object({
      installation: z.array(z.string()),
      dependencies: z.array(z.string()),
      modelDownload: z.string(),
    }),
    inferenceCode: z.string(),
    fineTuningCode: z.string().optional(),
    featureExtractionCode: z.string().optional(),
  }),

  usageExamples: z.array(z.object({
    scenario: z.string(),
    code: z.string(),
    expectedOutput: z.string(),
  })),

  performance: z.object({
    speed: z.string(),
    memory: z.string(),
    accuracy: z.string(),
  }),

  deployment: z.object({
    optimization_techniques: z.array(z.string()),
    deployment_code: z.string(),
    serving_recommendations: z.array(z.string()),
  }).optional(),

  recommendations: z.array(z.string()),
});

export type ModernFoundationModelOutput = z.infer<typeof ModernFoundationModelOutputSchema>;

/**
 * Modern Foundation Model Application Skill
 *
 * Supports latest CV foundation models:
 * - Depth Anything V3: Monocular depth estimation
 * - DINO v2: Self-supervised vision features
 * - SAM 2: Video and image segmentation
 * - Grounding DINO: Open-set object detection
 */
export const ModernFoundationModelSkill: SkillDefinition<
  ModernFoundationModelInput,
  ModernFoundationModelOutput
> = {
  id: 'cv.modern_foundation_model',
  name: 'Modern Foundation Model Application',
  category: 'cv',
  complexity: 'expert',
  description: 'Apply latest CV foundation models (Depth Anything V3, DINO v2, SAM 2, Grounding DINO)',
  tags: ['foundation-model', 'depth-estimation', 'segmentation', 'object-detection', 'self-supervised'],

  inputSchema: ModernFoundationModelInputSchema,
  outputSchema: ModernFoundationModelOutputSchema,

  requiredAgents: ['foundation_model_agent', 'cv_research_agent'],

  async execute(
    input: ModernFoundationModelInput,
    context: SkillContext,
  ): Promise<SkillResult<ModernFoundationModelOutput>> {
    const startTime = Date.now();

    try {
      // Step 1: Research model capabilities
      if (context.progressCallback) {
        context.progressCallback({
          step: 'research',
          message: `Researching ${input.model} capabilities...`,
          progress: 0.1,
        });
      }

      const modelInfo = this.getModelInfo(input.model, input.modelSize);

      // Step 2: Generate implementation
      if (context.progressCallback) {
        context.progressCallback({
          step: 'implementation',
          message: 'Generating implementation code...',
          progress: 0.3,
        });
      }

      const implementation = this.generateImplementation(input);

      // Step 3: Create usage examples
      if (context.progressCallback) {
        context.progressCallback({
          step: 'examples',
          message: 'Creating usage examples...',
          progress: 0.6,
        });
      }

      const usageExamples = this.generateUsageExamples(input);

      // Step 4: Performance analysis
      const performance = this.analyzePerformance(input);

      // Step 5: Deployment optimization
      const deployment = input.includeDeployment
        ? this.generateDeployment(input)
        : undefined;

      if (context.progressCallback) {
        context.progressCallback({
          step: 'finalization',
          message: 'Finalizing recommendations...',
          progress: 0.9,
        });
      }

      const output: ModernFoundationModelOutput = {
        modelInfo,
        implementation,
        usageExamples,
        performance,
        deployment,
        recommendations: this.generateRecommendations(input, modelInfo),
      };

      const duration = Date.now() - startTime;

      return {
        success: true,
        data: output,
        metadata: {
          executionTime: duration,
          agentsUsed: ['foundation_model_agent'],
          confidence: 0.94,
        },
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error occurred',
        metadata: {
          executionTime: Date.now() - startTime,
          agentsUsed: [],
          confidence: 0,
        },
      };
    }
  },

  getModelInfo(model: string, size: string) {
    const models: Record<string, any> = {
      depth_anything_v3: {
        name: 'Depth Anything V3',
        version: '3.0',
        description: 'State-of-the-art monocular depth estimation with improved metric depth prediction and zero-shot generalization',
        parameters: size === 'small' ? '24M' : size === 'base' ? '97M' : size === 'large' ? '335M' : '97M',
        capabilities: [
          'Monocular depth estimation from single images',
          'Metric depth prediction (absolute depth values)',
          'Zero-shot depth estimation across diverse scenes',
          'Real-time inference (up to 30 FPS on GPU)',
          'Robust to various lighting conditions',
          'Works with indoor and outdoor scenes',
          'Supports video depth estimation',
          'Fine-grained depth details',
        ],
        limitations: [
          'May struggle with transparent or reflective surfaces',
          'Depth scale ambiguity in texture-less regions',
          'Performance degrades with extreme motion blur',
          'Limited to forward-facing camera perspectives',
          'Requires GPU for real-time performance',
        ],
      },
      dinov2: {
        name: 'DINO v2',
        version: '2.0',
        description: 'Self-supervised vision transformer with superior feature quality, trained on 142M curated images',
        parameters: size === 'small' ? '21M' : size === 'base' ? '86M' : size === 'large' ? '300M' : size === 'huge' ? '1.1B' : '86M',
        capabilities: [
          'Self-supervised learning (no labels required)',
          'High-quality visual features for any task',
          'Strong zero-shot dense prediction (segmentation, depth)',
          'Linear probing achieves 78.9% on ImageNet',
          'Excellent transfer learning performance',
          'Emergent object segmentation in attention maps',
          'Robust to distribution shifts',
          'Supports global and local features',
        ],
        limitations: [
          'Large model size (especially ViT-g with 1.1B params)',
          'Requires substantial compute for fine-tuning',
          'Feature extraction can be memory-intensive',
          'Best performance requires proper input preprocessing',
        ],
      },
      sam2: {
        name: 'Segment Anything Model 2 (SAM 2)',
        version: '2.0',
        description: 'Unified promptable segmentation for both images and videos with temporal consistency',
        parameters: size === 'tiny' ? '38M' : size === 'small' ? '95M' : size === 'base' ? '224M' : size === 'large' ? '224M' : '224M',
        capabilities: [
          'Promptable segmentation with points, boxes, or masks',
          'Video object segmentation with temporal propagation',
          'Zero-shot segmentation for novel objects',
          'Interactive annotation with immediate feedback',
          'Memory-efficient streaming for long videos',
          'Handles occlusions and reappearances',
          'Supports multiple object tracking',
          'Real-time performance on modern GPUs',
        ],
        limitations: [
          'Video processing requires sequential frames',
          'Memory usage scales with video length',
          'May lose track during complete occlusions',
          'Challenging with high-speed motion',
          'Requires manual prompts for initialization',
        ],
      },
      grounding_dino: {
        name: 'Grounding DINO',
        version: '1.5',
        description: 'Open-set object detection with natural language queries, combining DINO with grounded pre-training',
        parameters: size === 'tiny' ? '28M' : size === 'small' ? '52M' : size === 'base' ? '218M' : '218M',
        capabilities: [
          'Open-vocabulary object detection (any text query)',
          'Zero-shot detection without fine-tuning',
          'Grounding arbitrary phrases to image regions',
          'Supports complex compositional queries',
          'High accuracy on novel object categories',
          'Integration with SAM for instance segmentation',
          'Visual grounding for referring expressions',
          'Multi-object detection with natural language',
        ],
        limitations: [
          'Performance varies with query phrasing',
          'May struggle with very small or occluded objects',
          'Inference speed slower than specialized detectors',
          'Requires good text prompts for best results',
          'Memory-intensive due to vision-language fusion',
        ],
      },
    };

    return models[model];
  },

  generateImplementation(input: ModernFoundationModelInput) {
    const setup = this.generateSetup(input.model);
    const inferenceCode = this.generateInferenceCode(input);
    const fineTuningCode = input.task === 'fine_tuning'
      ? this.generateFineTuningCode(input)
      : undefined;
    const featureExtractionCode = input.task === 'feature_extraction'
      ? this.generateFeatureExtractionCode(input)
      : undefined;

    return {
      setup,
      inferenceCode,
      fineTuningCode,
      featureExtractionCode,
    };
  },

  generateSetup(model: string) {
    const setups: Record<string, any> = {
      depth_anything_v3: {
        installation: [
          'pip install torch torchvision',
          'pip install transformers',
          'pip install depth-anything-v3',
          'pip install opencv-python pillow',
        ],
        dependencies: ['torch>=2.0.0', 'transformers>=4.35.0', 'depth-anything-v3>=1.0.0', 'opencv-python', 'pillow'],
        modelDownload: 'Model will be auto-downloaded from Hugging Face Hub on first use',
      },
      dinov2: {
        installation: [
          'pip install torch torchvision',
          'pip install transformers',
          'git clone https://github.com/facebookresearch/dinov2.git',
          'cd dinov2 && pip install -e .',
        ],
        dependencies: ['torch>=2.0.0', 'torchvision>=0.15.0', 'transformers>=4.30.0'],
        modelDownload: 'from transformers import AutoModel; model = AutoModel.from_pretrained("facebook/dinov2-base")',
      },
      sam2: {
        installation: [
          'pip install torch torchvision',
          'pip install git+https://github.com/facebookresearch/segment-anything-2.git',
          'pip install opencv-python matplotlib',
        ],
        dependencies: ['torch>=2.3.0', 'torchvision>=0.18.0', 'sam2>=1.0.0', 'opencv-python', 'matplotlib'],
        modelDownload: 'Model checkpoints will be auto-downloaded or specify local path',
      },
      grounding_dino: {
        installation: [
          'pip install torch torchvision',
          'pip install transformers',
          'pip install groundingdino-py',
          'pip install supervision  # For visualization',
        ],
        dependencies: ['torch>=2.0.0', 'transformers>=4.30.0', 'groundingdino-py>=0.1.0', 'supervision'],
        modelDownload: 'from groundingdino.util.inference import load_model; model = load_model("groundingdino/swint_ogc")',
      },
    };

    return setups[model];
  },

  generateInferenceCode(input: ModernFoundationModelInput): string {
    const codes: Record<string, string> = {
      depth_anything_v3: `# Depth Anything V3 - Monocular Depth Estimation
import torch
from transformers import pipeline
from PIL import Image
import numpy as np
import cv2

# Load model
device = "${input.computeResources?.device || 'cuda'}" if torch.cuda.is_available() else "cpu"
depth_estimator = pipeline(
    "depth-estimation",
    model="depth-anything/Depth-Anything-V3-${input.modelSize === 'small' ? 'Small' : input.modelSize === 'large' ? 'Large' : 'Base'}",
    device=device,
)

# Single image inference
image = Image.open("input.jpg")
result = depth_estimator(image)

# Get depth map
depth_map = result["depth"]  # PIL Image or numpy array
depth_array = np.array(depth_map)

# Visualize depth (closer = warmer colors)
depth_colored = cv2.applyColorMap(
    cv2.normalize(depth_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
    cv2.COLORMAP_INFERNO
)
cv2.imwrite("depth_output.jpg", depth_colored)

# Get metric depth values (in meters, if model supports it)
if "predicted_depth" in result:
    metric_depth = result["predicted_depth"]  # Absolute depth in meters
    print(f"Depth range: {metric_depth.min():.2f}m to {metric_depth.max():.2f}m")

# Video processing
import cv2

cap = cv2.VideoCapture("input_video.mp4")
out = cv2.VideoWriter("depth_video.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to PIL
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Estimate depth
    depth = depth_estimator(frame_pil)["depth"]
    depth_colored = cv2.applyColorMap(
        cv2.normalize(np.array(depth), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
        cv2.COLORMAP_INFERNO
    )

    out.write(depth_colored)

cap.release()
out.release()

print("Depth estimation complete!")
`,
      dinov2: `# DINO v2 - Self-Supervised Vision Features
import torch
from transformers import AutoModel, AutoImageProcessor
from PIL import Image

# Load model
model_name = "facebook/dinov2-${input.modelSize === 'small' ? 'small' : input.modelSize === 'large' ? 'large' : input.modelSize === 'huge' ? 'giant' : 'base'}"
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()
model.to("${input.computeResources?.device || 'cuda'}")

# Feature extraction
image = Image.open("input.jpg")
inputs = processor(images=image, return_tensors="pt").to("${input.computeResources?.device || 'cuda'}")

with torch.no_grad():
    outputs = model(**inputs)

# Get features
last_hidden_state = outputs.last_hidden_state  # [B, num_patches, hidden_dim]
pooled_output = outputs.pooler_output  # [B, hidden_dim] - CLS token

# Global features (for classification, retrieval)
global_features = pooled_output.cpu().numpy()
print(f"Global feature shape: {global_features.shape}")  # e.g., [1, 768]

# Patch features (for dense prediction tasks)
patch_features = last_hidden_state[:, 1:, :]  # Exclude CLS token
H = W = int(patch_features.size(1) ** 0.5)
patch_features = patch_features.reshape(1, H, W, -1)
print(f"Patch feature shape: {patch_features.shape}")  # e.g., [1, 14, 14, 768]

# Linear probing for classification
import torch.nn as nn

classifier = nn.Linear(model.config.hidden_size, num_classes)
classifier.to("${input.computeResources?.device || 'cuda'}")

# Freeze backbone
for param in model.parameters():
    param.requires_grad = False

# Training loop
optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for images, labels in dataloader:
    images, labels = images.to("${input.computeResources?.device || 'cuda'}"), labels.to("${input.computeResources?.device || 'cuda'}")

    with torch.no_grad():
        features = model(images).pooler_output

    logits = classifier(features)
    loss = criterion(logits, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("DINOv2 feature extraction complete!")
`,
      sam2: `# SAM 2 - Video and Image Segmentation
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.sam2_video_predictor import SAM2VideoPredictor
import numpy as np
import cv2

# Load model for images
checkpoint = "sam2_hiera_${input.modelSize === 'tiny' ? 'tiny' : input.modelSize === 'small' ? 'small' : 'base_plus'}.pt"
model_cfg = "sam2_hiera_${input.modelSize === 'tiny' ? 't' : input.modelSize === 'small' ? 's' : 'b+' }.yaml"

# Image segmentation
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

image = cv2.imread("input.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
predictor.set_image(image_rgb)

# Prompt with points
input_point = np.array([[500, 375]])  # x, y
input_label = np.array([1])  # 1 = foreground

masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)

# Use best mask
best_mask = masks[np.argmax(scores)]

# Visualize
overlay = image.copy()
overlay[best_mask] = overlay[best_mask] * 0.5 + np.array([0, 255, 0]) * 0.5
cv2.imwrite("segmentation_result.jpg", overlay)

# Video segmentation with temporal propagation
video_predictor = SAM2VideoPredictor(build_sam2(model_cfg, checkpoint))

# Initialize with video frames
video_path = "input_video.mp4"
frame_dir = "frames/"  # Extract frames first

with video_predictor.init_state(video_path=video_path) as inference_state:
    # Add points on first frame
    _, object_ids, mask_logits = video_predictor.add_new_points(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=0,
        points=input_point,
        labels=input_label,
    )

    # Propagate to all frames
    for frame_idx, object_ids, mask_logits in video_predictor.propagate_in_video(inference_state):
        # Get mask for current frame
        mask = (mask_logits[0] > 0.0).cpu().numpy()

        # Save or process mask
        frame = cv2.imread(f"{frame_dir}/frame_{frame_idx:04d}.jpg")
        overlay = frame.copy()
        overlay[mask] = overlay[mask] * 0.5 + np.array([0, 255, 0]) * 0.5
        cv2.imwrite(f"output/frame_{frame_idx:04d}.jpg", overlay)

print("SAM 2 segmentation complete!")
`,
      grounding_dino: `# Grounding DINO - Open-Set Object Detection
import torch
from groundingdino.util.inference import load_model, load_image, predict
from groundingdino.util import box_ops
import cv2
import supervision as sv

# Load model
model = load_model(
    "groundingdino/swint_ogc.py",
    "groundingdino_swint_ogc.pth"
)
model.to("${input.computeResources?.device || 'cuda'}")

# Load image
image_source, image = load_image("input.jpg")

# Text prompts (open vocabulary!)
TEXT_PROMPT = "${input.applicationContext.specificTask || 'person . car . dog . cat'}"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

# Inference
boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_THRESHOLD,
    text_threshold=TEXT_THRESHOLD,
    device="${input.computeResources?.device || 'cuda'}"
)

# Convert boxes to xyxy format
h, w, _ = image_source.shape
boxes = boxes * torch.Tensor([w, h, w, h])
xyxy = box_ops.box_cxcywh_to_xyxy(boxes).cpu().numpy()

# Visualize with supervision
detections = sv.Detections(xyxy=xyxy)
labels = [
    f"{phrase} {logit:.2f}"
    for phrase, logit in zip(phrases, logits)
]

box_annotator = sv.BoxAnnotator()
annotated_frame = box_annotator.annotate(scene=image_source.copy(), detections=detections, labels=labels)
cv2.imwrite("grounding_dino_output.jpg", annotated_frame)

# Integration with SAM for instance segmentation
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2

sam_predictor = SAM2ImagePredictor(build_sam2("sam2_hiera_b+.yaml", "sam2_hiera_base_plus.pt"))
sam_predictor.set_image(image_source)

# Convert boxes to SAM input format
masks = []
for box in xyxy:
    mask, _, _ = sam_predictor.predict(
        box=box[None, :],
        multimask_output=False,
    )
    masks.append(mask[0])

# Visualize masks
overlay = image_source.copy()
for i, mask in enumerate(masks):
    color = np.random.randint(0, 255, 3).tolist()
    overlay[mask] = overlay[mask] * 0.5 + np.array(color) * 0.5

cv2.imwrite("grounded_sam_output.jpg", overlay)

print(f"Detected {len(boxes)} objects: {phrases}")
print("Grounding DINO + SAM complete!")
`,
    };

    return codes[input.model];
  },

  generateFineTuningCode(input: ModernFoundationModelInput): string {
    return `# Fine-tuning ${input.model}
import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup

# Assuming model and processor are loaded
model.train()

# Prepare dataset
train_dataset = YourCustomDataset("${input.fineTuningConfig?.datasetPath || '/path/to/dataset'}")
train_loader = DataLoader(train_dataset, batch_size=${input.computeResources?.batchSize || 4}, shuffle=True)

# Optimizer
optimizer = AdamW(model.parameters(), lr=${input.fineTuningConfig?.learningRate || 1e-4})
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=len(train_loader) // 10,
    num_training_steps=len(train_loader) * ${input.fineTuningConfig?.numEpochs || 10}
)

# Training loop
for epoch in range(${input.fineTuningConfig?.numEpochs || 10}):
    total_loss = 0
    for batch in train_loader:
        inputs = batch['input'].to('${input.computeResources?.device || 'cuda'}')
        labels = batch['label'].to('${input.computeResources?.device || 'cuda'}')

        outputs = model(inputs, labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")

    # Save checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
    }, f"checkpoint_epoch_{epoch+1}.pt")

print("Fine-tuning complete!")
`;
  },

  generateFeatureExtractionCode(input: ModernFoundationModelInput): string {
    return `# Feature Extraction with ${input.model}
import torch
import numpy as np
from tqdm import tqdm

model.eval()
features_list = []
labels_list = []

with torch.no_grad():
    for images, labels in tqdm(dataloader):
        images = images.to('${input.computeResources?.device || 'cuda'}')

        # Extract features
        outputs = model(images)
        features = outputs.pooler_output  # or .last_hidden_state

        features_list.append(features.cpu().numpy())
        labels_list.append(labels.numpy())

# Concatenate all features
all_features = np.concatenate(features_list, axis=0)
all_labels = np.concatenate(labels_list, axis=0)

print(f"Extracted features shape: {all_features.shape}")

# Save features
np.save("features.npy", all_features)
np.save("labels.npy", all_labels)

# Use for downstream tasks
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(all_features, all_labels, test_size=0.2)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)

print(f"Linear probe accuracy: {accuracy*100:.2f}%")
`;
  },

  generateUsageExamples(input: ModernFoundationModelInput) {
    const examples: Record<string, any[]> = {
      depth_anything_v3: [
        {
          scenario: '3D Scene Reconstruction',
          code: `# Estimate depth for 3D reconstruction
depth_map = depth_estimator(image)["predicted_depth"]
point_cloud = depth_to_pointcloud(depth_map, camera_intrinsics)
save_ply("scene.ply", point_cloud)`,
          expectedOutput: '3D point cloud with metric depth values',
        },
        {
          scenario: 'AR/VR Applications',
          code: `# Real-time depth for AR occlusion
while True:
    frame = camera.capture()
    depth = depth_estimator(frame)["depth"]
    virtual_object = render_with_occlusion(depth, virtual_obj)
    display(virtual_object)`,
          expectedOutput: 'Real-time depth maps for AR occlusion handling',
        },
      ],
      dinov2: [
        {
          scenario: 'Image Retrieval',
          code: `# Extract features for image retrieval
database_features = [model(img).pooler_output for img in database]
query_feature = model(query_image).pooler_output
similarities = cosine_similarity(query_feature, database_features)
top_k = similarities.argsort()[-5:]`,
          expectedOutput: 'Top-5 similar images from database',
        },
        {
          scenario: 'Few-Shot Learning',
          code: `# Few-shot classification with frozen features
support_features = model(support_images).pooler_output
query_features = model(query_images).pooler_output
predictions = nearest_neighbor(query_features, support_features)`,
          expectedOutput: 'Classification with <10 examples per class',
        },
      ],
      sam2: [
        {
          scenario: 'Video Object Tracking',
          code: `# Track object through video
with video_predictor.init_state(video_path) as state:
    video_predictor.add_new_points(state, frame_idx=0, points=init_point)
    for frame_idx, masks in video_predictor.propagate_in_video(state):
        process_mask(masks)`,
          expectedOutput: 'Temporally consistent object segmentation masks',
        },
        {
          scenario: 'Interactive Annotation',
          code: `# Interactive segmentation with progressive refinement
mask1 = sam_predictor.predict(point_coords=positive_points)
mask2 = sam_predictor.predict(point_coords=positive_points + negative_points)
final_mask = refine_mask(mask2)`,
          expectedOutput: 'High-quality segmentation with minimal clicks',
        },
      ],
      grounding_dino: [
        {
          scenario: 'Zero-Shot Detection',
          code: `# Detect arbitrary objects with text
boxes, logits, phrases = predict(model, image, caption="red car . blue bag")
visualize_detections(image, boxes, phrases)`,
          expectedOutput: 'Bounding boxes for objects matching text queries',
        },
        {
          scenario: 'Visual Grounding',
          code: `# Ground referring expressions
query = "the person wearing a red shirt on the left"
boxes, scores, _ = predict(model, image, caption=query, box_threshold=0.3)
target_box = boxes[scores.argmax()]`,
          expectedOutput: 'Bounding box for the specific referred object',
        },
      ],
    };

    return examples[input.model] || [];
  },

  analyzePerformance(input: ModernFoundationModelInput) {
    const performance: Record<string, any> = {
      depth_anything_v3: {
        speed: `${input.modelSize === 'small' ? '~50' : input.modelSize === 'large' ? '~20' : '~35'} FPS on RTX 4090 (batch=1, fp16)`,
        memory: `${input.modelSize === 'small' ? '~2GB' : input.modelSize === 'large' ? '~6GB' : '~4GB'} VRAM`,
        accuracy: 'AbsRel 0.055 on NYU Depth V2, δ1 0.959 (better than previous versions)',
      },
      dinov2: {
        speed: `${input.modelSize === 'small' ? '~120' : input.modelSize === 'base' ? '~80' : input.modelSize === 'large' ? '~40' : '~15'} images/sec on A100`,
        memory: `${input.modelSize === 'small' ? '~1.5GB' : input.modelSize === 'base' ? '~3GB' : input.modelSize === 'large' ? '~8GB' : '~25GB'} VRAM`,
        accuracy: '78.9% linear probe on ImageNet (base), 83.5% (giant)',
      },
      sam2: {
        speed: `${input.modelSize === 'tiny' ? '~35' : input.modelSize === 'small' ? '~25' : '~15'} FPS for image, ~8 FPS for video (720p)`,
        memory: `${input.modelSize === 'tiny' ? '~2GB' : input.modelSize === 'small' ? '~4GB' : '~8GB'} VRAM`,
        accuracy: 'J&F 76.2 on DAVIS 2017, IoU 0.80 on interactive benchmarks',
      },
      grounding_dino: {
        speed: '~12 FPS on RTX 4090 (batch=1, fp16)',
        memory: '~8GB VRAM (including text encoder)',
        accuracy: 'AP 52.5 on COCO (zero-shot), AP 63.0 on LVIS (with minimal tuning)',
      },
    };

    return performance[input.model];
  },

  generateDeployment(input: ModernFoundationModelInput) {
    return {
      optimization_techniques: [
        'TorchScript compilation for faster inference',
        'ONNX export for cross-platform deployment',
        'TensorRT optimization for NVIDIA GPUs',
        'Quantization to INT8 for edge devices',
        'Model pruning to reduce size',
        'Batch inference for throughput',
        'Mixed precision (FP16) training and inference',
      ],
      deployment_code: `# Deployment optimization
import torch

# 1. TorchScript export
model.eval()
example_input = torch.randn(1, 3, 224, 224).to('${input.computeResources?.device || 'cuda'}')
traced_model = torch.jit.trace(model, example_input)
traced_model.save("model_traced.pt")

# 2. ONNX export
torch.onnx.export(
    model,
    example_input,
    "model.onnx",
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

# 3. Quantization
from torch.quantization import quantize_dynamic
quantized_model = quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)
torch.save(quantized_model.state_dict(), "model_quantized.pt")

# 4. TensorRT (NVIDIA GPUs)
import tensorrt as trt
# Convert ONNX to TensorRT engine for maximum performance

# 5. Serving with FastAPI
from fastapi import FastAPI, File, UploadFile
app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file)
    result = model(image)
    return {"prediction": result}
`,
      serving_recommendations: [
        'Use FastAPI or Flask for REST API deployment',
        'Implement batch processing for higher throughput',
        'Use Redis for caching frequent queries',
        'Deploy with Docker for reproducibility',
        'Use Kubernetes for scalability',
        'Monitor with Prometheus + Grafana',
        'Implement rate limiting and authentication',
        'Use CDN for static assets',
      ],
    };
  },

  generateRecommendations(input: ModernFoundationModelInput, modelInfo: any): string[] {
    const baseRecommendations = [
      `${modelInfo.name} is ideal for: ${input.applicationContext.specificTask}`,
      `Model size "${input.modelSize}" balances performance and efficiency`,
      `Use ${input.computeResources?.precision || 'fp16'} precision for optimal speed/accuracy tradeoff`,
    ];

    const modelSpecific: Record<string, string[]> = {
      depth_anything_v3: [
        'For real-time applications, use small model with FP16 precision',
        'Calibrate depth scale with known reference objects for metric depth',
        'Use temporal filtering for video to reduce flicker',
        'Consider Depth Anything V3 metric variant for absolute depth values',
      ],
      dinov2: [
        'For classification, use linear probing first before fine-tuning',
        'Extract patch features for dense prediction tasks (segmentation, detection)',
        'Use global features (CLS token) for image-level tasks',
        'DINOv2 giant (1.1B params) provides best features but requires significant compute',
      ],
      sam2: [
        'Initialize with good prompts on first frame for video',
        'Use memory bank for long videos to handle occlusions',
        'Combine with Grounding DINO for automatic prompt generation',
        'Enable temporal propagation for consistent video segmentation',
      ],
      grounding_dino: [
        'Use clear, specific text prompts for best detection results',
        'Adjust box_threshold and text_threshold based on your use case',
        'Combine with SAM for high-quality instance segmentation',
        'Fine-tune on domain-specific data for specialized applications',
      ],
    };

    return [
      ...baseRecommendations,
      ...modelSpecific[input.model],
      'Monitor GPU memory usage and adjust batch size accordingly',
      'Implement proper error handling and fallback mechanisms',
      'Validate model outputs with domain-specific metrics',
    ];
  },
};
