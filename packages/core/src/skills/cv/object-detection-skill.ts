/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { z } from 'zod';
import { SkillCategory, SkillComplexity, type SkillDefinition } from '../types.js';
import { AgentExecutor } from '../../agents/executor.js';
import { CVResearchAgent } from '../../agents/research/cv-research-agent.js';
import { ExperimentAgent } from '../../agents/research/experiment-agent.js';

const ObjectDetectionInputSchema = z.object({
  dataset: z.string().describe('Path to dataset'),
  classes: z.array(z.string()).describe('Object classes to detect'),
  architecture: z.enum(['yolov8', 'faster_rcnn', 'detr', 'retinanet']).default('yolov8'),
  imageSize: z.number().default(640),
  deploymentTarget: z.enum(['cloud', 'edge', 'mobile']).optional(),
});

const ObjectDetectionOutputSchema = z.object({
  success: z.boolean(),
  setupGuide: z.object({
    architecture: z.string(),
    configuration: z.record(z.any()),
    trainingCommand: z.string(),
  }),
  dataPreparation: z.object({
    format: z.string(),
    augmentations: z.array(z.string()),
    splits: z.object({
      train: z.number(),
      val: z.number(),
      test: z.number(),
    }),
  }),
  implementation: z.object({
    code: z.string(),
    dependencies: z.array(z.string()),
  }),
});

export const ObjectDetectionSkill: SkillDefinition = {
  id: 'cv.object_detection',
  name: 'Object Detection Setup',
  description: 'Complete setup for object detection with YOLO, Faster R-CNN, or DETR',
  usage: 'Set up object detection pipeline with data prep, training, and deployment',
  category: SkillCategory.CV,
  complexity: SkillComplexity.ADVANCED,
  version: '1.0.0',
  inputSchema: ObjectDetectionInputSchema,
  outputSchema: ObjectDetectionOutputSchema,
  requiredAgents: ['cv_research_agent', 'experiment_agent'],
  estimatedTime: '10-15 minutes',
  tags: ['cv', 'object-detection', 'yolo', 'computer-vision'],
  examples: [{
    title: 'YOLOv8 for custom objects',
    description: 'Set up YOLOv8 detection for custom dataset',
    input: {
      dataset: './data/custom',
      classes: ['person', 'car', 'bicycle'],
      architecture: 'yolov8',
      deploymentTarget: 'edge',
    },
  }],
  async execute(input, context, config) {
    const cvExecutor = await AgentExecutor.create(CVResearchAgent, config);
    const result = await cvExecutor.run({
      objective: `Setup ${input.architecture} for detecting ${input.classes.join(', ')}`,
      domain: 'object_detection',
      depth: 'thorough',
      includeImplementation: true,
    });

    const code = `
from ultralytics import YOLO
import torch

# Initialize model
model = YOLO('yolov8n.pt')  # or yolov8s, yolov8m, yolov8l, yolov8x

# Train
results = model.train(
    data='${input.dataset}/data.yaml',
    epochs=100,
    imgsz=${input.imageSize},
    batch=16,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    project='./runs/detect',
    name='${input.classes.join('_')}',
)

# Validate
metrics = model.val()
print(f"mAP50: {metrics.box.map50}")
print(f"mAP50-95: {metrics.box.map}")

# Export for deployment
model.export(format='${input.deploymentTarget === 'edge' ? 'onnx' : 'torchscript'}')
`;

    return {
      success: true,
      data: {
        success: true,
        setupGuide: {
          architecture: input.architecture,
          configuration: {
            image_size: input.imageSize,
            classes: input.classes,
            anchor_boxes: 'auto',
          },
          trainingCommand: `yolo detect train data=${input.dataset}/data.yaml model=yolov8n.pt epochs=100`,
        },
        dataPreparation: {
          format: 'YOLO format (txt annotations)',
          augmentations: ['mosaic', 'mixup', 'hsv', 'flip', 'scale', 'rotate'],
          splits: { train: 0.7, val: 0.2, test: 0.1 },
        },
        implementation: {
          code,
          dependencies: ['ultralytics', 'torch', 'opencv-python', 'numpy'],
        },
      },
      metadata: {
        skillName: 'Object Detection Setup',
        duration: 0,
        agentsUsed: ['cv_research_agent'],
      },
    };
  },
};
