/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { z } from 'zod';
import type { SkillDefinition, SkillContext, SkillResult } from '../types.js';

/**
 * Input schema for Foundation Model Training Skill
 */
const FoundationModelTrainingInputSchema = z.object({
  modelType: z.enum([
    'vision_transformer',
    'masked_autoencoder',
    'self_distillation',
    'segmentation',
  ]).describe('Type of foundation model to train'),

  modelSize: z.enum(['tiny', 'small', 'base', 'large', 'huge']).default('base')
    .describe('Model size (affects parameter count)'),

  trainingObjective: z.enum([
    'supervised',
    'self_supervised',
    'semi_supervised',
  ]).default('self_supervised')
    .describe('Training objective'),

  dataset: z.object({
    name: z.string().describe('Dataset name (e.g., ImageNet-1K, ImageNet-21K)'),
    size: z.string().describe('Estimated size'),
    customPath: z.string().optional().describe('Custom dataset path'),
  }).describe('Training dataset information'),

  computeResources: z.object({
    gpus: z.number().describe('Number of GPUs available'),
    gpuType: z.string().default('A100').describe('GPU type'),
    trainingTime: z.string().describe('Target training time (e.g., "3 days", "1 week")'),
  }).describe('Available compute resources'),

  objectives: z.array(z.string()).optional()
    .describe('Specific training objectives or goals'),
});

export type FoundationModelTrainingInput = z.infer<typeof FoundationModelTrainingInputSchema>;

/**
 * Output schema for Foundation Model Training Skill
 */
const FoundationModelTrainingOutputSchema = z.object({
  trainingPlan: z.object({
    modelArchitecture: z.object({
      type: z.string(),
      parameters: z.string(),
      layers: z.number(),
      hiddenSize: z.number(),
      heads: z.number(),
      patchSize: z.number().optional(),
    }),
    trainingConfig: z.object({
      objective: z.string(),
      epochs: z.number(),
      batchSize: z.number(),
      learningRate: z.number(),
      warmupSteps: z.number(),
      weightDecay: z.number(),
      optimizer: z.string(),
    }),
    dataConfig: z.object({
      augmentations: z.array(z.string()),
      preprocessing: z.array(z.string()),
      masking Ratio: z.number().optional(),
    }),
    estimatedTime: z.string(),
    estimatedCost: z.string(),
  }),

  implementation: z.object({
    framework: z.string(),
    codeTemplate: z.string(),
    dependencies: z.array(z.string()),
    setup_instructions: z.array(z.string()),
  }),

  trainingScript: z.object({
    mainScript: z.string(),
    configFile: z.string(),
    launchCommand: z.string(),
  }),

  monitoring: z.object({
    metrics: z.array(z.string()),
    checkpointing: z.object({
      frequency: z.string(),
      keep_best: z.number(),
    }),
    logging: z.array(z.string()),
  }),

  recommendations: z.array(z.string()),
});

export type FoundationModelTrainingOutput = z.infer<typeof FoundationModelTrainingOutputSchema>;

/**
 * Foundation Model Training Skill
 *
 * Sets up complete training pipeline for CV foundation models including:
 * - Vision Transformers (ViT)
 * - Masked Autoencoders (MAE)
 * - Self-distillation models (DINO)
 * - Segmentation models (SAM-style)
 */
export const FoundationModelTrainingSkill: SkillDefinition<
  FoundationModelTrainingInput,
  FoundationModelTrainingOutput
> = {
  id: 'cv.foundation_model_training',
  name: 'Foundation Model Training',
  category: 'cv',
  complexity: 'expert',
  description: 'Complete setup for training CV foundation models (ViT, MAE, DINO, SAM)',
  tags: ['foundation-model', 'vision-transformer', 'self-supervised', 'pretraining'],

  inputSchema: FoundationModelTrainingInputSchema,
  outputSchema: FoundationModelTrainingOutputSchema,

  requiredAgents: ['foundation_model_agent', 'cv_research_agent'],

  async execute(
    input: FoundationModelTrainingInput,
    context: SkillContext,
  ): Promise<SkillResult<FoundationModelTrainingOutput>> {
    const startTime = Date.now();

    try {
      // Step 1: Research optimal architecture and training strategy
      if (context.progressCallback) {
        context.progressCallback({
          step: 'research',
          message: 'Researching optimal architecture and training strategy...',
          progress: 0.1,
        });
      }

      const researchResult = await context.executeAgent('foundation_model_agent', {
        objective: `Recommend optimal ${input.modelType} architecture and training strategy for ${input.modelSize} model`,
        modelType: input.modelType,
        depth: 'thorough',
        focus: ['architecture', 'pretraining', 'implementation'],
        includeImplementation: true,
      });

      // Step 2: Design training configuration
      if (context.progressCallback) {
        context.progressCallback({
          step: 'design',
          message: 'Designing training configuration...',
          progress: 0.3,
        });
      }

      const trainingPlan = this.designTrainingPlan(input, researchResult);

      // Step 3: Generate implementation code
      if (context.progressCallback) {
        context.progressCallback({
          step: 'implementation',
          message: 'Generating implementation code...',
          progress: 0.6,
        });
      }

      const implementation = this.generateImplementation(input, trainingPlan);
      const trainingScript = this.generateTrainingScript(input, trainingPlan);

      // Step 4: Setup monitoring
      if (context.progressCallback) {
        context.progressCallback({
          step: 'monitoring',
          message: 'Setting up monitoring and checkpointing...',
          progress: 0.9,
        });
      }

      const monitoring = this.setupMonitoring(input.modelType);

      const output: FoundationModelTrainingOutput = {
        trainingPlan,
        implementation,
        trainingScript,
        monitoring,
        recommendations: this.generateRecommendations(input, trainingPlan),
      };

      const duration = Date.now() - startTime;

      return {
        success: true,
        data: output,
        metadata: {
          executionTime: duration,
          agentsUsed: ['foundation_model_agent'],
          confidence: 0.95,
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

  designTrainingPlan(
    input: FoundationModelTrainingInput,
    researchResult: any,
  ) {
    const modelSizes: Record<string, any> = {
      tiny: { params: '5M', layers: 12, hiddenSize: 192, heads: 3 },
      small: { params: '22M', layers: 12, hiddenSize: 384, heads: 6 },
      base: { params: '86M', layers: 12, hiddenSize: 768, heads: 12 },
      large: { params: '307M', layers: 24, hiddenSize: 1024, heads: 16 },
      huge: { params: '632M', layers: 32, hiddenSize: 1280, heads: 16 },
    };

    const modelSpec = modelSizes[input.modelSize];
    const gpuHours = input.computeResources.gpus * this.parseTime(input.computeResources.trainingTime);

    const trainingConfig = this.getTrainingConfig(input.modelType, input.modelSize);

    return {
      modelArchitecture: {
        type: input.modelType,
        parameters: modelSpec.params,
        layers: modelSpec.layers,
        hiddenSize: modelSpec.hiddenSize,
        heads: modelSpec.heads,
        patchSize: input.modelType.includes('transformer') ? 16 : undefined,
      },
      trainingConfig,
      dataConfig: this.getDataConfig(input.modelType, input.trainingObjective),
      estimatedTime: input.computeResources.trainingTime,
      estimatedCost: `$${Math.round(gpuHours * 3.0)} (at $3/GPU-hour for ${input.computeResources.gpuType})`,
    };
  },

  getTrainingConfig(modelType: string, modelSize: string) {
    const baseConfigs: Record<string, any> = {
      vision_transformer: {
        objective: 'Supervised classification',
        epochs: 300,
        batchSize: 4096,
        learningRate: 0.001,
        warmupSteps: 10000,
        weightDecay: 0.05,
        optimizer: 'AdamW',
      },
      masked_autoencoder: {
        objective: 'Masked image modeling (75% masking)',
        epochs: 1600,
        batchSize: 4096,
        learningRate: 0.00015,
        warmupSteps: 40,
        weightDecay: 0.05,
        optimizer: 'AdamW',
      },
      self_distillation: {
        objective: 'Self-distillation with momentum teacher',
        epochs: 800,
        batchSize: 2048,
        learningRate: 0.0005,
        warmupSteps: 10,
        weightDecay: 0.04,
        optimizer: 'AdamW',
      },
      segmentation: {
        objective: 'Interactive segmentation',
        epochs: 200,
        batchSize: 256,
        learningRate: 0.0001,
        warmupSteps: 250,
        weightDecay: 0.1,
        optimizer: 'AdamW',
      },
    };

    return baseConfigs[modelType] || baseConfigs.vision_transformer;
  },

  getDataConfig(modelType: string, objective: string) {
    const configs: Record<string, any> = {
      vision_transformer: {
        augmentations: ['RandomResizedCrop', 'RandomHorizontalFlip', 'ColorJitter', 'RandAugment'],
        preprocessing: ['Resize(224)', 'Normalize(ImageNet stats)'],
      },
      masked_autoencoder: {
        augmentations: ['RandomResizedCrop'],
        preprocessing: ['Resize(224)', 'Normalize(ImageNet stats)'],
        maskingRatio: 0.75,
      },
      self_distillation: {
        augmentations: ['MultiCrop', 'ColorJitter', 'GaussianBlur', 'Solarization'],
        preprocessing: ['Global crops(224)', 'Local crops(96)'],
      },
      segmentation: {
        augmentations: ['RandomScale', 'RandomCrop', 'RandomFlip'],
        preprocessing: ['Resize(1024)', 'Normalize'],
      },
    };

    return configs[modelType] || configs.vision_transformer;
  },

  generateImplementation(input: FoundationModelTrainingInput, plan: any) {
    const code = input.modelType === 'masked_autoencoder' ? this.getMaeCode(plan) : this.getViTCode(plan);

    return {
      framework: 'PyTorch + Hugging Face Transformers',
      codeTemplate: code,
      dependencies: [
        'torch>=2.0.0',
        'torchvision>=0.15.0',
        'transformers>=4.30.0',
        'timm>=0.9.0',
        'accelerate>=0.20.0',
        'wandb>=0.15.0',
      ],
      setup_instructions: [
        'Install dependencies: pip install -r requirements.txt',
        'Prepare dataset in ImageFolder format',
        'Configure wandb for logging: wandb login',
        'Set up distributed training: export MASTER_ADDR and MASTER_PORT',
      ],
    };
  },

  getMaeCode(plan: any): string {
    return `# Masked Autoencoder (MAE) Training Script
import torch
import torch.nn as nn
from transformers import ViTMAEForPreTraining, ViTImageProcessor
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import wandb

# Model configuration
model = ViTMAEForPreTraining.from_pretrained(
    "facebook/vit-mae-base",
    num_hidden_layers=${plan.modelArchitecture.layers},
    hidden_size=${plan.modelArchitecture.hiddenSize},
    num_attention_heads=${plan.modelArchitecture.heads},
    decoder_num_hidden_layers=8,
    decoder_hidden_size=512,
    mask_ratio=${plan.dataConfig.maskingRatio},
)

# Data preprocessing
processor = ViTImageProcessor.from_pretrained("facebook/vit-mae-base")
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
    transforms.ToTensor(),
])

# Dataset
train_dataset = datasets.ImageFolder(
    "${input.dataset.customPath || '/path/to/imagenet'}",
    transform=train_transforms,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=${plan.trainingConfig.batchSize},
    shuffle=True,
    num_workers=16,
    pin_memory=True,
)

# Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=${plan.trainingConfig.learningRate},
    betas=(0.9, 0.95),
    weight_decay=${plan.trainingConfig.weightDecay},
)

# Learning rate schedule
def get_lr_schedule(step, warmup_steps, total_steps, base_lr):
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    else:
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return base_lr * 0.5 * (1 + math.cos(math.pi * progress))

# Training loop
model.train()
model = model.cuda()
wandb.init(project="mae-pretraining", config=vars(args))

for epoch in range(${plan.trainingConfig.epochs}):
    for batch_idx, (images, _) in enumerate(train_loader):
        images = images.cuda()

        # Forward pass
        outputs = model(images)
        loss = outputs.loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Update learning rate
        step = epoch * len(train_loader) + batch_idx
        for param_group in optimizer.param_groups:
            param_group['lr'] = get_lr_schedule(
                step,
                ${plan.trainingConfig.warmupSteps},
                ${plan.trainingConfig.epochs} * len(train_loader),
                ${plan.trainingConfig.learningRate},
            )

        # Logging
        if batch_idx % 100 == 0:
            wandb.log({
                "loss": loss.item(),
                "lr": optimizer.param_groups[0]['lr'],
                "epoch": epoch,
            })
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

    # Save checkpoint
    if epoch % 50 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, f"mae_checkpoint_epoch_{epoch}.pt")

print("Training complete!")
`;
  },

  getViTCode(plan: any): string {
    return `# Vision Transformer Training Script
import torch
from transformers import ViTForImageClassification, ViTImageProcessor, TrainingArguments, Trainer
from datasets import load_dataset

# Load model
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=1000,  # ImageNet classes
    ignore_mismatched_sizes=True,
)

# Load dataset
dataset = load_dataset("imagenet-1k")

# Preprocessing
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

def preprocess(examples):
    images = [image.convert("RGB") for image in examples['image']]
    inputs = processor(images, return_tensors="pt")
    inputs['labels'] = examples['label']
    return inputs

train_dataset = dataset['train'].map(preprocess, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./vit-training",
    per_device_train_batch_size=${plan.trainingConfig.batchSize // input.computeResources.gpus},
    learning_rate=${plan.trainingConfig.learningRate},
    num_train_epochs=${plan.trainingConfig.epochs},
    warmup_steps=${plan.trainingConfig.warmupSteps},
    weight_decay=${plan.trainingConfig.weightDecay},
    logging_steps=100,
    save_steps=5000,
    evaluation_strategy="epoch",
    fp16=True,
    dataloader_num_workers=16,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

# Train
trainer.train()
`;
  },

  generateTrainingScript(input: FoundationModelTrainingInput, plan: any) {
    const configFile = this.generateConfigFile(input, plan);
    const launchCommand = `torchrun --nproc_per_node=${input.computeResources.gpus} \\
  --master_port=29500 \\
  train.py \\
  --config config.yaml \\
  --output_dir ./checkpoints`;

    return {
      mainScript: 'train.py',
      configFile,
      launchCommand,
    };
  },

  generateConfigFile(input: FoundationModelTrainingInput, plan: any): string {
    return `# Training Configuration
model:
  type: ${input.modelType}
  size: ${input.modelSize}
  parameters: ${plan.modelArchitecture.parameters}

training:
  epochs: ${plan.trainingConfig.epochs}
  batch_size: ${plan.trainingConfig.batchSize}
  learning_rate: ${plan.trainingConfig.learningRate}
  warmup_steps: ${plan.trainingConfig.warmupSteps}
  weight_decay: ${plan.trainingConfig.weightDecay}
  optimizer: ${plan.trainingConfig.optimizer}

dataset:
  name: ${input.dataset.name}
  path: ${input.dataset.customPath || '/path/to/dataset'}

compute:
  gpus: ${input.computeResources.gpus}
  gpu_type: ${input.computeResources.gpuType}
  mixed_precision: true

logging:
  wandb_project: foundation-model-training
  log_interval: 100
  checkpoint_interval: 5000
`;
  },

  setupMonitoring(modelType: string) {
    return {
      metrics: [
        'training_loss',
        'learning_rate',
        'gradient_norm',
        'throughput (images/sec)',
        'GPU_memory_usage',
        modelType === 'masked_autoencoder' ? 'reconstruction_loss' : 'classification_accuracy',
      ],
      checkpointing: {
        frequency: 'every 5000 steps',
        keep_best: 5,
      },
      logging: [
        'Log to Weights & Biases (wandb)',
        'TensorBoard for local visualization',
        'Save training curves as PNG',
        'Monitor GPU utilization with nvidia-smi',
      ],
    };
  },

  generateRecommendations(input: FoundationModelTrainingInput, plan: any): string[] {
    return [
      `For ${input.modelType} with ${input.modelSize} size, expect training time: ${plan.estimatedTime}`,
      `Estimated cost: ${plan.estimatedCost}`,
      `Use mixed precision (FP16) to reduce memory and speed up training`,
      `Monitor gradient norms to detect training instabilities`,
      `Save checkpoints regularly - training can take days`,
      input.trainingObjective === 'self_supervised'
        ? 'Self-supervised pretraining works well with unlabeled data - consider using larger datasets'
        : 'For supervised training, data quality matters more than quantity',
      `Consider using gradient accumulation if batch size is too large for GPU memory`,
      `Use distributed training across all ${input.computeResources.gpus} GPUs for optimal throughput`,
    ];
  },

  parseTime(timeStr: string): number {
    const match = timeStr.match(/(\d+)\s*(hour|day|week)/);
    if (!match) return 24; // default 1 day

    const value = parseInt(match[1]);
    const unit = match[2];

    const hours: Record<string, number> = {
      hour: value,
      day: value * 24,
      week: value * 24 * 7,
    };

    return hours[unit] || 24;
  },
};
