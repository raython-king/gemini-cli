/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @fileoverview Library of proven training recipes for various deep learning tasks.
 * Each recipe encapsulates best practices, optimal hyperparameters, and implementation details.
 */

import { z } from 'zod';

/**
 * Training recipe schema for type safety.
 */
const TrainingRecipeSchema = z.object({
  name: z.string(),
  description: z.string(),
  version: z.string(),
  source: z.string(),
  task: z.string(),
  targetPerformance: z.string(),
  optimizer: z.object({
    type: z.string(),
    learningRate: z.number(),
    weightDecay: z.number(),
    momentum: z.number().optional(),
    beta1: z.number().optional(),
    beta2: z.number().optional(),
    epsilon: z.number().optional(),
    additionalParams: z.record(z.any()).optional(),
  }),
  scheduler: z.object({
    type: z.string(),
    warmupEpochs: z.number().optional(),
    warmupSteps: z.number().optional(),
    totalEpochs: z.number().optional(),
    totalSteps: z.number().optional(),
    minLr: z.number().optional(),
    additionalParams: z.record(z.any()).optional(),
  }),
  batchSize: z.object({
    perGpu: z.number(),
    effective: z.number(),
    gradientAccumulation: z.number().optional(),
  }),
  augmentation: z.array(z.object({
    name: z.string(),
    params: z.record(z.any()),
  })).optional(),
  regularization: z.object({
    dropout: z.number().optional(),
    labelSmoothing: z.number().optional(),
    mixup: z.number().optional(),
    cutmix: z.number().optional(),
    stochasticDepth: z.number().optional(),
    weightDecay: z.number().optional(),
  }).optional(),
  mixedPrecision: z.object({
    enabled: z.boolean(),
    dtype: z.string(),
  }).optional(),
  ema: z.object({
    enabled: z.boolean(),
    decay: z.number(),
  }).optional(),
  implementation: z.string(),
  notes: z.array(z.string()),
});

export type TrainingRecipe = z.infer<typeof TrainingRecipeSchema>;

/**
 * ImageNet classification training recipes.
 */
export const ImageNetRecipes: Record<string, TrainingRecipe> = {
  /**
   * ResNet-50 recipe achieving ~79% top-1 accuracy.
   * Based on "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour"
   */
  resnet50_baseline: {
    name: 'ResNet-50 ImageNet Baseline',
    description: 'Standard ResNet-50 training achieving ~76.5% top-1 accuracy',
    version: '1.0.0',
    source: 'PyTorch torchvision defaults',
    task: 'ImageNet-1K classification',
    targetPerformance: '76.5% top-1 accuracy',
    optimizer: {
      type: 'SGD',
      learningRate: 0.1,
      weightDecay: 1e-4,
      momentum: 0.9,
    },
    scheduler: {
      type: 'StepLR',
      totalEpochs: 90,
      additionalParams: {
        stepSize: 30,
        gamma: 0.1,
      },
    },
    batchSize: {
      perGpu: 32,
      effective: 256,
      gradientAccumulation: 1,
    },
    augmentation: [
      { name: 'RandomResizedCrop', params: { size: 224 } },
      { name: 'RandomHorizontalFlip', params: {} },
      { name: 'Normalize', params: { mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225] } },
    ],
    regularization: {
      weightDecay: 1e-4,
    },
    mixedPrecision: {
      enabled: false,
      dtype: 'fp32',
    },
    implementation: `
import torch
import torch.nn as nn
import torchvision.models as models
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR

model = models.resnet50(weights=None)
optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
criterion = nn.CrossEntropyLoss()

for epoch in range(90):
    train_one_epoch(model, train_loader, optimizer, criterion)
    validate(model, val_loader)
    scheduler.step()
`,
    notes: [
      'Uses batch size 256 across 8 GPUs (32 per GPU)',
      'Learning rate warmup can improve stability',
      'Consider augmentation additions for better results',
    ],
  },

  /**
   * ResNet-50 A1/A2/A3 recipes from "ResNet strikes back"
   */
  resnet50_a1: {
    name: 'ResNet-50 A1 Recipe',
    description: 'Improved ResNet-50 training from "ResNet strikes back" achieving 80.4% top-1',
    version: '1.0.0',
    source: 'ResNet strikes back: An improved training procedure in timm (Wightman et al.)',
    task: 'ImageNet-1K classification',
    targetPerformance: '80.4% top-1 accuracy',
    optimizer: {
      type: 'LAMB',
      learningRate: 0.008,
      weightDecay: 0.02,
      beta1: 0.9,
      beta2: 0.999,
      epsilon: 1e-6,
    },
    scheduler: {
      type: 'CosineLRScheduler',
      warmupEpochs: 5,
      totalEpochs: 600,
      minLr: 1e-5,
      additionalParams: {
        warmupLrInit: 1e-6,
      },
    },
    batchSize: {
      perGpu: 256,
      effective: 2048,
      gradientAccumulation: 1,
    },
    augmentation: [
      { name: 'RandomResizedCrop', params: { size: 176, interpolation: 'bicubic' } },
      { name: 'RandomHorizontalFlip', params: {} },
      { name: 'RandAugment', params: { num_ops: 2, magnitude: 9 } },
      { name: 'RandomErasing', params: { probability: 0.25 } },
    ],
    regularization: {
      labelSmoothing: 0.1,
      mixup: 0.2,
      cutmix: 1.0,
      stochasticDepth: 0.05,
      weightDecay: 0.02,
    },
    mixedPrecision: {
      enabled: true,
      dtype: 'fp16',
    },
    ema: {
      enabled: true,
      decay: 0.9999,
    },
    implementation: `
import timm
from timm.scheduler import CosineLRScheduler
from timm.optim import Lamb
from timm.data import Mixup

model = timm.create_model('resnet50', pretrained=False, drop_path_rate=0.05)
optimizer = Lamb(model.parameters(), lr=0.008, weight_decay=0.02)
scheduler = CosineLRScheduler(optimizer, t_initial=600, warmup_t=5,
                              warmup_lr_init=1e-6, lr_min=1e-5)

mixup_fn = Mixup(mixup_alpha=0.2, cutmix_alpha=1.0, label_smoothing=0.1)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Training loop with EMA model update
for epoch in range(600):
    train_one_epoch_with_mixup(model, train_loader, optimizer, criterion, mixup_fn)
    ema_model.update(model)
    validate(ema_model.module, val_loader)
    scheduler.step(epoch)
`,
    notes: [
      'Requires 8 GPUs with 256 batch size per GPU for 2048 effective batch',
      'Uses LAMB optimizer for large batch stability',
      'Training crop size 176 with test crop 224',
      'EMA model achieves best results',
      'Long training (600 epochs) is crucial',
    ],
  },

  /**
   * Vision Transformer recipe
   */
  vit_base_patch16: {
    name: 'ViT-Base/16 ImageNet Recipe',
    description: 'Vision Transformer Base training from scratch on ImageNet-1K',
    version: '1.0.0',
    source: 'DeiT: Training data-efficient image transformers',
    task: 'ImageNet-1K classification',
    targetPerformance: '81.8% top-1 accuracy',
    optimizer: {
      type: 'AdamW',
      learningRate: 1e-3,
      weightDecay: 0.05,
      beta1: 0.9,
      beta2: 0.999,
      epsilon: 1e-8,
    },
    scheduler: {
      type: 'CosineAnnealing',
      warmupEpochs: 5,
      totalEpochs: 300,
      minLr: 1e-5,
    },
    batchSize: {
      perGpu: 128,
      effective: 1024,
      gradientAccumulation: 1,
    },
    augmentation: [
      { name: 'RandomResizedCrop', params: { size: 224 } },
      { name: 'RandomHorizontalFlip', params: {} },
      { name: 'RandAugment', params: { num_ops: 2, magnitude: 9 } },
      { name: 'ColorJitter', params: { brightness: 0.3, contrast: 0.3, saturation: 0.3 } },
      { name: 'RandomErasing', params: { probability: 0.25 } },
    ],
    regularization: {
      dropout: 0.0,
      labelSmoothing: 0.1,
      mixup: 0.8,
      cutmix: 1.0,
      stochasticDepth: 0.1,
      weightDecay: 0.05,
    },
    mixedPrecision: {
      enabled: true,
      dtype: 'fp16',
    },
    ema: {
      enabled: true,
      decay: 0.9999,
    },
    implementation: `
import timm
from timm.optim import AdamW
from timm.scheduler import CosineLRScheduler
from timm.data import Mixup

model = timm.create_model('vit_base_patch16_224', pretrained=False,
                          drop_path_rate=0.1)
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
scheduler = CosineLRScheduler(optimizer, t_initial=300, warmup_t=5,
                              warmup_lr_init=1e-6, lr_min=1e-5)

mixup_fn = Mixup(mixup_alpha=0.8, cutmix_alpha=1.0, label_smoothing=0.1)

for epoch in range(300):
    train_one_epoch_with_mixup(model, train_loader, optimizer, criterion, mixup_fn)
    ema_model.update(model)
    validate(ema_model.module, val_loader)
    scheduler.step(epoch)
`,
    notes: [
      'ViT requires strong regularization when trained from scratch',
      'Mixup and CutMix are essential for good performance',
      'Stochastic depth helps prevent overfitting',
      'Consider using pretrained weights for faster convergence',
      'Layer-wise learning rate decay can help (lower LR for early layers)',
    ],
  },
};

/**
 * Language model training recipes.
 */
export const LanguageModelRecipes: Record<string, TrainingRecipe> = {
  /**
   * GPT-style autoregressive language model training.
   */
  gpt_small: {
    name: 'GPT-Small (125M) Training Recipe',
    description: 'Training recipe for a 125M parameter GPT-style model',
    version: '1.0.0',
    source: 'Adapted from GPT-2/GPT-3 papers and nanoGPT',
    task: 'Autoregressive language modeling',
    targetPerformance: '~30 perplexity on Wikitext-103',
    optimizer: {
      type: 'AdamW',
      learningRate: 6e-4,
      weightDecay: 0.1,
      beta1: 0.9,
      beta2: 0.95,
      epsilon: 1e-8,
    },
    scheduler: {
      type: 'CosineAnnealing',
      warmupSteps: 2000,
      totalSteps: 600000,
      minLr: 6e-5,
    },
    batchSize: {
      perGpu: 12,
      effective: 480,
      gradientAccumulation: 5,
    },
    regularization: {
      dropout: 0.1,
      weightDecay: 0.1,
    },
    mixedPrecision: {
      enabled: true,
      dtype: 'bf16',
    },
    implementation: `
import torch
from torch.optim import AdamW

# Model config
config = {
    'vocab_size': 50257,
    'n_layer': 12,
    'n_head': 12,
    'n_embd': 768,
    'block_size': 1024,
    'dropout': 0.1,
}

model = GPT(config)
optimizer = AdamW(
    model.parameters(),
    lr=6e-4,
    betas=(0.9, 0.95),
    weight_decay=0.1
)

# Cosine schedule with warmup
def get_lr(step):
    if step < 2000:
        return 6e-4 * step / 2000
    if step > 600000:
        return 6e-5
    decay_ratio = (step - 2000) / (600000 - 2000)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return 6e-5 + coeff * (6e-4 - 6e-5)

# Training with gradient accumulation
for step in range(600000):
    optimizer.zero_grad()
    for micro_step in range(5):
        loss = model(batch) / 5
        loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
`,
    notes: [
      'Gradient clipping at 1.0 is essential for stability',
      'AdamW with beta2=0.95 (not 0.999) for LLMs',
      'Weight decay 0.1 is standard for transformers',
      'Use BF16 if available (better for LLMs than FP16)',
      '10% learning rate is the minimum after cosine decay',
      'Context length 1024 for small models',
    ],
  },

  /**
   * LLaMA-style training recipe
   */
  llama_7b: {
    name: 'LLaMA-7B Training Recipe',
    description: 'Training recipe for LLaMA-style 7B parameter model',
    version: '1.0.0',
    source: 'LLaMA: Open and Efficient Foundation Language Models',
    task: 'Autoregressive language modeling',
    targetPerformance: 'Competitive with larger models',
    optimizer: {
      type: 'AdamW',
      learningRate: 3e-4,
      weightDecay: 0.1,
      beta1: 0.9,
      beta2: 0.95,
      epsilon: 1e-8,
    },
    scheduler: {
      type: 'CosineAnnealing',
      warmupSteps: 2000,
      totalSteps: 1000000,
      minLr: 3e-5,
    },
    batchSize: {
      perGpu: 2,
      effective: 4000000,
      gradientAccumulation: 32,
    },
    regularization: {
      dropout: 0.0,
      weightDecay: 0.1,
    },
    mixedPrecision: {
      enabled: true,
      dtype: 'bf16',
    },
    implementation: `
# LLaMA 7B configuration
config = {
    'vocab_size': 32000,
    'hidden_size': 4096,
    'intermediate_size': 11008,
    'num_hidden_layers': 32,
    'num_attention_heads': 32,
    'max_position_embeddings': 2048,
    'rms_norm_eps': 1e-6,
}

# Use distributed training (FSDP or DeepSpeed ZeRO-3)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

model = FSDP(LLaMA(config))
optimizer = AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)

# Total tokens: 1T+ tokens
# Effective batch size: ~4M tokens
# Sequence length: 2048
# Requires 8+ A100-80GB GPUs with tensor parallelism
`,
    notes: [
      'No dropout - rely on data diversity for regularization',
      'RMSNorm instead of LayerNorm',
      'SwiGLU activation function',
      'Rotary positional embeddings (RoPE)',
      'Requires multi-node distributed training',
      'Uses tensor parallelism degree 8 minimum',
      '4M token effective batch size',
      'Pre-normalization architecture',
    ],
  },

  /**
   * BERT-style masked language model training
   */
  bert_base: {
    name: 'BERT-Base Training Recipe',
    description: 'Training recipe for BERT-Base masked language model',
    version: '1.0.0',
    source: 'BERT: Pre-training of Deep Bidirectional Transformers',
    task: 'Masked language modeling + Next sentence prediction',
    targetPerformance: 'Strong transfer learning performance',
    optimizer: {
      type: 'AdamW',
      learningRate: 1e-4,
      weightDecay: 0.01,
      beta1: 0.9,
      beta2: 0.999,
      epsilon: 1e-6,
    },
    scheduler: {
      type: 'LinearWithWarmup',
      warmupSteps: 10000,
      totalSteps: 1000000,
    },
    batchSize: {
      perGpu: 32,
      effective: 256,
    },
    regularization: {
      dropout: 0.1,
      weightDecay: 0.01,
    },
    mixedPrecision: {
      enabled: true,
      dtype: 'fp16',
    },
    implementation: `
from transformers import BertConfig, BertForPreTraining
from transformers import AdamW, get_linear_schedule_with_warmup

config = BertConfig(
    vocab_size=30522,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
)

model = BertForPreTraining(config)
optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=10000, num_training_steps=1000000
)

# Two-phase training:
# Phase 1: 90% of training with sequence length 128
# Phase 2: 10% of training with sequence length 512
`,
    notes: [
      'Two-phase training for efficiency (short sequences first)',
      '15% masking rate with 80/10/10 mask/random/unchanged',
      'Next Sentence Prediction task (optional in modern variants)',
      'Consider using whole word masking',
      'Sequence packing can improve throughput',
    ],
  },
};

/**
 * Diffusion model training recipes.
 */
export const DiffusionModelRecipes: Record<string, TrainingRecipe> = {
  /**
   * Stable Diffusion style training
   */
  stable_diffusion: {
    name: 'Stable Diffusion Training Recipe',
    description: 'Training recipe for latent diffusion models',
    version: '1.0.0',
    source: 'High-Resolution Image Synthesis with Latent Diffusion Models',
    task: 'Text-to-image generation',
    targetPerformance: 'High-quality 512x512 image generation',
    optimizer: {
      type: 'AdamW',
      learningRate: 1e-4,
      weightDecay: 0.01,
      beta1: 0.9,
      beta2: 0.999,
      epsilon: 1e-8,
    },
    scheduler: {
      type: 'Constant',
      warmupSteps: 10000,
      totalSteps: 1000000,
    },
    batchSize: {
      perGpu: 4,
      effective: 2048,
      gradientAccumulation: 64,
    },
    regularization: {
      weightDecay: 0.01,
    },
    mixedPrecision: {
      enabled: true,
      dtype: 'fp16',
    },
    ema: {
      enabled: true,
      decay: 0.9999,
    },
    implementation: `
from diffusers import UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel

# Components
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
unet = UNet2DConditionModel(
    sample_size=64,  # 64x64 latent for 512x512 image
    in_channels=4,
    out_channels=4,
    layers_per_block=2,
    block_out_channels=(320, 640, 1280, 1280),
    cross_attention_dim=768,
)

noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
optimizer = AdamW(unet.parameters(), lr=1e-4, weight_decay=0.01)

# EMA for inference quality
ema_unet = EMAModel(unet.parameters(), decay=0.9999)

# Training loop
for step in range(1000000):
    latents = vae.encode(images).latent_dist.sample() * 0.18215
    noise = torch.randn_like(latents)
    timesteps = torch.randint(0, 1000, (batch_size,))
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    encoder_hidden_states = text_encoder(text_input_ids)[0]
    noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

    loss = F.mse_loss(noise_pred, noise)
    loss.backward()
    optimizer.step()
    ema_unet.step(unet.parameters())
`,
    notes: [
      'VAE and text encoder are frozen during UNet training',
      'v-prediction can improve results over epsilon-prediction',
      'Classifier-free guidance trained with 10% unconditional',
      'EMA model used for inference',
      'Min-SNR weighting improves training',
      'Resolution: train at 256, finetune at 512',
    ],
  },

  /**
   * DiT (Diffusion Transformer) training
   */
  dit_xl: {
    name: 'DiT-XL/2 Training Recipe',
    description: 'Diffusion Transformer training recipe for class-conditional generation',
    version: '1.0.0',
    source: 'Scalable Diffusion Models with Transformers',
    task: 'Class-conditional image generation',
    targetPerformance: 'FID 2.27 on ImageNet 256x256',
    optimizer: {
      type: 'AdamW',
      learningRate: 1e-4,
      weightDecay: 0.0,
      beta1: 0.9,
      beta2: 0.999,
      epsilon: 1e-8,
    },
    scheduler: {
      type: 'Constant',
      totalSteps: 7000000,
    },
    batchSize: {
      perGpu: 32,
      effective: 256,
    },
    mixedPrecision: {
      enabled: true,
      dtype: 'fp16',
    },
    ema: {
      enabled: true,
      decay: 0.9999,
    },
    implementation: `
# DiT-XL/2 configuration
config = {
    'input_size': 32,  # 256/8 latent size
    'patch_size': 2,
    'hidden_size': 1152,
    'depth': 28,
    'num_heads': 16,
    'mlp_ratio': 4.0,
    'class_dropout_prob': 0.1,
    'num_classes': 1000,
}

model = DiT(**config)
optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0)
ema = EMAModel(model.parameters(), decay=0.9999)

# No learning rate decay - constant LR throughout
for step in range(7000000):
    loss = compute_diffusion_loss(model, batch)
    loss.backward()
    optimizer.step()
    ema.step(model.parameters())
`,
    notes: [
      'No weight decay or learning rate decay',
      'AdaLN-Zero conditioning mechanism',
      'Uses pretrained VAE (same as SD)',
      'Classifier-free guidance at inference',
      'Very long training (7M steps)',
      'Larger batch sizes improve results',
    ],
  },
};

/**
 * Reinforcement learning training recipes.
 */
export const ReinforcementLearningRecipes: Record<string, TrainingRecipe> = {
  /**
   * PPO for continuous control
   */
  ppo_continuous: {
    name: 'PPO Continuous Control Recipe',
    description: 'Proximal Policy Optimization for continuous action spaces',
    version: '1.0.0',
    source: 'Proximal Policy Optimization Algorithms (Schulman et al.)',
    task: 'Continuous control (MuJoCo, etc.)',
    targetPerformance: 'Humanoid-v4: 6000+ reward',
    optimizer: {
      type: 'Adam',
      learningRate: 3e-4,
      epsilon: 1e-5,
    },
    scheduler: {
      type: 'Linear',
      totalSteps: 1000000,
    },
    batchSize: {
      perGpu: 2048,
      effective: 2048,
    },
    implementation: `
import torch
import torch.nn as nn

# PPO hyperparameters
config = {
    'lr': 3e-4,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_epsilon': 0.2,
    'value_coef': 0.5,
    'entropy_coef': 0.01,
    'max_grad_norm': 0.5,
    'num_envs': 8,
    'num_steps': 2048,
    'num_epochs': 10,
    'num_minibatches': 32,
}

# Network architecture
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))

# Training loop with GAE
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-5)

for iteration in range(num_iterations):
    # Collect rollouts
    rollouts = collect_rollouts(env, model, num_steps=2048)

    # Compute GAE
    advantages = compute_gae(rollouts, gamma=0.99, gae_lambda=0.95)

    # PPO update
    for epoch in range(10):
        for batch in get_minibatches(rollouts, num_minibatches=32):
            # Clipped surrogate objective
            ratio = (new_log_prob - old_log_prob).exp()
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1-0.2, 1+0.2) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(values, returns)
            entropy_loss = -entropy.mean()

            loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
`,
    notes: [
      'GAE (lambda=0.95) for advantage estimation',
      'Clip ratio 0.2 is standard',
      'Entropy bonus encourages exploration',
      'Gradient clipping at 0.5',
      'Orthogonal weight initialization helps',
      'Observation normalization is important',
      'Reward scaling/clipping may help',
    ],
  },

  /**
   * DQN for discrete control
   */
  dqn_atari: {
    name: 'DQN Atari Recipe',
    description: 'Deep Q-Network for Atari games',
    version: '1.0.0',
    source: 'Human-level control through deep reinforcement learning',
    task: 'Atari 2600 games',
    targetPerformance: 'Human-level on many games',
    optimizer: {
      type: 'Adam',
      learningRate: 1e-4,
      epsilon: 1.5e-4,
    },
    scheduler: {
      type: 'Constant',
      totalSteps: 50000000,
    },
    batchSize: {
      perGpu: 32,
      effective: 32,
    },
    implementation: `
# DQN hyperparameters
config = {
    'lr': 1e-4,
    'gamma': 0.99,
    'buffer_size': 1000000,
    'batch_size': 32,
    'learning_starts': 50000,
    'target_update_freq': 10000,
    'train_freq': 4,
    'epsilon_start': 1.0,
    'epsilon_end': 0.01,
    'epsilon_decay_steps': 1000000,
}

# Nature DQN architecture
class DQN(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(3136, 512), nn.ReLU(),
            nn.Linear(512, num_actions),
        )

# Training with experience replay
replay_buffer = ReplayBuffer(1000000)
optimizer = torch.optim.Adam(q_network.parameters(), lr=1e-4, eps=1.5e-4)

for step in range(50000000):
    # Epsilon-greedy action selection
    epsilon = linear_schedule(step, 1.0, 0.01, 1000000)
    action = epsilon_greedy(q_network, obs, epsilon)

    # Store transition
    replay_buffer.add(obs, action, reward, next_obs, done)

    # Train every 4 steps
    if step % 4 == 0 and step > 50000:
        batch = replay_buffer.sample(32)

        # Double DQN target
        with torch.no_grad():
            next_actions = q_network(next_obs).argmax(1)
            next_q = target_network(next_obs).gather(1, next_actions)
            target_q = rewards + 0.99 * next_q * (1 - dones)

        current_q = q_network(obs).gather(1, actions)
        loss = F.smooth_l1_loss(current_q, target_q)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(q_network.parameters(), 10)
        optimizer.step()

    # Update target network
    if step % 10000 == 0:
        target_network.load_state_dict(q_network.state_dict())
`,
    notes: [
      '4 frames stacked as input',
      'Frame skipping (repeat action 4 times)',
      'Reward clipping to [-1, 1]',
      'Huber loss for stability',
      'Double DQN improves performance',
      'Prioritized replay further improves',
      'Dueling architecture helps',
    ],
  },
};

/**
 * Fine-tuning recipes for pretrained models.
 */
export const FineTuningRecipes: Record<string, TrainingRecipe> = {
  /**
   * BERT fine-tuning for classification
   */
  bert_classification: {
    name: 'BERT Classification Fine-tuning',
    description: 'Fine-tuning BERT for text classification tasks',
    version: '1.0.0',
    source: 'BERT paper + community best practices',
    task: 'Text classification (GLUE, etc.)',
    targetPerformance: 'Task-specific SOTA',
    optimizer: {
      type: 'AdamW',
      learningRate: 2e-5,
      weightDecay: 0.01,
      beta1: 0.9,
      beta2: 0.999,
      epsilon: 1e-8,
    },
    scheduler: {
      type: 'LinearWithWarmup',
      warmupSteps: 0,
      totalEpochs: 3,
      additionalParams: {
        warmupRatio: 0.06,
      },
    },
    batchSize: {
      perGpu: 16,
      effective: 32,
    },
    regularization: {
      dropout: 0.1,
      weightDecay: 0.01,
    },
    mixedPrecision: {
      enabled: true,
      dtype: 'fp16',
    },
    implementation: `
from transformers import BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Different LR for BERT vs classifier
optimizer = AdamW([
    {'params': model.bert.parameters(), 'lr': 2e-5},
    {'params': model.classifier.parameters(), 'lr': 1e-4}
], weight_decay=0.01)

total_steps = len(train_dataloader) * 3
warmup_steps = int(0.06 * total_steps)
scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

for epoch in range(3):
    model.train()
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    evaluate(model, eval_dataloader)
`,
    notes: [
      'Learning rates: 2e-5, 3e-5, 5e-5 work well',
      'Epochs: 2-4 for most tasks',
      'Higher LR for classifier head (10x)',
      'Gradient clipping at 1.0',
      '6-10% warmup ratio',
      'Try multiple random seeds',
      'Max sequence length task-dependent',
    ],
  },

  /**
   * Vision model fine-tuning
   */
  vit_finetune: {
    name: 'ViT Fine-tuning Recipe',
    description: 'Fine-tuning pretrained Vision Transformers',
    version: '1.0.0',
    source: 'How to train your ViT? (Steiner et al.)',
    task: 'Image classification fine-tuning',
    targetPerformance: 'Task-specific improvements',
    optimizer: {
      type: 'SGD',
      learningRate: 0.01,
      weightDecay: 0.0,
      momentum: 0.9,
    },
    scheduler: {
      type: 'CosineAnnealing',
      warmupEpochs: 0,
      totalEpochs: 20,
      minLr: 0,
    },
    batchSize: {
      perGpu: 128,
      effective: 512,
    },
    augmentation: [
      { name: 'RandomResizedCrop', params: { size: 384 } },
      { name: 'RandomHorizontalFlip', params: {} },
      { name: 'MixUp', params: { alpha: 0.5 } },
    ],
    regularization: {
      mixup: 0.5,
    },
    mixedPrecision: {
      enabled: true,
      dtype: 'fp16',
    },
    implementation: `
import timm

model = timm.create_model('vit_base_patch16_384', pretrained=True, num_classes=num_classes)

# SGD works well for fine-tuning
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

# Optional: Layer-wise learning rate decay
def get_layer_lr_decay_params(model, lr, lr_decay=0.65):
    param_groups = []
    num_layers = len(model.blocks)
    for i, block in enumerate(model.blocks):
        layer_lr = lr * (lr_decay ** (num_layers - i - 1))
        param_groups.append({'params': block.parameters(), 'lr': layer_lr})
    param_groups.append({'params': model.head.parameters(), 'lr': lr})
    return param_groups

for epoch in range(20):
    train_one_epoch(model, train_loader, optimizer, criterion)
    validate(model, val_loader)
    scheduler.step()
`,
    notes: [
      'SGD often better than AdamW for fine-tuning',
      'Higher resolution (384) improves results',
      'Layer-wise LR decay (0.65-0.75) helps',
      'Lighter augmentation than pretraining',
      'Fewer epochs needed (10-20)',
      'Consider linear probe then fine-tune',
    ],
  },

  /**
   * LLM instruction fine-tuning
   */
  llm_instruction_tuning: {
    name: 'LLM Instruction Tuning Recipe',
    description: 'Fine-tuning LLMs for instruction following',
    version: '1.0.0',
    source: 'FLAN, InstructGPT, Alpaca',
    task: 'Instruction following',
    targetPerformance: 'Helpful, harmless, honest responses',
    optimizer: {
      type: 'AdamW',
      learningRate: 2e-5,
      weightDecay: 0.0,
      beta1: 0.9,
      beta2: 0.95,
      epsilon: 1e-8,
    },
    scheduler: {
      type: 'CosineAnnealing',
      warmupSteps: 100,
      totalEpochs: 3,
      minLr: 0,
    },
    batchSize: {
      perGpu: 4,
      effective: 128,
      gradientAccumulation: 4,
    },
    regularization: {
      dropout: 0.0,
      weightDecay: 0.0,
    },
    mixedPrecision: {
      enabled: true,
      dtype: 'bf16',
    },
    implementation: `
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

# Use LoRA for efficient fine-tuning
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
)
model = get_peft_model(model, lora_config)

optimizer = AdamW(model.parameters(), lr=2e-5, betas=(0.9, 0.95))
scheduler = get_cosine_schedule_with_warmup(optimizer, 100, num_training_steps)

# Training
for epoch in range(3):
    for batch in train_loader:
        # Only compute loss on response tokens
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
`,
    notes: [
      'LoRA is more efficient than full fine-tuning',
      'Only compute loss on response tokens',
      'Diverse instruction dataset important',
      'Quality > quantity for training data',
      'Consider RLHF as follow-up',
      'NEFTune (noise embeddings) can help',
      'Lower LR than pretraining (1e-5 to 5e-5)',
    ],
  },
};

/**
 * Few-shot and meta-learning recipes.
 */
export const FewShotLearningRecipes: Record<string, TrainingRecipe> = {
  /**
   * MAML (Model-Agnostic Meta-Learning)
   */
  maml: {
    name: 'MAML Recipe',
    description: 'Model-Agnostic Meta-Learning for few-shot classification',
    version: '1.0.0',
    source: 'Model-Agnostic Meta-Learning for Fast Adaptation',
    task: 'Few-shot classification',
    targetPerformance: '~65% on 5-way 1-shot Mini-ImageNet',
    optimizer: {
      type: 'Adam',
      learningRate: 1e-3,
      beta1: 0.9,
      beta2: 0.999,
    },
    scheduler: {
      type: 'Constant',
      totalSteps: 60000,
    },
    batchSize: {
      perGpu: 4,
      effective: 4,
    },
    implementation: `
import torch
import torch.nn as nn
from copy import deepcopy

class MAML:
    def __init__(self, model, inner_lr=0.01, num_inner_steps=5):
        self.model = model
        self.inner_lr = inner_lr
        self.num_inner_steps = num_inner_steps
        self.meta_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def adapt(self, support_x, support_y):
        """Inner loop adaptation."""
        adapted_model = deepcopy(self.model)
        for _ in range(self.num_inner_steps):
            loss = F.cross_entropy(adapted_model(support_x), support_y)
            grads = torch.autograd.grad(loss, adapted_model.parameters())
            for param, grad in zip(adapted_model.parameters(), grads):
                param.data -= self.inner_lr * grad
        return adapted_model

    def meta_train_step(self, tasks):
        """Outer loop optimization."""
        self.meta_optimizer.zero_grad()
        meta_loss = 0

        for support_x, support_y, query_x, query_y in tasks:
            adapted_model = self.adapt(support_x, support_y)
            query_loss = F.cross_entropy(adapted_model(query_x), query_y)
            meta_loss += query_loss

        meta_loss /= len(tasks)
        meta_loss.backward()
        self.meta_optimizer.step()
        return meta_loss.item()

# Training
maml = MAML(model, inner_lr=0.01, num_inner_steps=5)
for iteration in range(60000):
    tasks = sample_tasks(train_dataset, num_tasks=4, k_shot=1, k_query=15)
    loss = maml.meta_train_step(tasks)
`,
    notes: [
      'Inner LR: 0.01 for 1-shot, 0.1 for 5-shot',
      'Inner steps: 5-10 for training, more for testing',
      'First-order MAML (FOMAML) is faster',
      'Task augmentation helps',
      'Consider ANIL (almost no inner loop) variant',
      'Meta-batch size 4-8 tasks typical',
    ],
  },

  /**
   * Prototypical Networks
   */
  prototypical_networks: {
    name: 'Prototypical Networks Recipe',
    description: 'Metric-based few-shot learning with class prototypes',
    version: '1.0.0',
    source: 'Prototypical Networks for Few-shot Learning',
    task: 'Few-shot classification',
    targetPerformance: '~68% on 5-way 1-shot Mini-ImageNet',
    optimizer: {
      type: 'Adam',
      learningRate: 1e-3,
    },
    scheduler: {
      type: 'StepLR',
      totalEpochs: 200,
      additionalParams: {
        stepSize: 20,
        gamma: 0.5,
      },
    },
    batchSize: {
      perGpu: 1,
      effective: 1,
    },
    implementation: `
import torch
import torch.nn as nn
import torch.nn.functional as F

class PrototypicalNetworks(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, support_x, support_y, query_x):
        # Encode all images
        support_embeddings = self.encoder(support_x)
        query_embeddings = self.encoder(query_x)

        # Compute class prototypes
        classes = torch.unique(support_y)
        prototypes = []
        for c in classes:
            class_embeddings = support_embeddings[support_y == c]
            prototype = class_embeddings.mean(dim=0)
            prototypes.append(prototype)
        prototypes = torch.stack(prototypes)

        # Compute distances to prototypes
        distances = torch.cdist(query_embeddings, prototypes)

        # Convert to log probabilities
        log_probs = F.log_softmax(-distances, dim=1)
        return log_probs

# 4-layer ConvNet encoder
encoder = nn.Sequential(
    ConvBlock(3, 64), ConvBlock(64, 64),
    ConvBlock(64, 64), ConvBlock(64, 64),
    nn.Flatten(), nn.Linear(64, 64)
)

model = PrototypicalNetworks(encoder)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

# Episode-based training
for epoch in range(200):
    for episode in sample_episodes(train_dataset, num_episodes=100):
        support_x, support_y, query_x, query_y = episode
        log_probs = model(support_x, support_y, query_x)
        loss = F.nll_loss(log_probs, query_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()
`,
    notes: [
      'Euclidean distance works well',
      'Train with more ways than test for better generalization',
      'Larger embedding dimension can help',
      'Consider using pretrained backbone',
      'Higher shots in training than testing helps',
      'Simpler than MAML, often competitive',
    ],
  },
};

/**
 * Get all available recipes organized by category.
 */
export function getAllRecipes(): Record<string, Record<string, TrainingRecipe>> {
  return {
    ImageNet: ImageNetRecipes,
    LanguageModels: LanguageModelRecipes,
    DiffusionModels: DiffusionModelRecipes,
    ReinforcementLearning: ReinforcementLearningRecipes,
    FineTuning: FineTuningRecipes,
    FewShotLearning: FewShotLearningRecipes,
  };
}

/**
 * Find recipes matching specific criteria.
 */
export function findRecipes(criteria: {
  task?: string;
  modelSize?: string;
  optimizerType?: string;
}): TrainingRecipe[] {
  const allRecipes = getAllRecipes();
  const matchingRecipes: TrainingRecipe[] = [];

  for (const category of Object.values(allRecipes)) {
    for (const recipe of Object.values(category)) {
      let matches = true;

      if (criteria.task && !recipe.task.toLowerCase().includes(criteria.task.toLowerCase())) {
        matches = false;
      }

      if (criteria.optimizerType && recipe.optimizer.type !== criteria.optimizerType) {
        matches = false;
      }

      if (matches) {
        matchingRecipes.push(recipe);
      }
    }
  }

  return matchingRecipes;
}

/**
 * Get recipe recommendations based on model and task.
 */
export function getRecipeRecommendation(
  modelType: string,
  task: string
): TrainingRecipe | undefined {
  const allRecipes = getAllRecipes();

  // Simple matching logic - can be made more sophisticated
  if (modelType.toLowerCase().includes('resnet') && task.toLowerCase().includes('imagenet')) {
    return allRecipes.ImageNet.resnet50_a1;
  }

  if (modelType.toLowerCase().includes('vit') && task.toLowerCase().includes('imagenet')) {
    return allRecipes.ImageNet.vit_base_patch16;
  }

  if (modelType.toLowerCase().includes('gpt') || modelType.toLowerCase().includes('llama')) {
    return allRecipes.LanguageModels.gpt_small;
  }

  if (modelType.toLowerCase().includes('bert')) {
    if (task.toLowerCase().includes('pretrain')) {
      return allRecipes.LanguageModels.bert_base;
    }
    return allRecipes.FineTuning.bert_classification;
  }

  if (task.toLowerCase().includes('diffusion')) {
    return allRecipes.DiffusionModels.stable_diffusion;
  }

  if (task.toLowerCase().includes('reinforcement') || task.toLowerCase().includes('rl')) {
    return allRecipes.ReinforcementLearning.ppo_continuous;
  }

  return undefined;
}
