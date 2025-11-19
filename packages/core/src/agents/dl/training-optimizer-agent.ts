/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { z } from 'zod';
import type { AgentDefinition } from '../types.js';

/**
 * Input schema for Training Optimizer Agent.
 */
const TrainingOptimizerInputSchema = z.object({
  /** Model architecture description */
  modelArchitecture: z.string().describe('Description of model architecture (e.g., ResNet-50, GPT-2, ViT-Large)'),

  /** Task type */
  taskType: z.enum([
    'classification',
    'regression',
    'generation',
    'segmentation',
    'detection',
    'translation',
    'summarization',
    'reinforcement_learning',
    'contrastive_learning',
    'other'
  ]).describe('Type of ML task'),

  /** Dataset characteristics */
  datasetInfo: z.object({
    size: z.string().describe('Dataset size (e.g., 1M samples, 100GB)'),
    inputShape: z.string().optional().describe('Input dimensions'),
    numClasses: z.number().optional().describe('Number of output classes'),
    isImbalanced: z.boolean().optional().describe('Whether dataset is class-imbalanced'),
    dataType: z.enum(['image', 'text', 'audio', 'tabular', 'multimodal', 'other']).optional(),
  }).describe('Dataset characteristics'),

  /** Hardware constraints */
  hardware: z.object({
    gpuType: z.string().describe('GPU model (e.g., A100-80GB, V100-32GB, RTX 4090)'),
    numGpus: z.number().describe('Number of GPUs available'),
    gpuMemory: z.string().describe('Per-GPU memory'),
    cpuMemory: z.string().optional().describe('System RAM available'),
    interconnect: z.enum(['NVLink', 'PCIe', 'InfiniBand', 'Ethernet']).optional(),
  }).describe('Available hardware'),

  /** Current training configuration (if any) */
  currentConfig: z.object({
    optimizer: z.string().optional(),
    learningRate: z.number().optional(),
    batchSize: z.number().optional(),
    epochs: z.number().optional(),
    scheduler: z.string().optional(),
  }).optional().describe('Current training configuration to optimize'),

  /** Optimization goals */
  goals: z.array(z.enum([
    'maximize_throughput',
    'minimize_memory',
    'fastest_convergence',
    'best_generalization',
    'stable_training',
    'reproduce_paper'
  ])).describe('Training optimization goals'),

  /** Training budget constraints */
  budget: z.object({
    maxTime: z.string().optional().describe('Maximum training time'),
    maxCost: z.string().optional().describe('Maximum compute cost'),
    targetMetric: z.string().optional().describe('Target performance metric'),
  }).optional().describe('Training budget constraints'),

  /** Framework being used */
  framework: z.enum(['pytorch', 'tensorflow', 'jax', 'other']).optional()
    .describe('Deep learning framework'),
});

/**
 * Output schema for Training Optimizer Agent.
 */
const TrainingOptimizerOutputSchema = z.object({
  /** Executive summary */
  Summary: z.string(),

  /** Analysis of current configuration */
  ConfigurationAnalysis: z.object({
    strengths: z.array(z.string()),
    weaknesses: z.array(z.string()),
    bottlenecks: z.array(z.string()),
    recommendations: z.array(z.string()),
  }),

  /** Optimized optimizer configuration */
  OptimizerConfig: z.object({
    optimizer: z.string(),
    learningRate: z.number(),
    weightDecay: z.number(),
    momentum: z.number().optional(),
    beta1: z.number().optional(),
    beta2: z.number().optional(),
    epsilon: z.number().optional(),
    amsgrad: z.boolean().optional(),
    decoupled: z.boolean().optional(),
    rationale: z.string(),
    alternativeOptimizers: z.array(z.object({
      name: z.string(),
      useCase: z.string(),
      tradeoffs: z.string(),
    })),
  }),

  /** Learning rate schedule */
  LearningRateSchedule: z.object({
    scheduleType: z.string(),
    warmupSteps: z.number(),
    warmupType: z.enum(['linear', 'exponential', 'cosine', 'constant']),
    initialLr: z.number(),
    peakLr: z.number(),
    finalLr: z.number(),
    totalSteps: z.number(),
    decayStrategy: z.string(),
    restarts: z.number().optional(),
    cycleMultiplier: z.number().optional(),
    implementation: z.string(),
    rationale: z.string(),
  }),

  /** Batch size optimization */
  BatchSizeConfig: z.object({
    effectiveBatchSize: z.number(),
    microBatchSize: z.number(),
    gradientAccumulationSteps: z.number(),
    memoryPerSample: z.string(),
    throughputEstimate: z.string(),
    scalingStrategy: z.string(),
    rationale: z.string(),
    dynamicBatching: z.boolean().optional(),
  }),

  /** Mixed precision training */
  MixedPrecisionConfig: z.object({
    enabled: z.boolean(),
    dtype: z.enum(['fp16', 'bf16', 'fp8', 'int8']),
    lossScaling: z.enum(['static', 'dynamic', 'none']),
    initialScaleFactor: z.number().optional(),
    scalingBackoffFactor: z.number().optional(),
    scalingGrowthInterval: z.number().optional(),
    layerSpecificPrecision: z.array(z.object({
      layerPattern: z.string(),
      precision: z.string(),
      reason: z.string(),
    })).optional(),
    memoryReduction: z.string(),
    speedup: z.string(),
    implementation: z.string(),
  }),

  /** Memory optimization strategies */
  MemoryOptimization: z.object({
    techniques: z.array(z.object({
      technique: z.string(),
      description: z.string(),
      memoryReduction: z.string(),
      computeOverhead: z.string(),
      implementation: z.string(),
    })),
    checkpointing: z.object({
      enabled: z.boolean(),
      strategy: z.enum(['selective', 'periodic', 'sqrt', 'full']),
      checkpointLayers: z.array(z.string()).optional(),
      memoryTradeoff: z.string(),
    }),
    activationOptimization: z.object({
      enabled: z.boolean(),
      technique: z.string(),
      implementation: z.string(),
    }).optional(),
    estimatedMemoryUsage: z.object({
      modelParams: z.string(),
      gradients: z.string(),
      optimizerStates: z.string(),
      activations: z.string(),
      total: z.string(),
    }),
  }),

  /** Gradient handling */
  GradientConfig: z.object({
    clipNorm: z.number().optional(),
    clipValue: z.number().optional(),
    clipGlobalNorm: z.boolean(),
    skipNanGrads: z.boolean(),
    gradientNoiseScale: z.number().optional(),
    gradientCentralization: z.boolean().optional(),
    lazyGradients: z.boolean().optional(),
    implementation: z.string(),
    rationale: z.string(),
  }),

  /** Regularization techniques */
  Regularization: z.object({
    dropout: z.object({
      rate: z.number(),
      type: z.enum(['standard', 'spatial', 'attention', 'droppath', 'dropblock']),
      schedule: z.string().optional(),
    }).optional(),
    labelSmoothing: z.number().optional(),
    mixup: z.object({
      enabled: z.boolean(),
      alpha: z.number(),
    }).optional(),
    cutmix: z.object({
      enabled: z.boolean(),
      alpha: z.number(),
    }).optional(),
    randAugment: z.object({
      enabled: z.boolean(),
      numOps: z.number(),
      magnitude: z.number(),
    }).optional(),
    stochasticDepth: z.object({
      enabled: z.boolean(),
      dropRate: z.number(),
    }).optional(),
    ema: z.object({
      enabled: z.boolean(),
      decay: z.number(),
      updateFrequency: z.number(),
    }).optional(),
    rationale: z.string(),
  }),

  /** Data loading optimization */
  DataLoadingConfig: z.object({
    numWorkers: z.number(),
    prefetchFactor: z.number(),
    pinMemory: z.boolean(),
    persistentWorkers: z.boolean(),
    dropLast: z.boolean(),
    shuffleBufferSize: z.number().optional(),
    cachingStrategy: z.string(),
    estimatedOverhead: z.string(),
  }),

  /** Complete training configuration */
  CompleteConfig: z.object({
    configYaml: z.string(),
    pytorchCode: z.string().optional(),
    tensorflowCode: z.string().optional(),
    jaxCode: z.string().optional(),
  }),

  /** Performance estimates */
  PerformanceEstimates: z.object({
    samplesPerSecond: z.number(),
    stepsPerSecond: z.number(),
    timePerEpoch: z.string(),
    totalTrainingTime: z.string(),
    gpuUtilization: z.string(),
    memoryUtilization: z.string(),
    expectedConvergence: z.string(),
  }),

  /** Potential issues and mitigations */
  PotentialIssues: z.array(z.object({
    issue: z.string(),
    likelihood: z.enum(['high', 'medium', 'low']),
    symptoms: z.array(z.string()),
    mitigation: z.string(),
  })),

  /** Monitoring recommendations */
  MonitoringRecommendations: z.array(z.object({
    metric: z.string(),
    frequency: z.string(),
    expectedBehavior: z.string(),
    alertThreshold: z.string().optional(),
  })),

  /** Next steps */
  NextSteps: z.array(z.object({
    priority: z.enum(['immediate', 'short_term', 'long_term']),
    action: z.string(),
    expectedBenefit: z.string(),
  })),
});

/**
 * System prompt for Training Optimizer Agent.
 */
const SYSTEM_PROMPT = `You are a world-class deep learning training optimization expert with comprehensive knowledge of:

**Optimizer Theory and Practice:**
- SGD with momentum, Nesterov acceleration, and learning rate coupling
- Adam family: Adam, AdamW, NAdam, RAdam, AdaFactor
- Second-order approximations: K-FAC, Shampoo, natural gradients
- Large batch optimizers: LAMB, LARS, NVLAMB
- Sharpness-aware: SAM, ASAM, GSAM, LookSAM
- Memory-efficient: Adagrad, RMSprop, 8-bit Adam
- Layer-wise learning rate adaptation: LLRD, discriminative fine-tuning

**Learning Rate Schedules:**
- Linear warmup and its variants
- Cosine annealing with warm restarts (SGDR)
- One-cycle policy (super-convergence)
- Cyclic learning rates with triangular, triangular2, exp_range modes
- Step decay, exponential decay, polynomial decay
- Inverse square root scheduling (Transformers)
- Linear probe then fine-tune (LP-FT)
- REX (Revised Exponential) scheduling

**Batch Size Theory:**
- Linear scaling rule and square root scaling
- Critical batch size and gradient noise scale
- LARS trust ratio for large batch training
- Progressive batch size increase
- Adaptive batch sizing based on gradient variance

**Mixed Precision Training:**
- FP16/BF16 training with automatic loss scaling
- Master weights in FP32
- FP8 training (NVIDIA H100+)
- Quantization-aware training
- Layer-specific precision requirements
- Numerical stability considerations

**Memory Optimization:**
- Gradient checkpointing (activation checkpointing)
- Memory-efficient attention (Flash Attention, xFormers)
- Offloading (CPU offload, NVMe offload)
- Memory pools and caching
- Tensor parallelism memory distribution
- Activation compression (AC-GC, GACT)

**Gradient Handling:**
- Gradient clipping (norm, value, adaptive)
- Gradient accumulation for large effective batch sizes
- Gradient noise injection for regularization
- Gradient centralization for faster convergence
- Gradient scaling and normalization

**Training Dynamics:**
- Loss landscape analysis
- Sharp vs flat minima
- Generalization bounds and implicit regularization
- Mode connectivity and lottery tickets
- Neural network lottery hypothesis
- Training instabilities and mitigations

**Framework-Specific Optimizations:**
- PyTorch: torch.compile, CUDA graphs, channels_last
- JAX: JIT compilation, XLA optimizations
- TensorFlow: tf.function, XLA, mixed precision policy

Your goal is to analyze training configurations and provide comprehensive optimization recommendations
that maximize training efficiency while maintaining or improving model quality.

Analysis Approach:
1. **Understand the full context**: Model size, dataset, hardware, goals
2. **Identify bottlenecks**: Memory, compute, I/O, communication
3. **Apply appropriate optimizations**: Match techniques to the specific situation
4. **Consider trade-offs**: Speed vs memory, convergence vs generalization
5. **Provide implementations**: Complete, runnable code for recommendations
6. **Anticipate issues**: Warn about potential problems and solutions

Always provide:
- Specific numerical recommendations (not ranges)
- Complete implementation code
- Rationale grounded in research and practice
- Performance estimates with assumptions stated
- Monitoring strategies to validate recommendations

Use tools to:
- Search for relevant papers and best practices
- Analyze existing code and configurations
- Find proven recipes for similar setups
- Generate complete configuration files`;

/**
 * Training Optimizer Agent - Master agent for training optimization.
 *
 * Specializes in:
 * - Optimizer configuration and selection
 * - Learning rate schedule optimization
 * - Batch size and gradient accumulation
 * - Mixed precision training
 * - Memory optimization techniques
 * - Training efficiency maximization
 */
export const TrainingOptimizerAgent: AgentDefinition<typeof TrainingOptimizerOutputSchema> = {
  name: 'training_optimizer_agent',
  description: 'Expert training optimization agent for configuring efficient deep learning training',
  version: '1.0.0',
  system_prompt: SYSTEM_PROMPT,
  input_schema: TrainingOptimizerInputSchema,
  output_schema: TrainingOptimizerOutputSchema,
  max_turns: 30,
  max_time_minutes: 30,
  thinking: {
    type: 'enabled',
    budget_tokens: 10000,
  },
  tool_config: {
    allowed_tools: [
      'read_file',
      'read_many_files',
      'grep',
      'glob',
      'write_file',
      'web_search',
    ],
    parallel_tool_calls: true,
  },
  model: 'gemini-2.0-flash-thinking-exp-01-21',
};

// Export utility types
export type TrainingOptimizerInput = z.infer<typeof TrainingOptimizerInputSchema>;
export type TrainingOptimizerOutput = z.infer<typeof TrainingOptimizerOutputSchema>;
