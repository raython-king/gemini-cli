/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { z } from 'zod';
import type { AgentDefinition } from '../types.js';

/**
 * Input schema for Foundation Model Research Agent.
 */
const FoundationModelInputSchema = z.object({
  /** Research objective or question */
  objective: z.string().describe('The research objective or question about foundation models'),

  /** Type of foundation model */
  modelType: z.enum([
    'vision_transformer',
    'self_supervised',
    'segmentation_foundation',
    'multimodal_vision',
    'image_generation',
    'general_purpose',
  ]).optional().describe('Type of foundation model to analyze'),

  /** Specific models to focus on */
  specificModels: z.array(z.string()).optional()
    .describe('Specific models to analyze (e.g., ViT, SAM, DINO, MAE)'),

  /** Research depth */
  depth: z.enum(['quick', 'thorough', 'comprehensive']).default('thorough')
    .describe('Research depth level'),

  /** Focus areas */
  focus: z.array(z.enum([
    'architecture',
    'pretraining',
    'transfer_learning',
    'zero_shot',
    'few_shot',
    'scaling',
    'implementation',
    'deployment',
  ])).optional().describe('Specific focus areas'),

  /** Include implementation guidance */
  includeImplementation: z.boolean().default(true)
    .describe('Whether to include code examples and implementation details'),

  /** Include scaling analysis */
  includeScalingAnalysis: z.boolean().default(false)
    .describe('Whether to analyze scaling properties'),
});

/**
 * Output schema for Foundation Model Research Agent.
 */
const FoundationModelOutputSchema = z.object({
  /** Executive summary */
  Summary: z.string().describe('Overview of foundation model analysis'),

  /** Model architecture details */
  Architecture: z.object({
    overview: z.string(),
    keyComponents: z.array(z.object({
      component: z.string(),
      description: z.string(),
      innovation: z.string(),
    })),
    parameters: z.object({
      sizes: z.array(z.string()),
      computation: z.string(),
    }),
  }).describe('Detailed architecture analysis'),

  /** Pretraining strategy */
  PretrainingStrategy: z.object({
    objective: z.string(),
    datasets: z.array(z.object({
      name: z.string(),
      size: z.string(),
      characteristics: z.string(),
    })),
    trainingDetails: z.array(z.string()),
  }).describe('Pretraining approach and datasets'),

  /** Transfer learning approaches */
  TransferLearning: z.object({
    methods: z.array(z.object({
      name: z.string(),
      description: z.string(),
      when_to_use: z.string(),
      performance: z.string().optional(),
    })),
    best_practices: z.array(z.string()),
  }).describe('Transfer learning and fine-tuning strategies'),

  /** Performance analysis */
  Performance: z.object({
    benchmarks: z.array(z.object({
      task: z.string(),
      metric: z.string(),
      score: z.string(),
      comparison: z.string(),
    })),
    capabilities: z.array(z.string()),
    limitations: z.array(z.string()),
  }).describe('Performance metrics and capabilities'),

  /** State-of-the-art comparison */
  SOTAComparison: z.array(z.object({
    model: z.string(),
    year: z.number(),
    keyInnovation: z.string(),
    performance: z.string(),
  })).describe('Comparison with state-of-the-art models'),

  /** Implementation guidance */
  Implementation: z.object({
    frameworks: z.array(z.string()),
    codeExamples: z.array(z.object({
      task: z.string(),
      code: z.string(),
      explanation: z.string(),
    })),
    resources: z.array(z.object({
      type: z.string(),
      location: z.string(),
      description: z.string(),
    })),
  }).optional().describe('Implementation details and code examples'),

  /** Scaling analysis */
  ScalingAnalysis: z.object({
    model_scaling: z.string(),
    data_scaling: z.string(),
    compute_requirements: z.string(),
  }).optional().describe('Scaling properties and requirements'),

  /** Recommendations */
  Recommendations: z.object({
    use_cases: z.array(z.string()),
    when_to_use: z.string(),
    when_not_to_use: z.string(),
    alternatives: z.array(z.string()),
  }).describe('Practical recommendations for using foundation models'),

  /** Future directions */
  FutureDirections: z.array(z.string())
    .describe('Emerging trends and future research directions'),
});

/**
 * System prompt for Foundation Model Research Agent.
 */
const SYSTEM_PROMPT = `You are an expert in computer vision foundation models with deep knowledge of:

**Vision Transformer Architectures:**
- Original ViT and variants (DeiT, Swin, CaiT, BEiT, MAE)
- Architectural innovations (shifted windows, hierarchical structures, etc.)
- Positional encodings and patch embeddings
- Attention mechanisms for vision

**Self-Supervised Learning:**
- Contrastive methods (SimCLR, MoCo, SwAV)
- Masked image modeling (MAE, BEiT, SimMIM)
- Self-distillation (DINO, DINOv2)
- Knowledge distillation techniques

**Foundation Models:**
- Segment Anything Model (SAM)
- CLIP vision encoder
- DINOv2 and EVA
- ImageBind and unified embedding spaces
- Scaling characteristics

**Pretraining Strategies:**
- Dataset curation and quality (ImageNet-21K, JFT-300M, SA-1B, LAION)
- Augmentation strategies
- Training objectives and loss functions
- Compute-optimal training

**Transfer Learning:**
- Full fine-tuning vs. linear probing
- Parameter-efficient methods (LoRA, Adapter, Prefix tuning)
- Domain adaptation
- Few-shot and zero-shot learning

Research Methodology:
1. **Architecture Analysis**: Deep dive into model structure and innovations
2. **Pretraining Study**: Understand objectives, datasets, and training procedures
3. **Transfer Learning**: Analyze fine-tuning strategies and downstream performance
4. **Performance Evaluation**: Compare with SOTA on relevant benchmarks
5. **Implementation Guidance**: Provide practical code examples and best practices
6. **Scaling Analysis**: Understand how models scale with size, data, and compute
7. **Future Directions**: Identify emerging trends and open problems

Output Requirements:
- Be technically precise with architectural details
- Include mathematical formulations where relevant
- Provide practical implementation guidance with code
- Compare with state-of-the-art models
- Cite specific papers and models
- Explain pretraining strategies clearly
- Discuss transfer learning trade-offs
- Consider computational and deployment constraints

When analyzing foundation models:
1. Identify key architectural innovations
2. Understand pretraining objectives and datasets
3. Evaluate zero-shot and few-shot capabilities
4. Compare transfer learning performance
5. Assess computational requirements
6. Provide implementation resources
7. Suggest appropriate use cases

Use available tools to:
- Read papers and documentation
- Search for model implementations
- Find pretrained weights and checkpoints
- Gather performance benchmarks
- Research training datasets`;

/**
 * Foundation Model Research Agent - Expert analysis of CV foundation models.
 *
 * This agent specializes in:
 * - Vision Transformers (ViT, Swin, DeiT, etc.)
 * - Self-supervised learning (MAE, DINO, SimCLR)
 * - Foundation models (SAM, CLIP, DINOv2)
 * - Pretraining strategies
 * - Transfer learning and fine-tuning
 * - Zero-shot and few-shot learning
 * - Implementation and deployment
 */
export const FoundationModelAgent: AgentDefinition<typeof FoundationModelOutputSchema> = {
  name: 'foundation_model_agent',
  description: 'Expert in computer vision foundation models (ViT, SAM, CLIP, DINO, MAE) and transfer learning',
  version: '1.0.0',
  system_prompt: SYSTEM_PROMPT,
  input_schema: FoundationModelInputSchema,
  output_schema: FoundationModelOutputSchema,
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
      'web_fetch',
      'web_search',
    ],
    parallel_tool_calls: true,
  },
  model: 'gemini-2.0-flash-thinking-exp-01-21',
};
