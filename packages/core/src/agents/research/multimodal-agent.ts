/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { z } from 'zod';
import type { AgentDefinition } from '../types.js';

/**
 * Input schema for Multi-Modal Research Agent.
 */
const MultiModalInputSchema = z.object({
  /** Research objective or question */
  objective: z.string().describe('The research objective about multimodal models'),

  /** Type of multimodal model */
  modelType: z.enum([
    'contrastive',      // CLIP, ALIGN
    'generative',       // BLIP, Flamingo, GPT-4V
    'unified',          // ImageBind, LLaVA
    'video_language',   // Flamingo, VideoChat
  ]).optional().describe('Type of multimodal model'),

  /** Modalities to analyze */
  modalities: z.array(z.enum(['vision', 'text', 'audio', 'video'])).default(['vision', 'text'])
    .describe('Modalities to consider'),

  /** Specific models to focus on */
  specificModels: z.array(z.string()).optional()
    .describe('Specific models to analyze (e.g., CLIP, BLIP-2, Flamingo)'),

  /** Research depth */
  depth: z.enum(['quick', 'thorough', 'comprehensive']).default('thorough')
    .describe('Research depth level'),

  /** Target task */
  targetTask: z.enum([
    'image_text_retrieval',
    'visual_question_answering',
    'image_captioning',
    'visual_reasoning',
    'multimodal_generation',
  ]).optional().describe('Target task or application'),

  /** Focus areas */
  focus: z.array(z.enum([
    'architecture',
    'training',
    'alignment',
    'zero_shot',
    'few_shot',
    'reasoning',
    'implementation',
    'deployment',
  ])).optional().describe('Specific focus areas'),

  /** Include implementation guidance */
  includeImplementation: z.boolean().default(true)
    .describe('Whether to include code examples'),
});

/**
 * Output schema for Multi-Modal Research Agent.
 */
const MultiModalOutputSchema = z.object({
  /** Executive summary */
  Summary: z.string().describe('Overview of multimodal model analysis'),

  /** Model architecture */
  Architecture: z.object({
    overview: z.string(),
    encoders: z.array(z.object({
      modality: z.string(),
      architecture: z.string(),
      parameters: z.string(),
    })),
    fusion_mechanism: z.string(),
    key_innovations: z.array(z.string()),
  }).describe('Detailed architecture analysis'),

  /** Training strategy */
  TrainingStrategy: z.object({
    objectives: z.array(z.object({
      name: z.string(),
      description: z.string(),
      loss_function: z.string(),
    })),
    datasets: z.array(z.object({
      name: z.string(),
      size: z.string(),
      modalities: z.array(z.string()),
    })),
    training_stages: z.array(z.object({
      stage: z.string(),
      description: z.string(),
      duration: z.string(),
    })),
  }).describe('Training approach and datasets'),

  /** Modality alignment */
  ModalityAlignment: z.object({
    alignment_quality: z.string(),
    techniques: z.array(z.string()),
    evaluation_methods: z.array(z.string()),
  }).describe('Cross-modal alignment strategies'),

  /** Capabilities */
  Capabilities: z.object({
    zero_shot: z.array(z.string()),
    few_shot: z.array(z.string()),
    generative: z.array(z.string()),
    reasoning: z.array(z.string()),
  }).describe('Model capabilities across different settings'),

  /** Performance analysis */
  Performance: z.object({
    benchmarks: z.array(z.object({
      task: z.string(),
      dataset: z.string(),
      metric: z.string(),
      score: z.string(),
    })),
    strengths: z.array(z.string()),
    limitations: z.array(z.string()),
  }).describe('Performance metrics and analysis'),

  /** State-of-the-art comparison */
  SOTAComparison: z.array(z.object({
    model: z.string(),
    year: z.number(),
    key_innovation: z.string(),
    capabilities: z.array(z.string()),
  })).describe('Comparison with state-of-the-art models'),

  /** Application domains */
  Applications: z.object({
    domains: z.array(z.object({
      domain: z.string(),
      use_cases: z.array(z.string()),
      requirements: z.string(),
    })),
    deployment_considerations: z.array(z.string()),
  }).describe('Practical applications and deployment'),

  /** Implementation guidance */
  Implementation: z.object({
    frameworks: z.array(z.string()),
    code_examples: z.array(z.object({
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

  /** Recommendations */
  Recommendations: z.object({
    use_cases: z.array(z.string()),
    when_to_use: z.string(),
    when_not_to_use: z.string(),
    alternatives: z.array(z.string()),
  }).describe('Practical recommendations'),

  /** Future directions */
  FutureDirections: z.array(z.string())
    .describe('Emerging trends and future research'),
});

/**
 * System prompt for Multi-Modal Research Agent.
 */
const SYSTEM_PROMPT = `You are an expert in multi-modal learning with deep knowledge of:

**Contrastive Vision-Language Models:**
- CLIP: Contrastive learning at scale (400M pairs)
- ALIGN: Noisy image-text pairs (1.8B pairs)
- OpenCLIP: Open-source alternatives with LAION
- InfoNCE loss and temperature scaling
- Zero-shot classification via text prompts

**Generative Multi-Modal Models:**
- BLIP: Bootstrapping language-image pretraining
- BLIP-2: Q-Former and frozen LLMs (efficient alignment)
- Flamingo: In-context learning for vision (interleaved data)
- GPT-4V: Multimodal reasoning
- LLaVA: Visual instruction tuning

**Unified Multi-Modal Embeddings:**
- ImageBind: 6 modalities in unified space
- Meta-Transformer: Modality-agnostic processing
- Emergent cross-modal capabilities

**Cross-Modal Alignment:**
- Attention mechanisms for fusion
- Perceiver architectures
- Q-Former and cross-attention
- Contrastive alignment strategies
- Modality-specific vs. shared encoders

**Training Strategies:**
- Contrastive learning objectives
- Image-text matching
- Masked language/image modeling
- In-context learning from interleaved data
- Two-stage training (alignment + generation)

**Applications:**
- Zero-shot image classification
- Visual question answering
- Image captioning and generation
- Visual reasoning and grounding
- Cross-modal retrieval
- Video understanding

Research Methodology:
1. **Architecture Analysis**: Understand encoders, fusion, and output heads
2. **Training Study**: Analyze objectives, datasets, and training procedures
3. **Alignment Evaluation**: Assess cross-modal alignment quality
4. **Capability Assessment**: Test zero-shot, few-shot, and generative capabilities
5. **Performance Benchmarking**: Compare on standard tasks (VQA, retrieval, captioning)
6. **Implementation Guidance**: Provide code examples and best practices
7. **Application Analysis**: Identify suitable use cases and deployment strategies

Output Requirements:
- Be technically precise about architectures and training
- Explain alignment mechanisms clearly
- Include mathematical formulations for loss functions
- Provide practical code examples
- Compare with state-of-the-art models
- Cite specific papers and models
- Discuss computational requirements
- Consider deployment constraints
- Analyze cross-modal capabilities

When analyzing multimodal models:
1. Identify key architectural innovations
2. Understand training objectives and datasets
3. Evaluate alignment quality
4. Test zero-shot and few-shot capabilities
5. Compare performance on benchmarks
6. Assess computational and memory requirements
7. Provide implementation resources
8. Suggest appropriate applications

Use available tools to:
- Read papers and documentation
- Search for model implementations
- Find pretrained weights
- Gather performance benchmarks
- Research datasets (COCO, Flickr, LAION, etc.)
- Access code repositories`;

/**
 * Multi-Modal Research Agent - Expert analysis of vision-language and multimodal models.
 *
 * This agent specializes in:
 * - Contrastive models (CLIP, ALIGN, OpenCLIP)
 * - Generative models (BLIP, BLIP-2, Flamingo, GPT-4V, LLaVA)
 * - Unified embeddings (ImageBind)
 * - Cross-modal alignment and fusion
 * - Vision-language pretraining
 * - Visual question answering and reasoning
 * - Image captioning and generation
 * - Video-language understanding
 */
export const MultiModalAgent: AgentDefinition<typeof MultiModalOutputSchema> = {
  name: 'multimodal_agent',
  description: 'Expert in multi-modal learning (CLIP, BLIP, Flamingo) and vision-language models',
  version: '1.0.0',
  system_prompt: SYSTEM_PROMPT,
  input_schema: MultiModalInputSchema,
  output_schema: MultiModalOutputSchema,
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
