/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { z } from 'zod';
import type { AgentDefinition } from '../types.js';

/**
 * Input schema for CV Research Agent.
 */
const CVResearchInputSchema = z.object({
  /** Research objective */
  objective: z.string().describe('The computer vision research objective'),

  /** Specific CV domain */
  domain: z.enum([
    'image_classification',
    'object_detection',
    'semantic_segmentation',
    'instance_segmentation',
    'image_generation',
    'video_understanding',
    'multimodal',
    '3d_vision',
    'medical_imaging',
    'other'
  ]).optional().describe('Specific computer vision domain'),

  /** Papers or resources */
  papers: z.array(z.string()).optional().describe('Papers or resources to analyze'),

  /** Research depth */
  depth: z.enum(['quick', 'thorough', 'comprehensive']).default('thorough'),

  /** Focus areas */
  focus: z.array(z.enum([
    'architecture',
    'training',
    'data_augmentation',
    'loss_functions',
    'evaluation_metrics',
    'optimization',
    'inference',
    'edge_deployment',
    'preprocessing'
  ])).optional().describe('Specific areas to focus on'),

  /** Include implementation */
  includeImplementation: z.boolean().default(false),
});

/**
 * Output schema for CV Research Agent.
 */
const CVResearchOutputSchema = z.object({
  /** Research summary */
  Summary: z.string(),

  /** Key findings */
  KeyFindings: z.array(z.object({
    title: z.string(),
    description: z.string(),
    significance: z.enum(['critical', 'high', 'medium', 'low']),
    category: z.string(),
  })),

  /** Technical analysis */
  TechnicalAnalysis: z.object({
    architecture: z.string().optional(),
    backboneDesign: z.string().optional(),
    lossFunctions: z.string().optional(),
    dataAugmentation: z.string().optional(),
    trainingStrategy: z.string().optional(),
    inferenceOptimization: z.string().optional(),
  }),

  /** Model architecture details */
  ArchitectureDetails: z.object({
    inputSpecifications: z.object({
      imageSize: z.string().optional(),
      colorSpace: z.string().optional(),
      normalization: z.string().optional(),
    }).optional(),
    layerConfiguration: z.array(z.object({
      name: z.string(),
      type: z.string(),
      parameters: z.string(),
    })).optional(),
    outputSpecifications: z.string().optional(),
    computationalComplexity: z.string().optional(),
  }).optional(),

  /** SOTA comparison */
  SOTAComparison: z.array(z.object({
    model: z.string(),
    accuracy: z.string().optional(),
    speed: z.string().optional(),
    parameters: z.string().optional(),
    strengths: z.array(z.string()),
    weaknesses: z.array(z.string()),
  })).optional(),

  /** Dataset recommendations */
  DatasetRecommendations: z.array(z.object({
    name: z.string(),
    description: z.string(),
    size: z.string().optional(),
    format: z.string().optional(),
    useCase: z.string(),
  })).optional(),

  /** Implementation guidance */
  ImplementationGuidance: z.object({
    framework: z.string().optional(),
    keyComponents: z.array(z.string()).optional(),
    codeStructure: z.string().optional(),
    dependencies: z.array(z.string()).optional(),
    implementationChallenges: z.array(z.string()).optional(),
    bestPractices: z.array(z.string()).optional(),
  }).optional(),

  /** Training recommendations */
  TrainingRecommendations: z.object({
    optimizer: z.string().optional(),
    learningRate: z.string().optional(),
    batchSize: z.string().optional(),
    epochs: z.string().optional(),
    augmentations: z.array(z.string()).optional(),
    regularization: z.array(z.string()).optional(),
  }).optional(),

  /** Evaluation metrics */
  EvaluationMetrics: z.array(z.object({
    metric: z.string(),
    description: z.string(),
    whenToUse: z.string(),
    implementation: z.string().optional(),
  })).optional(),

  /** Suggested experiments */
  SuggestedExperiments: z.array(z.object({
    title: z.string(),
    objective: z.string(),
    methodology: z.string(),
    expectedOutcomes: z.string(),
    priority: z.enum(['high', 'medium', 'low']),
  })),

  /** Optimization strategies */
  OptimizationStrategies: z.array(z.object({
    strategy: z.string(),
    description: z.string(),
    expectedSpeedup: z.string().optional(),
    tradeoffs: z.string().optional(),
  })).optional(),

  /** Resources */
  Resources: z.object({
    papers: z.array(z.object({
      title: z.string(),
      authors: z.string().optional(),
      year: z.string().optional(),
      url: z.string().optional(),
      relevance: z.string(),
    })),
    codebases: z.array(z.string()).optional(),
    datasets: z.array(z.string()).optional(),
    pretrained_models: z.array(z.string()).optional(),
  }),

  /** Implementation roadmap */
  ImplementationRoadmap: z.array(z.object({
    phase: z.string(),
    tasks: z.array(z.string()),
    estimatedTime: z.string(),
    dependencies: z.array(z.string()),
  })).optional(),
});

/**
 * System prompt for CV Research Agent.
 */
const SYSTEM_PROMPT = `You are an expert computer vision researcher with deep knowledge of:
- CNN architectures (ResNet, EfficientNet, Vision Transformer, ConvNeXt)
- Object detection (YOLO, Faster R-CNN, DETR, etc.)
- Segmentation (U-Net, Mask R-CNN, Segment Anything)
- Image generation (GANs, Diffusion Models, VAEs)
- Video understanding and temporal modeling
- 3D vision and point cloud processing
- Medical imaging and specialized domains
- Model optimization and edge deployment
- Data augmentation and preprocessing
- Loss functions and training strategies
- Evaluation metrics and benchmarking
- Transfer learning and fine-tuning

Your role is to conduct expert-level computer vision research and provide actionable insights.

Research Methodology:
1. **Architecture Analysis**: Deep dive into model architectures, design choices, and innovations
2. **Performance Evaluation**: Analyze metrics, benchmarks, and real-world performance
3. **Training Strategies**: Examine data preparation, augmentation, and training techniques
4. **Mathematical Foundations**: Explain loss functions, optimization, and theoretical basis
5. **Implementation Details**: Provide practical guidance for implementation
6. **Comparative Analysis**: Compare with SOTA models across multiple dimensions
7. **Optimization**: Identify opportunities for speed and efficiency improvements
8. **Experimental Design**: Suggest rigorous experiments to validate approaches

Output Requirements:
- Be technically precise with mathematical rigor
- Provide visual architecture diagrams descriptions when helpful
- Include specific hyperparameters and configurations
- Suggest concrete datasets and benchmarks
- Consider computational constraints and deployment scenarios
- Explain trade-offs between accuracy, speed, and model size
- Provide actionable implementation guidance
- Reference specific papers and implementations

When analyzing CV models:
1. Understand the architecture in detail (backbone, neck, head)
2. Analyze the loss function and training procedure
3. Evaluate performance on standard benchmarks
4. Consider data requirements and augmentation strategies
5. Assess computational complexity (FLOPs, parameters, latency)
6. Identify optimization opportunities
7. Compare with SOTA approaches
8. Consider practical deployment constraints

Use available tools to:
- Read papers and technical documentation
- Analyze existing implementations
- Search for related architectures and techniques
- Find relevant datasets and benchmarks
- Gather performance comparisons`;

/**
 * CV Research Agent - Expert-level computer vision research.
 *
 * Specializes in:
 * - Deep analysis of CV architectures
 * - Model design and optimization
 * - Training strategies and data augmentation
 * - Performance evaluation and benchmarking
 * - Implementation guidance
 * - Edge deployment optimization
 */
export const CVResearchAgent: AgentDefinition<typeof CVResearchOutputSchema> = {
  name: 'cv_research_agent',
  description: 'Expert computer vision researcher for deep analysis of CV models and techniques',
  version: '1.0.0',
  system_prompt: SYSTEM_PROMPT,
  input_schema: CVResearchInputSchema,
  output_schema: CVResearchOutputSchema,
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
