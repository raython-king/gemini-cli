/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { z } from 'zod';
import type { AgentDefinition } from '../types.js';

/**
 * Input schema for Scaling Law Research Agent.
 */
const ScalingLawInputSchema = z.object({
  /** Analysis objective */
  objective: z.string().describe('The scaling law analysis objective'),

  /** Domain for scaling analysis */
  domain: z.enum(['llm', 'cv', 'multimodal', 'general']).default('general')
    .describe('Domain for scaling analysis'),

  /** Target metric */
  targetMetric: z.enum([
    'loss',
    'accuracy',
    'perplexity',
    'downstream_performance',
    'compute_efficiency',
  ]).default('loss').describe('Primary metric to analyze'),

  /** Constraints */
  constraints: z.object({
    maxParameters: z.string().optional(),
    maxComputeFLOPs: z.string().optional(),
    maxDataSize: z.string().optional(),
    trainingBudget: z.string().optional(),
  }).optional().describe('Resource constraints'),

  /** Analysis depth */
  depth: z.enum(['quick', 'thorough', 'comprehensive']).default('thorough')
    .describe('Analysis depth level'),

  /** Include implementation guidance */
  includeImplementation: z.boolean().default(true)
    .describe('Whether to include code for scaling calculations'),

  /** Focus areas */
  focus: z.array(z.enum([
    'model_scaling',
    'data_scaling',
    'compute_optimization',
    'performance_prediction',
    'cost_optimization',
  ])).optional().describe('Specific focus areas'),
});

/**
 * Output schema for Scaling Law Research Agent.
 */
const ScalingLawOutputSchema = z.object({
  /** Executive summary */
  Summary: z.string().describe('Overview of scaling law analysis'),

  /** Fundamental scaling laws */
  ScalingLaws: z.object({
    overview: z.string(),
    power_laws: z.array(z.object({
      relationship: z.string(),
      formula: z.string(),
      exponent: z.string(),
      description: z.string(),
    })),
    key_findings: z.array(z.string()),
  }).describe('Fundamental scaling law relationships'),

  /** Model scaling analysis */
  ModelScaling: z.object({
    parameter_impact: z.object({
      relationship: z.string(),
      optimal_range: z.string(),
      diminishing_returns: z.string(),
    }),
    architectural_considerations: z.array(z.object({
      choice: z.string(),
      scaling_behavior: z.string(),
      recommendation: z.string(),
    })),
  }).describe('How model size affects performance'),

  /** Data scaling analysis */
  DataScaling: z.object({
    dataset_impact: z.object({
      relationship: z.string(),
      optimal_ratio: z.string(),
      quality_vs_quantity: z.string(),
    }),
    data_efficiency: z.array(z.object({
      technique: z.string(),
      improvement: z.string(),
      applicability: z.string(),
    })),
  }).describe('How data size affects performance'),

  /** Compute optimization */
  ComputeOptimization: z.object({
    budget_allocation: z.string(),
    chinchilla_optimal: z.string(),
    practical_considerations: z.string(),
    efficiency_techniques: z.array(z.object({
      technique: z.string(),
      compute_savings: z.string(),
      performance_impact: z.string(),
    })),
  }).describe('Optimal compute allocation strategies'),

  /** Performance prediction */
  PerformancePrediction: z.object({
    loss_estimation: z.object({
      formula: z.string(),
      confidence: z.string(),
      assumptions: z.array(z.string()),
    }),
    downstream_correlation: z.string(),
    predictability_limits: z.string(),
  }).describe('Predicting performance from scale'),

  /** Empirical studies */
  EmpiricalStudies: z.array(z.object({
    study: z.string(),
    year: z.number(),
    key_finding: z.string(),
    implications: z.string(),
  })).describe('Key empirical studies on scaling laws'),

  /** Recommendations */
  Recommendations: z.object({
    model_size_selection: z.string(),
    data_requirements: z.string(),
    compute_allocation: z.string(),
    training_duration: z.string(),
    cost_optimization: z.string(),
  }).describe('Practical recommendations for scaling'),

  /** Implementation guidance */
  Implementation: z.object({
    scaling_strategy: z.string(),
    code_examples: z.array(z.object({
      task: z.string(),
      code: z.string(),
      explanation: z.string(),
    })),
    monitoring: z.array(z.string()),
  }).optional().describe('Implementation details and code'),

  /** Future directions */
  FutureDirections: z.array(z.string())
    .describe('Open problems and future research in scaling'),
});

/**
 * System prompt for Scaling Law Research Agent.
 */
const SYSTEM_PROMPT = `You are an expert in machine learning scaling laws with deep knowledge of:

**Foundational Scaling Laws:**
- Kaplan et al. (2020): Neural language models scale predictably
- Chinchilla (2022): Compute-optimal training balances model and data
- Power law relationships: L(N) = (N_c / N)^α
- Emergent abilities at scale
- Domain-specific scaling (LLM, CV, multimodal)

**Model Scaling:**
- Parameter count impact on performance
- Architectural efficiency (dense vs. sparse, MoE)
- Width vs. depth scaling
- Diminishing returns and saturation points
- Optimal model sizing for compute budget

**Data Scaling:**
- Dataset size impact on generalization
- Data quality vs. quantity tradeoffs
- Optimal model-data ratios (Chinchilla: ~20 tokens/param)
- Data efficiency techniques
- Scaling past traditional dataset sizes

**Compute Optimization:**
- FLOPs budget allocation
- Chinchilla-optimal sizing: N ∝ D ∝ C^0.5
- Training time vs. model size tradeoffs
- Inference cost considerations
- Cost-performance optimization

**Performance Prediction:**
- Loss estimation from scale parameters
- Downstream task performance correlation
- Confidence intervals and uncertainty quantification
- Extrapolation limits and failure modes
- Emergent capabilities prediction

**Domain-Specific Considerations:**
- Vision: ViT scaling, data augmentation effects, patch size impact
- Language: LLM scaling, context length, tokenization
- Multimodal: Joint scaling of modalities, alignment quality

Research Methodology:
1. **Mathematical Analysis**: Derive and explain power law relationships
2. **Empirical Validation**: Review key studies (Kaplan, Chinchilla, GPT-4, etc.)
3. **Compute Allocation**: Calculate optimal model and data sizes
4. **Performance Prediction**: Estimate metrics from scaling parameters
5. **Cost Optimization**: Balance performance with computational budget
6. **Implementation Guidance**: Provide code for scaling calculations
7. **Future Directions**: Identify open problems and research gaps

Output Requirements:
- Be mathematically rigorous with formulas and exponents
- Cite key papers (Kaplan 2020, Chinchilla 2022, etc.)
- Provide actionable recommendations for resource allocation
- Include code for scaling calculations when requested
- Explain confidence bounds and assumptions
- Consider cost-performance tradeoffs
- Discuss domain-specific considerations
- Provide concrete examples with numbers

When analyzing scaling laws:
1. Explain power law relationships with formulas
2. Calculate optimal model/data sizes for given compute
3. Predict performance from scaling parameters
4. Compare with empirical studies
5. Consider architectural and training choices
6. Provide cost-performance analysis
7. Suggest monitoring and adjustment strategies

Use available tools to:
- Read scaling law papers
- Search for empirical results
- Find performance benchmarks
- Gather cost estimates
- Research training configurations`;

/**
 * Scaling Law Research Agent - Expert analysis of ML scaling laws.
 *
 * This agent specializes in:
 * - Chinchilla and Kaplan scaling laws
 * - Power law relationships
 * - Compute-optimal model and data sizing
 * - Performance prediction from scale
 * - Data efficiency analysis
 * - Training compute budget allocation
 * - Cost-performance optimization
 * - Domain-specific scaling (LLM, CV, multimodal)
 */
export const ScalingLawAgent: AgentDefinition<typeof ScalingLawOutputSchema> = {
  name: 'scaling_law_agent',
  description: 'Expert in ML scaling laws, compute-optimal training, and performance prediction',
  version: '1.0.0',
  system_prompt: SYSTEM_PROMPT,
  input_schema: ScalingLawInputSchema,
  output_schema: ScalingLawOutputSchema,
  max_turns: 25,
  max_time_minutes: 25,
  thinking: {
    type: 'enabled',
    budget_tokens: 8000,
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
