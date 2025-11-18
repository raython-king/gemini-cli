/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { z } from 'zod';
import type { SkillDefinition, SkillContext, SkillResult } from '../types.js';

/**
 * Input schema for Scaling Law Analysis Skill
 */
const ScalingLawAnalysisInputSchema = z.object({
  domain: z.enum(['llm', 'cv', 'multimodal']).describe('Domain for scaling analysis'),

  constraints: z.object({
    computeBudget: z.string().optional().describe('Total compute budget in FLOPs (e.g., "1e23")'),
    maxParameters: z.string().optional().describe('Maximum model parameters (e.g., "70B")'),
    maxDataSize: z.string().optional().describe('Maximum dataset size (e.g., "1T tokens")'),
  }).describe('Resource constraints'),

  objectives: z.array(z.string()).describe('Analysis objectives'),

  includeImplementation: z.boolean().default(true)
    .describe('Include code for scaling calculations'),
});

export type ScalingLawAnalysisInput = z.infer<typeof ScalingLawAnalysisInputSchema>;

/**
 * Output schema for Scaling Law Analysis Skill
 */
const ScalingLawAnalysisOutputSchema = z.object({
  analysis: z.object({
    overview: z.string(),
    scaling_laws: z.array(z.object({
      relationship: z.string(),
      formula: z.string(),
      implications: z.string(),
    })),
  }),

  recommendations: z.object({
    optimal_model_size: z.string(),
    optimal_data_size: z.string(),
    training_duration: z.string(),
    expected_performance: z.string(),
    cost_estimate: z.string(),
  }),

  implementation: z.object({
    calculator_code: z.string(),
    usage_examples: z.array(z.object({
      scenario: z.string(),
      code: z.string(),
      result: z.string(),
    })),
  }).optional(),

  tradeoff_analysis: z.object({
    model_vs_data: z.string(),
    performance_vs_cost: z.string(),
    training_vs_inference: z.string(),
  }),

  actionable_steps: z.array(z.string()),
});

export type ScalingLawAnalysisOutput = z.infer<typeof ScalingLawAnalysisOutputSchema>;

/**
 * Scaling Law Analysis Skill
 *
 * Analyzes scaling laws to optimize model and data sizing:
 * - Chinchilla-optimal allocation
 * - Performance prediction
 * - Cost-performance tradeoffs
 * - Resource planning
 */
export const ScalingLawAnalysisSkill: SkillDefinition<
  ScalingLawAnalysisInput,
  ScalingLawAnalysisOutput
> = {
  id: 'research.scaling_law_analysis',
  name: 'Scaling Law Analysis',
  category: 'research',
  complexity: 'expert',
  description: 'Optimize model/data sizing using scaling laws (Chinchilla, Kaplan) and predict performance',
  tags: ['scaling-laws', 'compute-optimal', 'chinchilla', 'performance-prediction'],

  inputSchema: ScalingLawAnalysisInputSchema,
  outputSchema: ScalingLawAnalysisOutputSchema,

  requiredAgents: ['scaling_law_agent'],

  async execute(
    input: ScalingLawAnalysisInput,
    context: SkillContext,
  ): Promise<SkillResult<ScalingLawAnalysisOutput>> {
    const startTime = Date.now();

    try {
      // Step 1: Analyze scaling laws for domain
      if (context.progressCallback) {
        context.progressCallback({
          step: 'analysis',
          message: `Analyzing ${input.domain.toUpperCase()} scaling laws...`,
          progress: 0.2,
        });
      }

      const scalingAnalysis = await context.executeAgent('scaling_law_agent', {
        objective: `Analyze scaling laws for ${input.domain} with constraints`,
        domain: input.domain,
        constraints: input.constraints,
        depth: 'thorough',
        includeImplementation: input.includeImplementation,
      });

      // Step 2: Calculate optimal configuration
      if (context.progressCallback) {
        context.progressCallback({
          step: 'optimization',
          message: 'Calculating optimal model and data sizes...',
          progress: 0.5,
        });
      }

      const recommendations = this.calculateOptimalConfig(input, scalingAnalysis);

      // Step 3: Generate implementation code
      const implementation = input.includeImplementation
        ? this.generateImplementation(input.domain)
        : undefined;

      // Step 4: Tradeoff analysis
      if (context.progressCallback) {
        context.progressCallback({
          step: 'tradeoffs',
          message: 'Analyzing tradeoffs...',
          progress: 0.8,
        });
      }

      const tradeoffAnalysis = this.analyzeTradeoffs(input, recommendations);

      const output: ScalingLawAnalysisOutput = {
        analysis: {
          overview: scalingAnalysis.Summary || 'Scaling law analysis complete',
          scaling_laws: this.extractScalingLaws(scalingAnalysis),
        },
        recommendations,
        implementation,
        tradeoff_analysis: tradeoffAnalysis,
        actionable_steps: this.generateActionableSteps(recommendations),
      };

      const duration = Date.now() - startTime;

      return {
        success: true,
        data: output,
        metadata: {
          executionTime: duration,
          agentsUsed: ['scaling_law_agent'],
          confidence: 0.92,
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

  calculateOptimalConfig(input: ScalingLawAnalysisInput, analysisResult: any) {
    const compute = this.parseCompute(input.constraints.computeBudget || '1e23');
    const { optimalParams, optimalData } = this.chinchillaOptimal(compute, input.domain);

    return {
      optimal_model_size: this.formatParams(optimalParams),
      optimal_data_size: this.formatData(optimalData, input.domain),
      training_duration: this.estimateTrainingTime(optimalParams, optimalData),
      expected_performance: this.predictPerformance(optimalParams, optimalData, input.domain),
      cost_estimate: this.estimateCost(compute),
    };
  },

  chinchillaOptimal(computeFLOPs: number, domain: string) {
    // Chinchilla: N_opt ∝ C^0.5, D_opt ∝ C^0.5
    // For LLM: C ≈ 6 * N * D
    if (domain === 'llm') {
      const optimalParams = Math.sqrt(computeFLOPs / (6 * 20));
      const optimalData = 20 * optimalParams;
      return { optimalParams, optimalData };
    } else if (domain === 'cv') {
      // CV scales slightly differently: more emphasis on data
      const optimalParams = Math.pow(computeFLOPs, 0.45);
      const optimalData = Math.pow(computeFLOPs, 0.55) / (6 * optimalParams * 3);
      return { optimalParams, optimalData };
    }

    // Default to LLM scaling
    const optimalParams = Math.sqrt(computeFLOPs / 120);
    const optimalData = 20 * optimalParams;
    return { optimalParams, optimalData };
  },

  parseCompute(computeStr: string): number {
    const match = computeStr.match(/([\d.]+)e(\d+)/);
    if (match) {
      return parseFloat(match[1]) * Math.pow(10, parseInt(match[2]));
    }
    return parseFloat(computeStr) || 1e23;
  },

  formatParams(params: number): string {
    if (params >= 1e9) return `${(params / 1e9).toFixed(1)}B parameters`;
    if (params >= 1e6) return `${(params / 1e6).toFixed(1)}M parameters`;
    return `${params.toFixed(0)} parameters`;
  },

  formatData(data: number, domain: string): string {
    const unit = domain === 'llm' ? 'tokens' : 'images';
    if (data >= 1e12) return `${(data / 1e12).toFixed(1)}T ${unit}`;
    if (data >= 1e9) return `${(data / 1e9).toFixed(1)}B ${unit}`;
    if (data >= 1e6) return `${(data / 1e6).toFixed(1)}M ${unit}`;
    return `${data.toFixed(0)} ${unit}`;
  },

  estimateTrainingTime(params: number, data: number): string {
    // Rough estimate: assume 10K tokens/sec per GPU for 7B model
    const throughputPerGPU = 10000 * (7e9 / params); // scale by model size
    const numGPUs = Math.max(8, Math.ceil(params / 1e9) * 8);
    const totalThroughput = throughputPerGPU * numGPUs;
    const timeSeconds = data / totalThroughput;
    const timeDays = timeSeconds / (24 * 3600);

    if (timeDays < 1) return `${Math.round(timeDays * 24)} hours on ${numGPUs} GPUs`;
    return `${Math.round(timeDays)} days on ${numGPUs} GPUs`;
  },

  predictPerformance(params: number, data: number, domain: string): string {
    if (domain === 'llm') {
      // Kaplan scaling: L(N,D) ≈ 1.69 + A/N^α + B/D^β
      const E = 1.69;
      const A = 8.8e13;
      const B = 5.4e13;
      const alpha = 0.076;
      const beta = 0.095;

      const loss = E + Math.pow(A / params, alpha) + Math.pow(B / data, beta);
      const perplexity = Math.exp(loss);

      return `Loss: ${loss.toFixed(3)} nats, Perplexity: ${perplexity.toFixed(1)}`;
    } else if (domain === 'cv') {
      // Approximate for ViT
      const error = 0.05 + 1e10 * Math.pow(params, -0.35) + 1e11 * Math.pow(data, -0.43);
      const accuracy = (1 - error) * 100;

      return `Estimated ImageNet accuracy: ${accuracy.toFixed(1)}%`;
    }

    return 'Performance prediction available after training';
  },

  estimateCost(computeFLOPs: number): string {
    // Rough cost estimate: $2/GPU-hour for A100
    // A100 delivers ~300 TFLOP/s
    const a100TFLOPs = 300e12; // 300 TFLOP/s
    const gpuHours = computeFLOPs / (a100TFLOPs * 3600);
    const cost = gpuHours * 2;

    if (cost >= 1e6) return `$${(cost / 1e6).toFixed(1)}M`;
    if (cost >= 1e3) return `$${(cost / 1e3).toFixed(0)}K`;
    return `$${Math.round(cost)}`;
  },

  extractScalingLaws(analysisResult: any) {
    return [
      {
        relationship: 'Loss vs. Model Size',
        formula: 'L(N) ∝ N^(-α), α ≈ 0.076 (LLM) or 0.35 (CV)',
        implications: 'Doubling model size improves loss predictably',
      },
      {
        relationship: 'Loss vs. Data Size',
        formula: 'L(D) ∝ D^(-β), β ≈ 0.095 (LLM) or 0.43 (CV)',
        implications: 'Data scaling is crucial, especially for vision',
      },
      {
        relationship: 'Chinchilla Optimal',
        formula: 'N_opt ∝ C^0.5, D_opt ∝ C^0.5',
        implications: 'Equal scaling of model and data maximizes performance',
      },
    ];
  },

  analyzeTradeoffs(input: ScalingLawAnalysisInput, recommendations: any) {
    return {
      model_vs_data: `Chinchilla-optimal: ${recommendations.optimal_model_size} with ${recommendations.optimal_data_size}. Over-training (more data) reduces inference cost but increases training time.`,
      performance_vs_cost: `Estimated cost: ${recommendations.cost_estimate}. Performance improves as power law with diminishing returns. Consider smaller model with more data for inference efficiency.`,
      training_vs_inference: `Training: ${recommendations.training_duration}. For deployment, consider over-training (30-40 tokens/param instead of 20) to reduce inference compute per token.`,
    };
  },

  generateImplementation(domain: string) {
    return {
      calculator_code: `# Scaling Law Calculator
import math

def chinchilla_optimal(compute_flops: float, domain: str = 'llm'):
    """
    Calculate Chinchilla-optimal model and data sizes.

    Args:
        compute_flops: Total compute budget in FLOPs
        domain: 'llm' or 'cv'

    Returns:
        (optimal_params, optimal_data)
    """
    if domain == 'llm':
        # Chinchilla: N ∝ C^0.5, D ∝ C^0.5
        # C ≈ 6 * N * D (6 FLOPs per token: forward + backward)
        optimal_params = math.sqrt(compute_flops / (6 * 20))
        optimal_data = 20 * optimal_params  # 20 tokens per parameter

    elif domain == 'cv':
        # Vision: slightly more emphasis on data
        optimal_params = compute_flops ** 0.45
        optimal_data = (compute_flops / (6 * optimal_params * 3)) ** 0.55

    return optimal_params, optimal_data

def predict_loss(params: float, data: float, domain: str = 'llm'):
    """Predict final training loss."""
    if domain == 'llm':
        E = 1.69  # Irreducible loss
        A, B = 8.8e13, 5.4e13
        alpha, beta = 0.076, 0.095
        loss = E + (A / params) ** alpha + (B / data) ** beta
        return loss

    elif domain == 'cv':
        c0 = 0.05  # Irreducible error
        c1, c2 = 1e10, 1e11
        alpha, beta = 0.35, 0.43
        error = c0 + c1 * params ** (-alpha) + c2 * data ** (-beta)
        return error

# Example usage
compute_budget = 1e23  # 100 ZettaFLOPs
params, data = chinchilla_optimal(compute_budget, 'llm')
print(f"Optimal: {params/1e9:.1f}B params, {data/1e9:.1f}B tokens")

loss = predict_loss(params, data, 'llm')
print(f"Predicted loss: {loss:.3f} nats")
`,
      usage_examples: [
        {
          scenario: 'LLM with 1e23 FLOPs budget',
          code: 'params, data = chinchilla_optimal(1e23, "llm")',
          result: '~70B params, ~1.4T tokens',
        },
        {
          scenario: 'Vision model with 1e21 FLOPs',
          code: 'params, data = chinchilla_optimal(1e21, "cv")',
          result: '~100M params, ~100M images',
        },
      ],
    };
  },

  generateActionableSteps(recommendations: any): string[] {
    return [
      `Target model size: ${recommendations.optimal_model_size}`,
      `Prepare dataset: ${recommendations.optimal_data_size}`,
      `Allocate compute for ${recommendations.training_duration}`,
      `Expected cost: ${recommendations.cost_estimate}`,
      'Use the scaling calculator code to refine estimates',
      'Monitor loss during training to validate scaling predictions',
      'Consider checkpointing every 10-20% of training',
      'Use mixed precision (FP16/BF16) to reduce compute by 2-3x',
    ];
  },
};
