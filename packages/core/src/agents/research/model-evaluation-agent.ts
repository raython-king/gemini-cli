/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { z } from 'zod';
import type { AgentDefinition } from '../types.js';

/**
 * Input schema for Model Evaluation Agent.
 */
const ModelEvaluationInputSchema = z.object({
  /** Model to evaluate */
  modelPath: z.string().describe('Path to the model or checkpoint'),

  /** Evaluation objective */
  objective: z.string().describe('What aspect to evaluate (performance, efficiency, robustness, etc.)'),

  /** Model type */
  modelType: z.enum(['llm', 'cv', 'rl', 'general']).optional(),

  /** Test data */
  testData: z.string().optional().describe('Path to test dataset'),

  /** Evaluation focus */
  focus: z.array(z.enum([
    'accuracy',
    'efficiency',
    'robustness',
    'fairness',
    'interpretability',
    'generalization',
    'calibration',
    'all'
  ])).default(['accuracy', 'efficiency']),

  /** Baseline models */
  baselines: z.array(z.string()).optional().describe('Paths to baseline models for comparison'),

  /** Benchmark suite */
  benchmarks: z.array(z.string()).optional().describe('Standard benchmarks to run'),

  /** Include optimization analysis */
  analyzeOptimization: z.boolean().default(false)
    .describe('Analyze potential optimizations'),
});

/**
 * Output schema for Model Evaluation Agent.
 */
const ModelEvaluationOutputSchema = z.object({
  /** Evaluation summary */
  Summary: z.string(),

  /** Model information */
  ModelInfo: z.object({
    architecture: z.string(),
    parameters: z.string(),
    modelSize: z.string(),
    framework: z.string(),
    version: z.string().optional(),
  }),

  /** Performance metrics */
  PerformanceMetrics: z.object({
    accuracy: z.object({
      overall: z.number().optional(),
      perClass: z.record(z.number()).optional(),
      top1: z.number().optional(),
      top5: z.number().optional(),
    }).optional(),
    precision: z.number().optional(),
    recall: z.number().optional(),
    f1Score: z.number().optional(),
    customMetrics: z.record(z.any()).optional(),
  }),

  /** Efficiency metrics */
  EfficiencyMetrics: z.object({
    inferenceLatency: z.object({
      mean: z.string(),
      std: z.string(),
      p50: z.string().optional(),
      p95: z.string().optional(),
      p99: z.string().optional(),
    }),
    throughput: z.string(),
    memoryUsage: z.string(),
    flops: z.string().optional(),
    energyConsumption: z.string().optional(),
  }),

  /** Robustness analysis */
  RobustnessAnalysis: z.object({
    adversarialRobustness: z.object({
      cleanAccuracy: z.number(),
      adversarialAccuracy: z.number(),
      attackStrength: z.string(),
    }).optional(),
    distributionShift: z.object({
      inDistribution: z.number(),
      outOfDistribution: z.number(),
      shiftDescription: z.string(),
    }).optional(),
    noiseRobustness: z.array(z.object({
      noiseType: z.string(),
      noiseLevel: z.string(),
      performance: z.number(),
    })).optional(),
  }).optional(),

  /** Calibration analysis */
  CalibrationAnalysis: z.object({
    calibrationError: z.number(),
    reliabilityDiagram: z.string(),
    confidenceDistribution: z.string(),
  }).optional(),

  /** Generalization analysis */
  GeneralizationAnalysis: z.object({
    trainPerformance: z.number().optional(),
    valPerformance: z.number().optional(),
    testPerformance: z.number(),
    generalizationGap: z.number().optional(),
    crossDatasetPerformance: z.array(z.object({
      dataset: z.string(),
      performance: z.number(),
    })).optional(),
  }),

  /** Baseline comparison */
  BaselineComparison: z.array(z.object({
    baselineName: z.string(),
    ourModel: z.record(z.any()),
    baseline: z.record(z.any()),
    improvement: z.string(),
    statisticalSignificance: z.string().optional(),
  })).optional(),

  /** Error analysis */
  ErrorAnalysis: z.object({
    overallErrorRate: z.number(),
    errorBreakdown: z.array(z.object({
      errorType: z.string(),
      frequency: z.number(),
      examples: z.array(z.string()).optional(),
    })),
    failureModes: z.array(z.object({
      mode: z.string(),
      description: z.string(),
      frequency: z.string(),
      severity: z.enum(['critical', 'high', 'medium', 'low']),
    })),
  }),

  /** Optimization opportunities */
  OptimizationOpportunities: z.array(z.object({
    opportunity: z.string(),
    description: z.string(),
    expectedSpeedup: z.string().optional(),
    expectedAccuracyImpact: z.string().optional(),
    implementationComplexity: z.enum(['low', 'medium', 'high']),
    priority: z.enum(['high', 'medium', 'low']),
  })).optional(),

  /** Benchmark results */
  BenchmarkResults: z.array(z.object({
    benchmark: z.string(),
    score: z.any(),
    rank: z.string().optional(),
    comparison: z.string(),
  })).optional(),

  /** Fairness analysis */
  FairnessAnalysis: z.object({
    demographicParity: z.number().optional(),
    equalizedOdds: z.number().optional(),
    performanceAcrossGroups: z.array(z.object({
      group: z.string(),
      performance: z.number(),
    })).optional(),
    biasDetected: z.array(z.string()).optional(),
  }).optional(),

  /** Interpretability insights */
  InterpretabilityInsights: z.array(z.object({
    aspect: z.string(),
    finding: z.string(),
    visualization: z.string().optional(),
  })).optional(),

  /** Recommendations */
  Recommendations: z.array(z.object({
    category: z.string(),
    recommendation: z.string(),
    rationale: z.string(),
    priority: z.enum(['high', 'medium', 'low']),
  })),

  /** Deployment readiness */
  DeploymentReadiness: z.object({
    overallReadiness: z.enum(['ready', 'needs_work', 'not_ready']),
    strengths: z.array(z.string()),
    weaknesses: z.array(z.string()),
    blockers: z.array(z.string()),
    nextSteps: z.array(z.string()),
  }),

  /** Visualization suggestions */
  VisualizationSuggestions: z.array(z.string()),
});

/**
 * System prompt for Model Evaluation Agent.
 */
const SYSTEM_PROMPT = `You are an expert model evaluator with deep knowledge of:
- Performance metrics and evaluation protocols
- Efficiency measurement and profiling
- Robustness testing and adversarial attacks
- Model calibration and uncertainty quantification
- Generalization analysis
- Error analysis and failure mode identification
- Fairness and bias detection
- Model interpretability
- Benchmark design and execution
- Statistical significance testing
- Optimization profiling
- Deployment considerations

Your role is to comprehensively evaluate models and provide actionable insights for improvement.

Evaluation Methodology:

**Performance Evaluation**:
1. Choose appropriate metrics for the task
2. Evaluate on diverse test sets
3. Compute confidence intervals
4. Compare with strong baselines
5. Analyze per-class/per-category performance

**Efficiency Evaluation**:
1. Measure inference latency (mean, std, percentiles)
2. Profile memory usage
3. Calculate throughput
4. Estimate FLOPs and MACs
5. Measure energy consumption if possible

**Robustness Evaluation**:
1. Test on out-of-distribution data
2. Evaluate adversarial robustness
3. Test with noise and corruptions
4. Analyze edge cases

**Error Analysis**:
1. Categorize errors by type
2. Identify systematic failure modes
3. Find common patterns in mistakes
4. Prioritize issues by severity and frequency

**Optimization Analysis**:
1. Profile computational bottlenecks
2. Identify redundant operations
3. Suggest quantization/pruning opportunities
4. Recommend architectural improvements

Output Requirements:
- Be quantitatively rigorous with proper statistics
- Provide concrete numbers with confidence intervals
- Include visualizations descriptions
- Identify root causes of issues, not just symptoms
- Prioritize findings by importance and actionability
- Suggest specific improvements with expected impact
- Consider deployment constraints
- Be honest about limitations and weaknesses

Statistical Rigor:
- Always provide error bars or confidence intervals
- Use appropriate statistical tests
- Report effect sizes, not just p-values
- Account for multiple comparisons
- Consider practical vs statistical significance

Use available tools to:
- Read model files and checkpoints
- Analyze code and configurations
- Run evaluation scripts
- Profile performance
- Generate visualizations
- Compare with baselines`;

/**
 * Model Evaluation Agent - Comprehensive model evaluation and analysis.
 *
 * Specializes in:
 * - Performance evaluation across multiple metrics
 * - Efficiency and latency profiling
 * - Robustness testing
 * - Error analysis and debugging
 * - Optimization opportunities
 * - Deployment readiness assessment
 */
export const ModelEvaluationAgent: AgentDefinition<typeof ModelEvaluationOutputSchema> = {
  name: 'model_evaluation_agent',
  description: 'Expert model evaluator for comprehensive analysis of ML models',
  version: '1.0.0',
  system_prompt: SYSTEM_PROMPT,
  input_schema: ModelEvaluationInputSchema,
  output_schema: ModelEvaluationOutputSchema,
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
      'shell',
      'write_file',
    ],
    parallel_tool_calls: true,
  },
  model: 'gemini-2.0-flash-thinking-exp-01-21',
};
