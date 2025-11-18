/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { z } from 'zod';
import type { AgentDefinition } from '../types.js';

/**
 * Input schema for Experiment Agent.
 */
const ExperimentInputSchema = z.object({
  /** Experiment objective */
  objective: z.string().describe('The main research question or hypothesis to test'),

  /** Algorithm/model type */
  algorithmType: z.enum(['llm', 'cv', 'rl', 'general']).optional()
    .describe('Type of algorithm being experimented with'),

  /** Experiment type */
  experimentType: z.enum([
    'baseline',
    'ablation',
    'hyperparameter_tuning',
    'architecture_search',
    'comparison',
    'reproducibility',
    'other'
  ]).optional().describe('Type of experiment to design'),

  /** Existing code/implementation */
  codebase: z.string().optional().describe('Path to existing code or implementation'),

  /** Dataset information */
  dataset: z.string().optional().describe('Dataset name or path'),

  /** Constraints */
  constraints: z.object({
    maxComputeBudget: z.string().optional(),
    timeLimit: z.string().optional(),
    hardwareAvailable: z.array(z.string()).optional(),
    otherConstraints: z.array(z.string()).optional(),
  }).optional().describe('Experimental constraints'),

  /** Prior results */
  priorResults: z.string().optional().describe('Previous experimental results or baselines'),
});

/**
 * Output schema for Experiment Agent.
 */
const ExperimentOutputSchema = z.object({
  /** Executive summary */
  Summary: z.string(),

  /** Research hypothesis */
  Hypothesis: z.object({
    nullHypothesis: z.string(),
    alternativeHypothesis: z.string(),
    rationale: z.string(),
  }),

  /** Experimental design */
  ExperimentalDesign: z.object({
    overview: z.string(),
    methodology: z.string(),
    controlVariables: z.array(z.string()),
    independentVariables: z.array(z.object({
      name: z.string(),
      values: z.array(z.string()),
      justification: z.string(),
    })),
    dependentVariables: z.array(z.object({
      name: z.string(),
      measurementMethod: z.string(),
    })),
  }),

  /** Experiment configurations */
  Configurations: z.array(z.object({
    name: z.string(),
    description: z.string(),
    parameters: z.record(z.any()),
    expectedOutcome: z.string(),
    priority: z.enum(['high', 'medium', 'low']),
  })),

  /** Data preparation */
  DataPreparation: z.object({
    datasetSplits: z.object({
      train: z.string().optional(),
      validation: z.string().optional(),
      test: z.string().optional(),
    }),
    preprocessing: z.array(z.string()),
    augmentation: z.array(z.string()).optional(),
    dataQualityChecks: z.array(z.string()),
  }),

  /** Training procedure */
  TrainingProcedure: z.object({
    initialization: z.string(),
    optimizer: z.string(),
    learningRateSchedule: z.string(),
    batchSize: z.string(),
    epochs: z.string(),
    earlyStoppingCriteria: z.string().optional(),
    checkpointing: z.string(),
  }),

  /** Evaluation metrics */
  EvaluationMetrics: z.array(z.object({
    metric: z.string(),
    description: z.string(),
    computationMethod: z.string(),
    statisticalTest: z.string().optional(),
  })),

  /** Baseline comparisons */
  Baselines: z.array(z.object({
    name: z.string(),
    description: z.string(),
    source: z.string().optional(),
    expectedPerformance: z.string().optional(),
  })),

  /** Implementation plan */
  ImplementationPlan: z.array(z.object({
    step: z.number(),
    task: z.string(),
    description: z.string(),
    code: z.string().optional(),
    estimatedTime: z.string(),
  })),

  /** Experiment tracking */
  ExperimentTracking: z.object({
    loggingStrategy: z.string(),
    metricsToTrack: z.array(z.string()),
    visualizations: z.array(z.string()),
    reproducibilityChecklist: z.array(z.string()),
  }),

  /** Statistical analysis plan */
  StatisticalAnalysis: z.object({
    significanceLevel: z.string(),
    statisticalTests: z.array(z.string()),
    multipleComparisonCorrection: z.string().optional(),
    confidenceIntervals: z.string(),
  }),

  /** Resource requirements */
  ResourceRequirements: z.object({
    computeResources: z.string(),
    estimatedTime: z.string(),
    storageNeeded: z.string(),
    estimatedCost: z.string().optional(),
  }),

  /** Risk assessment */
  RiskAssessment: z.array(z.object({
    risk: z.string(),
    probability: z.enum(['high', 'medium', 'low']),
    impact: z.enum(['high', 'medium', 'low']),
    mitigation: z.string(),
  })),

  /** Reproducibility guidelines */
  ReproducibilityGuidelines: z.array(z.string()),

  /** Next steps */
  NextSteps: z.array(z.object({
    action: z.string(),
    owner: z.string().optional(),
    deadline: z.string().optional(),
  })),
});

/**
 * System prompt for Experiment Agent.
 */
const SYSTEM_PROMPT = `You are an expert machine learning experimentalist with deep knowledge of:
- Experimental design and scientific methodology
- Statistical analysis and hypothesis testing
- Hyperparameter optimization and ablation studies
- Model evaluation and benchmarking
- Reproducibility best practices
- Experiment tracking and MLOps
- Data preparation and validation
- Training procedures and optimization
- Performance analysis and debugging
- Resource estimation and budgeting

Your role is to design rigorous, reproducible experiments that answer specific research questions.

Experimental Design Principles:
1. **Clear Hypothesis**: Formulate testable hypotheses with null and alternative forms
2. **Control Variables**: Identify and control for confounding factors
3. **Systematic Variation**: Design experiments to isolate effects of interest
4. **Statistical Rigor**: Plan appropriate statistical tests and significance levels
5. **Reproducibility**: Ensure experiments can be replicated by others
6. **Resource Efficiency**: Optimize for compute budget and time constraints
7. **Comprehensive Metrics**: Use multiple metrics to evaluate different aspects
8. **Baseline Comparisons**: Compare against strong, relevant baselines

Experiment Types:

**Baseline Experiments**:
- Establish performance of standard approaches
- Provide reference points for comparison
- Verify implementation correctness

**Ablation Studies**:
- Isolate contribution of individual components
- Remove/modify components systematically
- Understand what drives performance

**Hyperparameter Tuning**:
- Systematic or automated search strategies
- Cross-validation for generalization
- Consider computational budget

**Architecture Search**:
- Explore design space efficiently
- Balance exploration vs exploitation
- Consider architectural constraints

**Comparison Studies**:
- Fair comparison across methods
- Control for implementation quality
- Multiple runs for statistical significance

Output Requirements:
- Be methodologically rigorous and statistically sound
- Provide complete experimental protocols
- Include code snippets for key components
- Specify exact hyperparameters and configurations
- Plan for reproducibility from the start
- Consider computational constraints realistically
- Identify potential pitfalls and mitigation strategies
- Provide clear success criteria

Use available tools to:
- Analyze existing implementations
- Review prior experimental results
- Find relevant datasets and benchmarks
- Generate experiment tracking code
- Create visualization scripts`;

/**
 * Experiment Agent - Expert experimental design and execution.
 *
 * Specializes in:
 * - Rigorous experimental design
 * - Hypothesis formulation and testing
 * - Ablation studies
 * - Hyperparameter optimization
 * - Statistical analysis
 * - Reproducibility
 * - Resource planning
 */
export const ExperimentAgent: AgentDefinition<typeof ExperimentOutputSchema> = {
  name: 'experiment_agent',
  description: 'Expert experimentalist for designing rigorous ML/AI experiments',
  version: '1.0.0',
  system_prompt: SYSTEM_PROMPT,
  input_schema: ExperimentInputSchema,
  output_schema: ExperimentOutputSchema,
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
      'write_file',
      'web_search',
    ],
    parallel_tool_calls: true,
  },
  model: 'gemini-2.0-flash-thinking-exp-01-21',
};
