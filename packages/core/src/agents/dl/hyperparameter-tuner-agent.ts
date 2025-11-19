/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { z } from 'zod';
import type { AgentDefinition } from '../types.js';

/**
 * Input schema for Hyperparameter Tuner Agent.
 */
const HyperparameterTunerInputSchema = z.object({
  /** Model description */
  model: z.string().describe('Model architecture and size'),

  /** Task description */
  task: z.string().describe('Task description including dataset and metrics'),

  /** Hyperparameters to tune */
  hyperparameters: z.array(z.object({
    name: z.string(),
    type: z.enum(['continuous', 'discrete', 'categorical', 'conditional']),
    range: z.object({
      min: z.number().optional(),
      max: z.number().optional(),
      values: z.array(z.union([z.string(), z.number()])).optional(),
      log_scale: z.boolean().optional(),
      step: z.number().optional(),
    }),
    default: z.union([z.string(), z.number()]).optional(),
    importance: z.enum(['critical', 'high', 'medium', 'low']).optional(),
  })).describe('Hyperparameters to optimize'),

  /** Optimization objective */
  objective: z.object({
    metric: z.string().describe('Metric to optimize'),
    direction: z.enum(['minimize', 'maximize']),
    threshold: z.number().optional().describe('Target threshold for early stopping'),
    secondaryMetrics: z.array(z.object({
      metric: z.string(),
      direction: z.enum(['minimize', 'maximize']),
      weight: z.number().optional(),
    })).optional(),
  }).describe('Optimization objective'),

  /** Search budget */
  budget: z.object({
    maxTrials: z.number().describe('Maximum number of trials'),
    maxTimeHours: z.number().optional().describe('Maximum search time'),
    maxConcurrentTrials: z.number().optional().describe('Maximum parallel trials'),
    costPerTrial: z.number().optional().describe('Estimated cost per trial'),
  }).describe('Search budget'),

  /** Existing trial results */
  priorTrials: z.array(z.object({
    parameters: z.record(z.any()),
    metrics: z.record(z.number()),
    status: z.enum(['completed', 'failed', 'pruned']),
    duration: z.number().optional(),
  })).optional().describe('Results from previous trials'),

  /** Similar task results for transfer learning */
  similarTasks: z.array(z.object({
    taskName: z.string(),
    bestParameters: z.record(z.any()),
    performance: z.number(),
    similarity: z.number().optional(),
  })).optional().describe('Results from similar tasks for transfer'),

  /** Search constraints */
  constraints: z.object({
    memoryLimit: z.string().optional(),
    timePerTrialLimit: z.string().optional(),
    parameterConstraints: z.array(z.string()).optional(),
  }).optional().describe('Resource constraints'),

  /** Framework preferences */
  tuningFramework: z.enum(['optuna', 'ray_tune', 'wandb_sweeps', 'ax', 'custom']).optional()
    .describe('Preferred tuning framework'),
});

/**
 * Output schema for Hyperparameter Tuner Agent.
 */
const HyperparameterTunerOutputSchema = z.object({
  /** Executive summary */
  Summary: z.string(),

  /** Search space analysis */
  SearchSpaceAnalysis: z.object({
    dimensionality: z.number(),
    estimatedVolume: z.string(),
    complexityAssessment: z.string(),
    correlatedParameters: z.array(z.object({
      params: z.array(z.string()),
      correlationType: z.string(),
      recommendation: z.string(),
    })),
    parameterImportanceRanking: z.array(z.object({
      parameter: z.string(),
      estimatedImportance: z.number(),
      rationale: z.string(),
    })),
    searchSpaceRecommendations: z.array(z.string()),
  }),

  /** Recommended search strategy */
  SearchStrategy: z.object({
    primaryAlgorithm: z.object({
      name: z.string(),
      type: z.enum(['bayesian', 'evolutionary', 'bandit', 'grid', 'random', 'hybrid']),
      rationale: z.string(),
      hyperparameters: z.record(z.any()),
    }),
    acquisitionFunction: z.object({
      name: z.string(),
      explorationWeight: z.number(),
      rationale: z.string(),
    }).optional(),
    surrogate: z.object({
      model: z.string(),
      kernelType: z.string().optional(),
      numInitialPoints: z.number(),
    }).optional(),
    fallbackStrategies: z.array(z.object({
      condition: z.string(),
      strategy: z.string(),
    })),
    implementation: z.string(),
  }),

  /** Initial points recommendation */
  InitialPoints: z.object({
    warmStartPoints: z.array(z.object({
      parameters: z.record(z.any()),
      source: z.string(),
      expectedPerformance: z.string(),
    })),
    explorationPoints: z.array(z.object({
      parameters: z.record(z.any()),
      rationale: z.string(),
    })),
    latinHypercubeSamples: z.number().optional(),
    sobolSequence: z.boolean().optional(),
  }),

  /** Early stopping configuration */
  EarlyStoppingConfig: z.object({
    strategy: z.enum(['median', 'percentile', 'hyperband', 'asha', 'successive_halving', 'none']),
    parameters: z.object({
      minTrialsRequired: z.number(),
      graceEpochs: z.number(),
      reductionFactor: z.number().optional(),
      brackets: z.number().optional(),
      percentile: z.number().optional(),
    }),
    intermediateMetric: z.string(),
    reportingFrequency: z.string(),
    implementation: z.string(),
    rationale: z.string(),
  }),

  /** Multi-objective optimization */
  MultiObjectiveConfig: z.object({
    enabled: z.boolean(),
    algorithm: z.enum(['NSGA-II', 'MOEA/D', 'NSGA-III', 'ParEGO', 'weighted_sum']).optional(),
    objectives: z.array(z.object({
      metric: z.string(),
      direction: z.enum(['minimize', 'maximize']),
      weight: z.number().optional(),
      reference_point: z.number().optional(),
    })).optional(),
    paretoFrontStrategy: z.string().optional(),
    implementation: z.string().optional(),
  }),

  /** Transfer learning configuration */
  TransferLearningConfig: z.object({
    enabled: z.boolean(),
    sourceTasks: z.array(z.object({
      taskName: z.string(),
      transferMethod: z.enum(['warm_start', 'meta_learning', 'multi_task_gp', 'ranking_transfer']),
      weight: z.number(),
    })).optional(),
    metaFeatures: z.array(z.string()).optional(),
    adaptationStrategy: z.string().optional(),
    implementation: z.string().optional(),
  }),

  /** Suggested trial sequence */
  TrialSequence: z.array(z.object({
    trialNumber: z.number(),
    parameters: z.record(z.any()),
    rationale: z.string(),
    expectedOutcome: z.string(),
    priority: z.enum(['must_run', 'high', 'medium', 'exploratory']),
  })),

  /** Pruning strategies */
  PruningStrategies: z.object({
    hyperparameterPruning: z.array(z.object({
      condition: z.string(),
      action: z.string(),
    })),
    regionPruning: z.array(z.object({
      region: z.string(),
      reason: z.string(),
    })),
    adaptivePruning: z.object({
      enabled: z.boolean(),
      updateFrequency: z.number(),
      minDataPoints: z.number(),
    }),
  }),

  /** Analysis of prior trials */
  PriorTrialAnalysis: z.object({
    bestTrial: z.object({
      parameters: z.record(z.any()),
      metrics: z.record(z.number()),
      rank: z.number(),
    }).optional(),
    convergenceTrend: z.string(),
    parameterSensitivity: z.array(z.object({
      parameter: z.string(),
      sensitivity: z.enum(['very_high', 'high', 'medium', 'low', 'negligible']),
      optimalRange: z.string(),
    })).optional(),
    interactionEffects: z.array(z.object({
      parameters: z.array(z.string()),
      effect: z.string(),
    })).optional(),
    recommendations: z.array(z.string()),
  }),

  /** Complete implementation */
  Implementation: z.object({
    optunaCode: z.string().optional(),
    rayTuneCode: z.string().optional(),
    wandbConfig: z.string().optional(),
    axConfig: z.string().optional(),
    customCode: z.string().optional(),
    configFile: z.string(),
  }),

  /** Resource estimates */
  ResourceEstimates: z.object({
    totalTrialsNeeded: z.number(),
    expectedSearchTime: z.string(),
    computeCost: z.string().optional(),
    convergenceEstimate: z.string(),
    confidenceLevel: z.string(),
  }),

  /** Risk analysis */
  RiskAnalysis: z.array(z.object({
    risk: z.string(),
    probability: z.enum(['high', 'medium', 'low']),
    impact: z.string(),
    mitigation: z.string(),
  })),

  /** Monitoring and visualization */
  MonitoringConfig: z.object({
    metricsToTrack: z.array(z.string()),
    visualizations: z.array(z.object({
      name: z.string(),
      type: z.string(),
      parameters: z.array(z.string()),
    })),
    convergencePlots: z.array(z.string()),
    parameterImportancePlot: z.boolean(),
  }),

  /** Next steps */
  NextSteps: z.array(z.object({
    phase: z.enum(['immediate', 'after_initial_trials', 'refinement', 'final']),
    action: z.string(),
    trigger: z.string().optional(),
  })),
});

/**
 * System prompt for Hyperparameter Tuner Agent.
 */
const SYSTEM_PROMPT = `You are a world-class hyperparameter optimization expert with deep knowledge of:

**Bayesian Optimization:**
- Gaussian Processes (GP) with various kernels (Matern, RBF, rational quadratic)
- Tree-structured Parzen Estimators (TPE) and variants
- Sequential Model-based Algorithm Configuration (SMAC)
- Acquisition functions: EI, PI, UCB, EI per second, knowledge gradient
- Multi-fidelity Bayesian optimization with FABOLAS, BOHB
- Handling conditional hyperparameters and constraints
- Warm-starting and transfer learning across tasks
- Scalability: Random embeddings, Mondrian forests, ensemble GPs

**Evolutionary and Population-Based Methods:**
- CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
- PBT (Population-Based Training)
- NSGA-II/III for multi-objective optimization
- Regularized Evolution
- Differential Evolution
- Particle Swarm Optimization

**Multi-Fidelity and Early Stopping:**
- Successive Halving and Hyperband
- ASHA (Asynchronous Successive Halving Algorithm)
- BOHB (Bayesian Optimization with Hyperband)
- Learning curve extrapolation
- Median stopping rule
- Freeze-thaw Bayesian optimization

**Search Space Design:**
- Continuous, discrete, categorical, conditional spaces
- Log-scale vs linear transformations
- Hierarchical and nested search spaces
- Parameter correlations and constraints
- Effective dimensionality reduction
- Feature importance analysis: fANOVA, SHAP

**Multi-Objective Optimization:**
- Pareto frontier optimization
- NSGA-II, NSGA-III, MOEA/D
- Scalarization methods: weighted sum, Chebyshev
- Hypervolume indicator
- Expected Hypervolume Improvement

**Transfer Learning for HPO:**
- Meta-learning across tasks
- Task similarity metrics
- Warm-starting with prior knowledge
- Multi-task GPs
- Ranking-based transfer
- Zero-shot AutoML

**Practical Considerations:**
- Handling noisy evaluations
- Parallelization and distributed search
- Resource allocation and budgeting
- Dealing with failed trials
- Reproducibility and determinism
- Integration with experiment tracking

**Parameter Importance and Sensitivity:**
- fANOVA for functional ANOVA
- Local and global sensitivity analysis
- Interaction effects detection
- Critical hyperparameter identification
- Ablation path analysis

Your goal is to design optimal hyperparameter search strategies that efficiently find high-performing
configurations while respecting computational budgets.

Analysis Approach:
1. **Analyze search space**: Dimensionality, structure, correlations
2. **Leverage prior knowledge**: Transfer from similar tasks, known good regions
3. **Choose appropriate algorithm**: Match method to problem characteristics
4. **Configure early stopping**: Maximize efficiency without pruning winners
5. **Plan the search**: Initial points, exploration/exploitation balance
6. **Provide complete implementations**: Ready-to-run code for major frameworks

Always provide:
- Specific algorithm configurations (not generic advice)
- Complete implementation code for chosen framework
- Initial points to warm-start the search
- Expected resource usage and timeline
- Contingency plans for different scenarios

Use tools to:
- Find optimal hyperparameters from similar papers
- Analyze prior trial results
- Generate search configuration code
- Research best practices for specific models`;

/**
 * Hyperparameter Tuner Agent - Expert hyperparameter optimization.
 *
 * Specializes in:
 * - Search space definition and analysis
 * - Bayesian optimization configuration
 * - Early stopping strategies
 * - Multi-objective optimization
 * - Transfer learning across tasks
 * - Efficient search with limited budget
 */
export const HyperparameterTunerAgent: AgentDefinition<typeof HyperparameterTunerOutputSchema> = {
  name: 'hyperparameter_tuner_agent',
  description: 'Expert hyperparameter tuning agent for efficient search strategies',
  version: '1.0.0',
  system_prompt: SYSTEM_PROMPT,
  input_schema: HyperparameterTunerInputSchema,
  output_schema: HyperparameterTunerOutputSchema,
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
export type HyperparameterTunerInput = z.infer<typeof HyperparameterTunerInputSchema>;
export type HyperparameterTunerOutput = z.infer<typeof HyperparameterTunerOutputSchema>;
