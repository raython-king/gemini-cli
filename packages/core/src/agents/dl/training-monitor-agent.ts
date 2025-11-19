/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { z } from 'zod';
import type { AgentDefinition } from '../types.js';

/**
 * Input schema for Training Monitor Agent.
 */
const TrainingMonitorInputSchema = z.object({
  /** Model information */
  model: z.string().describe('Model architecture and configuration'),

  /** Training logs to analyze */
  trainingLogs: z.object({
    lossHistory: z.array(z.object({
      step: z.number(),
      trainLoss: z.number(),
      valLoss: z.number().optional(),
      epoch: z.number().optional(),
    })).optional().describe('Loss values over training'),

    metricHistory: z.array(z.object({
      step: z.number(),
      metrics: z.record(z.number()),
    })).optional().describe('Other metrics over training'),

    gradientStats: z.array(z.object({
      step: z.number(),
      gradNorm: z.number(),
      gradMin: z.number().optional(),
      gradMax: z.number().optional(),
      gradMean: z.number().optional(),
      gradStd: z.number().optional(),
      layerGradNorms: z.record(z.number()).optional(),
    })).optional().describe('Gradient statistics'),

    learningRateHistory: z.array(z.object({
      step: z.number(),
      lr: z.number(),
    })).optional().describe('Learning rate over training'),

    resourceUtilization: z.array(z.object({
      timestamp: z.string(),
      gpuUtilization: z.number().optional(),
      gpuMemoryUsed: z.number().optional(),
      gpuMemoryTotal: z.number().optional(),
      cpuUtilization: z.number().optional(),
      throughput: z.number().optional(),
    })).optional().describe('Resource utilization metrics'),

    logFile: z.string().optional().describe('Path to training log file'),
  }).describe('Training logs and metrics'),

  /** Training configuration */
  trainingConfig: z.object({
    optimizer: z.string(),
    learningRate: z.number(),
    batchSize: z.number(),
    totalSteps: z.number(),
    warmupSteps: z.number().optional(),
    scheduler: z.string().optional(),
  }).describe('Training configuration'),

  /** Expected behavior */
  expectations: z.object({
    targetLoss: z.number().optional(),
    targetMetric: z.object({
      name: z.string(),
      value: z.number(),
    }).optional(),
    convergenceStep: z.number().optional(),
    baselinePerformance: z.number().optional(),
  }).optional().describe('Expected training outcomes'),

  /** Analysis mode */
  analysisMode: z.enum([
    'diagnose_issues',
    'optimization_opportunities',
    'stability_check',
    'comprehensive',
    'real_time_alerts'
  ]).describe('Type of analysis to perform'),

  /** Current step */
  currentStep: z.number().optional().describe('Current training step'),
});

/**
 * Output schema for Training Monitor Agent.
 */
const TrainingMonitorOutputSchema = z.object({
  /** Executive summary */
  Summary: z.string(),

  /** Overall health assessment */
  HealthAssessment: z.object({
    overallStatus: z.enum(['healthy', 'warning', 'critical', 'needs_attention']),
    convergenceStatus: z.enum(['converging', 'converged', 'diverging', 'stalled', 'oscillating']),
    stabilityScore: z.number(),
    efficiencyScore: z.number(),
    projectedOutcome: z.string(),
  }),

  /** Loss curve analysis */
  LossCurveAnalysis: z.object({
    trainLossTrend: z.object({
      pattern: z.enum(['decreasing', 'plateaued', 'increasing', 'oscillating', 'diverging']),
      rate: z.string(),
      smoothness: z.enum(['smooth', 'noisy', 'very_noisy']),
      anomalies: z.array(z.object({
        step: z.number(),
        type: z.string(),
        severity: z.enum(['low', 'medium', 'high']),
        description: z.string(),
      })),
    }),
    valLossTrend: z.object({
      pattern: z.string(),
      gapWithTrain: z.string(),
      overfittingIndicator: z.number(),
    }).optional(),
    convergenceAnalysis: z.object({
      estimatedConvergenceStep: z.number().optional(),
      currentLossVsTarget: z.string(),
      improvementRate: z.string(),
      diminishingReturnsStep: z.number().optional(),
    }),
    recommendations: z.array(z.string()),
  }),

  /** Gradient analysis */
  GradientAnalysis: z.object({
    overallHealth: z.enum(['healthy', 'concerning', 'problematic']),
    gradientNormTrend: z.object({
      mean: z.number(),
      std: z.number(),
      trend: z.enum(['stable', 'increasing', 'decreasing', 'exploding', 'vanishing']),
      spikes: z.array(z.object({
        step: z.number(),
        value: z.number(),
        potentialCause: z.string(),
      })),
    }),
    layerWiseAnalysis: z.array(z.object({
      layer: z.string(),
      status: z.enum(['healthy', 'vanishing', 'exploding', 'dead']),
      averageNorm: z.number(),
      recommendation: z.string().optional(),
    })).optional(),
    vanishingGradientRisk: z.object({
      detected: z.boolean(),
      affectedLayers: z.array(z.string()),
      mitigation: z.string(),
    }),
    explodingGradientRisk: z.object({
      detected: z.boolean(),
      severity: z.enum(['none', 'mild', 'severe']),
      mitigation: z.string(),
    }),
    deadNeuronAnalysis: z.object({
      detected: z.boolean(),
      percentage: z.number().optional(),
      affectedLayers: z.array(z.string()).optional(),
    }).optional(),
    recommendations: z.array(z.string()),
  }),

  /** Learning rate analysis */
  LearningRateAnalysis: z.object({
    currentLr: z.number(),
    scheduleAdherence: z.enum(['as_expected', 'deviating', 'unknown']),
    effectivenessAssessment: z.object({
      tooHigh: z.boolean(),
      tooLow: z.boolean(),
      evidence: z.string(),
    }),
    warmupAnalysis: z.object({
      completed: z.boolean(),
      effectiveWarmup: z.boolean(),
      recommendations: z.string().optional(),
    }).optional(),
    suggestedAdjustments: z.array(z.object({
      adjustment: z.string(),
      rationale: z.string(),
      expectedImpact: z.string(),
    })),
  }),

  /** Resource utilization analysis */
  ResourceUtilization: z.object({
    gpuUtilization: z.object({
      average: z.number(),
      status: z.enum(['optimal', 'underutilized', 'bottlenecked']),
      bottleneck: z.string().optional(),
    }),
    memoryUtilization: z.object({
      average: z.number(),
      peak: z.number(),
      headroom: z.string(),
      memoryPressure: z.boolean(),
    }),
    throughput: z.object({
      current: z.string(),
      trend: z.enum(['stable', 'improving', 'degrading']),
      comparedToBaseline: z.string().optional(),
    }),
    ioAnalysis: z.object({
      dataPipelineBottleneck: z.boolean(),
      recommendations: z.array(z.string()),
    }),
    optimizationOpportunities: z.array(z.object({
      area: z.string(),
      currentState: z.string(),
      potentialImprovement: z.string(),
      implementation: z.string(),
    })),
  }),

  /** Anomaly detection */
  AnomalyDetection: z.object({
    detectedAnomalies: z.array(z.object({
      timestamp: z.string().optional(),
      step: z.number(),
      type: z.enum([
        'loss_spike',
        'gradient_explosion',
        'gradient_vanishing',
        'metric_regression',
        'learning_rate_anomaly',
        'memory_spike',
        'throughput_drop',
        'nan_inf',
        'other'
      ]),
      severity: z.enum(['info', 'warning', 'error', 'critical']),
      description: z.string(),
      potentialCauses: z.array(z.string()),
      suggestedActions: z.array(z.string()),
    })),
    riskAssessment: z.object({
      immediateRisks: z.array(z.string()),
      potentialFutureRisks: z.array(z.string()),
    }),
  }),

  /** Early warning indicators */
  EarlyWarnings: z.array(z.object({
    indicator: z.string(),
    currentValue: z.string(),
    threshold: z.string(),
    status: z.enum(['normal', 'approaching_threshold', 'exceeded']),
    predictedTimeToIssue: z.string().optional(),
    preventiveAction: z.string(),
  })),

  /** Validation metrics analysis */
  ValidationAnalysis: z.object({
    metricsHealth: z.record(z.object({
      trend: z.string(),
      currentValue: z.number(),
      bestValue: z.number(),
      bestStep: z.number(),
      stagnationSteps: z.number().optional(),
    })),
    overfittingAnalysis: z.object({
      detected: z.boolean(),
      severity: z.enum(['none', 'mild', 'moderate', 'severe']),
      trainValGap: z.number().optional(),
      onsetStep: z.number().optional(),
      recommendations: z.array(z.string()),
    }),
    generalizationProjection: z.string(),
  }).optional(),

  /** Comparative analysis */
  ComparativeAnalysis: z.object({
    vsBaseline: z.object({
      comparison: z.string(),
      percentageImprovement: z.number().optional(),
    }).optional(),
    vsExpected: z.object({
      onTrack: z.boolean(),
      deviation: z.string(),
      adjustments: z.array(z.string()),
    }).optional(),
    historicalComparison: z.string().optional(),
  }),

  /** Action recommendations */
  ActionRecommendations: z.array(z.object({
    priority: z.enum(['immediate', 'soon', 'when_convenient', 'consider']),
    category: z.enum([
      'learning_rate',
      'regularization',
      'architecture',
      'data',
      'optimization',
      'resources',
      'monitoring',
      'other'
    ]),
    action: z.string(),
    rationale: z.string(),
    expectedImpact: z.string(),
    implementation: z.string(),
    riskOfNotActing: z.string(),
  })),

  /** Monitoring dashboard configuration */
  MonitoringSetup: z.object({
    criticalMetrics: z.array(z.object({
      metric: z.string(),
      alertThreshold: z.string(),
      checkFrequency: z.string(),
    })),
    dashboardPanels: z.array(z.object({
      name: z.string(),
      metrics: z.array(z.string()),
      chartType: z.string(),
    })),
    alertRules: z.array(z.object({
      condition: z.string(),
      severity: z.enum(['info', 'warning', 'critical']),
      action: z.string(),
    })),
    loggingRecommendations: z.array(z.string()),
  }),

  /** Prediction and projection */
  Projections: z.object({
    estimatedTimeToTarget: z.string().optional(),
    finalPerformanceEstimate: z.string(),
    confidenceInterval: z.string(),
    riskOfFailure: z.string(),
    recommendedCheckpoint: z.string(),
  }),

  /** Debugging assistance */
  DebuggingAssistance: z.object({
    suggestedInvestigations: z.array(z.object({
      investigation: z.string(),
      command: z.string().optional(),
      expectedFindings: z.string(),
    })),
    diagnosticCode: z.string().optional(),
  }),

  /** Next steps */
  NextSteps: z.array(z.object({
    timeframe: z.enum(['now', 'next_checkpoint', 'end_of_run', 'next_run']),
    action: z.string(),
    trigger: z.string().optional(),
  })),
});

/**
 * System prompt for Training Monitor Agent.
 */
const SYSTEM_PROMPT = `You are a world-class deep learning training diagnostician with expertise in:

**Loss Curve Analysis:**
- Recognizing healthy convergence patterns
- Identifying plateau, oscillation, and divergence
- Detecting overfitting from train/val gap
- Optimal stopping point identification
- Learning rate decay timing from loss curvature
- Loss landscape interpretation
- Multi-objective loss balancing analysis

**Gradient Flow Diagnostics:**
- Gradient norm tracking and interpretation
- Vanishing gradient detection (per-layer analysis)
- Exploding gradient detection and clipping needs
- Dead neuron/ReLU identification
- Gradient noise scale estimation
- Layer-wise gradient health assessment
- Attention pattern analysis

**Training Dynamics:**
- Loss spike analysis and causes
- NaN/Inf detection and prevention
- Numerical stability monitoring
- Batch normalization statistics health
- Weight initialization effectiveness
- Mode collapse detection (GANs)
- Training/inference discrepancy detection

**Resource Efficiency Analysis:**
- GPU utilization optimization
- Memory usage patterns and leaks
- Data loading bottleneck detection
- Communication overhead in distributed training
- Throughput trend analysis
- Power efficiency considerations

**Anomaly Detection:**
- Statistical process control for training
- Sudden performance regression
- Distribution shift detection
- Out-of-distribution detection
- Reproducibility issues
- Hardware failure indicators

**Early Warning Systems:**
- Predictive convergence modeling
- Risk of divergence forecasting
- Overfitting early indicators
- Resource exhaustion prediction
- Training instability precursors
- Diminishing returns detection

**Optimization Opportunities:**
- Learning rate adjustment recommendations
- Batch size optimization signals
- Regularization strength tuning
- Model capacity assessment
- Data augmentation effectiveness
- Architecture bottleneck identification

**Monitoring Best Practices:**
- Essential metrics to track
- Appropriate logging frequencies
- Dashboard design for interpretability
- Alert threshold configuration
- Checkpoint strategies
- Experiment comparison frameworks

Your goal is to analyze training runs like an experienced ML engineer, quickly identifying issues
and optimization opportunities while providing actionable recommendations.

Analysis Approach:
1. **Holistic assessment**: Consider all metrics together
2. **Pattern recognition**: Identify known training pathologies
3. **Root cause analysis**: Don't just detect symptoms, find causes
4. **Actionable advice**: Provide specific, implementable solutions
5. **Prioritization**: Order recommendations by impact and urgency
6. **Prevention**: Set up monitoring to catch future issues early

Always provide:
- Clear health status with supporting evidence
- Specific numerical thresholds and comparisons
- Actionable recommendations with implementations
- Risk assessment for not taking action
- Monitoring setup to prevent recurrence

Think like a staff ML engineer reviewing a training run:
- What's working well?
- What's concerning?
- What should be changed immediately?
- What should be monitored closely?
- When should training be stopped?

Use tools to:
- Analyze log files in detail
- Generate visualization code
- Create monitoring configurations
- Find similar issues and solutions`;

/**
 * Training Monitor Agent - Expert training diagnostics and monitoring.
 *
 * Specializes in:
 * - Loss curve analysis and interpretation
 * - Gradient flow diagnostics
 * - Resource utilization optimization
 * - Anomaly detection and early warnings
 * - Training health assessment
 * - Actionable improvement recommendations
 */
export const TrainingMonitorAgent: AgentDefinition<typeof TrainingMonitorOutputSchema> = {
  name: 'training_monitor_agent',
  description: 'Expert training monitor agent for diagnosing issues and optimizing training runs',
  version: '1.0.0',
  system_prompt: SYSTEM_PROMPT,
  input_schema: TrainingMonitorInputSchema,
  output_schema: TrainingMonitorOutputSchema,
  max_turns: 25,
  max_time_minutes: 25,
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
export type TrainingMonitorInput = z.infer<typeof TrainingMonitorInputSchema>;
export type TrainingMonitorOutput = z.infer<typeof TrainingMonitorOutputSchema>;
