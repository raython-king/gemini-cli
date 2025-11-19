/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @fileoverview Deep Learning workflow orchestrations for multi-stage ML pipelines.
 * Defines comprehensive workflows for model development, research reproduction,
 * production deployment, debugging, and optimization.
 */

import type { AgentInputs } from '../agents/types.js';

// =============================================================================
// Core Types and Enums
// =============================================================================

/**
 * Workflow execution status.
 */
export enum WorkflowStatus {
  PENDING = 'pending',
  IN_PROGRESS = 'in_progress',
  COMPLETED = 'completed',
  FAILED = 'failed',
  BLOCKED = 'blocked',
  SKIPPED = 'skipped',
}

/**
 * Stage priority levels for workflow execution.
 */
export enum StagePriority {
  CRITICAL = 'critical', // Must complete successfully
  HIGH = 'high', // Important but can continue on failure
  MEDIUM = 'medium', // Standard priority
  LOW = 'low', // Optional, can be skipped
}

/**
 * Quality gate status.
 */
export enum QualityGateStatus {
  PASSED = 'passed',
  FAILED = 'failed',
  WARNING = 'warning',
  NOT_EVALUATED = 'not_evaluated',
}

/**
 * Deep learning task categories.
 */
export enum DLTaskCategory {
  DATA_PREPARATION = 'data_preparation',
  MODEL_ARCHITECTURE = 'model_architecture',
  TRAINING = 'training',
  EVALUATION = 'evaluation',
  OPTIMIZATION = 'optimization',
  DEPLOYMENT = 'deployment',
  DEBUGGING = 'debugging',
  RESEARCH = 'research',
}

// =============================================================================
// Workflow Stage Interfaces
// =============================================================================

/**
 * Defines a quality gate checkpoint for workflow stages.
 */
export interface QualityGate {
  /** Unique identifier for the gate */
  id: string;
  /** Human-readable name */
  name: string;
  /** Description of what this gate checks */
  description: string;
  /** Evaluation criteria */
  criteria: QualityCriteria[];
  /** Whether workflow can proceed on failure */
  blocking: boolean;
  /** Automated or manual evaluation */
  evaluationType: 'automated' | 'manual' | 'hybrid';
}

/**
 * Individual quality criteria within a gate.
 */
export interface QualityCriteria {
  /** Criteria identifier */
  id: string;
  /** What to check */
  check: string;
  /** Expected threshold or value */
  threshold?: number | string;
  /** Comparison operator */
  operator?: 'gt' | 'lt' | 'eq' | 'gte' | 'lte' | 'contains' | 'exists';
  /** Weight for scoring */
  weight: number;
}

/**
 * Input specification for a workflow stage.
 */
export interface StageInput {
  /** Input parameter name */
  name: string;
  /** Input type */
  type: 'string' | 'number' | 'boolean' | 'object' | 'array' | 'file' | 'model';
  /** Whether input is required */
  required: boolean;
  /** Description of the input */
  description: string;
  /** Source of the input (previous stage, user, config) */
  source?: string;
  /** Default value if not provided */
  defaultValue?: unknown;
}

/**
 * Output specification for a workflow stage.
 */
export interface StageOutput {
  /** Output parameter name */
  name: string;
  /** Output type */
  type: 'string' | 'number' | 'boolean' | 'object' | 'array' | 'file' | 'model' | 'metrics';
  /** Description of the output */
  description: string;
  /** Whether this output is persisted */
  persist: boolean;
  /** Validation schema for the output */
  validation?: Record<string, unknown>;
}

/**
 * Agent assignment for a workflow stage.
 */
export interface AgentAssignment {
  /** Primary agent for this stage */
  primaryAgent: string;
  /** Supporting agents that may be invoked */
  supportingAgents?: string[];
  /** Agent-specific configuration */
  agentConfig?: Record<string, unknown>;
  /** Maximum time allocated for agent execution */
  timeoutMinutes: number;
  /** Number of retry attempts on failure */
  maxRetries: number;
}

/**
 * Defines a single stage within a workflow.
 */
export interface WorkflowStage {
  /** Unique stage identifier */
  id: string;
  /** Human-readable stage name */
  name: string;
  /** Detailed description of stage objectives */
  description: string;
  /** Task category this stage belongs to */
  category: DLTaskCategory;
  /** Stage priority */
  priority: StagePriority;
  /** Input specifications */
  inputs: StageInput[];
  /** Output specifications */
  outputs: StageOutput[];
  /** Agent assignments */
  agents: AgentAssignment;
  /** Quality gates for this stage */
  qualityGates: QualityGate[];
  /** Dependencies on other stages (stage IDs) */
  dependencies: string[];
  /** Estimated duration in minutes */
  estimatedDuration: number;
  /** Whether stage can run in parallel with others */
  parallelizable: boolean;
  /** Stage-specific configuration */
  config?: Record<string, unknown>;
}

/**
 * Checkpoint for workflow state persistence.
 */
export interface WorkflowCheckpoint {
  /** Checkpoint identifier */
  id: string;
  /** When to create checkpoint */
  trigger: 'stage_complete' | 'quality_gate_passed' | 'time_interval' | 'manual';
  /** What to persist */
  persistence: {
    model: boolean;
    metrics: boolean;
    logs: boolean;
    artifacts: boolean;
  };
  /** Checkpoint description */
  description: string;
}

// =============================================================================
// Workflow Definition Interface
// =============================================================================

/**
 * Complete workflow definition.
 */
export interface DLWorkflow {
  /** Unique workflow identifier */
  id: string;
  /** Workflow name */
  name: string;
  /** Detailed description */
  description: string;
  /** Workflow version */
  version: string;
  /** Workflow category */
  category: DLTaskCategory;
  /** All stages in the workflow */
  stages: WorkflowStage[];
  /** Global workflow configuration */
  globalConfig: WorkflowGlobalConfig;
  /** Checkpoints for state persistence */
  checkpoints: WorkflowCheckpoint[];
  /** Metadata */
  metadata: WorkflowMetadata;
}

/**
 * Global configuration for a workflow.
 */
export interface WorkflowGlobalConfig {
  /** Maximum total execution time */
  maxExecutionTimeMinutes: number;
  /** Whether to continue on non-critical failures */
  continueOnFailure: boolean;
  /** Notification settings */
  notifications: {
    onStageComplete: boolean;
    onQualityGateFailed: boolean;
    onWorkflowComplete: boolean;
    channels: string[];
  };
  /** Resource constraints */
  resources: {
    maxGPUs: number;
    maxMemoryGB: number;
    maxCPUs: number;
  };
  /** Logging configuration */
  logging: {
    level: 'debug' | 'info' | 'warn' | 'error';
    persistLogs: boolean;
    logArtifacts: boolean;
  };
}

/**
 * Workflow metadata.
 */
export interface WorkflowMetadata {
  /** Workflow author */
  author: string;
  /** Creation date */
  createdAt: string;
  /** Last update date */
  updatedAt: string;
  /** Tags for categorization */
  tags: string[];
  /** Related workflows */
  relatedWorkflows: string[];
}

// =============================================================================
// Workflow Execution Interfaces
// =============================================================================

/**
 * Runtime state of a workflow execution.
 */
export interface WorkflowExecutionState {
  /** Execution ID */
  executionId: string;
  /** Workflow being executed */
  workflowId: string;
  /** Current status */
  status: WorkflowStatus;
  /** Current stage being executed */
  currentStageId: string | null;
  /** Completed stages */
  completedStages: string[];
  /** Failed stages */
  failedStages: string[];
  /** Stage outputs collected */
  stageOutputs: Record<string, Record<string, unknown>>;
  /** Quality gate results */
  qualityGateResults: Record<string, QualityGateResult>;
  /** Execution start time */
  startTime: Date;
  /** Execution end time */
  endTime: Date | null;
  /** Error information if failed */
  error?: WorkflowError;
}

/**
 * Result of quality gate evaluation.
 */
export interface QualityGateResult {
  /** Gate ID */
  gateId: string;
  /** Evaluation status */
  status: QualityGateStatus;
  /** Individual criteria results */
  criteriaResults: Array<{
    criteriaId: string;
    passed: boolean;
    actualValue: unknown;
    message: string;
  }>;
  /** Overall score (0-100) */
  score: number;
  /** Evaluation timestamp */
  evaluatedAt: Date;
}

/**
 * Workflow error information.
 */
export interface WorkflowError {
  /** Error code */
  code: string;
  /** Error message */
  message: string;
  /** Stage where error occurred */
  stageId: string;
  /** Stack trace */
  stack?: string;
  /** Recovery suggestions */
  recoverySuggestions: string[];
}

// =============================================================================
// Predefined Workflows
// =============================================================================

/**
 * Model Development Workflow
 *
 * A comprehensive workflow for developing deep learning models from scratch,
 * covering data preparation, architecture design, training, and evaluation.
 */
export const ModelDevelopmentWorkflow: DLWorkflow = {
  id: 'dl-model-development',
  name: 'Model Development Workflow',
  description: 'End-to-end workflow for developing deep learning models, from data preparation through evaluation and documentation.',
  version: '1.0.0',
  category: DLTaskCategory.MODEL_ARCHITECTURE,
  stages: [
    // Stage 1: Requirements Analysis
    {
      id: 'requirements-analysis',
      name: 'Requirements Analysis',
      description: 'Analyze project requirements, define success metrics, and establish constraints for the model development.',
      category: DLTaskCategory.RESEARCH,
      priority: StagePriority.CRITICAL,
      inputs: [
        {
          name: 'projectDescription',
          type: 'string',
          required: true,
          description: 'High-level description of the project and goals',
        },
        {
          name: 'constraints',
          type: 'object',
          required: false,
          description: 'Hardware, latency, and size constraints',
        },
      ],
      outputs: [
        {
          name: 'requirementsDocument',
          type: 'object',
          description: 'Structured requirements document',
          persist: true,
        },
        {
          name: 'successMetrics',
          type: 'array',
          description: 'List of success metrics with thresholds',
          persist: true,
        },
      ],
      agents: {
        primaryAgent: 'plan_agent',
        supportingAgents: ['explore_agent'],
        timeoutMinutes: 30,
        maxRetries: 2,
      },
      qualityGates: [
        {
          id: 'requirements-completeness',
          name: 'Requirements Completeness',
          description: 'Verify all necessary requirements are captured',
          criteria: [
            { id: 'has-metrics', check: 'Success metrics defined', weight: 1 },
            { id: 'has-constraints', check: 'Constraints documented', weight: 1 },
            { id: 'has-data-requirements', check: 'Data requirements specified', weight: 1 },
          ],
          blocking: true,
          evaluationType: 'hybrid',
        },
      ],
      dependencies: [],
      estimatedDuration: 30,
      parallelizable: false,
    },

    // Stage 2: Data Pipeline Setup
    {
      id: 'data-pipeline-setup',
      name: 'Data Pipeline Setup',
      description: 'Design and implement data loading, preprocessing, and augmentation pipelines.',
      category: DLTaskCategory.DATA_PREPARATION,
      priority: StagePriority.CRITICAL,
      inputs: [
        {
          name: 'dataSource',
          type: 'string',
          required: true,
          description: 'Path or identifier for the data source',
        },
        {
          name: 'requirementsDocument',
          type: 'object',
          required: true,
          description: 'Requirements from previous stage',
          source: 'requirements-analysis',
        },
      ],
      outputs: [
        {
          name: 'dataLoaderCode',
          type: 'file',
          description: 'Data loader implementation',
          persist: true,
        },
        {
          name: 'dataStats',
          type: 'metrics',
          description: 'Dataset statistics and distributions',
          persist: true,
        },
        {
          name: 'dataSplits',
          type: 'object',
          description: 'Train/val/test split configuration',
          persist: true,
        },
      ],
      agents: {
        primaryAgent: 'data_flow_agent',
        supportingAgents: ['code_reviewer'],
        timeoutMinutes: 60,
        maxRetries: 3,
      },
      qualityGates: [
        {
          id: 'data-quality',
          name: 'Data Quality Check',
          description: 'Verify data quality and pipeline correctness',
          criteria: [
            { id: 'no-data-leakage', check: 'No overlap between splits', weight: 2 },
            { id: 'balanced-splits', check: 'Class balance maintained', weight: 1 },
            { id: 'preprocessing-correct', check: 'Preprocessing invertible', weight: 1 },
          ],
          blocking: true,
          evaluationType: 'automated',
        },
      ],
      dependencies: ['requirements-analysis'],
      estimatedDuration: 60,
      parallelizable: false,
    },

    // Stage 3: Architecture Design
    {
      id: 'architecture-design',
      name: 'Architecture Design',
      description: 'Design model architecture based on requirements and data characteristics.',
      category: DLTaskCategory.MODEL_ARCHITECTURE,
      priority: StagePriority.CRITICAL,
      inputs: [
        {
          name: 'requirementsDocument',
          type: 'object',
          required: true,
          description: 'Requirements document',
          source: 'requirements-analysis',
        },
        {
          name: 'dataStats',
          type: 'metrics',
          required: true,
          description: 'Data statistics',
          source: 'data-pipeline-setup',
        },
      ],
      outputs: [
        {
          name: 'architectureSpec',
          type: 'object',
          description: 'Complete architecture specification',
          persist: true,
        },
        {
          name: 'modelCode',
          type: 'file',
          description: 'Model implementation code',
          persist: true,
        },
        {
          name: 'parameterCount',
          type: 'number',
          description: 'Total model parameters',
          persist: true,
        },
      ],
      agents: {
        primaryAgent: 'architecture_analyzer_agent',
        supportingAgents: ['layer_inspector_agent', 'code_reviewer'],
        timeoutMinutes: 90,
        maxRetries: 3,
      },
      qualityGates: [
        {
          id: 'architecture-review',
          name: 'Architecture Review',
          description: 'Review architecture design and implementation',
          criteria: [
            { id: 'param-budget', check: 'Within parameter budget', threshold: 1.0, operator: 'lte', weight: 2 },
            { id: 'code-quality', check: 'Code follows best practices', weight: 1 },
            { id: 'modularity', check: 'Architecture is modular', weight: 1 },
          ],
          blocking: true,
          evaluationType: 'hybrid',
        },
      ],
      dependencies: ['data-pipeline-setup'],
      estimatedDuration: 90,
      parallelizable: false,
    },

    // Stage 4: Training Configuration
    {
      id: 'training-configuration',
      name: 'Training Configuration',
      description: 'Configure optimizer, learning rate schedule, loss functions, and training hyperparameters.',
      category: DLTaskCategory.TRAINING,
      priority: StagePriority.HIGH,
      inputs: [
        {
          name: 'architectureSpec',
          type: 'object',
          required: true,
          description: 'Architecture specification',
          source: 'architecture-design',
        },
        {
          name: 'dataStats',
          type: 'metrics',
          required: true,
          description: 'Data statistics for batch size decisions',
          source: 'data-pipeline-setup',
        },
      ],
      outputs: [
        {
          name: 'trainingConfig',
          type: 'object',
          description: 'Complete training configuration',
          persist: true,
        },
        {
          name: 'trainingScript',
          type: 'file',
          description: 'Training script implementation',
          persist: true,
        },
      ],
      agents: {
        primaryAgent: 'experiment_agent',
        supportingAgents: ['debug_agent'],
        timeoutMinutes: 45,
        maxRetries: 2,
      },
      qualityGates: [
        {
          id: 'config-validation',
          name: 'Configuration Validation',
          description: 'Validate training configuration',
          criteria: [
            { id: 'lr-reasonable', check: 'Learning rate in valid range', weight: 1 },
            { id: 'batch-size-fits', check: 'Batch size fits in memory', weight: 2 },
            { id: 'loss-appropriate', check: 'Loss function matches task', weight: 1 },
          ],
          blocking: false,
          evaluationType: 'automated',
        },
      ],
      dependencies: ['architecture-design'],
      estimatedDuration: 45,
      parallelizable: true,
    },

    // Stage 5: Initial Training
    {
      id: 'initial-training',
      name: 'Initial Training',
      description: 'Run initial training to verify pipeline and establish baseline performance.',
      category: DLTaskCategory.TRAINING,
      priority: StagePriority.CRITICAL,
      inputs: [
        {
          name: 'modelCode',
          type: 'file',
          required: true,
          description: 'Model implementation',
          source: 'architecture-design',
        },
        {
          name: 'trainingConfig',
          type: 'object',
          required: true,
          description: 'Training configuration',
          source: 'training-configuration',
        },
        {
          name: 'dataLoaderCode',
          type: 'file',
          required: true,
          description: 'Data loader',
          source: 'data-pipeline-setup',
        },
      ],
      outputs: [
        {
          name: 'trainedModel',
          type: 'model',
          description: 'Trained model checkpoint',
          persist: true,
        },
        {
          name: 'trainingMetrics',
          type: 'metrics',
          description: 'Training metrics and curves',
          persist: true,
        },
        {
          name: 'trainingLogs',
          type: 'file',
          description: 'Detailed training logs',
          persist: true,
        },
      ],
      agents: {
        primaryAgent: 'debug_master_agent',
        supportingAgents: ['debug_agent'],
        timeoutMinutes: 180,
        maxRetries: 2,
      },
      qualityGates: [
        {
          id: 'training-success',
          name: 'Training Success',
          description: 'Verify training completed successfully',
          criteria: [
            { id: 'loss-decreasing', check: 'Loss decreased over training', weight: 2 },
            { id: 'no-nan', check: 'No NaN values in training', weight: 2 },
            { id: 'convergence', check: 'Model showing convergence', weight: 1 },
          ],
          blocking: true,
          evaluationType: 'automated',
        },
      ],
      dependencies: ['training-configuration'],
      estimatedDuration: 180,
      parallelizable: false,
    },

    // Stage 6: Evaluation
    {
      id: 'evaluation',
      name: 'Model Evaluation',
      description: 'Comprehensive evaluation on test set with multiple metrics.',
      category: DLTaskCategory.EVALUATION,
      priority: StagePriority.CRITICAL,
      inputs: [
        {
          name: 'trainedModel',
          type: 'model',
          required: true,
          description: 'Trained model',
          source: 'initial-training',
        },
        {
          name: 'successMetrics',
          type: 'array',
          required: true,
          description: 'Success metrics to evaluate',
          source: 'requirements-analysis',
        },
      ],
      outputs: [
        {
          name: 'evaluationReport',
          type: 'object',
          description: 'Complete evaluation report',
          persist: true,
        },
        {
          name: 'performanceMetrics',
          type: 'metrics',
          description: 'All performance metrics',
          persist: true,
        },
        {
          name: 'errorAnalysis',
          type: 'object',
          description: 'Detailed error analysis',
          persist: true,
        },
      ],
      agents: {
        primaryAgent: 'model_evaluation_agent',
        supportingAgents: ['summarizer_agent'],
        timeoutMinutes: 60,
        maxRetries: 2,
      },
      qualityGates: [
        {
          id: 'meets-requirements',
          name: 'Meets Requirements',
          description: 'Verify model meets success criteria',
          criteria: [
            { id: 'primary-metric', check: 'Primary metric meets threshold', weight: 3 },
            { id: 'secondary-metrics', check: 'Secondary metrics acceptable', weight: 1 },
            { id: 'no-regression', check: 'No regression from baseline', weight: 2 },
          ],
          blocking: true,
          evaluationType: 'automated',
        },
      ],
      dependencies: ['initial-training'],
      estimatedDuration: 60,
      parallelizable: false,
    },

    // Stage 7: Documentation
    {
      id: 'documentation',
      name: 'Documentation',
      description: 'Generate comprehensive documentation for the model.',
      category: DLTaskCategory.RESEARCH,
      priority: StagePriority.MEDIUM,
      inputs: [
        {
          name: 'architectureSpec',
          type: 'object',
          required: true,
          description: 'Architecture specification',
          source: 'architecture-design',
        },
        {
          name: 'evaluationReport',
          type: 'object',
          required: true,
          description: 'Evaluation results',
          source: 'evaluation',
        },
        {
          name: 'trainingConfig',
          type: 'object',
          required: true,
          description: 'Training configuration',
          source: 'training-configuration',
        },
      ],
      outputs: [
        {
          name: 'modelCard',
          type: 'file',
          description: 'Model card documentation',
          persist: true,
        },
        {
          name: 'technicalReport',
          type: 'file',
          description: 'Technical documentation',
          persist: true,
        },
      ],
      agents: {
        primaryAgent: 'summarizer_agent',
        supportingAgents: [],
        timeoutMinutes: 45,
        maxRetries: 2,
      },
      qualityGates: [
        {
          id: 'doc-completeness',
          name: 'Documentation Completeness',
          description: 'Verify documentation is complete',
          criteria: [
            { id: 'has-model-card', check: 'Model card generated', weight: 1 },
            { id: 'has-usage', check: 'Usage instructions included', weight: 1 },
            { id: 'has-limitations', check: 'Limitations documented', weight: 1 },
          ],
          blocking: false,
          evaluationType: 'hybrid',
        },
      ],
      dependencies: ['evaluation'],
      estimatedDuration: 45,
      parallelizable: true,
    },
  ],
  globalConfig: {
    maxExecutionTimeMinutes: 600,
    continueOnFailure: false,
    notifications: {
      onStageComplete: true,
      onQualityGateFailed: true,
      onWorkflowComplete: true,
      channels: ['slack', 'email'],
    },
    resources: {
      maxGPUs: 4,
      maxMemoryGB: 64,
      maxCPUs: 16,
    },
    logging: {
      level: 'info',
      persistLogs: true,
      logArtifacts: true,
    },
  },
  checkpoints: [
    {
      id: 'post-data-setup',
      trigger: 'stage_complete',
      persistence: { model: false, metrics: true, logs: true, artifacts: true },
      description: 'Checkpoint after data pipeline setup',
    },
    {
      id: 'post-training',
      trigger: 'stage_complete',
      persistence: { model: true, metrics: true, logs: true, artifacts: true },
      description: 'Checkpoint after initial training',
    },
    {
      id: 'final',
      trigger: 'stage_complete',
      persistence: { model: true, metrics: true, logs: true, artifacts: true },
      description: 'Final workflow checkpoint',
    },
  ],
  metadata: {
    author: 'Gemini CLI Team',
    createdAt: '2025-01-01',
    updatedAt: '2025-01-01',
    tags: ['deep-learning', 'model-development', 'end-to-end'],
    relatedWorkflows: ['dl-optimization', 'dl-deployment'],
  },
};

/**
 * Research Reproduction Workflow
 *
 * Workflow for reproducing results from deep learning research papers,
 * ensuring accurate implementation and fair comparison.
 */
export const ResearchReproductionWorkflow: DLWorkflow = {
  id: 'dl-research-reproduction',
  name: 'Research Reproduction Workflow',
  description: 'Systematic workflow for reproducing deep learning research papers with verification of results.',
  version: '1.0.0',
  category: DLTaskCategory.RESEARCH,
  stages: [
    // Stage 1: Paper Analysis
    {
      id: 'paper-analysis',
      name: 'Paper Analysis',
      description: 'Thorough analysis of the research paper to extract implementation details.',
      category: DLTaskCategory.RESEARCH,
      priority: StagePriority.CRITICAL,
      inputs: [
        {
          name: 'paperReference',
          type: 'string',
          required: true,
          description: 'Paper URL, arXiv ID, or PDF path',
        },
        {
          name: 'supplementaryMaterials',
          type: 'array',
          required: false,
          description: 'Additional materials (code, appendix)',
        },
      ],
      outputs: [
        {
          name: 'paperSummary',
          type: 'object',
          description: 'Structured summary of paper',
          persist: true,
        },
        {
          name: 'architectureExtraction',
          type: 'object',
          description: 'Extracted architecture details',
          persist: true,
        },
        {
          name: 'hyperparameters',
          type: 'object',
          description: 'All hyperparameters from paper',
          persist: true,
        },
        {
          name: 'ambiguities',
          type: 'array',
          description: 'Identified ambiguities requiring resolution',
          persist: true,
        },
      ],
      agents: {
        primaryAgent: 'literature_analyzer_agent',
        supportingAgents: ['explore_agent'],
        timeoutMinutes: 60,
        maxRetries: 2,
      },
      qualityGates: [
        {
          id: 'extraction-completeness',
          name: 'Extraction Completeness',
          description: 'Verify all necessary details extracted',
          criteria: [
            { id: 'architecture-complete', check: 'Architecture fully specified', weight: 2 },
            { id: 'hyperparams-complete', check: 'All hyperparameters found', weight: 2 },
            { id: 'training-details', check: 'Training procedure documented', weight: 1 },
          ],
          blocking: true,
          evaluationType: 'hybrid',
        },
      ],
      dependencies: [],
      estimatedDuration: 60,
      parallelizable: false,
    },

    // Stage 2: Ambiguity Resolution
    {
      id: 'ambiguity-resolution',
      name: 'Ambiguity Resolution',
      description: 'Resolve ambiguities through official code analysis or author contact.',
      category: DLTaskCategory.RESEARCH,
      priority: StagePriority.HIGH,
      inputs: [
        {
          name: 'ambiguities',
          type: 'array',
          required: true,
          description: 'List of ambiguities',
          source: 'paper-analysis',
        },
        {
          name: 'officialCode',
          type: 'string',
          required: false,
          description: 'Official code repository URL',
        },
      ],
      outputs: [
        {
          name: 'resolvedAmbiguities',
          type: 'object',
          description: 'Resolution for each ambiguity',
          persist: true,
        },
        {
          name: 'codeInsights',
          type: 'object',
          description: 'Insights from official code',
          persist: true,
        },
      ],
      agents: {
        primaryAgent: 'codebase_investigator',
        supportingAgents: ['explore_agent'],
        timeoutMinutes: 45,
        maxRetries: 2,
      },
      qualityGates: [
        {
          id: 'ambiguities-resolved',
          name: 'Ambiguities Resolved',
          description: 'All critical ambiguities addressed',
          criteria: [
            { id: 'critical-resolved', check: 'Critical ambiguities resolved', weight: 2 },
            { id: 'documented', check: 'Resolutions documented', weight: 1 },
          ],
          blocking: false,
          evaluationType: 'hybrid',
        },
      ],
      dependencies: ['paper-analysis'],
      estimatedDuration: 45,
      parallelizable: false,
    },

    // Stage 3: Implementation
    {
      id: 'implementation',
      name: 'Model Implementation',
      description: 'Implement the model architecture and training pipeline.',
      category: DLTaskCategory.MODEL_ARCHITECTURE,
      priority: StagePriority.CRITICAL,
      inputs: [
        {
          name: 'architectureExtraction',
          type: 'object',
          required: true,
          description: 'Architecture details',
          source: 'paper-analysis',
        },
        {
          name: 'hyperparameters',
          type: 'object',
          required: true,
          description: 'Hyperparameters',
          source: 'paper-analysis',
        },
        {
          name: 'resolvedAmbiguities',
          type: 'object',
          required: true,
          description: 'Resolved ambiguities',
          source: 'ambiguity-resolution',
        },
      ],
      outputs: [
        {
          name: 'modelImplementation',
          type: 'file',
          description: 'Complete model code',
          persist: true,
        },
        {
          name: 'trainingPipeline',
          type: 'file',
          description: 'Training pipeline code',
          persist: true,
        },
        {
          name: 'implementationNotes',
          type: 'object',
          description: 'Implementation decisions and notes',
          persist: true,
        },
      ],
      agents: {
        primaryAgent: 'architecture_analyzer_agent',
        supportingAgents: ['code_reviewer', 'layer_inspector_agent'],
        timeoutMinutes: 120,
        maxRetries: 3,
      },
      qualityGates: [
        {
          id: 'implementation-fidelity',
          name: 'Implementation Fidelity',
          description: 'Verify implementation matches paper',
          criteria: [
            { id: 'param-count-match', check: 'Parameter count matches paper', weight: 2 },
            { id: 'architecture-match', check: 'Architecture matches spec', weight: 2 },
            { id: 'code-quality', check: 'Code quality acceptable', weight: 1 },
          ],
          blocking: true,
          evaluationType: 'hybrid',
        },
      ],
      dependencies: ['ambiguity-resolution'],
      estimatedDuration: 120,
      parallelizable: false,
    },

    // Stage 4: Sanity Checks
    {
      id: 'sanity-checks',
      name: 'Sanity Checks',
      description: 'Run sanity checks to verify implementation correctness.',
      category: DLTaskCategory.DEBUGGING,
      priority: StagePriority.CRITICAL,
      inputs: [
        {
          name: 'modelImplementation',
          type: 'file',
          required: true,
          description: 'Model code',
          source: 'implementation',
        },
      ],
      outputs: [
        {
          name: 'sanityCheckResults',
          type: 'object',
          description: 'Results of all sanity checks',
          persist: true,
        },
        {
          name: 'debuggingNotes',
          type: 'array',
          description: 'Issues found and fixed',
          persist: true,
        },
      ],
      agents: {
        primaryAgent: 'debug_master_agent',
        supportingAgents: ['debug_agent'],
        timeoutMinutes: 60,
        maxRetries: 3,
      },
      qualityGates: [
        {
          id: 'sanity-passed',
          name: 'Sanity Checks Passed',
          description: 'All sanity checks pass',
          criteria: [
            { id: 'overfit-single', check: 'Can overfit single batch', weight: 2 },
            { id: 'gradients-flow', check: 'Gradients flow correctly', weight: 2 },
            { id: 'shapes-correct', check: 'All tensor shapes correct', weight: 1 },
          ],
          blocking: true,
          evaluationType: 'automated',
        },
      ],
      dependencies: ['implementation'],
      estimatedDuration: 60,
      parallelizable: false,
    },

    // Stage 5: Training Run
    {
      id: 'training-run',
      name: 'Full Training Run',
      description: 'Run full training with paper hyperparameters.',
      category: DLTaskCategory.TRAINING,
      priority: StagePriority.CRITICAL,
      inputs: [
        {
          name: 'modelImplementation',
          type: 'file',
          required: true,
          description: 'Model code',
          source: 'implementation',
        },
        {
          name: 'trainingPipeline',
          type: 'file',
          required: true,
          description: 'Training pipeline',
          source: 'implementation',
        },
        {
          name: 'hyperparameters',
          type: 'object',
          required: true,
          description: 'Training hyperparameters',
          source: 'paper-analysis',
        },
      ],
      outputs: [
        {
          name: 'trainedModel',
          type: 'model',
          description: 'Trained model checkpoint',
          persist: true,
        },
        {
          name: 'trainingCurves',
          type: 'metrics',
          description: 'Loss and metric curves',
          persist: true,
        },
      ],
      agents: {
        primaryAgent: 'experiment_agent',
        supportingAgents: ['debug_master_agent'],
        timeoutMinutes: 480,
        maxRetries: 2,
      },
      qualityGates: [
        {
          id: 'training-complete',
          name: 'Training Complete',
          description: 'Training completed successfully',
          criteria: [
            { id: 'convergence', check: 'Model converged', weight: 2 },
            { id: 'no-issues', check: 'No training issues', weight: 1 },
          ],
          blocking: true,
          evaluationType: 'automated',
        },
      ],
      dependencies: ['sanity-checks'],
      estimatedDuration: 480,
      parallelizable: false,
    },

    // Stage 6: Results Comparison
    {
      id: 'results-comparison',
      name: 'Results Comparison',
      description: 'Compare reproduction results with paper reported results.',
      category: DLTaskCategory.EVALUATION,
      priority: StagePriority.CRITICAL,
      inputs: [
        {
          name: 'trainedModel',
          type: 'model',
          required: true,
          description: 'Trained model',
          source: 'training-run',
        },
        {
          name: 'paperSummary',
          type: 'object',
          required: true,
          description: 'Paper results for comparison',
          source: 'paper-analysis',
        },
      ],
      outputs: [
        {
          name: 'comparisonReport',
          type: 'object',
          description: 'Detailed comparison with paper',
          persist: true,
        },
        {
          name: 'reproductionStatus',
          type: 'string',
          description: 'Overall reproduction status',
          persist: true,
        },
        {
          name: 'discrepancyAnalysis',
          type: 'object',
          description: 'Analysis of any discrepancies',
          persist: true,
        },
      ],
      agents: {
        primaryAgent: 'model_evaluation_agent',
        supportingAgents: ['summarizer_agent'],
        timeoutMinutes: 60,
        maxRetries: 2,
      },
      qualityGates: [
        {
          id: 'reproduction-success',
          name: 'Reproduction Success',
          description: 'Results match paper within tolerance',
          criteria: [
            { id: 'primary-match', check: 'Primary metric within 2% of paper', weight: 3 },
            { id: 'secondary-match', check: 'Secondary metrics reasonable', weight: 1 },
          ],
          blocking: false,
          evaluationType: 'automated',
        },
      ],
      dependencies: ['training-run'],
      estimatedDuration: 60,
      parallelizable: false,
    },
  ],
  globalConfig: {
    maxExecutionTimeMinutes: 960,
    continueOnFailure: false,
    notifications: {
      onStageComplete: true,
      onQualityGateFailed: true,
      onWorkflowComplete: true,
      channels: ['slack', 'email'],
    },
    resources: {
      maxGPUs: 8,
      maxMemoryGB: 128,
      maxCPUs: 32,
    },
    logging: {
      level: 'info',
      persistLogs: true,
      logArtifacts: true,
    },
  },
  checkpoints: [
    {
      id: 'post-analysis',
      trigger: 'stage_complete',
      persistence: { model: false, metrics: false, logs: true, artifacts: true },
      description: 'Checkpoint after paper analysis',
    },
    {
      id: 'post-implementation',
      trigger: 'stage_complete',
      persistence: { model: false, metrics: false, logs: true, artifacts: true },
      description: 'Checkpoint after implementation',
    },
    {
      id: 'post-training',
      trigger: 'stage_complete',
      persistence: { model: true, metrics: true, logs: true, artifacts: true },
      description: 'Checkpoint after training',
    },
  ],
  metadata: {
    author: 'Gemini CLI Team',
    createdAt: '2025-01-01',
    updatedAt: '2025-01-01',
    tags: ['deep-learning', 'research', 'reproduction', 'paper-implementation'],
    relatedWorkflows: ['dl-model-development'],
  },
};

/**
 * Production Deployment Workflow
 *
 * Workflow for deploying deep learning models to production,
 * including optimization, testing, and monitoring setup.
 */
export const ProductionDeploymentWorkflow: DLWorkflow = {
  id: 'dl-production-deployment',
  name: 'Production Deployment Workflow',
  description: 'End-to-end workflow for deploying deep learning models to production environments.',
  version: '1.0.0',
  category: DLTaskCategory.DEPLOYMENT,
  stages: [
    // Stage 1: Deployment Requirements
    {
      id: 'deployment-requirements',
      name: 'Deployment Requirements',
      description: 'Gather deployment requirements including latency, throughput, and infrastructure constraints.',
      category: DLTaskCategory.DEPLOYMENT,
      priority: StagePriority.CRITICAL,
      inputs: [
        {
          name: 'trainedModel',
          type: 'model',
          required: true,
          description: 'Model to deploy',
        },
        {
          name: 'targetEnvironment',
          type: 'string',
          required: true,
          description: 'Target deployment environment (cloud, edge, mobile)',
        },
        {
          name: 'performanceRequirements',
          type: 'object',
          required: true,
          description: 'Latency, throughput, and accuracy requirements',
        },
      ],
      outputs: [
        {
          name: 'deploymentPlan',
          type: 'object',
          description: 'Comprehensive deployment plan',
          persist: true,
        },
        {
          name: 'optimizationTargets',
          type: 'array',
          description: 'Required optimizations',
          persist: true,
        },
      ],
      agents: {
        primaryAgent: 'plan_agent',
        supportingAgents: ['explore_agent'],
        timeoutMinutes: 30,
        maxRetries: 2,
      },
      qualityGates: [
        {
          id: 'requirements-feasible',
          name: 'Requirements Feasibility',
          description: 'Verify requirements are achievable',
          criteria: [
            { id: 'latency-achievable', check: 'Latency target is achievable', weight: 2 },
            { id: 'resource-available', check: 'Required resources available', weight: 1 },
          ],
          blocking: true,
          evaluationType: 'hybrid',
        },
      ],
      dependencies: [],
      estimatedDuration: 30,
      parallelizable: false,
    },

    // Stage 2: Model Optimization
    {
      id: 'model-optimization',
      name: 'Model Optimization',
      description: 'Apply optimizations: quantization, pruning, distillation.',
      category: DLTaskCategory.OPTIMIZATION,
      priority: StagePriority.CRITICAL,
      inputs: [
        {
          name: 'trainedModel',
          type: 'model',
          required: true,
          description: 'Model to optimize',
        },
        {
          name: 'optimizationTargets',
          type: 'array',
          required: true,
          description: 'Required optimizations',
          source: 'deployment-requirements',
        },
      ],
      outputs: [
        {
          name: 'optimizedModel',
          type: 'model',
          description: 'Optimized model',
          persist: true,
        },
        {
          name: 'optimizationReport',
          type: 'object',
          description: 'Optimization results and tradeoffs',
          persist: true,
        },
        {
          name: 'performanceGains',
          type: 'metrics',
          description: 'Performance improvements',
          persist: true,
        },
      ],
      agents: {
        primaryAgent: 'architecture_analyzer_agent',
        supportingAgents: ['model_evaluation_agent'],
        timeoutMinutes: 120,
        maxRetries: 3,
      },
      qualityGates: [
        {
          id: 'optimization-success',
          name: 'Optimization Success',
          description: 'Optimizations meet requirements',
          criteria: [
            { id: 'latency-met', check: 'Latency requirement met', weight: 2 },
            { id: 'accuracy-preserved', check: 'Accuracy degradation < 2%', threshold: 2, operator: 'lt', weight: 2 },
            { id: 'size-reduced', check: 'Model size reduced', weight: 1 },
          ],
          blocking: true,
          evaluationType: 'automated',
        },
      ],
      dependencies: ['deployment-requirements'],
      estimatedDuration: 120,
      parallelizable: false,
    },

    // Stage 3: Export and Conversion
    {
      id: 'export-conversion',
      name: 'Export and Conversion',
      description: 'Export model to deployment format (ONNX, TensorRT, TFLite).',
      category: DLTaskCategory.DEPLOYMENT,
      priority: StagePriority.CRITICAL,
      inputs: [
        {
          name: 'optimizedModel',
          type: 'model',
          required: true,
          description: 'Optimized model',
          source: 'model-optimization',
        },
        {
          name: 'targetEnvironment',
          type: 'string',
          required: true,
          description: 'Target environment',
        },
      ],
      outputs: [
        {
          name: 'exportedModel',
          type: 'file',
          description: 'Exported model artifact',
          persist: true,
        },
        {
          name: 'conversionReport',
          type: 'object',
          description: 'Conversion details and validation',
          persist: true,
        },
      ],
      agents: {
        primaryAgent: 'architecture_analyzer_agent',
        supportingAgents: ['debug_agent'],
        timeoutMinutes: 60,
        maxRetries: 3,
      },
      qualityGates: [
        {
          id: 'export-valid',
          name: 'Export Validation',
          description: 'Exported model is valid',
          criteria: [
            { id: 'format-correct', check: 'Export format correct', weight: 1 },
            { id: 'outputs-match', check: 'Outputs match original', weight: 2 },
            { id: 'loadable', check: 'Model loads in target runtime', weight: 2 },
          ],
          blocking: true,
          evaluationType: 'automated',
        },
      ],
      dependencies: ['model-optimization'],
      estimatedDuration: 60,
      parallelizable: false,
    },

    // Stage 4: Inference Pipeline
    {
      id: 'inference-pipeline',
      name: 'Inference Pipeline',
      description: 'Build production inference pipeline with preprocessing and postprocessing.',
      category: DLTaskCategory.DEPLOYMENT,
      priority: StagePriority.HIGH,
      inputs: [
        {
          name: 'exportedModel',
          type: 'file',
          required: true,
          description: 'Exported model',
          source: 'export-conversion',
        },
        {
          name: 'deploymentPlan',
          type: 'object',
          required: true,
          description: 'Deployment plan',
          source: 'deployment-requirements',
        },
      ],
      outputs: [
        {
          name: 'inferencePipeline',
          type: 'file',
          description: 'Complete inference pipeline code',
          persist: true,
        },
        {
          name: 'apiSpec',
          type: 'object',
          description: 'API specification',
          persist: true,
        },
      ],
      agents: {
        primaryAgent: 'code_reviewer',
        supportingAgents: ['refactor_agent'],
        timeoutMinutes: 90,
        maxRetries: 2,
      },
      qualityGates: [
        {
          id: 'pipeline-quality',
          name: 'Pipeline Quality',
          description: 'Inference pipeline meets standards',
          criteria: [
            { id: 'error-handling', check: 'Proper error handling', weight: 1 },
            { id: 'input-validation', check: 'Input validation implemented', weight: 1 },
            { id: 'batching-support', check: 'Batching supported', weight: 1 },
          ],
          blocking: false,
          evaluationType: 'hybrid',
        },
      ],
      dependencies: ['export-conversion'],
      estimatedDuration: 90,
      parallelizable: false,
    },

    // Stage 5: Load Testing
    {
      id: 'load-testing',
      name: 'Load Testing',
      description: 'Perform load testing to verify performance under production conditions.',
      category: DLTaskCategory.EVALUATION,
      priority: StagePriority.HIGH,
      inputs: [
        {
          name: 'inferencePipeline',
          type: 'file',
          required: true,
          description: 'Inference pipeline',
          source: 'inference-pipeline',
        },
        {
          name: 'performanceRequirements',
          type: 'object',
          required: true,
          description: 'Performance requirements',
        },
      ],
      outputs: [
        {
          name: 'loadTestResults',
          type: 'metrics',
          description: 'Load test metrics',
          persist: true,
        },
        {
          name: 'scalingRecommendations',
          type: 'object',
          description: 'Scaling recommendations',
          persist: true,
        },
      ],
      agents: {
        primaryAgent: 'model_evaluation_agent',
        supportingAgents: ['debug_agent'],
        timeoutMinutes: 60,
        maxRetries: 2,
      },
      qualityGates: [
        {
          id: 'load-test-pass',
          name: 'Load Test Pass',
          description: 'Load test meets requirements',
          criteria: [
            { id: 'p99-latency', check: 'P99 latency meets target', weight: 2 },
            { id: 'throughput', check: 'Throughput meets target', weight: 2 },
            { id: 'error-rate', check: 'Error rate < 0.1%', threshold: 0.1, operator: 'lt', weight: 1 },
          ],
          blocking: true,
          evaluationType: 'automated',
        },
      ],
      dependencies: ['inference-pipeline'],
      estimatedDuration: 60,
      parallelizable: false,
    },

    // Stage 6: Monitoring Setup
    {
      id: 'monitoring-setup',
      name: 'Monitoring Setup',
      description: 'Set up production monitoring, alerting, and logging.',
      category: DLTaskCategory.DEPLOYMENT,
      priority: StagePriority.HIGH,
      inputs: [
        {
          name: 'inferencePipeline',
          type: 'file',
          required: true,
          description: 'Inference pipeline',
          source: 'inference-pipeline',
        },
        {
          name: 'deploymentPlan',
          type: 'object',
          required: true,
          description: 'Deployment plan',
          source: 'deployment-requirements',
        },
      ],
      outputs: [
        {
          name: 'monitoringConfig',
          type: 'file',
          description: 'Monitoring configuration',
          persist: true,
        },
        {
          name: 'alertingRules',
          type: 'object',
          description: 'Alerting rules',
          persist: true,
        },
        {
          name: 'dashboards',
          type: 'array',
          description: 'Monitoring dashboards',
          persist: true,
        },
      ],
      agents: {
        primaryAgent: 'plan_agent',
        supportingAgents: [],
        timeoutMinutes: 45,
        maxRetries: 2,
      },
      qualityGates: [
        {
          id: 'monitoring-complete',
          name: 'Monitoring Complete',
          description: 'Monitoring setup is complete',
          criteria: [
            { id: 'metrics-tracked', check: 'Key metrics tracked', weight: 1 },
            { id: 'alerts-configured', check: 'Alerts configured', weight: 1 },
            { id: 'dashboards-created', check: 'Dashboards created', weight: 1 },
          ],
          blocking: false,
          evaluationType: 'hybrid',
        },
      ],
      dependencies: ['load-testing'],
      estimatedDuration: 45,
      parallelizable: true,
    },

    // Stage 7: Deployment
    {
      id: 'deployment',
      name: 'Production Deployment',
      description: 'Deploy model to production environment.',
      category: DLTaskCategory.DEPLOYMENT,
      priority: StagePriority.CRITICAL,
      inputs: [
        {
          name: 'inferencePipeline',
          type: 'file',
          required: true,
          description: 'Inference pipeline',
          source: 'inference-pipeline',
        },
        {
          name: 'monitoringConfig',
          type: 'file',
          required: true,
          description: 'Monitoring configuration',
          source: 'monitoring-setup',
        },
      ],
      outputs: [
        {
          name: 'deploymentStatus',
          type: 'object',
          description: 'Deployment status and endpoints',
          persist: true,
        },
        {
          name: 'rollbackPlan',
          type: 'object',
          description: 'Rollback procedure',
          persist: true,
        },
      ],
      agents: {
        primaryAgent: 'plan_agent',
        supportingAgents: ['debug_agent'],
        timeoutMinutes: 30,
        maxRetries: 2,
      },
      qualityGates: [
        {
          id: 'deployment-success',
          name: 'Deployment Success',
          description: 'Deployment completed successfully',
          criteria: [
            { id: 'endpoints-live', check: 'Endpoints are live', weight: 2 },
            { id: 'health-checks', check: 'Health checks passing', weight: 2 },
            { id: 'canary-ok', check: 'Canary traffic successful', weight: 1 },
          ],
          blocking: true,
          evaluationType: 'automated',
        },
      ],
      dependencies: ['monitoring-setup'],
      estimatedDuration: 30,
      parallelizable: false,
    },
  ],
  globalConfig: {
    maxExecutionTimeMinutes: 540,
    continueOnFailure: false,
    notifications: {
      onStageComplete: true,
      onQualityGateFailed: true,
      onWorkflowComplete: true,
      channels: ['slack', 'pagerduty', 'email'],
    },
    resources: {
      maxGPUs: 2,
      maxMemoryGB: 32,
      maxCPUs: 8,
    },
    logging: {
      level: 'info',
      persistLogs: true,
      logArtifacts: true,
    },
  },
  checkpoints: [
    {
      id: 'post-optimization',
      trigger: 'stage_complete',
      persistence: { model: true, metrics: true, logs: true, artifacts: true },
      description: 'Checkpoint after model optimization',
    },
    {
      id: 'pre-deployment',
      trigger: 'quality_gate_passed',
      persistence: { model: true, metrics: true, logs: true, artifacts: true },
      description: 'Checkpoint before deployment',
    },
  ],
  metadata: {
    author: 'Gemini CLI Team',
    createdAt: '2025-01-01',
    updatedAt: '2025-01-01',
    tags: ['deep-learning', 'deployment', 'production', 'mlops'],
    relatedWorkflows: ['dl-optimization', 'dl-model-development'],
  },
};

/**
 * Debugging Workflow
 *
 * Systematic workflow for debugging deep learning training and inference issues.
 */
export const DebuggingWorkflow: DLWorkflow = {
  id: 'dl-debugging',
  name: 'Debugging Workflow',
  description: 'Systematic workflow for diagnosing and fixing deep learning issues.',
  version: '1.0.0',
  category: DLTaskCategory.DEBUGGING,
  stages: [
    // Stage 1: Issue Triage
    {
      id: 'issue-triage',
      name: 'Issue Triage',
      description: 'Classify and prioritize the reported issue.',
      category: DLTaskCategory.DEBUGGING,
      priority: StagePriority.CRITICAL,
      inputs: [
        {
          name: 'issueDescription',
          type: 'string',
          required: true,
          description: 'Description of the issue',
        },
        {
          name: 'errorLogs',
          type: 'string',
          required: false,
          description: 'Error logs if available',
        },
        {
          name: 'trainingCurves',
          type: 'file',
          required: false,
          description: 'Training curves if relevant',
        },
      ],
      outputs: [
        {
          name: 'issueClassification',
          type: 'object',
          description: 'Issue type and severity',
          persist: true,
        },
        {
          name: 'investigationPlan',
          type: 'array',
          description: 'Ordered list of investigations',
          persist: true,
        },
      ],
      agents: {
        primaryAgent: 'debug_master_agent',
        supportingAgents: ['explore_agent'],
        timeoutMinutes: 20,
        maxRetries: 1,
      },
      qualityGates: [
        {
          id: 'issue-classified',
          name: 'Issue Classified',
          description: 'Issue has been classified',
          criteria: [
            { id: 'type-identified', check: 'Issue type identified', weight: 1 },
            { id: 'plan-created', check: 'Investigation plan created', weight: 1 },
          ],
          blocking: true,
          evaluationType: 'automated',
        },
      ],
      dependencies: [],
      estimatedDuration: 20,
      parallelizable: false,
    },

    // Stage 2: Data Pipeline Investigation
    {
      id: 'data-investigation',
      name: 'Data Pipeline Investigation',
      description: 'Investigate data loading and preprocessing for issues.',
      category: DLTaskCategory.DEBUGGING,
      priority: StagePriority.HIGH,
      inputs: [
        {
          name: 'investigationPlan',
          type: 'array',
          required: true,
          description: 'Investigation plan',
          source: 'issue-triage',
        },
        {
          name: 'dataLoaderCode',
          type: 'file',
          required: false,
          description: 'Data loader implementation',
        },
      ],
      outputs: [
        {
          name: 'dataIssues',
          type: 'array',
          description: 'Identified data issues',
          persist: true,
        },
        {
          name: 'dataStats',
          type: 'metrics',
          description: 'Data statistics',
          persist: true,
        },
      ],
      agents: {
        primaryAgent: 'data_flow_agent',
        supportingAgents: ['debug_agent'],
        timeoutMinutes: 30,
        maxRetries: 2,
      },
      qualityGates: [
        {
          id: 'data-checked',
          name: 'Data Checked',
          description: 'Data pipeline has been checked',
          criteria: [
            { id: 'loading-verified', check: 'Data loading verified', weight: 1 },
            { id: 'preprocessing-checked', check: 'Preprocessing checked', weight: 1 },
          ],
          blocking: false,
          evaluationType: 'automated',
        },
      ],
      dependencies: ['issue-triage'],
      estimatedDuration: 30,
      parallelizable: true,
    },

    // Stage 3: Gradient Analysis
    {
      id: 'gradient-analysis',
      name: 'Gradient Analysis',
      description: 'Analyze gradients for vanishing, exploding, or dead neuron issues.',
      category: DLTaskCategory.DEBUGGING,
      priority: StagePriority.HIGH,
      inputs: [
        {
          name: 'investigationPlan',
          type: 'array',
          required: true,
          description: 'Investigation plan',
          source: 'issue-triage',
        },
        {
          name: 'modelCode',
          type: 'file',
          required: true,
          description: 'Model implementation',
        },
      ],
      outputs: [
        {
          name: 'gradientReport',
          type: 'object',
          description: 'Gradient analysis report',
          persist: true,
        },
        {
          name: 'gradientIssues',
          type: 'array',
          description: 'Identified gradient issues',
          persist: true,
        },
      ],
      agents: {
        primaryAgent: 'debug_agent',
        supportingAgents: ['layer_inspector_agent'],
        timeoutMinutes: 45,
        maxRetries: 2,
      },
      qualityGates: [
        {
          id: 'gradients-analyzed',
          name: 'Gradients Analyzed',
          description: 'Gradient analysis complete',
          criteria: [
            { id: 'flow-checked', check: 'Gradient flow checked', weight: 1 },
            { id: 'magnitudes-checked', check: 'Gradient magnitudes checked', weight: 1 },
          ],
          blocking: false,
          evaluationType: 'automated',
        },
      ],
      dependencies: ['issue-triage'],
      estimatedDuration: 45,
      parallelizable: true,
    },

    // Stage 4: Memory Analysis
    {
      id: 'memory-analysis',
      name: 'Memory Analysis',
      description: 'Check for memory leaks and inefficient memory usage.',
      category: DLTaskCategory.DEBUGGING,
      priority: StagePriority.MEDIUM,
      inputs: [
        {
          name: 'investigationPlan',
          type: 'array',
          required: true,
          description: 'Investigation plan',
          source: 'issue-triage',
        },
        {
          name: 'trainingCode',
          type: 'file',
          required: true,
          description: 'Training code',
        },
      ],
      outputs: [
        {
          name: 'memoryReport',
          type: 'object',
          description: 'Memory analysis report',
          persist: true,
        },
        {
          name: 'memoryIssues',
          type: 'array',
          description: 'Identified memory issues',
          persist: true,
        },
      ],
      agents: {
        primaryAgent: 'debug_agent',
        supportingAgents: [],
        timeoutMinutes: 30,
        maxRetries: 2,
      },
      qualityGates: [
        {
          id: 'memory-analyzed',
          name: 'Memory Analyzed',
          description: 'Memory analysis complete',
          criteria: [
            { id: 'leaks-checked', check: 'Memory leaks checked', weight: 1 },
            { id: 'usage-profiled', check: 'Memory usage profiled', weight: 1 },
          ],
          blocking: false,
          evaluationType: 'automated',
        },
      ],
      dependencies: ['issue-triage'],
      estimatedDuration: 30,
      parallelizable: true,
    },

    // Stage 5: Root Cause Analysis
    {
      id: 'root-cause-analysis',
      name: 'Root Cause Analysis',
      description: 'Synthesize findings to identify root cause.',
      category: DLTaskCategory.DEBUGGING,
      priority: StagePriority.CRITICAL,
      inputs: [
        {
          name: 'dataIssues',
          type: 'array',
          required: false,
          description: 'Data issues',
          source: 'data-investigation',
        },
        {
          name: 'gradientIssues',
          type: 'array',
          required: false,
          description: 'Gradient issues',
          source: 'gradient-analysis',
        },
        {
          name: 'memoryIssues',
          type: 'array',
          required: false,
          description: 'Memory issues',
          source: 'memory-analysis',
        },
      ],
      outputs: [
        {
          name: 'rootCause',
          type: 'object',
          description: 'Identified root cause',
          persist: true,
        },
        {
          name: 'fixRecommendations',
          type: 'array',
          description: 'Recommended fixes',
          persist: true,
        },
      ],
      agents: {
        primaryAgent: 'debug_master_agent',
        supportingAgents: ['summarizer_agent'],
        timeoutMinutes: 30,
        maxRetries: 2,
      },
      qualityGates: [
        {
          id: 'root-cause-found',
          name: 'Root Cause Found',
          description: 'Root cause has been identified',
          criteria: [
            { id: 'cause-identified', check: 'Root cause identified', weight: 2 },
            { id: 'fixes-proposed', check: 'Fixes proposed', weight: 1 },
          ],
          blocking: true,
          evaluationType: 'hybrid',
        },
      ],
      dependencies: ['data-investigation', 'gradient-analysis', 'memory-analysis'],
      estimatedDuration: 30,
      parallelizable: false,
    },

    // Stage 6: Fix Implementation
    {
      id: 'fix-implementation',
      name: 'Fix Implementation',
      description: 'Implement and test the recommended fixes.',
      category: DLTaskCategory.DEBUGGING,
      priority: StagePriority.CRITICAL,
      inputs: [
        {
          name: 'fixRecommendations',
          type: 'array',
          required: true,
          description: 'Recommended fixes',
          source: 'root-cause-analysis',
        },
        {
          name: 'rootCause',
          type: 'object',
          required: true,
          description: 'Root cause',
          source: 'root-cause-analysis',
        },
      ],
      outputs: [
        {
          name: 'implementedFixes',
          type: 'array',
          description: 'Implemented fixes',
          persist: true,
        },
        {
          name: 'verificationResults',
          type: 'object',
          description: 'Fix verification results',
          persist: true,
        },
      ],
      agents: {
        primaryAgent: 'refactor_agent',
        supportingAgents: ['debug_agent', 'test_runner'],
        timeoutMinutes: 60,
        maxRetries: 3,
      },
      qualityGates: [
        {
          id: 'fix-verified',
          name: 'Fix Verified',
          description: 'Fix has been verified',
          criteria: [
            { id: 'issue-resolved', check: 'Original issue resolved', weight: 2 },
            { id: 'no-regression', check: 'No regression introduced', weight: 2 },
            { id: 'tests-pass', check: 'Tests pass', weight: 1 },
          ],
          blocking: true,
          evaluationType: 'automated',
        },
      ],
      dependencies: ['root-cause-analysis'],
      estimatedDuration: 60,
      parallelizable: false,
    },
  ],
  globalConfig: {
    maxExecutionTimeMinutes: 300,
    continueOnFailure: true,
    notifications: {
      onStageComplete: false,
      onQualityGateFailed: true,
      onWorkflowComplete: true,
      channels: ['slack'],
    },
    resources: {
      maxGPUs: 2,
      maxMemoryGB: 32,
      maxCPUs: 8,
    },
    logging: {
      level: 'debug',
      persistLogs: true,
      logArtifacts: true,
    },
  },
  checkpoints: [
    {
      id: 'post-triage',
      trigger: 'stage_complete',
      persistence: { model: false, metrics: false, logs: true, artifacts: true },
      description: 'Checkpoint after issue triage',
    },
    {
      id: 'post-analysis',
      trigger: 'stage_complete',
      persistence: { model: false, metrics: true, logs: true, artifacts: true },
      description: 'Checkpoint after root cause analysis',
    },
  ],
  metadata: {
    author: 'Gemini CLI Team',
    createdAt: '2025-01-01',
    updatedAt: '2025-01-01',
    tags: ['deep-learning', 'debugging', 'troubleshooting'],
    relatedWorkflows: ['dl-model-development'],
  },
};

/**
 * Optimization Workflow
 *
 * Workflow for optimizing deep learning models for speed, memory, and accuracy.
 */
export const OptimizationWorkflow: DLWorkflow = {
  id: 'dl-optimization',
  name: 'Optimization Workflow',
  description: 'Comprehensive workflow for optimizing deep learning model performance.',
  version: '1.0.0',
  category: DLTaskCategory.OPTIMIZATION,
  stages: [
    // Stage 1: Performance Profiling
    {
      id: 'performance-profiling',
      name: 'Performance Profiling',
      description: 'Profile model performance to identify bottlenecks.',
      category: DLTaskCategory.OPTIMIZATION,
      priority: StagePriority.CRITICAL,
      inputs: [
        {
          name: 'model',
          type: 'model',
          required: true,
          description: 'Model to optimize',
        },
        {
          name: 'optimizationGoals',
          type: 'object',
          required: true,
          description: 'Target metrics (latency, memory, throughput)',
        },
      ],
      outputs: [
        {
          name: 'profilingReport',
          type: 'object',
          description: 'Detailed profiling report',
          persist: true,
        },
        {
          name: 'bottlenecks',
          type: 'array',
          description: 'Identified bottlenecks',
          persist: true,
        },
        {
          name: 'baselineMetrics',
          type: 'metrics',
          description: 'Baseline performance metrics',
          persist: true,
        },
      ],
      agents: {
        primaryAgent: 'architecture_analyzer_agent',
        supportingAgents: ['layer_inspector_agent'],
        timeoutMinutes: 45,
        maxRetries: 2,
      },
      qualityGates: [
        {
          id: 'profiling-complete',
          name: 'Profiling Complete',
          description: 'Profiling has been completed',
          criteria: [
            { id: 'bottlenecks-identified', check: 'Bottlenecks identified', weight: 1 },
            { id: 'baseline-established', check: 'Baseline metrics established', weight: 1 },
          ],
          blocking: true,
          evaluationType: 'automated',
        },
      ],
      dependencies: [],
      estimatedDuration: 45,
      parallelizable: false,
    },

    // Stage 2: Architecture Optimization
    {
      id: 'architecture-optimization',
      name: 'Architecture Optimization',
      description: 'Optimize model architecture for efficiency.',
      category: DLTaskCategory.OPTIMIZATION,
      priority: StagePriority.HIGH,
      inputs: [
        {
          name: 'model',
          type: 'model',
          required: true,
          description: 'Model to optimize',
        },
        {
          name: 'bottlenecks',
          type: 'array',
          required: true,
          description: 'Identified bottlenecks',
          source: 'performance-profiling',
        },
      ],
      outputs: [
        {
          name: 'optimizedArchitecture',
          type: 'model',
          description: 'Architecture-optimized model',
          persist: true,
        },
        {
          name: 'architectureChanges',
          type: 'array',
          description: 'Applied architecture changes',
          persist: true,
        },
      ],
      agents: {
        primaryAgent: 'architecture_analyzer_agent',
        supportingAgents: ['code_reviewer'],
        timeoutMinutes: 90,
        maxRetries: 3,
      },
      qualityGates: [
        {
          id: 'architecture-improved',
          name: 'Architecture Improved',
          description: 'Architecture optimization successful',
          criteria: [
            { id: 'complexity-reduced', check: 'Computational complexity reduced', weight: 1 },
            { id: 'accuracy-preserved', check: 'Accuracy preserved', weight: 2 },
          ],
          blocking: false,
          evaluationType: 'automated',
        },
      ],
      dependencies: ['performance-profiling'],
      estimatedDuration: 90,
      parallelizable: false,
    },

    // Stage 3: Quantization
    {
      id: 'quantization',
      name: 'Quantization',
      description: 'Apply quantization techniques to reduce model size and improve speed.',
      category: DLTaskCategory.OPTIMIZATION,
      priority: StagePriority.HIGH,
      inputs: [
        {
          name: 'optimizedArchitecture',
          type: 'model',
          required: true,
          description: 'Architecture-optimized model',
          source: 'architecture-optimization',
        },
        {
          name: 'calibrationData',
          type: 'array',
          required: true,
          description: 'Data for quantization calibration',
        },
      ],
      outputs: [
        {
          name: 'quantizedModel',
          type: 'model',
          description: 'Quantized model',
          persist: true,
        },
        {
          name: 'quantizationReport',
          type: 'object',
          description: 'Quantization analysis',
          persist: true,
        },
      ],
      agents: {
        primaryAgent: 'architecture_analyzer_agent',
        supportingAgents: ['model_evaluation_agent'],
        timeoutMinutes: 60,
        maxRetries: 3,
      },
      qualityGates: [
        {
          id: 'quantization-success',
          name: 'Quantization Success',
          description: 'Quantization meets targets',
          criteria: [
            { id: 'size-reduced', check: 'Model size reduced', weight: 1 },
            { id: 'accuracy-drop', check: 'Accuracy drop < 2%', threshold: 2, operator: 'lt', weight: 2 },
            { id: 'speed-improved', check: 'Inference speed improved', weight: 1 },
          ],
          blocking: true,
          evaluationType: 'automated',
        },
      ],
      dependencies: ['architecture-optimization'],
      estimatedDuration: 60,
      parallelizable: false,
    },

    // Stage 4: Pruning
    {
      id: 'pruning',
      name: 'Pruning',
      description: 'Apply pruning to remove unnecessary weights.',
      category: DLTaskCategory.OPTIMIZATION,
      priority: StagePriority.MEDIUM,
      inputs: [
        {
          name: 'quantizedModel',
          type: 'model',
          required: true,
          description: 'Quantized model',
          source: 'quantization',
        },
        {
          name: 'pruningTarget',
          type: 'number',
          required: true,
          description: 'Target sparsity percentage',
          defaultValue: 50,
        },
      ],
      outputs: [
        {
          name: 'prunedModel',
          type: 'model',
          description: 'Pruned model',
          persist: true,
        },
        {
          name: 'pruningReport',
          type: 'object',
          description: 'Pruning analysis',
          persist: true,
        },
      ],
      agents: {
        primaryAgent: 'architecture_analyzer_agent',
        supportingAgents: ['model_evaluation_agent'],
        timeoutMinutes: 90,
        maxRetries: 3,
      },
      qualityGates: [
        {
          id: 'pruning-success',
          name: 'Pruning Success',
          description: 'Pruning meets targets',
          criteria: [
            { id: 'sparsity-achieved', check: 'Target sparsity achieved', weight: 1 },
            { id: 'accuracy-preserved', check: 'Accuracy preserved', weight: 2 },
          ],
          blocking: false,
          evaluationType: 'automated',
        },
      ],
      dependencies: ['quantization'],
      estimatedDuration: 90,
      parallelizable: false,
    },

    // Stage 5: Final Evaluation
    {
      id: 'final-evaluation',
      name: 'Final Evaluation',
      description: 'Evaluate optimized model against baseline and targets.',
      category: DLTaskCategory.EVALUATION,
      priority: StagePriority.CRITICAL,
      inputs: [
        {
          name: 'prunedModel',
          type: 'model',
          required: true,
          description: 'Final optimized model',
          source: 'pruning',
        },
        {
          name: 'baselineMetrics',
          type: 'metrics',
          required: true,
          description: 'Baseline metrics',
          source: 'performance-profiling',
        },
        {
          name: 'optimizationGoals',
          type: 'object',
          required: true,
          description: 'Target metrics',
        },
      ],
      outputs: [
        {
          name: 'finalMetrics',
          type: 'metrics',
          description: 'Final performance metrics',
          persist: true,
        },
        {
          name: 'optimizationSummary',
          type: 'object',
          description: 'Complete optimization summary',
          persist: true,
        },
        {
          name: 'recommendations',
          type: 'array',
          description: 'Further optimization recommendations',
          persist: true,
        },
      ],
      agents: {
        primaryAgent: 'model_evaluation_agent',
        supportingAgents: ['summarizer_agent'],
        timeoutMinutes: 60,
        maxRetries: 2,
      },
      qualityGates: [
        {
          id: 'targets-met',
          name: 'Targets Met',
          description: 'Optimization targets achieved',
          criteria: [
            { id: 'latency-target', check: 'Latency target met', weight: 2 },
            { id: 'memory-target', check: 'Memory target met', weight: 1 },
            { id: 'accuracy-target', check: 'Accuracy target met', weight: 2 },
          ],
          blocking: true,
          evaluationType: 'automated',
        },
      ],
      dependencies: ['pruning'],
      estimatedDuration: 60,
      parallelizable: false,
    },
  ],
  globalConfig: {
    maxExecutionTimeMinutes: 420,
    continueOnFailure: false,
    notifications: {
      onStageComplete: true,
      onQualityGateFailed: true,
      onWorkflowComplete: true,
      channels: ['slack', 'email'],
    },
    resources: {
      maxGPUs: 4,
      maxMemoryGB: 64,
      maxCPUs: 16,
    },
    logging: {
      level: 'info',
      persistLogs: true,
      logArtifacts: true,
    },
  },
  checkpoints: [
    {
      id: 'post-profiling',
      trigger: 'stage_complete',
      persistence: { model: false, metrics: true, logs: true, artifacts: true },
      description: 'Checkpoint after profiling',
    },
    {
      id: 'post-quantization',
      trigger: 'stage_complete',
      persistence: { model: true, metrics: true, logs: true, artifacts: true },
      description: 'Checkpoint after quantization',
    },
    {
      id: 'final',
      trigger: 'stage_complete',
      persistence: { model: true, metrics: true, logs: true, artifacts: true },
      description: 'Final optimization checkpoint',
    },
  ],
  metadata: {
    author: 'Gemini CLI Team',
    createdAt: '2025-01-01',
    updatedAt: '2025-01-01',
    tags: ['deep-learning', 'optimization', 'quantization', 'pruning'],
    relatedWorkflows: ['dl-deployment', 'dl-model-development'],
  },
};

// =============================================================================
// Workflow Registry and Utilities
// =============================================================================

/**
 * Registry of all available DL workflows.
 */
export const DLWorkflowRegistry: Record<string, DLWorkflow> = {
  'model-development': ModelDevelopmentWorkflow,
  'research-reproduction': ResearchReproductionWorkflow,
  'production-deployment': ProductionDeploymentWorkflow,
  'debugging': DebuggingWorkflow,
  'optimization': OptimizationWorkflow,
};

/**
 * Get a workflow by ID.
 */
export function getWorkflow(workflowId: string): DLWorkflow | undefined {
  return DLWorkflowRegistry[workflowId];
}

/**
 * List all available workflows.
 */
export function listWorkflows(): Array<{ id: string; name: string; description: string }> {
  return Object.values(DLWorkflowRegistry).map((workflow) => ({
    id: workflow.id,
    name: workflow.name,
    description: workflow.description,
  }));
}

/**
 * Get workflow stages in execution order (respecting dependencies).
 */
export function getExecutionOrder(workflow: DLWorkflow): WorkflowStage[] {
  const stages = [...workflow.stages];
  const sorted: WorkflowStage[] = [];
  const visited = new Set<string>();

  function visit(stage: WorkflowStage): void {
    if (visited.has(stage.id)) return;

    // Visit dependencies first
    for (const depId of stage.dependencies) {
      const depStage = stages.find((s) => s.id === depId);
      if (depStage) {
        visit(depStage);
      }
    }

    visited.add(stage.id);
    sorted.push(stage);
  }

  for (const stage of stages) {
    visit(stage);
  }

  return sorted;
}

/**
 * Validate workflow configuration.
 */
export function validateWorkflow(workflow: DLWorkflow): { valid: boolean; errors: string[] } {
  const errors: string[] = [];

  // Check for unique stage IDs
  const stageIds = workflow.stages.map((s) => s.id);
  const uniqueIds = new Set(stageIds);
  if (stageIds.length !== uniqueIds.size) {
    errors.push('Duplicate stage IDs found');
  }

  // Check dependencies exist
  for (const stage of workflow.stages) {
    for (const depId of stage.dependencies) {
      if (!stageIds.includes(depId)) {
        errors.push(`Stage ${stage.id} has invalid dependency: ${depId}`);
      }
    }
  }

  // Check for circular dependencies
  try {
    getExecutionOrder(workflow);
  } catch {
    errors.push('Circular dependencies detected');
  }

  // Check quality gates have unique IDs within stages
  for (const stage of workflow.stages) {
    const gateIds = stage.qualityGates.map((g) => g.id);
    const uniqueGateIds = new Set(gateIds);
    if (gateIds.length !== uniqueGateIds.size) {
      errors.push(`Stage ${stage.id} has duplicate quality gate IDs`);
    }
  }

  return {
    valid: errors.length === 0,
    errors,
  };
}

/**
 * Estimate total workflow duration.
 */
export function estimateWorkflowDuration(workflow: DLWorkflow): number {
  // Simple estimation - sum of all stage durations
  // In practice, would account for parallelization
  return workflow.stages.reduce((total, stage) => total + stage.estimatedDuration, 0);
}

/**
 * Create workflow execution inputs from user parameters.
 */
export function createWorkflowInputs(
  workflow: DLWorkflow,
  userParams: Record<string, unknown>
): AgentInputs {
  const inputs: AgentInputs = {};

  // Get all required inputs from the first stage
  const firstStage = workflow.stages.find((s) => s.dependencies.length === 0);
  if (firstStage) {
    for (const input of firstStage.inputs) {
      if (input.required && !(input.name in userParams)) {
        throw new Error(`Missing required input: ${input.name}`);
      }
      inputs[input.name] = userParams[input.name] ?? input.defaultValue;
    }
  }

  return inputs;
}
