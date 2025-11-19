/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Deep Learning Tools for Gemini CLI
 *
 * This module provides a comprehensive set of tools for deep learning development,
 * analysis, and optimization. These tools help developers understand their models,
 * diagnose training issues, and improve performance.
 *
 * Available tools:
 * - NeuralNetworkAnalyzerTool: Analyze model architectures, parameters, and FLOPs
 * - GradientAnalyzerTool: Detect gradient flow issues and suggest solutions
 * - TrainingDiagnosticsTool: Diagnose training problems from logs
 * - ModelComparisonTool: Compare multiple models side-by-side
 * - DatasetAnalyzerTool: Analyze datasets for quality and suggest augmentations
 */

// Neural Network Architecture Analyzer
export {
  NeuralNetworkAnalyzerTool,
  type NeuralNetworkAnalyzerParams,
} from './neural-network-analyzer-tool.js';

// Gradient Flow Analyzer
export {
  GradientAnalyzerTool,
  type GradientAnalyzerParams,
} from './gradient-analyzer-tool.js';

// Training Diagnostics
export {
  TrainingDiagnosticsTool,
  type TrainingDiagnosticsParams,
} from './training-diagnostics-tool.js';

// Model Comparison
export {
  ModelComparisonTool,
  type ModelComparisonParams,
} from './model-comparison-tool.js';

// Dataset Analyzer
export {
  DatasetAnalyzerTool,
  type DatasetAnalyzerParams,
} from './dataset-analyzer-tool.js';

// Tool names for external reference
export const DL_TOOL_NAMES = {
  NEURAL_NETWORK_ANALYZER: 'analyze_neural_network',
  GRADIENT_ANALYZER: 'analyze_gradients',
  TRAINING_DIAGNOSTICS: 'training_diagnostics',
  MODEL_COMPARISON: 'compare_models',
  DATASET_ANALYZER: 'analyze_dataset',
} as const;

// Re-export all tool classes for convenience
import { NeuralNetworkAnalyzerTool } from './neural-network-analyzer-tool.js';
import { GradientAnalyzerTool } from './gradient-analyzer-tool.js';
import { TrainingDiagnosticsTool } from './training-diagnostics-tool.js';
import { ModelComparisonTool } from './model-comparison-tool.js';
import { DatasetAnalyzerTool } from './dataset-analyzer-tool.js';
import type { Config } from '../../config/config.js';
import type { MessageBus } from '../../confirmation-bus/message-bus.js';

/**
 * Create all deep learning tools with the given configuration
 * @param config The application configuration
 * @param messageBus Optional message bus for tool communication
 * @returns Array of all deep learning tool instances
 */
export function createDeepLearningTools(
  config: Config,
  messageBus?: MessageBus,
) {
  return [
    new NeuralNetworkAnalyzerTool(config, messageBus),
    new GradientAnalyzerTool(config, messageBus),
    new TrainingDiagnosticsTool(config, messageBus),
    new ModelComparisonTool(config, messageBus),
    new DatasetAnalyzerTool(config, messageBus),
  ];
}

/**
 * Tool information for documentation
 */
export const DL_TOOLS_INFO = [
  {
    name: 'Neural Network Analyzer',
    toolName: DL_TOOL_NAMES.NEURAL_NETWORK_ANALYZER,
    description: 'Analyze neural network architectures, calculate parameters, FLOPs, memory footprint, and detect patterns',
    useCases: [
      'Understanding model complexity',
      'Comparing architecture decisions',
      'Planning deployment resources',
      'Visualizing model structure',
    ],
  },
  {
    name: 'Gradient Analyzer',
    toolName: DL_TOOL_NAMES.GRADIENT_ANALYZER,
    description: 'Detect gradient flow issues like vanishing/exploding gradients, dead neurons, and bottlenecks',
    useCases: [
      'Debugging training failures',
      'Optimizing gradient flow',
      'Improving convergence',
      'Diagnosing numerical issues',
    ],
  },
  {
    name: 'Training Diagnostics',
    toolName: DL_TOOL_NAMES.TRAINING_DIAGNOSTICS,
    description: 'Analyze training logs to detect overfitting, learning rate issues, and suggest optimizations',
    useCases: [
      'Detecting overfitting/underfitting',
      'Optimizing learning rate',
      'Finding data leakage',
      'Improving GPU utilization',
    ],
  },
  {
    name: 'Model Comparison',
    toolName: DL_TOOL_NAMES.MODEL_COMPARISON,
    description: 'Compare multiple models side-by-side for parameters, FLOPs, memory, and architecture',
    useCases: [
      'Selecting best model for use case',
      'Understanding speed/accuracy tradeoffs',
      'Comparing architectural choices',
      'Resource planning',
    ],
  },
  {
    name: 'Dataset Analyzer',
    toolName: DL_TOOL_NAMES.DATASET_ANALYZER,
    description: 'Analyze datasets for class distribution, quality issues, and augmentation suggestions',
    useCases: [
      'Detecting class imbalance',
      'Finding data quality issues',
      'Planning augmentation strategy',
      'Understanding data distribution',
    ],
  },
];
