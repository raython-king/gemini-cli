/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Deep Learning Architecture Analysis, Training Optimization, Debugging and Profiling System
 *
 * A comprehensive suite of agents and utilities for:
 * - Analyzing, inspecting, comparing, and improving deep learning model architectures
 * - Optimizing training configurations for efficiency and performance
 * - Distributed training setup and hyperparameter tuning
 * - Training monitoring and diagnostics
 * - Debugging training failures, NaN/Inf issues, CUDA errors
 * - Profiling performance and memory usage
 * - Memory optimization and bottleneck identification
 */

// Export architecture analysis agents
export { ArchitectureAnalyzerAgent } from './architecture-analyzer-agent.js';
export { LayerInspectorAgent } from './layer-inspector-agent.js';
export { ArchitectureSuggesterAgent } from './architecture-suggester-agent.js';
export { ModelComparisonAgent } from './model-comparison-agent.js';

// Export training optimization agents
export {
  TrainingOptimizerAgent,
  type TrainingOptimizerInput,
  type TrainingOptimizerOutput,
} from './training-optimizer-agent.js';

export {
  HyperparameterTunerAgent,
  type HyperparameterTunerInput,
  type HyperparameterTunerOutput,
} from './hyperparameter-tuner-agent.js';

export {
  DistributedTrainingAgent,
  type DistributedTrainingInput,
  type DistributedTrainingOutput,
} from './distributed-training-agent.js';

export {
  TrainingMonitorAgent,
  type TrainingMonitorInput,
  type TrainingMonitorOutput,
} from './training-monitor-agent.js';

// Export debugging and profiling agents
export { DLDebugMasterAgent } from './debug-master-agent.js';
export { DLProfilerAgent } from './profiler-agent.js';
export { DLMemoryAnalyzerAgent } from './memory-analyzer-agent.js';

// Export training recipes library
export {
  type TrainingRecipe,
  ImageNetRecipes,
  LanguageModelRecipes,
  DiffusionModelRecipes,
  ReinforcementLearningRecipes,
  FineTuningRecipes,
  FewShotLearningRecipes,
  getAllRecipes,
  findRecipes,
  getRecipeRecommendation,
} from './training-recipes-library.js';

// Export optimization strategies
export {
  type OptimizerConfig,
  type LRScheduleConfig,
  WarmupStrategies,
  LearningRateSchedules,
  AdvancedOptimizers,
  GradientStrategies,
  getAllOptimizationStrategies,
  recommendStrategy,
  generateScheduleVisualization,
} from './optimization-strategies.js';

// Export architecture patterns library
export {
  // Pattern databases
  ATTENTION_PATTERNS,
  NORMALIZATION_PATTERNS,
  SKIP_CONNECTION_PATTERNS,
  EFFICIENT_CONV_PATTERNS,
  TRANSFORMER_VARIANTS,
  POSITION_ENCODING_PATTERNS,
  ACTIVATION_PATTERNS,
  ARCHITECTURE_PATTERNS,
  // Pattern types
  type AttentionPattern,
  type NormalizationPattern,
  type SkipConnectionPattern,
  type EfficientConvPattern,
  type TransformerVariant,
  type PositionEncodingPattern,
  type ActivationPattern,
  // Helper functions
  getAttentionPattern,
  getNormalizationPattern,
  getAllPatternCategories,
} from './architecture-patterns-library.js';

/**
 * All deep learning architecture analysis agents.
 */
export const DL_ARCHITECTURE_AGENTS = [
  'ArchitectureAnalyzerAgent',
  'LayerInspectorAgent',
  'ArchitectureSuggesterAgent',
  'ModelComparisonAgent',
] as const;

/**
 * All deep learning training optimization agents.
 */
export const DL_TRAINING_AGENTS = [
  'TrainingOptimizerAgent',
  'HyperparameterTunerAgent',
  'DistributedTrainingAgent',
  'TrainingMonitorAgent',
] as const;

/**
 * All deep learning debugging and profiling agents.
 */
export const DL_DEBUG_PROFILING_AGENTS = [
  'DLDebugMasterAgent',
  'DLProfilerAgent',
  'DLMemoryAnalyzerAgent',
] as const;

/**
 * All deep learning agents.
 */
export const DL_AGENTS = [
  ...DL_ARCHITECTURE_AGENTS,
  ...DL_TRAINING_AGENTS,
  ...DL_DEBUG_PROFILING_AGENTS,
] as const;

/**
 * Type for DL agent names.
 */
export type DLAgentName = (typeof DL_AGENTS)[number];
