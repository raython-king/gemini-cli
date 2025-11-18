/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Analysis module - Data flow processing and literature analysis.
 *
 * This module provides tools for automated data flow processing and
 * specialized analysis agents for summarization, literature review, and data analysis.
 *
 * See README.md in this directory for more information.
 */

// Export data pipeline utilities (ready to use)
export {
  DataPipeline,
  Transforms,
  Validators,
  type TransformFn,
  type ValidatorFn,
  type ValidationResult,
  type PipelineStep,
  type PipelineResult,
} from './data-pipeline.js';

// Export analysis agents
export { SummarizerAgent } from './summarizer-agent.js';
export { LiteratureAnalyzerAgent } from './literature-analyzer-agent.js';
export { DataFlowAgent } from './data-flow-agent.js';

// Export analysis orchestrator (adapted to new architecture)
export {
  AnalysisOrchestrator,
  AnalysisWorkflowType,
  type AnalysisTask,
  type AnalysisResult,
} from './analysis-orchestrator.js';
