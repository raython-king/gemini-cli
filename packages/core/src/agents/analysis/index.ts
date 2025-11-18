/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Analysis module - Data flow processing and literature analysis.
 *
 * This module provides tools for automated data flow processing.
 * Additional agents (DataFlowAgent, LiteratureAnalyzerAgent, SummarizerAgent,
 * AnalysisOrchestrator) are in development and will be added in a future update.
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

// Note: Agent exports temporarily disabled pending adaptation to new AgentDefinition structure
// - DataFlowAgent, LiteratureAnalyzerAgent, SummarizerAgent
// - AnalysisOrchestrator
// See data-flow-agent.ts, literature-analyzer-agent.ts, summarizer-agent.ts,
// and analysis-orchestrator.ts for prototypes
