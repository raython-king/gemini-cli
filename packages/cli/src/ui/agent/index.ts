/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * UI components for multi-agent system integration.
 *
 * This module provides React components for monitoring and displaying
 * agent execution in the Gemini CLI.
 */

export { AgentMonitor } from './AgentMonitor.js';
export { PlanDisplay } from './PlanDisplay.js';

export type {
  AgentInfo,
  AgentStatus,
  ToolCallRecord,
  Plan,
  PlanStep,
  Risk,
  Dependency,
  AlternativeApproach,
  ActivityCallback,
} from './types.js';
