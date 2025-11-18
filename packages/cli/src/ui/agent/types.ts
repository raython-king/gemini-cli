/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type { SubagentActivityEvent } from '@google/gemini-cli-core';

/**
 * Status of an agent during execution.
 */
export type AgentStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';

/**
 * Information about a running agent for monitoring.
 */
export interface AgentInfo {
  /** Unique identifier for the agent instance */
  id: string;
  /** Name of the agent */
  name: string;
  /** Display name of the agent */
  displayName?: string;
  /** Current status */
  status: AgentStatus;
  /** Current progress percentage (0-100) */
  progress?: number;
  /** Start time */
  startTime: Date;
  /** End time (if completed/failed) */
  endTime?: Date;
  /** Current activity description */
  currentActivity?: string;
  /** Error message if failed */
  error?: string;
  /** Tool call history */
  toolCalls: ToolCallRecord[];
}

/**
 * Record of a tool call made by an agent.
 */
export interface ToolCallRecord {
  /** Name of the tool */
  toolName: string;
  /** Start time of the tool call */
  startTime: Date;
  /** End time of the tool call (if completed) */
  endTime?: Date;
  /** Whether the tool call was successful */
  success?: boolean;
  /** Error message if failed */
  error?: string;
}

/**
 * Plan step structure from Plan Agent.
 */
export interface PlanStep {
  StepNumber: number;
  Description: string;
  AffectedFiles: string[];
  EstimatedComplexity: 'LOW' | 'MEDIUM' | 'HIGH';
  Dependencies: number[];
  Rationale: string;
}

/**
 * Risk assessment from Plan Agent.
 */
export interface Risk {
  Description: string;
  Severity: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  Mitigation: string;
}

/**
 * Dependency from Plan Agent.
 */
export interface Dependency {
  Type: 'LIBRARY' | 'API' | 'TOOL' | 'SERVICE' | 'FILE';
  Name: string;
  Reason: string;
}

/**
 * Alternative approach from Plan Agent.
 */
export interface AlternativeApproach {
  Description: string;
  Pros: string[];
  Cons: string[];
}

/**
 * Complete plan structure from Plan Agent.
 */
export interface Plan {
  Summary: string;
  Steps: PlanStep[];
  Risks: Risk[];
  Dependencies: Dependency[];
  EstimatedDuration: string;
  AlternativeApproaches?: AlternativeApproach[];
}

/**
 * Callback type for activity events.
 */
export type ActivityCallback = (event: SubagentActivityEvent) => void;
