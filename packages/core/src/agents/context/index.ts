/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Context Engineering Module
 *
 * This module provides advanced context management for multi-agent systems,
 * including shared context, memory, and intelligent optimization.
 */

// SharedContext exports
export {
  SharedContext,
  ContextType,
  ContextPriority,
  type ContextItem,
  type ContextFilter,
} from './shared-context.js';

// ContextOptimizer exports
export {
  ContextOptimizer,
  OptimizationStrategy,
  type OptimizedContext,
} from './context-optimizer.js';

// AgentMemory exports
export {
  AgentMemory,
  MemoryType,
  type Memory,
  type MemoryQuery,
} from './agent-memory.js';

// ContextualOrchestrator exports
export {
  ContextualOrchestrator,
  type ContextualStrategy,
  type ContextualExecutionResult,
} from './contextual-orchestrator.js';
