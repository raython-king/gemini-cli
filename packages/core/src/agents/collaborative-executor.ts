/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { MultiAgentCoordinator, type CoordinatorStatus } from './multi-agent-coordinator.js';
import { ContextualOrchestrator } from './context/contextual-orchestrator.js';
import type { Config } from '../config/config.js';
import type { SubagentActivityEvent } from './types.js';
import { debugLogger } from '../utils/debugLogger.js';

/**
 * Collaborative execution event.
 */
export interface CollaborativeEvent {
  type:
    | 'decomposition_complete'
    | 'stage_start'
    | 'stage_complete'
    | 'subtask_start'
    | 'subtask_complete'
    | 'subtask_failed'
    | 'agent_activity'
    | 'context_shared'
    | 'execution_complete';
  timestamp: Date;
  data: unknown;
}

/**
 * Collaborative execution options.
 */
export interface CollaborativeExecutionOptions {
  /** Maximum number of subtasks to create */
  maxSubtasks?: number;
  /** Maximum concurrent agents */
  maxConcurrency?: number;
  /** Enable context sharing between agents */
  enableContextSharing?: boolean;
  /** Enable agent memory */
  enableMemory?: boolean;
  /** Preview only (don't execute) */
  previewOnly?: boolean;
  /** Event callback */
  onEvent?: (event: CollaborativeEvent) => void;
}

/**
 * Collaborative execution result.
 */
export interface CollaborativeResult {
  /** Original task */
  task: string;
  /** Final status */
  status: CoordinatorStatus;
  /** Execution events */
  events: CollaborativeEvent[];
  /** Shared context summary (if enabled) */
  contextSummary?: string;
  /** Memory summary (if enabled) */
  memorySummary?: string;
}

/**
 * Collaborative Executor - High-level multi-agent collaboration system.
 *
 * This executor combines:
 * - Task decomposition
 * - Multi-agent coordination
 * - Context sharing
 * - Agent memory
 *
 * It provides a Claude Code-like experience where multiple agents work
 * together intelligently on complex tasks.
 */
export class CollaborativeExecutor {
  private coordinator: MultiAgentCoordinator;
  private contextualOrchestrator: ContextualOrchestrator;
  private events: CollaborativeEvent[] = [];

  constructor(private runtimeContext: Config) {
    this.coordinator = new MultiAgentCoordinator(runtimeContext);
    this.contextualOrchestrator = new ContextualOrchestrator({
      maxContextTokens: 8000,
    });
  }

  /**
   * Execute a task with multi-agent collaboration.
   *
   * @param task The user's task
   * @param options Execution options
   * @returns Collaborative execution result
   */
  async execute(
    task: string,
    options: CollaborativeExecutionOptions = {},
  ): Promise<CollaborativeResult> {
    this.events = [];
    const startTime = new Date();

    debugLogger.log(`[CollaborativeExecutor] Starting collaborative execution: ${task}`);

    // Step 1: Preview decomposition
    const decomposition = this.coordinator.preview(
      task,
      options.maxSubtasks ?? 10,
    );

    this.emitEvent('decomposition_complete', {
      subtaskCount: decomposition.subtasks.length,
      stageCount: decomposition.executionPlan.length,
      estimatedDuration: decomposition.estimatedDuration,
      decomposition,
    }, options.onEvent);

    // If preview only, return early
    if (options.previewOnly) {
      const previewStatus: CoordinatorStatus = {
        task,
        currentStage: 0,
        totalStages: decomposition.executionPlan.length,
        subtasks: decomposition.subtasks.map((st) => ({
          subtask: st,
          status: 'pending' as const,
        })),
        progress: 0,
        isComplete: false,
        success: false,
        startTime,
      };

      return {
        task,
        status: previewStatus,
        events: this.events,
      };
    }

    // Step 2: Execute with coordination
    const finalStatus = await this.coordinator.execute(task, {
      maxSubtasks: options.maxSubtasks,
      maxConcurrency: options.maxConcurrency,
      onStatusUpdate: (status) => {
        // Detect stage transitions
        if (status.currentStage > 0) {
          const currentStageSubtasks = status.subtasks.filter(
            (st) => st.status === 'running' || st.status === 'completed',
          );

          // Check if stage just started
          const justStartedRunning = currentStageSubtasks.some(
            (st) =>
              st.status === 'running' &&
              st.startTime &&
              Date.now() - st.startTime.getTime() < 1000,
          );

          if (justStartedRunning) {
            this.emitEvent('stage_start', {
              stage: status.currentStage,
              totalStages: status.totalStages,
            }, options.onEvent);
          }
        }
      },
      onAgentActivity: (subtaskId, agentName, event) => {
        this.emitEvent('agent_activity', {
          subtaskId,
          agentName,
          eventType: event.type,
          event,
        }, options.onEvent);

        // Handle context sharing
        if (
          options.enableContextSharing &&
          event.type === 'TOOL_CALL_END' &&
          event.data.result
        ) {
          this.handleContextSharing(
            agentName,
            event.data.result,
            options.onEvent,
          );
        }

        // Track subtask completion
        if (event.type === 'COMPLETE') {
          this.emitEvent('subtask_complete', {
            subtaskId,
            agentName,
            result: event.data,
          }, options.onEvent);
        } else if (event.type === 'ERROR') {
          this.emitEvent('subtask_failed', {
            subtaskId,
            agentName,
            error: event.data.error,
          }, options.onEvent);
        }
      },
    });

    this.emitEvent('execution_complete', {
      success: finalStatus.success,
      progress: finalStatus.progress,
      duration:
        finalStatus.endTime && finalStatus.startTime
          ? finalStatus.endTime.getTime() - finalStatus.startTime.getTime()
          : 0,
    }, options.onEvent);

    // Prepare result
    const result: CollaborativeResult = {
      task,
      status: finalStatus,
      events: this.events,
    };

    // Add context summary if enabled
    if (options.enableContextSharing) {
      const contextState = this.contextualOrchestrator.exportState();
      result.contextSummary = contextState.context;
      debugLogger.log(
        `[CollaborativeExecutor] Context shared: ${contextState.stats.context.totalItems} items`,
      );
    }

    // Add memory summary if enabled
    if (options.enableMemory) {
      const memoryState = this.contextualOrchestrator.exportState();
      result.memorySummary = memoryState.memory;
      debugLogger.log(
        `[CollaborativeExecutor] Memories stored: ${memoryState.stats.memory.totalMemories} memories`,
      );
    }

    debugLogger.log(
      `[CollaborativeExecutor] Execution complete. Success: ${finalStatus.success}`,
    );

    return result;
  }

  /**
   * Handle context sharing between agents.
   */
  private handleContextSharing(
    agentName: string,
    result: unknown,
    onEvent?: (event: CollaborativeEvent) => void,
  ): void {
    try {
      // Store agent output in shared context
      const output = result as { result?: string; terminate_reason?: string };

      if (output && output.result) {
        this.contextualOrchestrator.recordExecution(agentName, output as any);

        this.emitEvent('context_shared', {
          agentName,
          hasOutput: !!output.result,
        }, onEvent);
      }
    } catch (error) {
      debugLogger.log(
        `[CollaborativeExecutor] Error sharing context: ${error}`,
      );
    }
  }

  /**
   * Emit an event.
   */
  private emitEvent(
    type: CollaborativeEvent['type'],
    data: unknown,
    callback?: (event: CollaborativeEvent) => void,
  ): void {
    const event: CollaborativeEvent = {
      type,
      timestamp: new Date(),
      data,
    };

    this.events.push(event);

    if (callback) {
      callback(event);
    }
  }

  /**
   * Get the contextual orchestrator (for advanced usage).
   */
  getContextualOrchestrator(): ContextualOrchestrator {
    return this.contextualOrchestrator;
  }

  /**
   * Get the multi-agent coordinator (for advanced usage).
   */
  getCoordinator(): MultiAgentCoordinator {
    return this.coordinator;
  }

  /**
   * Clear context and memory.
   */
  reset(): void {
    this.events = [];
    this.contextualOrchestrator.cleanup({
      maxContextAge: 0,
      minMemoryImportance: 0,
    });
  }
}
