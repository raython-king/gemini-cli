/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { TaskDecomposer, type Subtask, type DecompositionResult } from './task-decomposer.js';
import { ParallelAgentExecutor, type AgentTask, type AgentTaskResult } from './parallel-executor.js';
import { AgentRegistry } from './registry.js';
import type { Config } from '../config/config.js';
import type { SubagentActivityEvent } from './types.js';
import { debugLogger } from '../utils/debugLogger.js';

/**
 * Status of a subtask execution.
 */
export enum SubtaskStatus {
  PENDING = 'pending',
  RUNNING = 'running',
  COMPLETED = 'completed',
  FAILED = 'failed',
  SKIPPED = 'skipped',
}

/**
 * Live status of a subtask.
 */
export interface SubtaskExecutionStatus {
  subtask: Subtask;
  status: SubtaskStatus;
  startTime?: Date;
  endTime?: Date;
  result?: AgentTaskResult;
  error?: string;
}

/**
 * Overall execution status.
 */
export interface CoordinatorStatus {
  /** Original task */
  task: string;
  /** Current stage being executed */
  currentStage: number;
  /** Total stages */
  totalStages: number;
  /** All subtasks with their status */
  subtasks: SubtaskExecutionStatus[];
  /** Overall progress (0-100) */
  progress: number;
  /** Whether execution is complete */
  isComplete: boolean;
  /** Whether execution succeeded */
  success: boolean;
  /** Start time */
  startTime: Date;
  /** End time (if complete) */
  endTime?: Date;
}

/**
 * Callback for status updates.
 */
export type StatusUpdateCallback = (status: CoordinatorStatus) => void;

/**
 * Multi-Agent Coordinator - Orchestrates complex multi-agent workflows.
 *
 * This coordinator combines task decomposition with parallel execution to
 * automatically break down complex tasks and execute them using multiple
 * specialized agents, similar to Claude Code's multi-agent system.
 */
export class MultiAgentCoordinator {
  private decomposer: TaskDecomposer;
  private registry: AgentRegistry;

  constructor(private runtimeContext: Config) {
    this.decomposer = new TaskDecomposer();
    this.registry = new AgentRegistry();
  }

  /**
   * Execute a complex task using multi-agent coordination.
   *
   * @param task The user's task
   * @param options Execution options
   * @returns Final coordinator status
   */
  async execute(
    task: string,
    options: {
      maxSubtasks?: number;
      maxConcurrency?: number;
      onStatusUpdate?: StatusUpdateCallback;
      onAgentActivity?: (
        subtaskId: string,
        agentName: string,
        event: SubagentActivityEvent,
      ) => void;
    } = {},
  ): Promise<CoordinatorStatus> {
    const startTime = new Date();
    debugLogger.log(`[MultiAgentCoordinator] Starting execution: ${task}`);

    // Step 1: Decompose the task
    const decomposition = this.decomposer.decompose(
      task,
      options.maxSubtasks ?? 10,
    );

    debugLogger.log(
      `[MultiAgentCoordinator] Decomposed into ${decomposition.subtasks.length} subtasks, ${decomposition.executionPlan.length} stages`,
    );

    // Step 2: Initialize status tracking
    const subtaskStatuses: SubtaskExecutionStatus[] =
      decomposition.subtasks.map((st) => ({
        subtask: st,
        status: SubtaskStatus.PENDING,
      }));

    let currentStage = 0;
    let overallSuccess = true;

    const getStatus = (): CoordinatorStatus => {
      const completed = subtaskStatuses.filter(
        (s) => s.status === SubtaskStatus.COMPLETED,
      ).length;
      const total = subtaskStatuses.length;
      const progress = total > 0 ? Math.round((completed / total) * 100) : 0;
      const isComplete = completed === total || currentStage >= decomposition.executionPlan.length;

      return {
        task,
        currentStage,
        totalStages: decomposition.executionPlan.length,
        subtasks: subtaskStatuses,
        progress,
        isComplete,
        success: overallSuccess,
        startTime,
        endTime: isComplete ? new Date() : undefined,
      };
    };

    // Step 3: Execute stages sequentially
    for (let i = 0; i < decomposition.executionPlan.length; i++) {
      const stage = decomposition.executionPlan[i];
      currentStage = stage.stage;

      debugLogger.log(
        `[MultiAgentCoordinator] Executing stage ${stage.stage}/${decomposition.executionPlan.length} ` +
          `with ${stage.subtasks.length} subtasks (parallel: ${stage.canRunInParallel})`,
      );

      // Update status
      if (options.onStatusUpdate) {
        options.onStatusUpdate(getStatus());
      }

      // Get subtasks for this stage
      const stageSubtasks = decomposition.subtasks.filter((st) =>
        stage.subtasks.includes(st.id),
      );

      // Execute subtasks
      const results = await this.executeStage(
        stageSubtasks,
        stage.canRunInParallel,
        options.maxConcurrency ?? 3,
        subtaskStatuses,
        options.onAgentActivity,
      );

      // Update status with results
      for (const result of results) {
        const statusEntry = subtaskStatuses.find(
          (s) => s.subtask.id === result.taskId,
        );
        if (statusEntry) {
          statusEntry.status = result.success
            ? SubtaskStatus.COMPLETED
            : SubtaskStatus.FAILED;
          statusEntry.endTime = new Date();
          statusEntry.result = result;

          if (!result.success) {
            overallSuccess = false;
            statusEntry.error = result.error?.message;

            debugLogger.log(
              `[MultiAgentCoordinator] Subtask ${result.taskId} failed: ${result.error?.message}`,
            );
          }
        }
      }

      // Update status after stage completion
      if (options.onStatusUpdate) {
        options.onStatusUpdate(getStatus());
      }

      // Stop if a critical failure occurred
      if (!overallSuccess && !stage.canRunInParallel) {
        debugLogger.log(
          '[MultiAgentCoordinator] Critical failure in sequential stage, stopping execution',
        );
        break;
      }
    }

    const finalStatus = getStatus();
    debugLogger.log(
      `[MultiAgentCoordinator] Execution complete. Success: ${finalStatus.success}, Progress: ${finalStatus.progress}%`,
    );

    return finalStatus;
  }

  /**
   * Execute a single stage (potentially with multiple subtasks).
   */
  private async executeStage(
    subtasks: Subtask[],
    parallel: boolean,
    maxConcurrency: number,
    allStatuses: SubtaskExecutionStatus[],
    onAgentActivity?: (
      subtaskId: string,
      agentName: string,
      event: SubagentActivityEvent,
    ) => void,
  ): Promise<AgentTaskResult[]> {
    // Convert subtasks to agent tasks
    const agentTasks: AgentTask[] = [];

    for (const subtask of subtasks) {
      // Update status to running
      const statusEntry = allStatuses.find((s) => s.subtask.id === subtask.id);
      if (statusEntry) {
        statusEntry.status = SubtaskStatus.RUNNING;
        statusEntry.startTime = new Date();
      }

      // Get the first recommended agent
      const agentName = subtask.recommendedAgents[0];
      if (!agentName) {
        debugLogger.log(
          `[MultiAgentCoordinator] Warning: No recommended agent for subtask ${subtask.id}`,
        );
        continue;
      }

      // Get agent definition from registry
      const agentDef = this.registry.getAgent(agentName);
      if (!agentDef) {
        debugLogger.log(
          `[MultiAgentCoordinator] Warning: Agent ${agentName} not found in registry`,
        );
        continue;
      }

      agentTasks.push({
        id: subtask.id,
        agentName,
        agentDefinition: agentDef,
        inputs: subtask.inputs,
        priority: subtask.priority,
      });
    }

    // Execute tasks
    const executor = new ParallelAgentExecutor(this.runtimeContext, maxConcurrency);

    const activityCallback = onAgentActivity
      ? (taskId: string, event: SubagentActivityEvent) => {
          const task = agentTasks.find((t) => t.id === taskId);
          if (task) {
            onAgentActivity(taskId, task.agentName, event);
          }
        }
      : undefined;

    if (parallel && agentTasks.length > 1) {
      return await executor.executeParallel(agentTasks, activityCallback);
    } else {
      // Sequential execution (concurrency = 1)
      const sequentialExecutor = new ParallelAgentExecutor(this.runtimeContext, 1);
      return await sequentialExecutor.executeParallel(
        agentTasks,
        activityCallback,
      );
    }
  }

  /**
   * Preview task decomposition without executing.
   *
   * @param task The user's task
   * @param maxSubtasks Maximum subtasks to create
   * @returns Decomposition result
   */
  preview(task: string, maxSubtasks = 10): DecompositionResult {
    return this.decomposer.decompose(task, maxSubtasks);
  }

  /**
   * Format decomposition result as human-readable text.
   */
  formatDecomposition(decomposition: DecompositionResult): string {
    let output = `Task: ${decomposition.originalTask}\n\n`;
    output += `Type: ${decomposition.classification.type}\n`;
    output += `Complexity: ${decomposition.classification.complexity}\n`;
    output += `Estimated Duration: ${decomposition.estimatedDuration}\n\n`;

    output += `Execution Plan (${decomposition.executionPlan.length} stages):\n`;
    output += `${'='.repeat(60)}\n\n`;

    for (const stage of decomposition.executionPlan) {
      output += `Stage ${stage.stage} (${stage.canRunInParallel ? 'Parallel' : 'Sequential'}):\n`;

      const stageSubtasks = decomposition.subtasks.filter((st) =>
        stage.subtasks.includes(st.id),
      );

      for (const subtask of stageSubtasks) {
        output += `  - ${subtask.title}\n`;
        output += `    Agent: ${subtask.recommendedAgents.join(', ')}\n`;
        output += `    Complexity: ${subtask.complexity}\n`;
        if (subtask.dependencies.length > 0) {
          output += `    Dependencies: ${subtask.dependencies.join(', ')}\n`;
        }
        output += '\n';
      }
    }

    return output;
  }

  /**
   * Format execution status as human-readable text.
   */
  formatStatus(status: CoordinatorStatus): string {
    let output = `Task: ${status.task}\n`;
    output += `Progress: ${status.progress}% (Stage ${status.currentStage}/${status.totalStages})\n`;
    output += `Status: ${status.isComplete ? 'Complete' : 'In Progress'}\n`;
    if (status.isComplete) {
      output += `Success: ${status.success ? 'Yes' : 'No'}\n`;
    }
    output += '\n';

    const pending = status.subtasks.filter(
      (s) => s.status === SubtaskStatus.PENDING,
    );
    const running = status.subtasks.filter(
      (s) => s.status === SubtaskStatus.RUNNING,
    );
    const completed = status.subtasks.filter(
      (s) => s.status === SubtaskStatus.COMPLETED,
    );
    const failed = status.subtasks.filter(
      (s) => s.status === SubtaskStatus.FAILED,
    );

    output += `Subtasks:\n`;
    output += `  Pending: ${pending.length}\n`;
    output += `  Running: ${running.length}\n`;
    output += `  Completed: ${completed.length}\n`;
    output += `  Failed: ${failed.length}\n`;
    output += '\n';

    if (running.length > 0) {
      output += 'Currently Running:\n';
      for (const st of running) {
        output += `  - ${st.subtask.title}\n`;
      }
      output += '\n';
    }

    if (failed.length > 0) {
      output += 'Failed Subtasks:\n';
      for (const st of failed) {
        output += `  - ${st.subtask.title}\n`;
        if (st.error) {
          output += `    Error: ${st.error}\n`;
        }
      }
      output += '\n';
    }

    return output;
  }
}
