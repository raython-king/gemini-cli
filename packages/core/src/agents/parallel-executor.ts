/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { AgentExecutor } from './executor.js';
import type {
  AgentDefinition,
  AgentInputs,
  SubagentActivityEvent,
  OutputObject,
} from './types.js';
import type { Config } from '../config/config.js';
import { debugLogger } from '../utils/debugLogger.js';
import { type z } from 'zod';

/**
 * Represents a task for an agent to execute.
 */
export interface AgentTask<TOutput extends z.ZodTypeAny = z.ZodUnknown> {
  /** Unique identifier for this task */
  id: string;
  /** Name of the agent to execute */
  agentName: string;
  /** Definition of the agent */
  agentDefinition: AgentDefinition<TOutput>;
  /** Input parameters for the agent */
  inputs: AgentInputs;
  /** Priority level (higher = higher priority) */
  priority?: number;
}

/**
 * Result of an agent task execution.
 */
export interface AgentTaskResult {
  /** Task ID */
  taskId: string;
  /** Agent name */
  agentName: string;
  /** Whether execution was successful */
  success: boolean;
  /** Output from the agent */
  output?: OutputObject;
  /** Error if execution failed */
  error?: Error;
  /** Execution duration in milliseconds */
  durationMs: number;
}

/**
 * Callback for agent activity events during parallel execution.
 */
export type ParallelActivityCallback = (
  taskId: string,
  event: SubagentActivityEvent,
) => void;

/**
 * Parallel Agent Executor - Executes multiple agents concurrently.
 *
 * This executor manages the parallel execution of multiple agent tasks,
 * respecting concurrency limits and providing aggregated results.
 */
export class ParallelAgentExecutor {
  constructor(
    private readonly runtimeContext: Config,
    private readonly maxConcurrency: number = 3,
  ) {}

  /**
   * Execute multiple agent tasks in parallel.
   *
   * @param tasks Array of agent tasks to execute
   * @param activityCallback Optional callback for agent activity events
   * @returns Promise that resolves with array of task results
   */
  async executeParallel(
    tasks: AgentTask[],
    activityCallback?: ParallelActivityCallback,
  ): Promise<AgentTaskResult[]> {
    if (tasks.length === 0) {
      return [];
    }

    debugLogger.log(
      `[ParallelAgentExecutor] Starting execution of ${tasks.length} tasks with concurrency ${this.maxConcurrency}`,
    );

    // Sort tasks by priority (higher priority first)
    const sortedTasks = [...tasks].sort(
      (a, b) => (b.priority ?? 0) - (a.priority ?? 0),
    );

    // Execute tasks with concurrency limit
    const results: AgentTaskResult[] = [];
    const executing: Array<Promise<void>> = [];

    for (const task of sortedTasks) {
      // Wait if we've reached max concurrency
      if (executing.length >= this.maxConcurrency) {
        await Promise.race(executing);
      }

      // Start executing this task
      const promise = this.executeTask(task, activityCallback)
        .then((result) => {
          results.push(result);
        })
        .finally(() => {
          // Remove from executing list when done
          const index = executing.indexOf(promise);
          if (index > -1) {
            executing.splice(index, 1);
          }
        });

      executing.push(promise);
    }

    // Wait for all remaining tasks to complete
    await Promise.all(executing);

    debugLogger.log(
      `[ParallelAgentExecutor] Completed all ${tasks.length} tasks. ` +
        `Success: ${results.filter((r) => r.success).length}, ` +
        `Failed: ${results.filter((r) => !r.success).length}`,
    );

    return results;
  }

  /**
   * Execute multiple agent tasks using Promise.allSettled (all at once).
   * Use this when you want maximum parallelism without concurrency limits.
   *
   * @param tasks Array of agent tasks to execute
   * @param activityCallback Optional callback for agent activity events
   * @returns Promise that resolves with array of task results
   */
  async executeAllSettled(
    tasks: AgentTask[],
    activityCallback?: ParallelActivityCallback,
  ): Promise<AgentTaskResult[]> {
    if (tasks.length === 0) {
      return [];
    }

    debugLogger.log(
      `[ParallelAgentExecutor] Starting unlimited parallel execution of ${tasks.length} tasks`,
    );

    const promises = tasks.map((task) =>
      this.executeTask(task, activityCallback),
    );

    const results = await Promise.allSettled(promises);

    // Convert PromiseSettledResult to AgentTaskResult
    const taskResults: AgentTaskResult[] = results.map((result, index) => {
      if (result.status === 'fulfilled') {
        return result.value;
      } else {
        // Promise was rejected
        return {
          taskId: tasks[index].id,
          agentName: tasks[index].agentName,
          success: false,
          error: result.reason,
          durationMs: 0,
        };
      }
    });

    debugLogger.log(
      `[ParallelAgentExecutor] Completed all ${tasks.length} tasks. ` +
        `Success: ${taskResults.filter((r) => r.success).length}, ` +
        `Failed: ${taskResults.filter((r) => !r.success).length}`,
    );

    return taskResults;
  }

  /**
   * Execute a single agent task.
   */
  private async executeTask(
    task: AgentTask,
    activityCallback?: ParallelActivityCallback,
  ): Promise<AgentTaskResult> {
    const startTime = Date.now();

    debugLogger.log(
      `[ParallelAgentExecutor] Starting task ${task.id} (${task.agentName})`,
    );

    try {
      // Create a wrapped activity callback that includes the task ID
      const wrappedCallback = activityCallback
        ? (event: SubagentActivityEvent) => {
            activityCallback(task.id, event);
          }
        : undefined;

      // Create and execute the agent
      const executor = await AgentExecutor.create(
        task.agentDefinition,
        this.runtimeContext,
        wrappedCallback,
      );

      // Create an AbortController for this task
      const abortController = new AbortController();
      const output = await executor.run(task.inputs, abortController.signal);

      const durationMs = Date.now() - startTime;

      debugLogger.log(
        `[ParallelAgentExecutor] Task ${task.id} completed successfully in ${durationMs}ms`,
      );

      return {
        taskId: task.id,
        agentName: task.agentName,
        success: true,
        output,
        durationMs,
      };
    } catch (error) {
      const durationMs = Date.now() - startTime;

      debugLogger.log(
        `[ParallelAgentExecutor] Task ${task.id} failed after ${durationMs}ms: ${error}`,
      );

      return {
        taskId: task.id,
        agentName: task.agentName,
        success: false,
        error: error instanceof Error ? error : new Error(String(error)),
        durationMs,
      };
    }
  }

  /**
   * Create agent tasks from a list of agent definitions and inputs.
   * Helper method to easily create tasks for parallel execution.
   */
  static createTasks(
    definitions: Array<{
      agentDefinition: AgentDefinition;
      inputs: AgentInputs;
      priority?: number;
    }>,
  ): AgentTask[] {
    return definitions.map((def, index) => ({
      id: `task_${index}_${Date.now()}`,
      agentName: def.agentDefinition.name,
      agentDefinition: def.agentDefinition,
      inputs: def.inputs,
      priority: def.priority,
    }));
  }
}
