/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type { AgentInputs } from './types.js';
import { debugLogger } from '../utils/debugLogger.js';
import {
  AgentOrchestrator,
  TaskType,
  TaskComplexity,
} from './orchestrator.js';
import type { TaskClassification } from './orchestrator.js';

/**
 * Represents a decomposed subtask.
 */
export interface Subtask {
  /** Unique ID for this subtask */
  id: string;
  /** Human-readable title */
  title: string;
  /** Detailed description */
  description: string;
  /** Recommended agent(s) for this subtask */
  recommendedAgents: string[];
  /** Dependencies (IDs of other subtasks that must complete first) */
  dependencies: string[];
  /** Estimated complexity */
  complexity: TaskComplexity;
  /** Priority (higher = more important) */
  priority: number;
  /** Inputs for the agent */
  inputs: AgentInputs;
  /** Whether this subtask can run in parallel with others */
  canParallelize: boolean;
}

/**
 * Result of task decomposition.
 */
export interface DecompositionResult {
  /** Original task */
  originalTask: string;
  /** Task classification */
  classification: TaskClassification;
  /** List of subtasks */
  subtasks: Subtask[];
  /** Execution plan (stages of parallel execution) */
  executionPlan: {
    stage: number;
    subtasks: string[]; // Subtask IDs
    canRunInParallel: boolean;
  }[];
  /** Estimated total time (based on complexity) */
  estimatedDuration: string;
}

/**
 * Task Decomposer - Automatically breaks down complex tasks into manageable subtasks.
 *
 * This is inspired by Claude Code's ability to automatically decompose complex
 * requests into multiple coordinated agent tasks.
 */
export class TaskDecomposer {
  private orchestrator: AgentOrchestrator;
  private subtaskCounter = 0;

  constructor() {
    this.orchestrator = new AgentOrchestrator();
  }

  /**
   * Decompose a complex task into subtasks.
   *
   * @param userTask The user's request
   * @param maxSubtasks Maximum number of subtasks to create (default: 10)
   * @returns Decomposition result with subtasks and execution plan
   */
  decompose(userTask: string, maxSubtasks = 10): DecompositionResult {
    debugLogger.log(`[TaskDecomposer] Decomposing task: ${userTask}`);

    // Classify the task
    const classification = this.orchestrator.classifyTask(userTask);

    // Determine if decomposition is needed
    const needsDecomposition = this.shouldDecompose(classification);

    if (!needsDecomposition) {
      // Simple task - create a single subtask
      const singleSubtask = this.createSimpleSubtask(userTask, classification);
      return {
        originalTask: userTask,
        classification,
        subtasks: [singleSubtask],
        executionPlan: [
          {
            stage: 1,
            subtasks: [singleSubtask.id],
            canRunInParallel: false,
          },
        ],
        estimatedDuration: this.estimateDuration([singleSubtask]),
      };
    }

    // Complex task - decompose into subtasks
    const subtasks = this.decomposeComplex(
      userTask,
      classification,
      maxSubtasks,
    );

    // Create execution plan
    const executionPlan = this.createExecutionPlan(subtasks);

    debugLogger.log(
      `[TaskDecomposer] Created ${subtasks.length} subtasks in ${executionPlan.length} stages`,
    );

    return {
      originalTask: userTask,
      classification,
      subtasks,
      executionPlan,
      estimatedDuration: this.estimateDuration(subtasks),
    };
  }

  /**
   * Determine if a task should be decomposed.
   */
  private shouldDecompose(classification: TaskClassification): boolean {
    // Decompose if:
    // 1. High or very high complexity
    // 2. Implementation or refactoring tasks
    // 3. Multiple agents recommended
    return (
      classification.complexity === TaskComplexity.HIGH ||
      classification.complexity === TaskComplexity.VERY_HIGH ||
      classification.type === TaskType.IMPLEMENTATION ||
      (classification.type === TaskType.REFACTORING &&
        classification.complexity !== TaskComplexity.LOW) ||
      classification.recommendedAgents.length > 1
    );
  }

  /**
   * Create a simple single subtask for non-complex tasks.
   */
  private createSimpleSubtask(
    task: string,
    classification: TaskClassification,
  ): Subtask {
    const agent = classification.recommendedAgents[0] || 'explore_agent';

    return {
      id: this.generateSubtaskId(),
      title: task.slice(0, 60) + (task.length > 60 ? '...' : ''),
      description: task,
      recommendedAgents: [agent],
      dependencies: [],
      complexity: classification.complexity,
      priority: 1,
      inputs: this.createInputsForAgent(agent, task, classification),
      canParallelize: false,
    };
  }

  /**
   * Decompose a complex task into multiple subtasks.
   */
  private decomposeComplex(
    task: string,
    classification: TaskClassification,
    maxSubtasks: number,
  ): Subtask[] {
    const subtasks: Subtask[] = [];
    const lowerTask = task.toLowerCase();

    // Strategy based on task type
    switch (classification.type) {
      case TaskType.IMPLEMENTATION:
        subtasks.push(...this.decomposeImplementation(task, classification));
        break;

      case TaskType.REFACTORING:
        subtasks.push(...this.decomposeRefactoring(task, classification));
        break;

      case TaskType.DEBUGGING:
        subtasks.push(...this.decomposeDebugging(task, classification));
        break;

      case TaskType.CODE_REVIEW:
        subtasks.push(...this.decomposeCodeReview(task, classification));
        break;

      case TaskType.TESTING:
        subtasks.push(...this.decomposeTesting(task, classification));
        break;

      default:
        // Generic decomposition
        subtasks.push(...this.decomposeGeneric(task, classification));
        break;
    }

    // Limit to maxSubtasks
    return subtasks.slice(0, maxSubtasks);
  }

  /**
   * Decompose an implementation task.
   */
  private decomposeImplementation(
    task: string,
    classification: TaskClassification,
  ): Subtask[] {
    const subtasks: Subtask[] = [];

    // Step 1: Planning
    if (classification.requiresPlan) {
      subtasks.push({
        id: this.generateSubtaskId(),
        title: 'Create implementation plan',
        description: `Plan how to implement: ${task}`,
        recommendedAgents: ['plan_agent'],
        dependencies: [],
        complexity: TaskComplexity.MEDIUM,
        priority: 10,
        inputs: { task, context: '' },
        canParallelize: false,
      });
    }

    // Step 2: Explore existing code
    subtasks.push({
      id: this.generateSubtaskId(),
      title: 'Explore existing codebase',
      description:
        'Understand relevant existing code and architecture patterns',
      recommendedAgents: ['explore_agent'],
      dependencies: [],
      complexity: TaskComplexity.LOW,
      priority: 9,
      inputs: {
        query: `Find code related to: ${task}`,
        thoroughness: 'medium',
      },
      canParallelize: !classification.requiresPlan, // Can run in parallel if no planning
    });

    // Step 3: Implementation (placeholder - actual implementation done by user or main agent)
    subtasks.push({
      id: this.generateSubtaskId(),
      title: 'Implement the feature',
      description: task,
      recommendedAgents: ['plan_agent'], // Planning agent provides implementation steps
      dependencies: subtasks.map((s) => s.id),
      complexity: classification.complexity,
      priority: 8,
      inputs: { task, context: 'Based on exploration and plan' },
      canParallelize: false,
    });

    // Step 4: Code review
    subtasks.push({
      id: this.generateSubtaskId(),
      title: 'Review implemented code',
      description: 'Automated code review for quality and security',
      recommendedAgents: ['code_reviewer'],
      dependencies: [subtasks[subtasks.length - 1].id],
      complexity: TaskComplexity.LOW,
      priority: 7,
      inputs: { files: '', focus: 'all' },
      canParallelize: false,
    });

    // Step 5: Testing
    subtasks.push({
      id: this.generateSubtaskId(),
      title: 'Run tests',
      description: 'Ensure all tests pass',
      recommendedAgents: ['test_runner'],
      dependencies: [subtasks[subtasks.length - 1].id],
      complexity: TaskComplexity.LOW,
      priority: 6,
      inputs: { testCommand: 'npm test', maxAttempts: 3 },
      canParallelize: false,
    });

    return subtasks;
  }

  /**
   * Decompose a refactoring task.
   */
  private decomposeRefactoring(
    task: string,
    classification: TaskClassification,
  ): Subtask[] {
    const subtasks: Subtask[] = [];

    // Step 1: Understand current code
    subtasks.push({
      id: this.generateSubtaskId(),
      title: 'Analyze code to refactor',
      description: 'Understand current implementation and identify issues',
      recommendedAgents: ['explore_agent'],
      dependencies: [],
      complexity: TaskComplexity.LOW,
      priority: 10,
      inputs: {
        query: `Analyze code for refactoring: ${task}`,
        thoroughness: 'thorough',
      },
      canParallelize: false,
    });

    // Step 2: Initial code review
    subtasks.push({
      id: this.generateSubtaskId(),
      title: 'Review code quality',
      description: 'Identify specific quality issues and refactoring opportunities',
      recommendedAgents: ['code_reviewer'],
      dependencies: [subtasks[0].id],
      complexity: TaskComplexity.LOW,
      priority: 9,
      inputs: { files: '', focus: 'all' },
      canParallelize: false,
    });

    // Step 3: Refactor
    subtasks.push({
      id: this.generateSubtaskId(),
      title: 'Perform refactoring',
      description: task,
      recommendedAgents: ['refactor_agent'],
      dependencies: [subtasks[1].id],
      complexity: classification.complexity,
      priority: 8,
      inputs: {
        target: '',
        strategy: 'comprehensive',
        testCommand: 'npm test',
      },
      canParallelize: false,
    });

    // Step 4: Verify with tests
    subtasks.push({
      id: this.generateSubtaskId(),
      title: 'Verify refactoring with tests',
      description: 'Ensure refactoring didn\'t break anything',
      recommendedAgents: ['test_runner'],
      dependencies: [subtasks[2].id],
      complexity: TaskComplexity.LOW,
      priority: 7,
      inputs: { testCommand: 'npm test', maxAttempts: 2 },
      canParallelize: false,
    });

    return subtasks;
  }

  /**
   * Decompose a debugging task.
   */
  private decomposeDebugging(
    task: string,
    classification: TaskClassification,
  ): Subtask[] {
    const subtasks: Subtask[] = [];

    // Step 1: Explore to understand context
    subtasks.push({
      id: this.generateSubtaskId(),
      title: 'Explore bug context',
      description: 'Understand where the bug might be located',
      recommendedAgents: ['explore_agent'],
      dependencies: [],
      complexity: TaskComplexity.LOW,
      priority: 10,
      inputs: {
        query: `Find code related to bug: ${task}`,
        thoroughness: 'medium',
      },
      canParallelize: false,
    });

    // Step 2: Debug
    subtasks.push({
      id: this.generateSubtaskId(),
      title: 'Debug and fix',
      description: task,
      recommendedAgents: ['debug_agent'],
      dependencies: [subtasks[0].id],
      complexity: classification.complexity,
      priority: 9,
      inputs: {
        bugDescription: task,
        reproduceSteps: '',
        context: '',
      },
      canParallelize: false,
    });

    // Step 3: Test the fix
    subtasks.push({
      id: this.generateSubtaskId(),
      title: 'Test the fix',
      description: 'Verify the bug is fixed and no regressions',
      recommendedAgents: ['test_runner'],
      dependencies: [subtasks[1].id],
      complexity: TaskComplexity.LOW,
      priority: 8,
      inputs: { testCommand: 'npm test', maxAttempts: 3 },
      canParallelize: false,
    });

    return subtasks;
  }

  /**
   * Decompose a code review task.
   */
  private decomposeCodeReview(
    task: string,
    classification: TaskClassification,
  ): Subtask[] {
    // Code review can often be parallelized by reviewing different aspects
    const subtasks: Subtask[] = [];

    // Review security
    subtasks.push({
      id: this.generateSubtaskId(),
      title: 'Security review',
      description: 'Check for security vulnerabilities',
      recommendedAgents: ['code_reviewer'],
      dependencies: [],
      complexity: TaskComplexity.MEDIUM,
      priority: 10,
      inputs: { files: '', focus: 'security' },
      canParallelize: true,
    });

    // Review performance
    subtasks.push({
      id: this.generateSubtaskId(),
      title: 'Performance review',
      description: 'Check for performance issues',
      recommendedAgents: ['code_reviewer'],
      dependencies: [],
      complexity: TaskComplexity.MEDIUM,
      priority: 9,
      inputs: { files: '', focus: 'performance' },
      canParallelize: true,
    });

    // Review testing
    subtasks.push({
      id: this.generateSubtaskId(),
      title: 'Test coverage review',
      description: 'Check test coverage and quality',
      recommendedAgents: ['code_reviewer'],
      dependencies: [],
      complexity: TaskComplexity.LOW,
      priority: 8,
      inputs: { files: '', focus: 'testing' },
      canParallelize: true,
    });

    return subtasks;
  }

  /**
   * Decompose a testing task.
   */
  private decomposeTesting(
    task: string,
    classification: TaskClassification,
  ): Subtask[] {
    const subtasks: Subtask[] = [];

    // Run tests
    subtasks.push({
      id: this.generateSubtaskId(),
      title: 'Run test suite',
      description: task,
      recommendedAgents: ['test_runner'],
      dependencies: [],
      complexity: TaskComplexity.LOW,
      priority: 10,
      inputs: { testCommand: 'npm test', maxAttempts: 3 },
      canParallelize: false,
    });

    return subtasks;
  }

  /**
   * Generic decomposition for unknown task types.
   */
  private decomposeGeneric(
    task: string,
    classification: TaskClassification,
  ): Subtask[] {
    const subtasks: Subtask[] = [];

    // Start with exploration
    subtasks.push({
      id: this.generateSubtaskId(),
      title: 'Explore and understand',
      description: task,
      recommendedAgents: classification.recommendedAgents,
      dependencies: [],
      complexity: classification.complexity,
      priority: 10,
      inputs: { query: task, thoroughness: 'medium' },
      canParallelize: false,
    });

    return subtasks;
  }

  /**
   * Create an execution plan with stages for parallel execution.
   */
  private createExecutionPlan(subtasks: Subtask[]): Array<{
    stage: number;
    subtasks: string[];
    canRunInParallel: boolean;
  }> {
    const plan: Array<{
      stage: number;
      subtasks: string[];
      canRunInParallel: boolean;
    }> = [];

    const completed = new Set<string>();
    let stage = 1;

    while (completed.size < subtasks.length) {
      // Find subtasks that can run in this stage
      const readySubtasks = subtasks.filter((st) => {
        // Already completed
        if (completed.has(st.id)) return false;

        // Check if all dependencies are completed
        return st.dependencies.every((depId) => completed.has(depId));
      });

      if (readySubtasks.length === 0) {
        // No more subtasks can run - break to avoid infinite loop
        debugLogger.log(
          '[TaskDecomposer] Warning: Circular dependency detected or unreachable subtasks',
        );
        break;
      }

      // Check if these subtasks can run in parallel
      const canParallelize =
        readySubtasks.length > 1 &&
        readySubtasks.every((st) => st.canParallelize);

      plan.push({
        stage,
        subtasks: readySubtasks.map((st) => st.id),
        canRunInParallel: canParallelize,
      });

      // Mark as completed
      readySubtasks.forEach((st) => completed.add(st.id));

      stage++;
    }

    return plan;
  }

  /**
   * Estimate total duration based on subtasks.
   */
  private estimateDuration(subtasks: Subtask[]): string {
    // Simple estimation based on complexity
    const timePerComplexity = {
      [TaskComplexity.LOW]: 2,
      [TaskComplexity.MEDIUM]: 5,
      [TaskComplexity.HIGH]: 10,
      [TaskComplexity.VERY_HIGH]: 20,
    };

    const totalMinutes = subtasks.reduce(
      (sum, st) => sum + timePerComplexity[st.complexity],
      0,
    );

    if (totalMinutes < 60) {
      return `~${totalMinutes} minutes`;
    } else {
      const hours = Math.round(totalMinutes / 60);
      return `~${hours} hour${hours > 1 ? 's' : ''}`;
    }
  }

  /**
   * Create inputs for a specific agent.
   */
  private createInputsForAgent(
    agentName: string,
    task: string,
    classification: TaskClassification,
  ): AgentInputs {
    // Delegate to orchestrator
    return this.orchestrator['createInputsForAgent'](
      agentName,
      task,
      classification,
    );
  }

  /**
   * Generate a unique subtask ID.
   */
  private generateSubtaskId(): string {
    return `subtask_${++this.subtaskCounter}_${Date.now()}`;
  }

  /**
   * Reset the subtask counter (useful for testing).
   */
  resetCounter(): void {
    this.subtaskCounter = 0;
  }
}
