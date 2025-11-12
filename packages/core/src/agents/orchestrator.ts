/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type { AgentInputs } from './types.js';
import { debugLogger } from '../utils/debugLogger.js';

/**
 * Task classification for determining agent selection.
 */
export enum TaskType {
  PLANNING = 'planning',
  EXPLORATION = 'exploration',
  CODE_REVIEW = 'code_review',
  TESTING = 'testing',
  DEBUGGING = 'debugging',
  REFACTORING = 'refactoring',
  IMPLEMENTATION = 'implementation',
  UNKNOWN = 'unknown',
}

/**
 * Task complexity level.
 */
export enum TaskComplexity {
  LOW = 'low', // Single file, simple change
  MEDIUM = 'medium', // Multiple files, moderate complexity
  HIGH = 'high', // System-wide changes, complex logic
  VERY_HIGH = 'very_high', // Architecture changes, major refactoring
}

/**
 * Result of task classification.
 */
export interface TaskClassification {
  type: TaskType;
  complexity: TaskComplexity;
  recommendedAgents: string[]; // Agent names
  requiresPlan: boolean; // Whether plan mode should be triggered
  canParallelize: boolean; // Whether subtasks can run in parallel
}

/**
 * Orchestration strategy for executing a task.
 */
export interface OrchestrationStrategy {
  /** Whether to use plan mode first */
  usePlanMode: boolean;
  /** Agents to invoke (in order if sequential, or all at once if parallel) */
  agents: Array<{
    agentName: string;
    inputs: AgentInputs;
    priority?: number;
  }>;
  /** Whether agents should run in parallel */
  parallel: boolean;
  /** Maximum concurrency if parallel */
  maxConcurrency?: number;
}

/**
 * Agent Orchestrator - Intelligent agent selection and coordination.
 *
 * This orchestrator analyzes user requests, classifies tasks, and determines
 * the optimal execution strategy (which agents, sequential vs parallel, etc.).
 */
export class AgentOrchestrator {
  constructor() {}

  /**
   * Classify a task based on user message.
   *
   * @param userMessage The user's request or message
   * @returns Task classification with recommendations
   */
  classifyTask(userMessage: string): TaskClassification {
    const lowerMessage = userMessage.toLowerCase();

    // Keywords for each task type
    const keywords = {
      planning: [
        'plan',
        'design',
        'architect',
        'strategy',
        'approach',
        'how should',
        'what steps',
      ],
      exploration: [
        'where is',
        'how does',
        'find',
        'locate',
        'explore',
        'understand',
        'explain',
        'what is',
        'show me',
      ],
      codeReview: [
        'review',
        'check',
        'audit',
        'quality',
        'security',
        'analyze code',
        'look at',
      ],
      testing: ['test', 'spec', 'unit test', 'integration test', 'run tests'],
      debugging: [
        'bug',
        'error',
        'fix',
        'broken',
        'not working',
        'crash',
        'issue',
      ],
      refactoring: [
        'refactor',
        'improve',
        'cleanup',
        'simplify',
        'optimize',
        'reorganize',
      ],
      implementation: [
        'implement',
        'add',
        'create',
        'build',
        'develop',
        'write',
        'code',
      ],
    };

    // Determine task type
    let taskType = TaskType.UNKNOWN;
    let maxMatches = 0;

    for (const [type, words] of Object.entries(keywords)) {
      const matches = words.filter((keyword) =>
        lowerMessage.includes(keyword),
      ).length;
      if (matches > maxMatches) {
        maxMatches = matches;
        taskType = type as TaskType;
      }
    }

    // Estimate complexity based on various signals
    const complexity = this.estimateComplexity(userMessage, lowerMessage);

    // Determine recommended agents
    const recommendedAgents = this.selectAgentsForTask(taskType, complexity);

    // Determine if plan mode is needed
    const requiresPlan =
      complexity === TaskComplexity.HIGH ||
      complexity === TaskComplexity.VERY_HIGH ||
      taskType === TaskType.IMPLEMENTATION ||
      lowerMessage.includes('plan');

    // Determine if parallelization is possible
    const canParallelize = this.canParallelizeTask(taskType);

    return {
      type: taskType,
      complexity,
      recommendedAgents,
      requiresPlan,
      canParallelize,
    };
  }

  /**
   * Create an orchestration strategy for a user request.
   *
   * @param userMessage The user's request
   * @param classification Optional pre-computed classification
   * @returns Orchestration strategy
   */
  createStrategy(
    userMessage: string,
    classification?: TaskClassification,
  ): OrchestrationStrategy {
    const taskClass = classification ?? this.classifyTask(userMessage);

    debugLogger.log(
      `[AgentOrchestrator] Task classified as ${taskClass.type} (${taskClass.complexity})`,
    );

    // If plan mode is required, use Plan Agent first
    if (taskClass.requiresPlan) {
      return {
        usePlanMode: true,
        agents: [
          {
            agentName: 'plan_agent',
            inputs: {
              task: userMessage,
              context: '',
            },
          },
        ],
        parallel: false,
      };
    }

    // Create agent invocations based on recommended agents
    const agents = taskClass.recommendedAgents.map((agentName, index) => ({
      agentName,
      inputs: this.createInputsForAgent(agentName, userMessage, taskClass),
      priority: taskClass.recommendedAgents.length - index, // Higher index = lower priority
    }));

    // Determine parallelization strategy
    const parallel = taskClass.canParallelize && agents.length > 1;

    return {
      usePlanMode: false,
      agents,
      parallel,
      maxConcurrency: parallel ? 3 : 1,
    };
  }

  /**
   * Estimate task complexity from user message.
   */
  private estimateComplexity(
    originalMessage: string,
    lowerMessage: string,
  ): TaskComplexity {
    let complexityScore = 0;

    // Length-based signals
    if (originalMessage.length > 500) complexityScore += 2;
    else if (originalMessage.length > 200) complexityScore += 1;

    // Keyword-based signals for complexity
    const complexityKeywords = {
      veryHigh: [
        'entire',
        'all',
        'system-wide',
        'architecture',
        'redesign',
        'migrate',
      ],
      high: [
        'multiple',
        'several',
        'across',
        'integrate',
        'comprehensive',
        'major',
      ],
      medium: ['some', 'few', 'modify', 'update', 'change'],
    };

    if (complexityKeywords.veryHigh.some((kw) => lowerMessage.includes(kw))) {
      complexityScore += 4;
    } else if (
      complexityKeywords.high.some((kw) => lowerMessage.includes(kw))
    ) {
      complexityScore += 3;
    } else if (
      complexityKeywords.medium.some((kw) => lowerMessage.includes(kw))
    ) {
      complexityScore += 2;
    } else {
      complexityScore += 1; // Default to low-medium
    }

    // Determine final complexity
    if (complexityScore >= 5) return TaskComplexity.VERY_HIGH;
    if (complexityScore >= 4) return TaskComplexity.HIGH;
    if (complexityScore >= 2) return TaskComplexity.MEDIUM;
    return TaskComplexity.LOW;
  }

  /**
   * Select appropriate agents for a task type and complexity.
   */
  private selectAgentsForTask(
    taskType: TaskType,
    complexity: TaskComplexity,
  ): string[] {
    const isComplex =
      complexity === TaskComplexity.HIGH ||
      complexity === TaskComplexity.VERY_HIGH;

    switch (taskType) {
      case TaskType.PLANNING:
        return ['plan_agent'];

      case TaskType.EXPLORATION:
        // Use Explore for quick queries, CodebaseInvestigator for deep analysis
        return isComplex ? ['codebase_investigator'] : ['explore_agent'];

      case TaskType.CODE_REVIEW:
        return ['code_reviewer'];

      case TaskType.TESTING:
        return ['test_runner'];

      case TaskType.DEBUGGING:
        return ['debug_agent'];

      case TaskType.REFACTORING:
        // For complex refactoring, review first, then refactor
        return isComplex
          ? ['code_reviewer', 'refactor_agent']
          : ['refactor_agent'];

      case TaskType.IMPLEMENTATION:
        // For complex implementation, plan first (handled by requiresPlan)
        return ['plan_agent'];

      case TaskType.UNKNOWN:
      default:
        // Default to exploration for unknown tasks
        return ['explore_agent'];
    }
  }

  /**
   * Determine if a task type can be parallelized.
   */
  private canParallelizeTask(taskType: TaskType): boolean {
    // Some tasks naturally support parallelization
    switch (taskType) {
      case TaskType.CODE_REVIEW:
      case TaskType.EXPLORATION:
      case TaskType.TESTING:
        return true; // Can review/explore/test multiple files in parallel

      case TaskType.DEBUGGING:
      case TaskType.REFACTORING:
      case TaskType.IMPLEMENTATION:
      case TaskType.PLANNING:
        return false; // These typically need sequential execution

      default:
        return false;
    }
  }

  /**
   * Create appropriate inputs for a specific agent.
   */
  private createInputsForAgent(
    agentName: string,
    userMessage: string,
    taskClass: TaskClassification,
  ): AgentInputs {
    // Base inputs that most agents need
    const baseInputs: AgentInputs = {};

    switch (agentName) {
      case 'plan_agent':
        return {
          task: userMessage,
          context: `Task type: ${taskClass.type}, Complexity: ${taskClass.complexity}`,
        };

      case 'explore_agent':
        return {
          query: userMessage,
          thoroughness:
            taskClass.complexity === TaskComplexity.HIGH ||
            taskClass.complexity === TaskComplexity.VERY_HIGH
              ? 'thorough'
              : 'medium',
        };

      case 'codebase_investigator':
        return {
          objective: userMessage,
        };

      case 'code_reviewer':
        // For code review, extract file paths if mentioned
        return {
          files: this.extractFilePaths(userMessage),
          focus: 'all',
        };

      case 'test_runner':
        return {
          testCommand: this.inferTestCommand(),
          maxAttempts: 3,
        };

      case 'debug_agent':
        return {
          bugDescription: userMessage,
          reproduceSteps: '',
          context: '',
        };

      case 'refactor_agent':
        return {
          target: this.extractRefactorTarget(userMessage),
          strategy: 'comprehensive',
        };

      default:
        return baseInputs;
    }
  }

  /**
   * Extract file paths mentioned in a message.
   */
  private extractFilePaths(message: string): string {
    // Simple regex to find file paths (can be improved)
    const pathPattern =
      /(?:^|\s)([\w\-/.]+\.(?:ts|js|tsx|jsx|py|java|go|rs|cpp|c|h))/gi;
    const matches = message.match(pathPattern);
    return matches ? matches.map((m) => m.trim()).join(', ') : '';
  }

  /**
   * Extract refactoring target from message.
   */
  private extractRefactorTarget(message: string): string {
    // Extract file paths or function names
    const filePaths = this.extractFilePaths(message);
    if (filePaths) return filePaths;

    // Try to extract quoted text as target
    const quotedPattern = /"([^"]+)"|'([^']+)'/;
    const match = message.match(quotedPattern);
    return match ? match[1] || match[2] : message;
  }

  /**
   * Infer test command from project structure.
   * This is a placeholder - in real implementation, would check package.json, etc.
   */
  private inferTestCommand(): string {
    // Default to npm test - actual implementation would detect project type
    return 'npm test';
  }

  /**
   * Check if Plan Mode should be triggered for a user message.
   * This is a convenience method for quick checks.
   */
  shouldUsePlanMode(userMessage: string): boolean {
    const classification = this.classifyTask(userMessage);
    return classification.requiresPlan;
  }
}
