/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  AgentOrchestrator,
  TaskType,
  TaskComplexity,
} from '../orchestrator.js';
import type {
  TaskClassification,
  OrchestrationStrategy,
} from '../orchestrator.js';
import type { AgentInputs, OutputObject } from '../types.js';
import {
  SharedContext,
  ContextType,
  ContextPriority,
} from './shared-context.js';
import { ContextOptimizer, OptimizationStrategy } from './context-optimizer.js';
import { AgentMemory } from './agent-memory.js';
import { debugLogger } from '../../utils/debugLogger.js';

/**
 * Enhanced orchestration strategy with context awareness.
 */
export interface ContextualStrategy extends OrchestrationStrategy {
  /** Relevant context for this execution */
  context: {
    summary: string;
    items: number;
    optimizedPrompt: string;
  };
  /** Relevant memories */
  memories: {
    count: number;
    summary: string;
  };
  /** Confidence score for this strategy (0-1) */
  confidence: number;
}

/**
 * Execution result with context tracking.
 */
export interface ContextualExecutionResult {
  agentName: string;
  output: OutputObject;
  contextUsed: {
    itemCount: number;
    tokenCount: number;
  };
  memoriesCreated: string[];
  insights: string[];
}

/**
 * ContextualOrchestrator - Context-aware agent orchestration.
 *
 * This orchestrator extends the base AgentOrchestrator with context
 * management capabilities, enabling agents to share knowledge and
 * build upon each other's work.
 */
export class ContextualOrchestrator extends AgentOrchestrator {
  private sharedContext: SharedContext;
  private contextOptimizer: ContextOptimizer;
  private memory: AgentMemory;

  constructor(
    options: {
      maxContextTokens?: number;
      optimizationStrategy?: OptimizationStrategy;
    } = {},
  ) {
    super();
    this.sharedContext = new SharedContext();
    this.contextOptimizer = new ContextOptimizer(
      options.maxContextTokens ?? 4000,
      options.optimizationStrategy ?? OptimizationStrategy.BALANCED,
    );
    this.memory = new AgentMemory();
  }

  /**
   * Create a context-aware orchestration strategy.
   */
  override createStrategy(
    userMessage: string,
    classification?: TaskClassification,
  ): ContextualStrategy {
    // Get base strategy from parent
    const baseStrategy = super.createStrategy(userMessage, classification);
    const taskClass = classification ?? this.classifyTask(userMessage);

    // Extract relevant context
    const relevantContext = this.getRelevantContext(taskClass);
    const optimized = this.contextOptimizer.optimize(relevantContext, {
      preserveTypes: [ContextType.CODE_INSIGHTS, ContextType.ISSUES],
      maxItems: 10,
    });

    // Retrieve relevant memories
    const relevantMemories = this.getRelevantMemories(taskClass);

    // Calculate confidence based on available context
    const confidence = this.calculateConfidence(
      taskClass,
      optimized.items.length,
      relevantMemories.length,
    );

    debugLogger.log(
      `[ContextualOrchestrator] Strategy created with ${optimized.items.length} context items, ` +
        `${relevantMemories.length} memories, confidence: ${confidence.toFixed(2)}`,
    );

    return {
      ...baseStrategy,
      context: {
        summary: this.contextOptimizer.summarize(optimized.items),
        items: optimized.items.length,
        optimizedPrompt: optimized.promptText,
      },
      memories: {
        count: relevantMemories.length,
        summary: this.memory.exportSummary({
          limit: 5,
          sortBy: 'importance',
        }),
      },
      confidence,
    };
  }

  /**
   * Enhanced agent inputs with context injection.
   */
  enhanceInputsWithContext(
    baseInputs: AgentInputs,
    agentName: string,
    taskClass: TaskClassification,
  ): AgentInputs {
    const enhancedInputs = { ...baseInputs };

    // Get relevant context
    const relevantContext = this.getRelevantContext(taskClass);
    const optimized = this.contextOptimizer.optimize(relevantContext);

    // Inject optimized context as additional input
    if (optimized.items.length > 0) {
      enhancedInputs['_contextPrompt'] = optimized.promptText;
    }

    // Inject relevant memories
    const memories = this.getRelevantMemories(taskClass);
    if (memories.length > 0) {
      enhancedInputs['_memorySummary'] = this.memory.exportSummary({
        limit: 5,
        sortBy: 'importance',
      });
    }

    // Add metadata about context
    enhancedInputs['_contextMetadata'] = {
      itemCount: optimized.items.length,
      estimatedTokens: optimized.estimatedTokens,
      memoryCount: memories.length,
      agentName,
    };

    return enhancedInputs;
  }

  /**
   * Record agent execution results in context and memory.
   */
  recordExecution(
    agentName: string,
    output: OutputObject,
    extractedInsights?: {
      files?: string[];
      issues?: string[];
      patterns?: string[];
    },
  ): ContextualExecutionResult {
    const memoriesCreated: string[] = [];
    const insights: string[] = [];

    // Store in shared context
    this.sharedContext.addAgentOutput(agentName, output);

    // Store in memory
    const memoryId = this.memory.storeAgentExecution(agentName, output, {
      importance: output.terminate_reason === 'GOAL' ? 0.8 : 0.5,
      tags: [agentName, output.terminate_reason],
    });
    memoriesCreated.push(memoryId);

    // Store extracted insights
    if (extractedInsights) {
      if (extractedInsights.files || extractedInsights.patterns) {
        this.sharedContext.addCodeInsights(agentName, {
          files: extractedInsights.files,
          patterns: extractedInsights.patterns,
        });

        const insightSummary = `Found ${extractedInsights.files?.length ?? 0} files, ${extractedInsights.patterns?.length ?? 0} patterns`;
        insights.push(insightSummary);

        const insightMemId = this.memory.storeFact(
          agentName,
          insightSummary,
          extractedInsights,
          {
            tags: ['code', 'insights'],
            importance: 0.7,
          },
        );
        memoriesCreated.push(insightMemId);
      }

      if (extractedInsights.issues && extractedInsights.issues.length > 0) {
        const issueObjects = extractedInsights.issues.map((issue) => ({
          severity: 'MEDIUM',
          description: issue,
        }));

        this.sharedContext.addIssues(agentName, issueObjects);
        insights.push(`Identified ${extractedInsights.issues.length} issues`);
      }
    }

    return {
      agentName,
      output,
      contextUsed: {
        itemCount: 0, // Will be filled when used
        tokenCount: 0,
      },
      memoriesCreated,
      insights,
    };
  }

  /**
   * Get relevant context items for a task classification.
   */
  private getRelevantContext(taskClass: TaskClassification) {
    const filters: Array<{
      types?: ContextType[];
      tags?: string[];
      minPriority?: ContextPriority;
    }> = [];

    // Add type-specific filters
    switch (taskClass.type) {
      case TaskType.EXPLORATION:
      case TaskType.PLANNING:
        filters.push({ types: [ContextType.CODE_INSIGHTS] });
        break;

      case TaskType.CODE_REVIEW:
      case TaskType.REFACTORING:
        filters.push({
          types: [ContextType.ISSUES, ContextType.CODE_INSIGHTS],
        });
        break;

      case TaskType.DEBUGGING:
        filters.push({
          types: [ContextType.ISSUES],
          minPriority: ContextPriority.MEDIUM,
        });
        break;

      case TaskType.TESTING:
        filters.push({
          types: [ContextType.ISSUES, ContextType.AGENT_OUTPUT],
          tags: ['test', 'testing'],
        });
        break;

      default:
        // For unknown or implementation tasks, include all types
        break;
    }

    // Combine results from all filters
    const allItems = new Set<string>();
    const results: ReturnType<typeof this.sharedContext.get> = [];

    filters.forEach((filter) => {
      const items = this.sharedContext.get(filter);
      items.forEach((item) => {
        if (!allItems.has(item.id)) {
          allItems.add(item.id);
          results.push(item);
        }
      });
    });

    // Always include recent high-priority items
    const highPriority = this.sharedContext.getHighPriority();
    highPriority.forEach((item) => {
      if (!allItems.has(item.id)) {
        results.push(item);
      }
    });

    return results;
  }

  /**
   * Get relevant memories for a task classification.
   */
  private getRelevantMemories(taskClass: TaskClassification) {
    const tags: string[] = [];

    // Add task-type specific tags
    switch (taskClass.type) {
      case TaskType.EXPLORATION:
        tags.push('code', 'architecture', 'insights');
        break;
      case TaskType.CODE_REVIEW:
        tags.push('issues', 'quality', 'security');
        break;
      case TaskType.DEBUGGING:
        tags.push('bug', 'error', 'fix');
        break;
      case TaskType.TESTING:
        tags.push('test', 'testing', 'verification');
        break;
      case TaskType.REFACTORING:
        tags.push('refactor', 'quality', 'improvement');
        break;
      default:
        // For unknown or other task types, use general tags
        tags.push('general', 'execution');
        break;
    }

    return this.memory.getRelevant(tags, 10);
  }

  /**
   * Calculate confidence score for a strategy.
   */
  private calculateConfidence(
    taskClass: TaskClassification,
    contextCount: number,
    memoryCount: number,
  ): number {
    let confidence = 0.5; // Base confidence

    // Increase confidence based on available context
    if (contextCount > 0) {
      confidence += Math.min(0.2, contextCount * 0.02);
    }

    // Increase confidence based on relevant memories
    if (memoryCount > 0) {
      confidence += Math.min(0.15, memoryCount * 0.015);
    }

    // Adjust based on task complexity
    if (taskClass.complexity === TaskComplexity.LOW) {
      confidence += 0.1;
    } else if (taskClass.complexity === TaskComplexity.VERY_HIGH) {
      confidence -= 0.1;
    }

    // Ensure confidence is between 0 and 1
    return Math.max(0, Math.min(1, confidence));
  }

  /**
   * Perform memory consolidation (move important working memory to long-term).
   */
  consolidateMemories(importanceThreshold: number = 0.6): number {
    return this.memory.consolidate(importanceThreshold);
  }

  /**
   * Clean up old or unimportant context and memories.
   */
  cleanup(
    options: {
      maxContextAge?: number;
      minMemoryImportance?: number;
    } = {},
  ): { contextRemoved: number; memoriesRemoved: number } {
    const contextRemoved = options.maxContextAge
      ? this.sharedContext.clearOld(options.maxContextAge)
      : 0;

    const memoriesRemoved = options.minMemoryImportance
      ? this.memory.forget(options.minMemoryImportance)
      : 0;

    return { contextRemoved, memoriesRemoved };
  }

  /**
   * Get statistics about context and memory usage.
   */
  getStats() {
    return {
      context: this.sharedContext.getStats(),
      memory: this.memory.getStats(),
    };
  }

  /**
   * Export full context and memory state (for debugging or persistence).
   */
  exportState() {
    return {
      context: this.sharedContext.exportSummary(),
      memory: this.memory.exportSummary(),
      stats: this.getStats(),
    };
  }

  /**
   * Get shared context instance for external access.
   */
  getSharedContext(): SharedContext {
    return this.sharedContext;
  }

  /**
   * Get memory instance for external access.
   */
  getMemory(): AgentMemory {
    return this.memory;
  }
}
