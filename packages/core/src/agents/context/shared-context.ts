/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type { OutputObject } from '../types.js';

/**
 * Types of context information that can be shared between agents.
 */
export enum ContextType {
  /** Output from a previous agent execution */
  AGENT_OUTPUT = 'agent_output',
  /** Discovered code locations and insights */
  CODE_INSIGHTS = 'code_insights',
  /** Identified issues or problems */
  ISSUES = 'issues',
  /** Task-related metadata */
  TASK_METADATA = 'task_metadata',
  /** User preferences and constraints */
  USER_PREFERENCES = 'user_preferences',
  /** Execution history and patterns */
  EXECUTION_HISTORY = 'execution_history',
}

/**
 * Priority level for context items (higher = more important).
 */
export enum ContextPriority {
  LOW = 1,
  MEDIUM = 2,
  HIGH = 3,
  CRITICAL = 4,
}

/**
 * A single piece of context information.
 */
export interface ContextItem {
  /** Unique identifier for this context item */
  id: string;
  /** Type of context */
  type: ContextType;
  /** Priority level */
  priority: ContextPriority;
  /** Source agent that created this context */
  source: string;
  /** Timestamp when created */
  timestamp: Date;
  /** The actual context data */
  data: unknown;
  /** Optional metadata */
  metadata?: Record<string, unknown>;
  /** Tags for categorization and filtering */
  tags?: string[];
}

/**
 * Filter criteria for retrieving context items.
 */
export interface ContextFilter {
  types?: ContextType[];
  sources?: string[];
  minPriority?: ContextPriority;
  tags?: string[];
  since?: Date;
}

/**
 * SharedContext - Manages context sharing between multiple agents.
 *
 * This class provides a centralized store for context information that
 * can be shared across multiple agent executions, enabling better
 * collaboration and knowledge transfer.
 */
export class SharedContext {
  private items: Map<string, ContextItem> = new Map();
  private nextId = 0;

  /**
   * Add a new context item.
   */
  add(
    type: ContextType,
    data: unknown,
    options: {
      priority?: ContextPriority;
      source?: string;
      metadata?: Record<string, unknown>;
      tags?: string[];
    } = {},
  ): string {
    const id = `ctx_${this.nextId++}_${Date.now()}`;
    const item: ContextItem = {
      id,
      type,
      priority: options.priority ?? ContextPriority.MEDIUM,
      source: options.source ?? 'unknown',
      timestamp: new Date(),
      data,
      metadata: options.metadata,
      tags: options.tags,
    };

    this.items.set(id, item);
    return id;
  }

  /**
   * Add context from an agent's output.
   */
  addAgentOutput(agentName: string, output: OutputObject): string {
    return this.add(ContextType.AGENT_OUTPUT, output, {
      priority: ContextPriority.HIGH,
      source: agentName,
      metadata: {
        terminateReason: output.terminate_reason,
      },
      tags: [agentName, 'output'],
    });
  }

  /**
   * Add code insights discovered during exploration.
   */
  addCodeInsights(
    source: string,
    insights: {
      files?: string[];
      functions?: string[];
      patterns?: string[];
      issues?: string[];
    },
  ): string {
    return this.add(ContextType.CODE_INSIGHTS, insights, {
      priority: ContextPriority.HIGH,
      source,
      tags: ['code', 'insights'],
    });
  }

  /**
   * Add identified issues.
   */
  addIssues(
    source: string,
    issues: Array<{
      severity: string;
      description: string;
      location?: string;
    }>,
  ): string {
    const priority = issues.some(
      (i) => i.severity === 'CRITICAL' || i.severity === 'HIGH',
    )
      ? ContextPriority.CRITICAL
      : ContextPriority.MEDIUM;

    return this.add(ContextType.ISSUES, issues, {
      priority,
      source,
      tags: ['issues', 'problems'],
    });
  }

  /**
   * Retrieve context items matching the filter.
   */
  get(filter: ContextFilter = {}): ContextItem[] {
    let results = Array.from(this.items.values());

    // Filter by type
    if (filter.types && filter.types.length > 0) {
      results = results.filter((item) => filter.types!.includes(item.type));
    }

    // Filter by source
    if (filter.sources && filter.sources.length > 0) {
      results = results.filter((item) => filter.sources!.includes(item.source));
    }

    // Filter by minimum priority
    if (filter.minPriority !== undefined) {
      results = results.filter((item) => item.priority >= filter.minPriority!);
    }

    // Filter by tags
    if (filter.tags && filter.tags.length > 0) {
      results = results.filter(
        (item) =>
          item.tags && filter.tags!.some((tag) => item.tags!.includes(tag)),
      );
    }

    // Filter by timestamp
    if (filter.since) {
      results = results.filter((item) => item.timestamp >= filter.since!);
    }

    // Sort by priority (descending) and timestamp (descending)
    results.sort((a, b) => {
      if (a.priority !== b.priority) {
        return b.priority - a.priority;
      }
      return b.timestamp.getTime() - a.timestamp.getTime();
    });

    return results;
  }

  /**
   * Get a single context item by ID.
   */
  getById(id: string): ContextItem | undefined {
    return this.items.get(id);
  }

  /**
   * Get the most recent context items.
   */
  getRecent(limit: number = 10, filter?: ContextFilter): ContextItem[] {
    const results = this.get(filter);
    return results.slice(0, limit);
  }

  /**
   * Get high-priority context items.
   */
  getHighPriority(
    minPriority: ContextPriority = ContextPriority.HIGH,
  ): ContextItem[] {
    return this.get({ minPriority });
  }

  /**
   * Clear all context items.
   */
  clear(): void {
    this.items.clear();
  }

  /**
   * Remove context items older than the specified age (in milliseconds).
   */
  clearOld(maxAge: number): number {
    const cutoff = new Date(Date.now() - maxAge);
    const oldItems = Array.from(this.items.entries()).filter(
      ([, item]) => item.timestamp < cutoff,
    );

    oldItems.forEach(([id]) => this.items.delete(id));
    return oldItems.length;
  }

  /**
   * Get summary statistics about the context store.
   */
  getStats(): {
    totalItems: number;
    byType: Record<string, number>;
    byPriority: Record<string, number>;
    bySource: Record<string, number>;
  } {
    const items = Array.from(this.items.values());

    const byType: Record<string, number> = {};
    const byPriority: Record<string, number> = {};
    const bySource: Record<string, number> = {};

    items.forEach((item) => {
      byType[item.type] = (byType[item.type] || 0) + 1;
      byPriority[item.priority] = (byPriority[item.priority] || 0) + 1;
      bySource[item.source] = (bySource[item.source] || 0) + 1;
    });

    return {
      totalItems: items.length,
      byType,
      byPriority,
      bySource,
    };
  }

  /**
   * Export context as a formatted summary string for use in prompts.
   */
  exportSummary(filter?: ContextFilter): string {
    const items = this.get(filter);

    if (items.length === 0) {
      return 'No relevant context available.';
    }

    let summary = `# Context Summary (${items.length} items)\n\n`;

    // Group by type
    const byType = new Map<ContextType, ContextItem[]>();
    items.forEach((item) => {
      if (!byType.has(item.type)) {
        byType.set(item.type, []);
      }
      byType.get(item.type)!.push(item);
    });

    // Format each group
    byType.forEach((groupItems, type) => {
      summary += `## ${type.replace(/_/g, ' ').toUpperCase()}\n\n`;

      groupItems.forEach((item, idx) => {
        summary += `${idx + 1}. **[${item.priority === ContextPriority.CRITICAL ? '🔴' : item.priority === ContextPriority.HIGH ? '🟡' : '⚪'}] From ${item.source}**\n`;

        // Format data based on type
        if (typeof item.data === 'string') {
          summary += `   ${item.data}\n`;
        } else if (typeof item.data === 'object' && item.data !== null) {
          const dataStr = JSON.stringify(item.data, null, 2);
          // Truncate if too long
          const lines = dataStr.split('\n');
          if (lines.length > 10) {
            summary += `   ${lines.slice(0, 10).join('\n')}...\n`;
          } else {
            summary += `   ${dataStr}\n`;
          }
        }

        summary += '\n';
      });
    });

    return summary;
  }
}
