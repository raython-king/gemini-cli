/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type { ContextItem } from './shared-context.js';
import { ContextPriority, ContextType } from './shared-context.js';

/**
 * Strategy for optimizing context.
 */
export enum OptimizationStrategy {
  /** Prioritize recent items */
  RECENCY = 'recency',
  /** Prioritize high-priority items */
  PRIORITY = 'priority',
  /** Balance between recency and priority */
  BALANCED = 'balanced',
  /** Preserve diversity of context types */
  DIVERSE = 'diverse',
}

/**
 * Result of context optimization.
 */
export interface OptimizedContext {
  /** Selected context items */
  items: ContextItem[];
  /** Formatted prompt string */
  promptText: string;
  /** Estimated token count */
  estimatedTokens: number;
  /** Statistics about optimization */
  stats: {
    totalItems: number;
    selectedItems: number;
    droppedItems: number;
    compressionRatio: number;
  };
}

/**
 * ContextOptimizer - Intelligently optimizes context for agent prompts.
 *
 * This class helps manage context window limits by selecting the most
 * relevant context items and formatting them efficiently.
 */
export class ContextOptimizer {
  private readonly maxTokens: number;
  private readonly strategy: OptimizationStrategy;

  constructor(
    maxTokens: number = 4000,
    strategy: OptimizationStrategy = OptimizationStrategy.BALANCED,
  ) {
    this.maxTokens = maxTokens;
    this.strategy = strategy;
  }

  /**
   * Optimize a list of context items for inclusion in a prompt.
   */
  optimize(
    items: ContextItem[],
    options: {
      preserveTypes?: ContextType[];
      maxItems?: number;
    } = {},
  ): OptimizedContext {
    const totalItems = items.length;

    // Step 1: Sort items based on strategy
    const sorted = this.sortByStrategy(items);

    // Step 2: Preserve required types
    const preserved: ContextItem[] = [];
    const remaining: ContextItem[] = [];

    if (options.preserveTypes && options.preserveTypes.length > 0) {
      sorted.forEach((item) => {
        if (options.preserveTypes!.includes(item.type)) {
          preserved.push(item);
        } else {
          remaining.push(item);
        }
      });
    } else {
      remaining.push(...sorted);
    }

    // Step 3: Select items within token budget
    const selected: ContextItem[] = [...preserved];
    let currentTokens = this.estimateTokens(selected);

    for (const item of remaining) {
      const itemTokens = this.estimateItemTokens(item);

      if (
        currentTokens + itemTokens <= this.maxTokens &&
        (options.maxItems === undefined || selected.length < options.maxItems)
      ) {
        selected.push(item);
        currentTokens += itemTokens;
      }
    }

    // Step 4: Format as prompt text
    const promptText = this.formatAsPrompt(selected);
    const estimatedTokens = this.estimateTokens(selected);

    return {
      items: selected,
      promptText,
      estimatedTokens,
      stats: {
        totalItems,
        selectedItems: selected.length,
        droppedItems: totalItems - selected.length,
        compressionRatio: totalItems > 0 ? selected.length / totalItems : 1,
      },
    };
  }

  /**
   * Sort items based on optimization strategy.
   */
  private sortByStrategy(items: ContextItem[]): ContextItem[] {
    const sorted = [...items];

    switch (this.strategy) {
      case OptimizationStrategy.RECENCY:
        // Sort by timestamp (newest first)
        sorted.sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());
        break;

      case OptimizationStrategy.PRIORITY:
        // Sort by priority (highest first)
        sorted.sort((a, b) => b.priority - a.priority);
        break;

      case OptimizationStrategy.BALANCED:
        // Balance priority and recency
        sorted.sort((a, b) => {
          const aPriority = a.priority * 1000;
          const aRecency = a.timestamp.getTime() / 1000000;
          const aScore = aPriority + aRecency;

          const bPriority = b.priority * 1000;
          const bRecency = b.timestamp.getTime() / 1000000;
          const bScore = bPriority + bRecency;

          return bScore - aScore;
        });
        break;

      case OptimizationStrategy.DIVERSE:
        // Ensure diversity of context types
        sorted.sort((a, b) => {
          // Higher priority first
          if (a.priority !== b.priority) {
            return b.priority - a.priority;
          }
          // Then different types preferred
          if (a.type !== b.type) {
            return a.type.localeCompare(b.type);
          }
          // Finally by timestamp
          return b.timestamp.getTime() - a.timestamp.getTime();
        });
        break;

      default:
        // Default to balanced strategy
        break;
    }

    return sorted;
  }

  /**
   * Estimate token count for a single context item.
   */
  private estimateItemTokens(item: ContextItem): number {
    // Rough estimation: ~4 characters per token
    let text = '';

    if (typeof item.data === 'string') {
      text = item.data;
    } else if (typeof item.data === 'object' && item.data !== null) {
      text = JSON.stringify(item.data);
    }

    // Add overhead for metadata
    text += `Source: ${item.source}, Priority: ${item.priority}, Type: ${item.type}`;

    return Math.ceil(text.length / 4);
  }

  /**
   * Estimate total token count for a list of items.
   */
  private estimateTokens(items: ContextItem[]): number {
    return items.reduce((sum, item) => sum + this.estimateItemTokens(item), 0);
  }

  /**
   * Format context items as a structured prompt section.
   */
  private formatAsPrompt(items: ContextItem[]): string {
    if (items.length === 0) {
      return '';
    }

    let prompt = '# Relevant Context\n\n';
    prompt += `The following context has been gathered from previous agent executions:\n\n`;

    // Group by type for better organization
    const byType = new Map<ContextType, ContextItem[]>();
    items.forEach((item) => {
      if (!byType.has(item.type)) {
        byType.set(item.type, []);
      }
      byType.get(item.type)!.push(item);
    });

    // Format each group
    const typeNames: Record<ContextType, string> = {
      [ContextType.AGENT_OUTPUT]: 'Previous Agent Outputs',
      [ContextType.CODE_INSIGHTS]: 'Code Insights',
      [ContextType.ISSUES]: 'Identified Issues',
      [ContextType.TASK_METADATA]: 'Task Information',
      [ContextType.USER_PREFERENCES]: 'User Preferences',
      [ContextType.EXECUTION_HISTORY]: 'Execution History',
    };

    byType.forEach((groupItems, type) => {
      prompt += `## ${typeNames[type]}\n\n`;

      groupItems.forEach((item, idx) => {
        const priorityIcon =
          item.priority === ContextPriority.CRITICAL
            ? '🔴'
            : item.priority === ContextPriority.HIGH
              ? '🟡'
              : item.priority === ContextPriority.MEDIUM
                ? '🟢'
                : '⚪';

        prompt += `${idx + 1}. ${priorityIcon} **From ${item.source}**\n`;

        // Format data
        if (typeof item.data === 'string') {
          prompt += `   ${item.data}\n`;
        } else if (typeof item.data === 'object' && item.data !== null) {
          const formatted = this.formatData(item.data);
          prompt += `   ${formatted}\n`;
        }

        prompt += '\n';
      });
    });

    prompt += `---\n\n`;
    prompt += `Use the above context to inform your analysis and decisions. Prioritize items marked with 🔴 (CRITICAL) and 🟡 (HIGH).\n\n`;

    return prompt;
  }

  /**
   * Format data object as readable text.
   */
  private formatData(data: unknown): string {
    if (typeof data !== 'object' || data === null) {
      return String(data);
    }

    // For objects, create a compact representation
    const obj = data as Record<string, unknown>;
    const lines: string[] = [];

    for (const [key, value] of Object.entries(obj)) {
      if (
        typeof value === 'string' ||
        typeof value === 'number' ||
        typeof value === 'boolean'
      ) {
        lines.push(`${key}: ${value}`);
      } else if (Array.isArray(value)) {
        if (value.length <= 3) {
          lines.push(`${key}: [${value.join(', ')}]`);
        } else {
          lines.push(
            `${key}: [${value.slice(0, 3).join(', ')}, ... (${value.length} items)]`,
          );
        }
      } else if (typeof value === 'object' && value !== null) {
        lines.push(`${key}: ${JSON.stringify(value)}`);
      }
    }

    return lines.join('\n   ');
  }

  /**
   * Create a summary of multiple context items.
   */
  summarize(items: ContextItem[], maxLength: number = 500): string {
    if (items.length === 0) {
      return 'No context available.';
    }

    let summary = `Context from ${items.length} sources: `;

    // Group by source
    const bySource = new Map<string, ContextItem[]>();
    items.forEach((item) => {
      if (!bySource.has(item.source)) {
        bySource.set(item.source, []);
      }
      bySource.get(item.source)!.push(item);
    });

    const sourceSummaries: string[] = [];
    bySource.forEach((sourceItems, source) => {
      const types = new Set(sourceItems.map((i) => i.type));
      sourceSummaries.push(
        `${source} (${sourceItems.length} items, types: ${Array.from(types).join(', ')})`,
      );
    });

    summary += sourceSummaries.join('; ');

    // Truncate if too long
    if (summary.length > maxLength) {
      summary = summary.substring(0, maxLength - 3) + '...';
    }

    return summary;
  }
}
