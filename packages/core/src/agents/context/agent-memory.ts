/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type { OutputObject } from '../types.js';

/**
 * Types of memories that can be stored.
 */
export enum MemoryType {
  /** Short-term working memory (current session) */
  WORKING = 'working',
  /** Long-term memory (persists across sessions) */
  LONG_TERM = 'long_term',
  /** Episodic memory (specific events/executions) */
  EPISODIC = 'episodic',
  /** Semantic memory (facts and knowledge) */
  SEMANTIC = 'semantic',
}

/**
 * A single memory entry.
 */
export interface Memory {
  id: string;
  type: MemoryType;
  timestamp: Date;
  /** Agent that created this memory */
  agent: string;
  /** Summary of the memory */
  summary: string;
  /** Detailed content */
  content: unknown;
  /** Associated tags for retrieval */
  tags: string[];
  /** Importance score (0-1) */
  importance: number;
  /** How many times this memory has been accessed */
  accessCount: number;
  /** Last access timestamp */
  lastAccessed?: Date;
}

/**
 * Query parameters for memory retrieval.
 */
export interface MemoryQuery {
  /** Filter by memory type */
  types?: MemoryType[];
  /** Filter by tags (any match) */
  tags?: string[];
  /** Filter by minimum importance */
  minImportance?: number;
  /** Filter by agent */
  agents?: string[];
  /** Filter by date range */
  since?: Date;
  until?: Date;
  /** Maximum number of results */
  limit?: number;
  /** Sort order */
  sortBy?: 'timestamp' | 'importance' | 'accessCount';
  sortOrder?: 'asc' | 'desc';
}

/**
 * AgentMemory - Manages persistent memory across agent executions.
 *
 * This class implements a memory system inspired by human memory models,
 * with working memory, long-term memory, and retrieval mechanisms.
 */
export class AgentMemory {
  private memories: Map<string, Memory> = new Map();
  private nextId = 0;

  /**
   * Store a new memory.
   */
  store(
    agent: string,
    summary: string,
    content: unknown,
    options: {
      type?: MemoryType;
      tags?: string[];
      importance?: number;
    } = {},
  ): string {
    const id = `mem_${this.nextId++}_${Date.now()}`;
    const memory: Memory = {
      id,
      type: options.type ?? MemoryType.WORKING,
      timestamp: new Date(),
      agent,
      summary,
      content,
      tags: options.tags ?? [],
      importance: options.importance ?? 0.5,
      accessCount: 0,
    };

    this.memories.set(id, memory);
    return id;
  }

  /**
   * Store agent execution output as episodic memory.
   */
  storeAgentExecution(
    agentName: string,
    output: OutputObject,
    options: {
      tags?: string[];
      importance?: number;
    } = {},
  ): string {
    return this.store(
      agentName,
      `Execution completed: ${output.terminate_reason}`,
      output,
      {
        type: MemoryType.EPISODIC,
        tags: [...(options.tags ?? []), 'execution', agentName],
        importance: options.importance ?? 0.7,
      },
    );
  }

  /**
   * Store a discovered fact as semantic memory.
   */
  storeFact(
    agent: string,
    fact: string,
    details?: unknown,
    options: {
      tags?: string[];
      importance?: number;
    } = {},
  ): string {
    return this.store(agent, fact, details, {
      type: MemoryType.SEMANTIC,
      tags: [...(options.tags ?? []), 'fact'],
      importance: options.importance ?? 0.6,
    });
  }

  /**
   * Retrieve memories matching the query.
   */
  retrieve(query: MemoryQuery = {}): Memory[] {
    let results = Array.from(this.memories.values());

    // Filter by type
    if (query.types && query.types.length > 0) {
      results = results.filter((mem) => query.types!.includes(mem.type));
    }

    // Filter by tags
    if (query.tags && query.tags.length > 0) {
      results = results.filter((mem) =>
        query.tags!.some((tag) => mem.tags.includes(tag)),
      );
    }

    // Filter by importance
    if (query.minImportance !== undefined) {
      results = results.filter((mem) => mem.importance >= query.minImportance!);
    }

    // Filter by agent
    if (query.agents && query.agents.length > 0) {
      results = results.filter((mem) => query.agents!.includes(mem.agent));
    }

    // Filter by date range
    if (query.since) {
      results = results.filter((mem) => mem.timestamp >= query.since!);
    }
    if (query.until) {
      results = results.filter((mem) => mem.timestamp <= query.until!);
    }

    // Sort results
    const sortBy = query.sortBy ?? 'importance';
    const sortOrder = query.sortOrder ?? 'desc';

    results.sort((a, b) => {
      let comparison = 0;

      switch (sortBy) {
        case 'timestamp':
          comparison = a.timestamp.getTime() - b.timestamp.getTime();
          break;
        case 'importance':
          comparison = a.importance - b.importance;
          break;
        case 'accessCount':
          comparison = a.accessCount - b.accessCount;
          break;
        default:
          comparison = 0;
          break;
      }

      return sortOrder === 'asc' ? comparison : -comparison;
    });

    // Update access counts
    results.forEach((mem) => {
      mem.accessCount++;
      mem.lastAccessed = new Date();
    });

    // Apply limit
    if (query.limit) {
      results = results.slice(0, query.limit);
    }

    return results;
  }

  /**
   * Get the most relevant memories for a given context.
   */
  getRelevant(contextTags: string[], limit: number = 5): Memory[] {
    return this.retrieve({
      tags: contextTags,
      sortBy: 'importance',
      sortOrder: 'desc',
      limit,
    });
  }

  /**
   * Get recent working memory.
   */
  getWorkingMemory(limit: number = 10): Memory[] {
    return this.retrieve({
      types: [MemoryType.WORKING],
      sortBy: 'timestamp',
      sortOrder: 'desc',
      limit,
    });
  }

  /**
   * Consolidate working memory into long-term memory.
   *
   * This simulates the process of memory consolidation, where important
   * short-term memories are converted to long-term storage.
   */
  consolidate(importanceThreshold: number = 0.6): number {
    const workingMemories = Array.from(this.memories.values()).filter(
      (mem) =>
        mem.type === MemoryType.WORKING &&
        mem.importance >= importanceThreshold,
    );

    workingMemories.forEach((mem) => {
      mem.type = MemoryType.LONG_TERM;
    });

    return workingMemories.length;
  }

  /**
   * Forget (remove) low-importance memories.
   *
   * This helps manage memory size by removing less important items.
   */
  forget(importanceThreshold: number = 0.3): number {
    const toForget = Array.from(this.memories.entries()).filter(
      ([, mem]) => mem.importance < importanceThreshold && mem.accessCount < 2,
    );

    toForget.forEach(([id]) => this.memories.delete(id));

    return toForget.length;
  }

  /**
   * Update the importance of a memory based on access patterns.
   *
   * Memories that are accessed frequently become more important.
   */
  updateImportance(memoryId: string): void {
    const memory = this.memories.get(memoryId);
    if (!memory) return;

    // Increase importance based on access count
    const accessBonus = Math.min(0.2, memory.accessCount * 0.02);
    memory.importance = Math.min(1.0, memory.importance + accessBonus);
  }

  /**
   * Search memories by content similarity (simple keyword matching).
   */
  search(keywords: string[]): Memory[] {
    const lowerKeywords = keywords.map((k) => k.toLowerCase());

    const matches = Array.from(this.memories.values()).filter((mem) => {
      const summaryLower = mem.summary.toLowerCase();
      const contentStr =
        typeof mem.content === 'string'
          ? mem.content.toLowerCase()
          : JSON.stringify(mem.content).toLowerCase();

      return lowerKeywords.some(
        (keyword) =>
          summaryLower.includes(keyword) || contentStr.includes(keyword),
      );
    });

    // Sort by importance
    matches.sort((a, b) => b.importance - a.importance);

    return matches;
  }

  /**
   * Export memories as a summary for use in prompts.
   */
  exportSummary(query?: MemoryQuery): string {
    const memories = this.retrieve(query);

    if (memories.length === 0) {
      return 'No relevant memories found.';
    }

    let summary = `# Memory Recall (${memories.length} memories)\n\n`;

    memories.forEach((mem, idx) => {
      const importanceStars = '★'.repeat(Math.ceil(mem.importance * 5));
      summary += `${idx + 1}. [${mem.type}] ${importanceStars} **${mem.summary}**\n`;
      summary += `   Agent: ${mem.agent}, Accessed: ${mem.accessCount} times\n`;

      if (typeof mem.content === 'string' && mem.content.length < 200) {
        summary += `   Content: ${mem.content}\n`;
      } else if (typeof mem.content === 'object' && mem.content !== null) {
        const contentStr = JSON.stringify(mem.content);
        if (contentStr.length < 200) {
          summary += `   ${contentStr}\n`;
        }
      }

      summary += '\n';
    });

    return summary;
  }

  /**
   * Get statistics about the memory store.
   */
  getStats(): {
    totalMemories: number;
    byType: Record<string, number>;
    byAgent: Record<string, number>;
    avgImportance: number;
    avgAccessCount: number;
  } {
    const memories = Array.from(this.memories.values());

    const byType: Record<string, number> = {};
    const byAgent: Record<string, number> = {};
    let totalImportance = 0;
    let totalAccess = 0;

    memories.forEach((mem) => {
      byType[mem.type] = (byType[mem.type] || 0) + 1;
      byAgent[mem.agent] = (byAgent[mem.agent] || 0) + 1;
      totalImportance += mem.importance;
      totalAccess += mem.accessCount;
    });

    return {
      totalMemories: memories.length,
      byType,
      byAgent,
      avgImportance:
        memories.length > 0 ? totalImportance / memories.length : 0,
      avgAccessCount: memories.length > 0 ? totalAccess / memories.length : 0,
    };
  }

  /**
   * Clear all memories (useful for testing or reset).
   */
  clear(): void {
    this.memories.clear();
  }
}
