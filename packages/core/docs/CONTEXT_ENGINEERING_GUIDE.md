# Context Engineering Guide

This guide explains how to use the Context Engineering system in the Gemini CLI
multi-agent framework. The Context Engineering system enables agents to share
knowledge, maintain memory across executions, and build upon each other's work
for more effective collaboration.

## Table of Contents

1. [Overview](#overview)
2. [Core Components](#core-components)
3. [Quick Start](#quick-start)
4. [SharedContext](#sharedcontext)
5. [ContextOptimizer](#contextoptimizer)
6. [AgentMemory](#agentmemory)
7. [ContextualOrchestrator](#contextualorchestrator)
8. [Advanced Usage](#advanced-usage)
9. [Best Practices](#best-practices)
10. [Examples](#examples)

## Overview

The Context Engineering system consists of four main components:

- **SharedContext**: Centralized storage for sharing information between agents
- **ContextOptimizer**: Intelligent optimization of context within token limits
- **AgentMemory**: Persistent memory system inspired by human memory models
- **ContextualOrchestrator**: Context-aware agent orchestration and coordination

These components work together to enable multi-agent systems to maintain
continuity, learn from past executions, and make informed decisions based on
accumulated knowledge.

## Core Components

### Architecture

```
┌─────────────────────────────────────────────────────┐
│          ContextualOrchestrator                     │
│  (Coordinates agents with context awareness)        │
└──────────────┬────────────────────┬─────────────────┘
               │                    │
       ┌───────▼──────┐     ┌──────▼────────┐
       │ SharedContext│     │ AgentMemory   │
       │ (Live Data)  │     │ (Long-term)   │
       └──────┬───────┘     └───────┬───────┘
              │                     │
              └──────┬──────────────┘
                     │
              ┌──────▼──────────┐
              │ ContextOptimizer│
              │  (Optimizes)    │
              └─────────────────┘
```

## Quick Start

### Basic Usage

```typescript
import {
  ContextualOrchestrator,
  SharedContext,
  ContextType,
  ContextPriority,
  OptimizationStrategy,
} from '@google/gemini-cli-core';

// Create a context-aware orchestrator
const orchestrator = new ContextualOrchestrator({
  maxContextTokens: 4000,
  optimizationStrategy: OptimizationStrategy.BALANCED,
});

// Classify a task and create a strategy
const userMessage =
  'Find and fix all security vulnerabilities in the authentication module';
const strategy = orchestrator.createStrategy(userMessage);

console.log('Task type:', strategy.type);
console.log('Context items:', strategy.context.items);
console.log('Relevant memories:', strategy.memories.count);
console.log('Confidence:', strategy.confidence);

// Execute agents and record results
// (Agent execution code here)

// Record execution results
orchestrator.recordExecution('debug_agent', agentOutput, {
  issues: ['SQL injection vulnerability', 'Missing CSRF protection'],
  files: ['src/auth/login.ts', 'src/auth/session.ts'],
});

// Get statistics
const stats = orchestrator.getStats();
console.log('Context stats:', stats.context);
console.log('Memory stats:', stats.memory);
```

## SharedContext

SharedContext is a centralized store for sharing information between agents
during a session.

### Context Types

```typescript
export enum ContextType {
  AGENT_OUTPUT = 'agent_output', // Output from agent executions
  CODE_INSIGHTS = 'code_insights', // Discovered code patterns/structure
  ISSUES = 'issues', // Identified problems
  TASK_METADATA = 'task_metadata', // Task-related information
  USER_PREFERENCES = 'user_preferences', // User preferences
  EXECUTION_HISTORY = 'execution_history', // Past execution records
}
```

### Priority Levels

```typescript
export enum ContextPriority {
  LOW = 1, // Nice to have
  MEDIUM = 2, // Useful
  HIGH = 3, // Important
  CRITICAL = 4, // Must be included
}
```

### Usage Examples

```typescript
import {
  SharedContext,
  ContextType,
  ContextPriority,
} from '@google/gemini-cli-core';

const context = new SharedContext();

// Add agent output
const outputId = context.addAgentOutput('explore_agent', {
  terminate_reason: 'GOAL',
  result: 'Found 15 TypeScript files in src/core',
});

// Add code insights
const insightId = context.addCodeInsights('codebase_investigator', {
  files: ['src/index.ts', 'src/core/client.ts'],
  patterns: ['Factory pattern', 'Singleton pattern'],
});

// Add issues
const issueId = context.addIssues('code_reviewer', [
  {
    severity: 'HIGH',
    description: 'Potential memory leak in event handler',
    location: 'src/events.ts:42',
  },
]);

// Query context
const highPriorityItems = context.getHighPriority(ContextPriority.HIGH);
const codeInsights = context.get({ types: [ContextType.CODE_INSIGHTS] });
const recentIssues = context.get({
  types: [ContextType.ISSUES],
  tags: ['security'],
  limit: 5,
});

// Export summary
const summary = context.exportSummary({
  minPriority: ContextPriority.MEDIUM,
});
console.log(summary);

// Get statistics
const stats = context.getStats();
console.log('Total items:', stats.totalItems);
console.log('By type:', stats.byType);
console.log('By priority:', stats.byPriority);
```

## ContextOptimizer

ContextOptimizer intelligently selects and formats context items to fit within
token limits while maximizing relevance.

### Optimization Strategies

```typescript
export enum OptimizationStrategy {
  RECENCY = 'recency', // Prioritize recent items
  PRIORITY = 'priority', // Prioritize high-priority items
  BALANCED = 'balanced', // Balance recency and priority
  DIVERSE = 'diverse', // Ensure diversity of context types
}
```

### Usage Examples

```typescript
import {
  ContextOptimizer,
  OptimizationStrategy,
  SharedContext,
  ContextType,
} from '@google/gemini-cli-core';

const optimizer = new ContextOptimizer(
  4000, // Max tokens
  OptimizationStrategy.BALANCED,
);

const context = new SharedContext();
// ... add items to context ...

const items = context.get();

// Optimize for inclusion in prompt
const optimized = optimizer.optimize(items, {
  preserveTypes: [ContextType.ISSUES], // Always include issues
  maxItems: 10,
});

console.log('Selected items:', optimized.items.length);
console.log('Estimated tokens:', optimized.estimatedTokens);
console.log('Compression ratio:', optimized.stats.compressionRatio);

// Use the optimized prompt
const promptText = optimized.promptText;
// Add to agent input...

// Create a summary
const summary = optimizer.summarize(items, 300);
console.log(summary);
```

## AgentMemory

AgentMemory provides persistent memory across agent executions, inspired by
human memory models.

### Memory Types

```typescript
export enum MemoryType {
  WORKING = 'working', // Short-term (current session)
  LONG_TERM = 'long_term', // Persists across sessions
  EPISODIC = 'episodic', // Specific events/executions
  SEMANTIC = 'semantic', // Facts and knowledge
}
```

### Usage Examples

```typescript
import { AgentMemory, MemoryType } from '@google/gemini-cli-core';

const memory = new AgentMemory();

// Store working memory
const memId1 = memory.store(
  'explore_agent',
  'Discovered authentication system structure',
  {
    files: ['src/auth/login.ts', 'src/auth/session.ts'],
    complexity: 'medium',
  },
  {
    type: MemoryType.WORKING,
    tags: ['auth', 'architecture'],
    importance: 0.7,
  },
);

// Store agent execution as episodic memory
const memId2 = memory.storeAgentExecution('test_runner', agentOutput, {
  tags: ['tests', 'success'],
  importance: 0.8,
});

// Store a fact as semantic memory
const memId3 = memory.storeFact(
  'code_reviewer',
  'Project uses JWT for authentication',
  { standard: 'RFC 7519' },
  {
    tags: ['auth', 'security'],
    importance: 0.9,
  },
);

// Retrieve relevant memories
const authMemories = memory.getRelevant(['auth', 'security'], 5);
const workingMemory = memory.getWorkingMemory(10);

// Search by keywords
const results = memory.search(['authentication', 'JWT']);

// Consolidate important memories to long-term
const consolidated = memory.consolidate(0.7); // importance >= 0.7
console.log(`Consolidated ${consolidated} memories`);

// Forget unimportant memories
const forgotten = memory.forget(0.3); // importance < 0.3
console.log(`Forgot ${forgotten} memories`);

// Export summary for prompts
const memorySummary = memory.exportSummary({
  types: [MemoryType.SEMANTIC, MemoryType.EPISODIC],
  minImportance: 0.6,
  limit: 10,
});

// Get statistics
const stats = memory.getStats();
console.log('Total memories:', stats.totalMemories);
console.log('By type:', stats.byType);
console.log('Average importance:', stats.avgImportance);
```

## ContextualOrchestrator

ContextualOrchestrator extends AgentOrchestrator with context-awareness,
enabling intelligent agent coordination.

### Features

- Context-aware task classification
- Automatic context and memory injection into agent inputs
- Execution recording and knowledge extraction
- Memory consolidation and cleanup
- Confidence scoring for strategies

### Usage Examples

```typescript
import {
  ContextualOrchestrator,
  OptimizationStrategy,
  TaskType,
} from '@google/gemini-cli-core';

// Create orchestrator
const orchestrator = new ContextualOrchestrator({
  maxContextTokens: 4000,
  optimizationStrategy: OptimizationStrategy.BALANCED,
});

// Create a context-aware strategy
const userMessage = 'Refactor the user service to improve performance';
const strategy = orchestrator.createStrategy(userMessage);

console.log('Task type:', strategy.type); // TaskType.REFACTORING
console.log(
  'Agents:',
  strategy.agents.map((a) => a.agentName),
);
console.log('Context summary:', strategy.context.summary);
console.log('Memories:', strategy.memories.summary);
console.log('Confidence:', strategy.confidence);

// Enhance agent inputs with context
const classification = orchestrator.classifyTask(userMessage);
const baseInputs = { target: 'src/services/user.ts' };
const enhancedInputs = orchestrator.enhanceInputsWithContext(
  baseInputs,
  'refactor_agent',
  classification,
);

// Enhanced inputs now include:
// - _contextPrompt: Formatted context from previous executions
// - _memorySummary: Relevant memories
// - _contextMetadata: Metadata about context

// Execute agent (pseudo-code)
const agentOutput = await executeAgent('refactor_agent', enhancedInputs);

// Record execution results
const result = orchestrator.recordExecution('refactor_agent', agentOutput, {
  files: ['src/services/user.ts'],
  patterns: ['Caching pattern', 'Lazy loading'],
});

console.log('Memories created:', result.memoriesCreated);
console.log('Insights:', result.insights);

// Periodic maintenance
const consolidated = orchestrator.consolidateMemories(0.7);
console.log(`Consolidated ${consolidated} memories`);

const cleanup = orchestrator.cleanup({
  maxContextAge: 3600000, // 1 hour in ms
  minMemoryImportance: 0.3,
});
console.log('Cleaned up:', cleanup);

// Export state for debugging
const state = orchestrator.exportState();
console.log('Context:', state.context);
console.log('Memory:', state.memory);
console.log('Statistics:', state.stats);

// Access underlying components
const sharedContext = orchestrator.getSharedContext();
const memory = orchestrator.getMemory();
```

## Advanced Usage

### Custom Context Integration

```typescript
// Manually add context during agent execution
const orchestrator = new ContextualOrchestrator();
const context = orchestrator.getSharedContext();

// Agent discovers important information
context.add(
  ContextType.CODE_INSIGHTS,
  {
    pattern: 'Observer pattern',
    files: ['src/events.ts', 'src/listeners.ts'],
    complexity: 'high',
  },
  {
    source: 'pattern_detector',
    priority: ContextPriority.HIGH,
    tags: ['architecture', 'patterns'],
  },
);

// Later agents can access this context automatically
const strategy = orchestrator.createStrategy('Explain the event system');
// Strategy will include the observer pattern insight
```

### Memory Importance Tuning

```typescript
const memory = new AgentMemory();

// Store with custom importance calculation
function calculateImportance(output: OutputObject): number {
  let importance = 0.5;

  // Boost importance for successful executions
  if (output.terminate_reason === 'GOAL') {
    importance += 0.2;
  }

  // Boost for critical findings
  if (output.result?.includes('security')) {
    importance += 0.3;
  }

  return Math.min(1.0, importance);
}

const memoryId = memory.storeAgentExecution('security_scanner', output, {
  importance: calculateImportance(output),
  tags: ['security', 'scan'],
});

// Update importance based on access patterns
memory.updateImportance(memoryId);
```

### Multi-Stage Orchestration

```typescript
const orchestrator = new ContextualOrchestrator();

// Stage 1: Exploration
const exploreStrategy = orchestrator.createStrategy(
  'Understand the authentication system',
);

// Execute exploration agent
const exploreOutput = await executeAgent(
  exploreStrategy.agents[0].agentName,
  exploreStrategy.agents[0].inputs,
);

// Record results
orchestrator.recordExecution('explore_agent', exploreOutput, {
  files: ['src/auth/login.ts', 'src/auth/session.ts'],
  patterns: ['JWT authentication', 'Session management'],
});

// Stage 2: Code Review (benefits from exploration context)
const reviewStrategy = orchestrator.createStrategy(
  'Review authentication code for security issues',
);

// Review strategy automatically includes context from exploration
console.log('Context items:', reviewStrategy.context.items); // > 0

const reviewOutput = await executeAgent(
  reviewStrategy.agents[0].agentName,
  reviewStrategy.agents[0].inputs,
);

orchestrator.recordExecution('code_reviewer', reviewOutput, {
  issues: ['Missing rate limiting', 'Weak password policy'],
});

// Stage 3: Fix (benefits from both previous stages)
const fixStrategy = orchestrator.createStrategy(
  'Fix the security issues in authentication',
);

// Fix strategy has context from exploration and review
console.log('Memories:', fixStrategy.memories.count); // > 0
```

## Best Practices

### 1. Context Cleanup

Regularly clean up old or unimportant context to avoid memory bloat:

```typescript
// Every hour or after major task completion
setInterval(() => {
  orchestrator.cleanup({
    maxContextAge: 3600000, // 1 hour
    minMemoryImportance: 0.3,
  });
}, 3600000);
```

### 2. Memory Consolidation

Consolidate important working memory to long-term storage:

```typescript
// After completing a major task
const consolidated = orchestrator.consolidateMemories(0.7);
console.log(`Consolidated ${consolidated} important memories`);
```

### 3. Importance Scoring

Use consistent importance scoring across your system:

```typescript
const IMPORTANCE_LEVELS = {
  CRITICAL_FINDING: 0.9, // Security issues, bugs
  IMPORTANT_INSIGHT: 0.7, // Architecture, patterns
  USEFUL_INFO: 0.5, // General information
  ROUTINE: 0.3, // Routine operations
};
```

### 4. Context Types

Use appropriate context types for different information:

- `AGENT_OUTPUT`: Raw agent execution results
- `CODE_INSIGHTS`: Discovered patterns, architecture
- `ISSUES`: Bugs, vulnerabilities, problems
- `TASK_METADATA`: Task-related information
- `USER_PREFERENCES`: User-specified preferences
- `EXECUTION_HISTORY`: Historical execution records

### 5. Optimization Strategy Selection

Choose optimization strategy based on use case:

- `RECENCY`: For rapidly changing codebases
- `PRIORITY`: For critical-issue focused tasks
- `BALANCED`: General-purpose (recommended default)
- `DIVERSE`: For comprehensive analysis tasks

### 6. Tag Consistently

Use consistent tagging for better memory retrieval:

```typescript
const STANDARD_TAGS = {
  security: ['security', 'vulnerability', 'auth'],
  performance: ['performance', 'optimization', 'speed'],
  architecture: ['architecture', 'design', 'patterns'],
  testing: ['test', 'testing', 'qa'],
};

memory.store(agent, summary, data, {
  tags: STANDARD_TAGS.security,
});
```

## Examples

### Example 1: Multi-Agent Bug Fix

```typescript
const orchestrator = new ContextualOrchestrator();

// Step 1: Explore codebase
const exploreOutput = await executeAgent('explore_agent', {
  query: 'Find all authentication-related files',
});

orchestrator.recordExecution('explore_agent', exploreOutput, {
  files: exploreOutput.result.files,
});

// Step 2: Debug (with context from exploration)
const debugStrategy = orchestrator.createStrategy('Debug login failure issue');

const debugOutput = await executeAgent(
  'debug_agent',
  debugStrategy.agents[0].inputs,
);

orchestrator.recordExecution('debug_agent', debugOutput, {
  issues: ['Null pointer in session handler'],
  files: ['src/auth/session.ts'],
});

// Step 3: Fix (with context from both previous steps)
const fixStrategy = orchestrator.createStrategy('Fix the session handler bug');

// Execute fix with full context
```

### Example 2: Code Quality Improvement

```typescript
const orchestrator = new ContextualOrchestrator();

// Review code
const reviewOutput = await executeAgent('code_reviewer', {
  files: 'src/**/*.ts',
  focus: 'all',
});

orchestrator.recordExecution('code_reviewer', reviewOutput, {
  issues: reviewOutput.result.issues,
  patterns: reviewOutput.result.patterns,
});

// Refactor based on review context
const refactorStrategy = orchestrator.createStrategy(
  'Refactor code based on review findings',
);

// Strategy includes context from review
const refactorOutput = await executeAgent(
  'refactor_agent',
  refactorStrategy.agents[0].inputs,
);

orchestrator.recordExecution('refactor_agent', refactorOutput);

// Consolidate learnings
orchestrator.consolidateMemories(0.7);
```

### Example 3: Long-Running Session

```typescript
const orchestrator = new ContextualOrchestrator();

async function executeTaskWithContext(taskDescription: string) {
  // Get strategy with accumulated context
  const strategy = orchestrator.createStrategy(taskDescription);

  console.log(`Confidence: ${strategy.confidence}`);
  console.log(`Context items: ${strategy.context.items}`);
  console.log(`Memories: ${strategy.memories.count}`);

  // Execute with context
  const output = await executeAgent(
    strategy.agents[0].agentName,
    strategy.agents[0].inputs,
  );

  // Record for future tasks
  orchestrator.recordExecution(strategy.agents[0].agentName, output);

  return output;
}

// Task 1
await executeTaskWithContext('Explore the codebase');

// Task 2 (benefits from Task 1 context)
await executeTaskWithContext('Find performance issues');

// Task 3 (benefits from Task 1 & 2 context)
await executeTaskWithContext('Optimize slow functions');

// Periodic cleanup
orchestrator.cleanup({
  maxContextAge: 7200000, // 2 hours
  minMemoryImportance: 0.4,
});

// View accumulated knowledge
const state = orchestrator.exportState();
console.log('Total context items:', state.stats.context.totalItems);
console.log('Total memories:', state.stats.memory.totalMemories);
```

## Conclusion

The Context Engineering system enables sophisticated multi-agent collaboration
by providing:

1. **Shared Knowledge**: Agents can share discoveries through SharedContext
2. **Long-term Memory**: Important information persists across executions via
   AgentMemory
3. **Intelligent Optimization**: ContextOptimizer ensures relevant context fits
   within limits
4. **Smart Orchestration**: ContextualOrchestrator coordinates agents with
   context awareness

By leveraging these components, you can build multi-agent systems that learn
from experience, maintain continuity across tasks, and make increasingly
informed decisions.

For more examples and patterns, see the agent implementations in
`packages/core/src/agents/`.
