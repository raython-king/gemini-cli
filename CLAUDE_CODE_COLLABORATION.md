# Multi-Agent Collaboration System

## Overview

Gemini CLI now features a powerful multi-agent collaboration system inspired by Claude Code, enabling intelligent task decomposition and coordinated execution across multiple specialized agents.

## Key Features

### 1. **Intelligent Task Decomposition**

The system automatically analyzes complex tasks and breaks them down into manageable subtasks:

```typescript
import { TaskDecomposer } from '@google/gemini-cli-core';

const decomposer = new TaskDecomposer();
const result = decomposer.decompose(
  'Implement user authentication with JWT tokens',
);

console.log(`Subtasks: ${result.subtasks.length}`);
console.log(`Stages: ${result.executionPlan.length}`);
console.log(`Estimated duration: ${result.estimatedDuration}`);
```

**Features:**
- Automatic complexity assessment
- Dependency tracking between subtasks
- Parallel execution planning
- Smart agent selection per subtask

### 2. **Multi-Agent Coordination**

Coordinate multiple agents working together on decomposed tasks:

```typescript
import { MultiAgentCoordinator } from '@google/gemini-cli-core';

const coordinator = new MultiAgentCoordinator(runtimeContext);

const status = await coordinator.execute(
  'Refactor the authentication module',
  {
    maxSubtasks: 10,
    maxConcurrency: 3,
    onStatusUpdate: (status) => {
      console.log(`Progress: ${status.progress}%`);
    },
  },
);
```

**Features:**
- Stage-based execution (sequential and parallel)
- Real-time progress tracking
- Automatic dependency resolution
- Failure handling and recovery

### 3. **Collaborative Execution**

High-level executor that combines decomposition, coordination, and context sharing:

```typescript
import { CollaborativeExecutor } from '@google/gemini-cli-core';

const executor = new CollaborativeExecutor(runtimeContext);

const result = await executor.execute(
  'Implement and test new user registration flow',
  {
    maxSubtasks: 8,
    maxConcurrency: 3,
    enableContextSharing: true,
    enableMemory: true,
    onEvent: (event) => {
      console.log(`Event: ${event.type}`, event.data);
    },
  },
);
```

**Features:**
- Context sharing between agents
- Agent memory system
- Event-driven architecture
- Comprehensive execution tracking

## CLI Usage

Use the `collaborate` command for quick multi-agent execution:

```bash
# Execute a complex task
gemini collaborate "implement user authentication with JWT"

# Preview the decomposition without executing
gemini collaborate --preview "refactor the login module"

# Control parallelism
gemini collaborate --max-concurrency 5 "review all code in src/"

# Limit subtask count
gemini collaborate --max-subtasks 5 "debug authentication issues"

# Verbose mode for detailed output
gemini collaborate --verbose "add comprehensive tests"
```

### Command Options

- `--preview` - Show decomposition without executing
- `--max-subtasks <n>` - Maximum number of subtasks (default: 10)
- `--max-concurrency <n>` - Max concurrent agents (default: 3)
- `--enable-context` - Enable context sharing (default: true)
- `--enable-memory` - Enable agent memory (default: true)
- `--verbose` - Show detailed agent activity

## Architecture

### Task Decomposition Flow

```
User Task
    ↓
Task Classification (type, complexity)
    ↓
Task Decomposition (subtasks, dependencies)
    ↓
Execution Planning (stages, parallelization)
    ↓
Agent Selection (per subtask)
    ↓
Parallel/Sequential Execution
    ↓
Result Aggregation
```

### Decomposition Strategies

The system uses different strategies based on task type:

#### Implementation Tasks
1. **Planning** - Create implementation plan
2. **Exploration** - Understand existing code
3. **Implementation** - Build the feature
4. **Code Review** - Review for quality/security
5. **Testing** - Run and fix tests

#### Refactoring Tasks
1. **Analysis** - Understand current code
2. **Code Review** - Identify quality issues
3. **Refactoring** - Apply improvements
4. **Testing** - Verify no regressions

#### Debugging Tasks
1. **Exploration** - Understand bug context
2. **Debugging** - Find and fix the bug
3. **Testing** - Verify the fix

#### Code Review Tasks
- **Security Review** (parallel)
- **Performance Review** (parallel)
- **Test Coverage Review** (parallel)

## Advanced Features

### Context Sharing

Agents automatically share context and insights:

```typescript
const executor = new CollaborativeExecutor(runtimeContext);
const orchestrator = executor.getContextualOrchestrator();

// Agents automatically populate shared context
await executor.execute('complex task', {
  enableContextSharing: true,
});

// Access shared context
const context = orchestrator.getSharedContext();
const insights = context.get({ types: ['CODE_INSIGHTS'] });
```

### Agent Memory

The system maintains memory across executions:

```typescript
const orchestrator = executor.getContextualOrchestrator();
const memory = orchestrator.getMemory();

// Store facts
memory.storeFact(
  'agent_name',
  'Important insight about the codebase',
  { files: ['src/auth.ts'] },
);

// Retrieve relevant memories
const relevant = memory.getRelevant(['authentication', 'security'], 10);

// Consolidate important memories
const consolidated = orchestrator.consolidateMemories(0.7);
```

### Custom Event Handling

Monitor execution with detailed events:

```typescript
await executor.execute(task, {
  onEvent: (event) => {
    switch (event.type) {
      case 'decomposition_complete':
        console.log('Task decomposed:', event.data);
        break;

      case 'stage_start':
        console.log('Starting stage:', event.data);
        break;

      case 'subtask_complete':
        console.log('Subtask done:', event.data);
        break;

      case 'agent_activity':
        console.log('Agent activity:', event.data);
        break;

      case 'context_shared':
        console.log('Context updated:', event.data);
        break;
    }
  },
});
```

## Example Workflows

### 1. Full Feature Implementation

```bash
gemini collaborate "implement real-time notifications using WebSockets"
```

This will:
1. Create a plan
2. Explore existing architecture
3. Guide implementation steps
4. Review the code
5. Run tests

### 2. Comprehensive Code Review

```bash
gemini collaborate "review all authentication code for security and performance"
```

This will run parallel reviews for:
- Security vulnerabilities
- Performance issues
- Test coverage

### 3. Bug Investigation and Fix

```bash
gemini collaborate "debug why login fails on iOS Safari"
```

This will:
1. Explore related code
2. Debug the issue
3. Apply a fix
4. Test the fix

### 4. Large-Scale Refactoring

```bash
gemini collaborate "refactor the entire user module to use TypeScript strict mode"
```

This will:
1. Analyze current code
2. Identify quality issues
3. Apply refactorings
4. Run comprehensive tests

## Comparison with Claude Code

| Feature                      | Gemini CLI Multi-Agent | Claude Code |
|------------------------------|------------------------|-------------|
| Automatic Task Decomposition | ✅                     | ✅          |
| Parallel Agent Execution     | ✅                     | ✅          |
| Context Sharing              | ✅                     | ✅          |
| Agent Memory                 | ✅                     | ✅          |
| Specialized Agents           | 9 agents               | ~10+ agents |
| CLI Integration              | ✅ `collaborate`       | Built-in    |
| Event-Driven Architecture    | ✅                     | ✅          |
| Dependency Resolution        | ✅                     | ✅          |
| Stage-Based Execution        | ✅                     | ✅          |
| Real-Time Progress Tracking  | ✅                     | ✅          |

## Best Practices

### 1. Use for Complex Tasks

Multi-agent collaboration is best for:
- ✅ Feature implementations
- ✅ Large refactorings
- ✅ Comprehensive reviews
- ✅ Complex debugging

Avoid for:
- ❌ Simple queries
- ❌ Single-file changes
- ❌ Quick explorations

### 2. Preview First

Always preview complex tasks:

```bash
gemini collaborate --preview "your complex task"
```

This shows the execution plan before running.

### 3. Adjust Concurrency

For resource-intensive tasks:

```bash
gemini collaborate --max-concurrency 2 "memory-intensive task"
```

For fast, parallel tasks:

```bash
gemini collaborate --max-concurrency 5 "review all modules"
```

### 4. Limit Subtasks

For very complex tasks, limit subtasks to avoid overwhelming the system:

```bash
gemini collaborate --max-subtasks 5 "huge refactoring"
```

### 5. Enable Verbose Mode

For debugging or understanding execution:

```bash
gemini collaborate --verbose "task"
```

## Programmatic API

### Basic Usage

```typescript
import { CollaborativeExecutor } from '@google/gemini-cli-core';

const executor = new CollaborativeExecutor(config);

const result = await executor.execute('your task', {
  maxSubtasks: 10,
  maxConcurrency: 3,
  enableContextSharing: true,
  enableMemory: true,
});

console.log(`Success: ${result.status.success}`);
console.log(`Progress: ${result.status.progress}%`);
```

### Advanced Usage

```typescript
import {
  TaskDecomposer,
  MultiAgentCoordinator,
  CollaborativeExecutor,
} from '@google/gemini-cli-core';

// Manual decomposition
const decomposer = new TaskDecomposer();
const decomposition = decomposer.decompose('complex task');

// Manual coordination
const coordinator = new MultiAgentCoordinator(config);
const status = await coordinator.execute('task', {
  onStatusUpdate: (s) => console.log(s),
  onAgentActivity: (id, name, event) => console.log(event),
});

// Full collaboration with context
const executor = new CollaborativeExecutor(config);
const result = await executor.execute('task', {
  enableContextSharing: true,
  onEvent: (e) => handleEvent(e),
});

// Access shared context and memory
const orchestrator = executor.getContextualOrchestrator();
const context = orchestrator.getSharedContext();
const memory = orchestrator.getMemory();
```

## Troubleshooting

### Task Takes Too Long

- Reduce `--max-subtasks`
- Increase `--max-concurrency` for parallel tasks
- Preview first to understand the plan

### Agents Fail

- Check agent logs with `--verbose`
- Review subtask dependencies
- Simplify the task or break it down manually

### Memory Issues

- Reduce `--max-concurrency`
- Limit `--max-subtasks`
- Clean up context: `orchestrator.cleanup()`

### Context Not Shared

- Ensure `--enable-context` is true
- Check that agents are completing successfully
- Use `--verbose` to see context sharing events

## Future Enhancements

- [ ] Agent-to-agent direct communication
- [ ] Dynamic agent creation
- [ ] Persistent memory across sessions
- [ ] Learning from past executions
- [ ] UI dashboard for monitoring
- [ ] Custom decomposition strategies
- [ ] Agent performance metrics

## License

Copyright 2025 Google LLC. Licensed under Apache-2.0.
