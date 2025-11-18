/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 *
 * Multi-Agent Collaboration Examples
 *
 * This file contains comprehensive examples of using the multi-agent
 * collaboration system inspired by Claude Code.
 */

import {
  TaskDecomposer,
  MultiAgentCoordinator,
  CollaborativeExecutor,
  type Config,
  type DecompositionResult,
  type CoordinatorStatus,
  type CollaborativeResult,
} from '@google/gemini-cli-core';

// ============================================================================
// Example 1: Simple Task Decomposition
// ============================================================================

export async function example1_SimpleDecomposition() {
  console.log('Example 1: Simple Task Decomposition\n');

  const decomposer = new TaskDecomposer();

  // Decompose a complex task
  const result: DecompositionResult = decomposer.decompose(
    'Implement user authentication with JWT tokens and refresh token rotation',
    10, // max subtasks
  );

  console.log(`Task Type: ${result.classification.type}`);
  console.log(`Complexity: ${result.classification.complexity}`);
  console.log(`Estimated Duration: ${result.estimatedDuration}`);
  console.log(`\nSubtasks (${result.subtasks.length}):`);

  result.subtasks.forEach((subtask, index) => {
    console.log(`\n${index + 1}. ${subtask.title}`);
    console.log(`   Agent: ${subtask.recommendedAgents.join(', ')}`);
    console.log(`   Complexity: ${subtask.complexity}`);
    console.log(`   Can Parallelize: ${subtask.canParallelize}`);
    if (subtask.dependencies.length > 0) {
      console.log(`   Dependencies: ${subtask.dependencies.join(', ')}`);
    }
  });

  console.log(`\n\nExecution Plan (${result.executionPlan.length} stages):`);
  result.executionPlan.forEach((stage) => {
    console.log(
      `\nStage ${stage.stage} (${stage.canRunInParallel ? 'Parallel' : 'Sequential'}):`,
    );
    console.log(`  Subtasks: ${stage.subtasks.length}`);
  });
}

// ============================================================================
// Example 2: Multi-Agent Coordination
// ============================================================================

export async function example2_MultiAgentCoordination(config: Config) {
  console.log('Example 2: Multi-Agent Coordination\n');

  const coordinator = new MultiAgentCoordinator(config);

  // Track progress
  let lastProgress = 0;

  const status: CoordinatorStatus = await coordinator.execute(
    'Review the authentication module for security issues and performance problems',
    {
      maxSubtasks: 5,
      maxConcurrency: 3,
      onStatusUpdate: (status) => {
        if (status.progress !== lastProgress) {
          console.log(`Progress: ${status.progress}%`);
          console.log(
            `Stage: ${status.currentStage}/${status.totalStages}`,
          );
          lastProgress = status.progress;
        }
      },
      onAgentActivity: (subtaskId, agentName, event) => {
        if (event.type === 'TOOL_CALL_START') {
          console.log(
            `  [${agentName}] Calling tool: ${event.data.toolName}`,
          );
        }
      },
    },
  );

  console.log('\n\nFinal Status:');
  console.log(`Success: ${status.success}`);
  console.log(`Progress: ${status.progress}%`);
  console.log(`Duration: ${status.endTime && status.startTime ? (status.endTime.getTime() - status.startTime.getTime()) / 1000 : 0}s`);

  console.log('\n\nSubtask Results:');
  status.subtasks.forEach((st) => {
    console.log(
      `  ${st.status === 'completed' ? '✓' : st.status === 'failed' ? '✗' : '○'} ${st.subtask.title}`,
    );
    if (st.error) {
      console.log(`    Error: ${st.error}`);
    }
  });
}

// ============================================================================
// Example 3: Full Collaborative Execution
// ============================================================================

export async function example3_CollaborativeExecution(config: Config) {
  console.log('Example 3: Full Collaborative Execution\n');

  const executor = new CollaborativeExecutor(config);

  const result: CollaborativeResult = await executor.execute(
    'Refactor the user registration flow to use async/await and add comprehensive error handling',
    {
      maxSubtasks: 8,
      maxConcurrency: 3,
      enableContextSharing: true,
      enableMemory: true,
      onEvent: (event) => {
        switch (event.type) {
          case 'decomposition_complete':
            console.log(
              `✓ Task decomposed: ${(event.data as any).subtaskCount} subtasks`,
            );
            break;

          case 'stage_start':
            console.log(
              `→ Starting stage ${(event.data as any).stage}/${(event.data as any).totalStages}`,
            );
            break;

          case 'subtask_complete':
            console.log(
              `  ✓ ${(event.data as any).agentName} completed`,
            );
            break;

          case 'subtask_failed':
            console.log(
              `  ✗ ${(event.data as any).agentName} failed: ${(event.data as any).error}`,
            );
            break;

          case 'context_shared':
            console.log(
              `  ↔ Context shared by ${(event.data as any).agentName}`,
            );
            break;

          case 'execution_complete':
            console.log(
              `\n✓ Execution complete (${Math.round((event.data as any).duration / 1000)}s)`,
            );
            break;
        }
      },
    },
  );

  console.log('\n\nExecution Summary:');
  console.log(`Task: ${result.task}`);
  console.log(`Success: ${result.status.success}`);
  console.log(`Progress: ${result.status.progress}%`);
  console.log(`Events: ${result.events.length}`);

  if (result.contextSummary) {
    console.log('\n\nContext Summary:');
    console.log(result.contextSummary);
  }

  if (result.memorySummary) {
    console.log('\n\nMemory Summary:');
    console.log(result.memorySummary);
  }
}

// ============================================================================
// Example 4: Preview Mode
// ============================================================================

export async function example4_PreviewMode(config: Config) {
  console.log('Example 4: Preview Mode\n');

  const executor = new CollaborativeExecutor(config);

  // Preview without executing
  const result = await executor.execute(
    'Implement a complete CI/CD pipeline with GitHub Actions, Docker, and automated testing',
    {
      maxSubtasks: 12,
      previewOnly: true,
    },
  );

  console.log('Preview Results:');
  console.log(`Task: ${result.task}`);
  console.log(`Subtasks: ${result.status.subtasks.length}`);
  console.log(`Stages: ${result.status.totalStages}`);

  console.log('\n\nPlanned Subtasks:');
  result.status.subtasks.forEach((st, index) => {
    console.log(`\n${index + 1}. ${st.subtask.title}`);
    console.log(`   Agent: ${st.subtask.recommendedAgents.join(', ')}`);
    console.log(`   Complexity: ${st.subtask.complexity}`);
    console.log(`   Priority: ${st.subtask.priority}`);
  });
}

// ============================================================================
// Example 5: Context Sharing and Memory
// ============================================================================

export async function example5_ContextAndMemory(config: Config) {
  console.log('Example 5: Context Sharing and Memory\n');

  const executor = new CollaborativeExecutor(config);

  // Execute with context sharing
  const result = await executor.execute(
    'Find and fix all authentication bugs',
    {
      enableContextSharing: true,
      enableMemory: true,
      maxConcurrency: 2,
    },
  );

  // Access shared context
  const orchestrator = executor.getContextualOrchestrator();
  const sharedContext = orchestrator.getSharedContext();
  const memory = orchestrator.getMemory();

  console.log('\n\nShared Context:');
  const stats = sharedContext.getStats();
  console.log(`Total Items: ${stats.totalItems}`);
  console.log(`By Type:`);
  Object.entries(stats.byType).forEach(([type, count]) => {
    console.log(`  ${type}: ${count}`);
  });

  console.log('\n\nAgent Memory:');
  const memoryStats = memory.getStats();
  console.log(`Total Memories: ${memoryStats.totalMemories}`);
  console.log(
    `Working Memory: ${memoryStats.workingMemoryCount}`,
  );
  console.log(
    `Long-term Memory: ${memoryStats.longTermMemoryCount}`,
  );

  // Get relevant memories
  const relevantMemories = memory.getRelevant(['authentication', 'bug'], 5);
  console.log(`\n\nRelevant Memories (${relevantMemories.length}):`);
  relevantMemories.forEach((mem) => {
    console.log(`  - ${mem.content.slice(0, 100)}...`);
  });

  // Consolidate memories
  const consolidated = orchestrator.consolidateMemories(0.6);
  console.log(`\nConsolidated ${consolidated} important memories`);

  // Cleanup
  const cleanup = orchestrator.cleanup({
    maxContextAge: 3600000, // 1 hour
    minMemoryImportance: 0.3,
  });
  console.log(
    `Cleaned up ${cleanup.contextRemoved} context items and ${cleanup.memoriesRemoved} memories`,
  );
}

// ============================================================================
// Example 6: Custom Decomposition Strategy
// ============================================================================

export async function example6_CustomStrategy() {
  console.log('Example 6: Custom Decomposition Strategy\n');

  const decomposer = new TaskDecomposer();

  // Different task types show different decomposition strategies
  const tasks = [
    {
      task: 'Implement a new feature',
      type: 'Implementation',
    },
    {
      task: 'Refactor legacy code',
      type: 'Refactoring',
    },
    {
      task: 'Fix a critical bug',
      type: 'Debugging',
    },
    {
      task: 'Review code for issues',
      type: 'Code Review',
    },
  ];

  for (const { task, type } of tasks) {
    console.log(`\n${type}: "${task}"`);
    const result = decomposer.decompose(task, 10);

    console.log(`  Subtasks: ${result.subtasks.length}`);
    console.log(`  Stages: ${result.executionPlan.length}`);
    console.log(`  Duration: ${result.estimatedDuration}`);

    console.log('  Steps:');
    result.subtasks.forEach((st, i) => {
      console.log(`    ${i + 1}. ${st.title} (${st.recommendedAgents[0]})`);
    });
  }
}

// ============================================================================
// Example 7: Parallel vs Sequential Execution
// ============================================================================

export async function example7_ParallelVsSequential(config: Config) {
  console.log('Example 7: Parallel vs Sequential Execution\n');

  const coordinator = new MultiAgentCoordinator(config);

  // Task that benefits from parallelization
  console.log('Parallel Task: Code Review');
  const parallelStart = Date.now();
  await coordinator.execute(
    'Review all modules for security, performance, and test coverage',
    {
      maxConcurrency: 3, // Allow parallel execution
    },
  );
  const parallelDuration = Date.now() - parallelStart;
  console.log(`Parallel execution took: ${parallelDuration}ms`);

  // Task that requires sequential execution
  console.log('\n\nSequential Task: Implementation');
  const sequentialStart = Date.now();
  await coordinator.execute(
    'Implement user authentication step by step',
    {
      maxConcurrency: 1, // Force sequential
    },
  );
  const sequentialDuration = Date.now() - sequentialStart;
  console.log(
    `Sequential execution took: ${sequentialDuration}ms`,
  );
}

// ============================================================================
// Example 8: Error Handling and Recovery
// ============================================================================

export async function example8_ErrorHandling(config: Config) {
  console.log('Example 8: Error Handling and Recovery\n');

  const executor = new CollaborativeExecutor(config);

  const result = await executor.execute(
    'Process a complex task that might fail',
    {
      maxSubtasks: 6,
      onEvent: (event) => {
        if (event.type === 'subtask_failed') {
          console.log(
            `Subtask failed: ${(event.data as any).agentName}`,
          );
          console.log(`Error: ${(event.data as any).error}`);
        }
      },
    },
  );

  console.log('\n\nError Analysis:');
  const failed = result.status.subtasks.filter(
    (st) => st.status === 'failed',
  );
  const completed = result.status.subtasks.filter(
    (st) => st.status === 'completed',
  );

  console.log(`Total Subtasks: ${result.status.subtasks.length}`);
  console.log(`Completed: ${completed.length}`);
  console.log(`Failed: ${failed.length}`);
  console.log(`Success Rate: ${Math.round((completed.length / result.status.subtasks.length) * 100)}%`);

  if (failed.length > 0) {
    console.log('\n\nFailed Subtasks:');
    failed.forEach((st) => {
      console.log(`  - ${st.subtask.title}`);
      console.log(`    Error: ${st.error}`);
    });
  }
}

// ============================================================================
// Example 9: Formatting and Display
// ============================================================================

export async function example9_FormattingDisplay(config: Config) {
  console.log('Example 9: Formatting and Display\n');

  const coordinator = new MultiAgentCoordinator(config);

  // Get decomposition
  const decomposition = coordinator.preview(
    'Build a real-time chat application',
    10,
  );

  // Format decomposition
  const formattedDecomp = coordinator.formatDecomposition(decomposition);
  console.log('Decomposition Plan:');
  console.log(formattedDecomp);

  // Execute and format status
  const status = await coordinator.execute(
    'Build a real-time chat application',
    {
      maxSubtasks: 10,
      maxConcurrency: 3,
    },
  );

  const formattedStatus = coordinator.formatStatus(status);
  console.log('\n\nExecution Status:');
  console.log(formattedStatus);
}

// ============================================================================
// Example 10: Integration with Existing Agents
// ============================================================================

export async function example10_AgentIntegration(config: Config) {
  console.log('Example 10: Integration with Existing Agents\n');

  const executor = new CollaborativeExecutor(config);

  // Complex task that uses multiple specialized agents
  const result = await executor.execute(
    'Analyze the data processing pipeline, identify bottlenecks, and create optimization recommendations',
    {
      maxSubtasks: 8,
      maxConcurrency: 3,
      enableContextSharing: true,
      onEvent: (event) => {
        if (event.type === 'agent_activity') {
          const data = event.data as any;
          console.log(
            `[${data.agentName}] ${data.eventType}`,
          );
        }
      },
    },
  );

  console.log('\n\nAgents Used:');
  const agentsUsed = new Set<string>();
  result.status.subtasks.forEach((st) => {
    st.subtask.recommendedAgents.forEach((agent) => {
      agentsUsed.add(agent);
    });
  });

  agentsUsed.forEach((agent) => {
    console.log(`  - ${agent}`);
  });

  console.log(`\nTotal Unique Agents: ${agentsUsed.size}`);
}

// ============================================================================
// Run All Examples
// ============================================================================

export async function runAllExamples(config: Config) {
  console.log('='.repeat(60));
  console.log('Multi-Agent Collaboration Examples');
  console.log('='.repeat(60));
  console.log();

  try {
    await example1_SimpleDecomposition();
    console.log('\n' + '='.repeat(60) + '\n');

    await example2_MultiAgentCoordination(config);
    console.log('\n' + '='.repeat(60) + '\n');

    await example3_CollaborativeExecution(config);
    console.log('\n' + '='.repeat(60) + '\n');

    await example4_PreviewMode(config);
    console.log('\n' + '='.repeat(60) + '\n');

    await example5_ContextAndMemory(config);
    console.log('\n' + '='.repeat(60) + '\n');

    await example6_CustomStrategy();
    console.log('\n' + '='.repeat(60) + '\n');

    await example7_ParallelVsSequential(config);
    console.log('\n' + '='.repeat(60) + '\n');

    await example8_ErrorHandling(config);
    console.log('\n' + '='.repeat(60) + '\n');

    await example9_FormattingDisplay(config);
    console.log('\n' + '='.repeat(60) + '\n');

    await example10_AgentIntegration(config);
    console.log('\n' + '='.repeat(60) + '\n');

    console.log('All examples completed successfully!');
  } catch (error) {
    console.error('Error running examples:', error);
  }
}
