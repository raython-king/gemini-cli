/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Practical examples of using the Context Engineering system.
 *
 * These examples demonstrate how to use SharedContext, ContextOptimizer,
 * AgentMemory, and ContextualOrchestrator in real-world scenarios.
 */

import {
  SharedContext,
  ContextType,
  ContextPriority,
  ContextOptimizer,
  OptimizationStrategy,
  AgentMemory,
  MemoryType,
  ContextualOrchestrator,
  type OutputObject,
} from '../src/index.js';

// ============================================================================
// Example 1: Basic SharedContext Usage
// ============================================================================

export function example1_BasicSharedContext() {
  console.log('\n=== Example 1: Basic SharedContext Usage ===\n');

  const context = new SharedContext();

  // Add agent output
  const output: OutputObject = {
    terminate_reason: 'GOAL',
    result: 'Found 15 TypeScript files in src/core directory',
  };

  const outputId = context.addAgentOutput('explore_agent', output);
  console.log('Added agent output:', outputId);

  // Add code insights
  const insightId = context.addCodeInsights('codebase_investigator', {
    files: ['src/index.ts', 'src/core/client.ts', 'src/agents/executor.ts'],
    patterns: ['Factory pattern', 'Observer pattern', 'Dependency injection'],
    architecture: 'Modular with clear separation of concerns',
  });
  console.log('Added code insights:', insightId);

  // Add issues
  const issueId = context.addIssues('code_reviewer', [
    {
      severity: 'HIGH',
      description: 'Potential memory leak in event handler',
      location: 'src/events.ts:42',
      suggestion: 'Add cleanup in component unmount',
    },
    {
      severity: 'MEDIUM',
      description: 'Missing error handling in async function',
      location: 'src/core/client.ts:128',
    },
  ]);
  console.log('Added issues:', issueId);

  // Query context
  console.log('\nQuerying context:');

  const highPriorityItems = context.getHighPriority(ContextPriority.HIGH);
  console.log('High priority items:', highPriorityItems.length);

  const codeInsights = context.get({ types: [ContextType.CODE_INSIGHTS] });
  console.log('Code insights:', codeInsights.length);

  const issues = context.get({
    types: [ContextType.ISSUES],
    minPriority: ContextPriority.MEDIUM,
  });
  console.log('Issues (medium+):', issues.length);

  // Export summary
  console.log('\nContext Summary:');
  console.log(context.exportSummary({ minPriority: ContextPriority.LOW }));

  // Get statistics
  const stats = context.getStats();
  console.log('\nStatistics:');
  console.log('Total items:', stats.totalItems);
  console.log('By type:', stats.byType);
  console.log('By priority:', stats.byPriority);
}

// ============================================================================
// Example 2: Context Optimization
// ============================================================================

export function example2_ContextOptimization() {
  console.log('\n=== Example 2: Context Optimization ===\n');

  const context = new SharedContext();

  // Add multiple context items
  for (let i = 0; i < 20; i++) {
    context.add(
      ContextType.AGENT_OUTPUT,
      { message: `Agent execution ${i} result` },
      {
        source: `agent_${i % 3}`,
        priority: [
          ContextPriority.LOW,
          ContextPriority.MEDIUM,
          ContextPriority.HIGH,
        ][i % 3],
        tags: ['execution', i % 2 === 0 ? 'success' : 'partial'],
      },
    );
  }

  // Add critical items
  context.addIssues('security_scanner', [
    {
      severity: 'CRITICAL',
      description: 'SQL injection vulnerability',
      location: 'src/db/query.ts:45',
    },
  ]);

  const items = context.get();
  console.log('Total context items:', items.length);

  // Try different optimization strategies
  const strategies = [
    OptimizationStrategy.RECENCY,
    OptimizationStrategy.PRIORITY,
    OptimizationStrategy.BALANCED,
    OptimizationStrategy.DIVERSE,
  ];

  for (const strategy of strategies) {
    const optimizer = new ContextOptimizer(2000, strategy); // 2000 tokens max

    const optimized = optimizer.optimize(items, {
      preserveTypes: [ContextType.ISSUES], // Always include issues
      maxItems: 10,
    });

    console.log(`\n${strategy.toUpperCase()} Strategy:`);
    console.log('  Selected items:', optimized.stats.selectedItems);
    console.log('  Dropped items:', optimized.stats.droppedItems);
    console.log('  Estimated tokens:', optimized.estimatedTokens);
    console.log(
      '  Compression ratio:',
      optimized.stats.compressionRatio.toFixed(2),
    );

    // Show summary
    const summary = optimizer.summarize(optimized.items, 200);
    console.log('  Summary:', summary);
  }
}

// ============================================================================
// Example 3: Agent Memory System
// ============================================================================

export function example3_AgentMemory() {
  console.log('\n=== Example 3: Agent Memory System ===\n');

  const memory = new AgentMemory();

  // Store different types of memories
  console.log('Storing memories...');

  // Working memory - current session
  memory.store(
    'explore_agent',
    'Discovered authentication system structure',
    {
      files: [
        'src/auth/login.ts',
        'src/auth/session.ts',
        'src/auth/middleware.ts',
      ],
      patterns: ['JWT authentication', 'Session management', 'Rate limiting'],
    },
    {
      type: MemoryType.WORKING,
      tags: ['auth', 'architecture', 'discovery'],
      importance: 0.7,
    },
  );

  // Episodic memory - specific execution
  const agentOutput: OutputObject = {
    terminate_reason: 'GOAL',
    result: 'Successfully refactored user service, reduced complexity by 40%',
  };

  memory.storeAgentExecution('refactor_agent', agentOutput, {
    tags: ['refactor', 'success', 'performance'],
    importance: 0.8,
  });

  // Semantic memory - facts
  memory.storeFact(
    'code_reviewer',
    'Project uses PostgreSQL for database',
    {
      version: '14.x',
      orm: 'TypeORM',
      migrations: 'src/migrations',
    },
    {
      tags: ['database', 'infrastructure'],
      importance: 0.9,
    },
  );

  memory.storeFact(
    'security_scanner',
    'Authentication uses JWT with RS256 signing',
    {
      standard: 'RFC 7519',
      keyStorage: 'Environment variables',
    },
    {
      tags: ['auth', 'security'],
      importance: 0.85,
    },
  );

  // Retrieve memories
  console.log('\nRetrieving memories...');

  const authMemories = memory.getRelevant(['auth', 'security'], 5);
  console.log('Auth-related memories:', authMemories.length);
  authMemories.forEach((mem) => {
    console.log(
      `  - [${mem.type}] ${mem.summary} (importance: ${mem.importance})`,
    );
  });

  const workingMem = memory.getWorkingMemory(5);
  console.log('\nWorking memory:', workingMem.length);

  // Search by keywords
  const searchResults = memory.search(['authentication', 'JWT']);
  console.log(
    '\nSearch results for "authentication JWT":',
    searchResults.length,
  );
  searchResults.forEach((mem) => {
    console.log(`  - ${mem.summary}`);
  });

  // Memory consolidation
  console.log('\nMemory consolidation...');
  const consolidated = memory.consolidate(0.7); // Move important memories to long-term
  console.log(`Consolidated ${consolidated} memories to long-term storage`);

  // Memory cleanup
  const forgotten = memory.forget(0.3); // Remove unimportant memories
  console.log(`Forgot ${forgotten} low-importance memories`);

  // Export summary
  console.log('\nMemory Summary:');
  console.log(
    memory.exportSummary({
      types: [MemoryType.SEMANTIC, MemoryType.EPISODIC],
      minImportance: 0.7,
      limit: 10,
      sortBy: 'importance',
    }),
  );

  // Statistics
  const stats = memory.getStats();
  console.log('\nMemory Statistics:');
  console.log('Total memories:', stats.totalMemories);
  console.log('By type:', stats.byType);
  console.log('By agent:', stats.byAgent);
  console.log('Average importance:', stats.avgImportance.toFixed(2));
  console.log('Average access count:', stats.avgAccessCount.toFixed(2));
}

// ============================================================================
// Example 4: Contextual Orchestration
// ============================================================================

export function example4_ContextualOrchestration() {
  console.log('\n=== Example 4: Contextual Orchestration ===\n');

  const orchestrator = new ContextualOrchestrator({
    maxContextTokens: 4000,
    optimizationStrategy: OptimizationStrategy.BALANCED,
  });

  // Scenario: Multi-stage bug fix workflow

  // Stage 1: Exploration
  console.log('Stage 1: Exploration');
  const exploreStrategy = orchestrator.createStrategy(
    'Find all authentication-related files and understand the structure',
  );

  console.log('Task type:', exploreStrategy.type);
  console.log(
    'Recommended agents:',
    exploreStrategy.agents.map((a) => a.agentName),
  );
  console.log('Context items:', exploreStrategy.context.items);
  console.log('Confidence:', exploreStrategy.confidence.toFixed(2));

  // Simulate exploration results
  const exploreOutput: OutputObject = {
    terminate_reason: 'GOAL',
    result: 'Found 8 authentication files, identified JWT flow',
  };

  const exploreResult = orchestrator.recordExecution(
    'explore_agent',
    exploreOutput,
    {
      files: [
        'src/auth/login.ts',
        'src/auth/session.ts',
        'src/auth/middleware.ts',
        'src/auth/jwt.ts',
      ],
      patterns: ['JWT authentication', 'Express middleware', 'Session cookies'],
    },
  );

  console.log('Memories created:', exploreResult.memoriesCreated.length);
  console.log('Insights:', exploreResult.insights);

  // Stage 2: Code Review (benefits from exploration context)
  console.log('\nStage 2: Code Review');
  const reviewStrategy = orchestrator.createStrategy(
    'Review authentication code for security vulnerabilities',
  );

  console.log('Task type:', reviewStrategy.type);
  console.log(
    'Context items:',
    reviewStrategy.context.items,
    '(includes exploration results)',
  );
  console.log('Memories:', reviewStrategy.memories.count);
  console.log('Confidence:', reviewStrategy.confidence.toFixed(2));

  // Simulate review results
  const reviewOutput: OutputObject = {
    terminate_reason: 'GOAL',
    result: 'Found 3 security issues',
  };

  orchestrator.recordExecution('code_reviewer', reviewOutput, {
    issues: [
      'Missing rate limiting on login endpoint',
      'JWT tokens do not expire',
      'No CSRF protection',
    ],
    files: ['src/auth/login.ts', 'src/auth/jwt.ts'],
  });

  // Stage 3: Bug Fix (benefits from both previous stages)
  console.log('\nStage 3: Bug Fix');
  const fixStrategy = orchestrator.createStrategy(
    'Fix the security vulnerabilities in authentication',
  );

  console.log('Task type:', fixStrategy.type);
  console.log(
    'Context items:',
    fixStrategy.context.items,
    '(includes all previous)',
  );
  console.log('Memories:', fixStrategy.memories.count);
  console.log('Confidence:', fixStrategy.confidence.toFixed(2));
  console.log('Context summary:', fixStrategy.context.summary);

  // Enhanced inputs include context
  const classification = orchestrator.classifyTask(
    'Fix the security vulnerabilities in authentication',
  );
  const enhancedInputs = orchestrator.enhanceInputsWithContext(
    { target: 'src/auth' },
    'debug_agent',
    classification,
  );

  console.log('\nEnhanced inputs include:');
  console.log('- _contextPrompt:', !!enhancedInputs['_contextPrompt']);
  console.log('- _memorySummary:', !!enhancedInputs['_memorySummary']);
  console.log('- _contextMetadata:', enhancedInputs['_contextMetadata']);

  // Simulate fix completion
  const fixOutput: OutputObject = {
    terminate_reason: 'GOAL',
    result: 'Fixed all 3 security issues',
  };

  orchestrator.recordExecution('debug_agent', fixOutput, {
    files: ['src/auth/login.ts', 'src/auth/jwt.ts', 'src/auth/middleware.ts'],
  });

  // Maintenance
  console.log('\nMaintenance:');
  const consolidated = orchestrator.consolidateMemories(0.7);
  console.log(`Consolidated ${consolidated} important memories`);

  const cleanup = orchestrator.cleanup({
    maxContextAge: 3600000, // 1 hour
    minMemoryImportance: 0.3,
  });
  console.log('Cleanup:', cleanup);

  // View accumulated state
  console.log('\nFinal State:');
  const stats = orchestrator.getStats();
  console.log('Context:', stats.context);
  console.log('Memory:', stats.memory);
}

// ============================================================================
// Example 5: Long-Running Session with Context Accumulation
// ============================================================================

export function example5_LongRunningSession() {
  console.log('\n=== Example 5: Long-Running Session ===\n');

  const orchestrator = new ContextualOrchestrator({
    maxContextTokens: 3000,
    optimizationStrategy: OptimizationStrategy.BALANCED,
  });

  // Simulate a series of related tasks
  const tasks = [
    'Explore the user service module',
    'Find performance bottlenecks in user service',
    'Analyze database queries in user service',
    'Optimize slow database queries',
    'Review optimized code for correctness',
  ];

  console.log('Executing series of related tasks...\n');

  tasks.forEach((task, index) => {
    console.log(`\nTask ${index + 1}: ${task}`);
    console.log('─'.repeat(60));

    const strategy = orchestrator.createStrategy(task);

    console.log('Agent:', strategy.agents[0]?.agentName || 'None');
    console.log('Context items:', strategy.context.items);
    console.log('Memories available:', strategy.memories.count);
    console.log('Confidence:', strategy.confidence.toFixed(2));

    // Simulate execution
    const output: OutputObject = {
      terminate_reason: 'GOAL',
      result: `Completed: ${task}`,
    };

    // Task-specific insights
    let insights: {
      files?: string[];
      issues?: string[];
      patterns?: string[];
    } = {};

    switch (index) {
      case 0: // Explore
        insights = {
          files: ['src/services/user.ts', 'src/models/user.ts'],
          patterns: ['Service pattern', 'Repository pattern'],
        };
        break;
      case 1: // Find bottlenecks
        insights = {
          issues: ['N+1 query problem in getUserWithOrders'],
        };
        break;
      case 2: // Analyze queries
        insights = {
          patterns: ['Multiple sequential queries', 'Missing indexes'],
        };
        break;
      case 3: // Optimize
        insights = {
          files: ['src/services/user.ts'],
          patterns: ['Eager loading', 'Query batching'],
        };
        break;
    }

    orchestrator.recordExecution(
      strategy.agents[0]?.agentName || 'generic_agent',
      output,
      insights,
    );

    // Periodic consolidation
    if (index === 2) {
      const consolidated = orchestrator.consolidateMemories(0.6);
      console.log(`[Maintenance] Consolidated ${consolidated} memories`);
    }
  });

  // Final state
  console.log('\n' + '='.repeat(60));
  console.log('Final Session State');
  console.log('='.repeat(60));

  const finalStats = orchestrator.getStats();
  console.log('\nContext Statistics:');
  console.log('  Total items:', finalStats.context.totalItems);
  console.log('  By type:', finalStats.context.byType);
  console.log('  By priority:', finalStats.context.byPriority);

  console.log('\nMemory Statistics:');
  console.log('  Total memories:', finalStats.memory.totalMemories);
  console.log('  By type:', finalStats.memory.byType);
  console.log(
    '  Average importance:',
    finalStats.memory.avgImportance.toFixed(2),
  );

  // Export knowledge
  const state = orchestrator.exportState();
  console.log('\nAccumulated Knowledge Summary:');
  console.log(state.context);
  console.log('\n' + state.memory);
}

// ============================================================================
// Run All Examples
// ============================================================================

export function runAllExamples() {
  console.log('╔════════════════════════════════════════════════════════════╗');
  console.log('║     Context Engineering System - Usage Examples           ║');
  console.log('╚════════════════════════════════════════════════════════════╝');

  try {
    example1_BasicSharedContext();
    example2_ContextOptimization();
    example3_AgentMemory();
    example4_ContextualOrchestration();
    example5_LongRunningSession();

    console.log('\n' + '='.repeat(60));
    console.log('All examples completed successfully!');
    console.log('='.repeat(60) + '\n');
  } catch (error) {
    console.error('Error running examples:', error);
    throw error;
  }
}

// Run examples if executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  runAllExamples();
}
