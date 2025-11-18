/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Practical examples of using the Analysis System.
 *
 * These examples demonstrate DataFlowAgent, LiteratureAnalyzerAgent,
 * SummarizerAgent, DataPipeline, and AnalysisOrchestrator.
 */

import {
  DataPipeline,
  Transforms,
  Validators,
  AnalysisOrchestrator,
  AnalysisWorkflowType,
  type AnalysisTask,
  type PipelineResult,
} from '../src/index.js';

// ============================================================================
// Example 1: Basic Data Pipeline
// ============================================================================

export async function example1_BasicDataPipeline() {
  console.log('\n=== Example 1: Basic Data Pipeline ===\n');

  // Sample user data
  const rawData = JSON.stringify([
    {
      id: 1,
      name: 'Alice',
      email: 'alice@example.com',
      active: true,
      loginCount: 42,
    },
    {
      id: 2,
      name: 'Bob',
      email: 'bob@example.com',
      active: false,
      loginCount: 5,
    },
    {
      id: 3,
      name: 'Charlie',
      email: 'charlie@example.com',
      active: true,
      loginCount: 128,
    },
    {
      id: 4,
      name: 'Diana',
      email: 'diana@example.com',
      active: true,
      loginCount: 73,
    },
  ]);

  // Create a data processing pipeline
  const pipeline = new DataPipeline()
    .addStep({
      name: 'parse',
      description: 'Parse JSON string',
      transform: Transforms.parseJSON<Array<Record<string, unknown>>>,
    })
    .addStep({
      name: 'filter-active',
      description: 'Filter active users only',
      transform: Transforms.filter((user: Record<string, unknown>) =>
        Boolean(user.active),
      ),
      validate: Validators.nonEmpty,
    })
    .addStep({
      name: 'extract-fields',
      description: 'Extract relevant fields',
      transform: Transforms.extractFields(['id', 'name', 'loginCount']),
    })
    .addStep({
      name: 'sort',
      description: 'Sort by login count',
      transform: Transforms.sort(
        (a: Record<string, unknown>, b: Record<string, unknown>) =>
          (b.loginCount as number) - (a.loginCount as number),
      ),
    });

  console.log('Pipeline steps:');
  pipeline.getSteps().forEach((step, idx) => {
    console.log(`  ${idx + 1}. ${step.name}: ${step.description}`);
  });

  // Execute pipeline
  const result: PipelineResult = await pipeline.execute(rawData);

  console.log('\nExecution result:');
  console.log('  Success:', result.success);
  console.log('  Steps executed:', result.metadata.stepsExecuted);
  console.log('  Execution time:', `${result.metadata.executionTime}ms`);

  if (result.success) {
    console.log('\nProcessed data:');
    console.log(JSON.stringify(result.data, null, 2));
  } else {
    console.log('\nErrors:');
    result.errors.forEach((err) => {
      console.log(`  ${err.step}: ${err.message}`);
    });
  }
}

// ============================================================================
// Example 2: Advanced Data Pipeline with Validation
// ============================================================================

export async function example2_AdvancedPipeline() {
  console.log('\n=== Example 2: Advanced Pipeline with Validation ===\n');

  // Sample event log data
  const eventLogs = [
    { id: 1, timestamp: Date.now(), event: 'login', userId: 'user1' },
    { id: 2, timestamp: Date.now(), event: 'click', userId: 'user2' },
    { id: 3, timestamp: Date.now(), event: 'login', userId: 'user1' },
    {
      id: 4,
      timestamp: Date.now(),
      event: 'purchase',
      userId: 'user3',
      amount: 99.99,
    },
    { id: 5, timestamp: Date.now(), event: 'login', userId: 'user2' },
    {
      id: 6,
      timestamp: Date.now(),
      event: 'purchase',
      userId: 'user1',
      amount: 149.5,
    },
  ];

  const pipeline = new DataPipeline()
    .addStep({
      name: 'validate-structure',
      description: 'Validate required fields',
      transform: (data: typeof eventLogs) => data,
      validate: Validators.all(
        Validators.nonEmpty,
        Validators.custom(
          (logs: typeof eventLogs) =>
            logs.every(
              (log) => 'id' in log && 'event' in log && 'userId' in log,
            ),
          'Missing required fields',
        ),
      ),
    })
    .addStep({
      name: 'group-by-user',
      description: 'Group events by user',
      transform: Transforms.groupBy(
        (log: (typeof eventLogs)[number]) => log.userId,
      ),
    })
    .addStep({
      name: 'calculate-stats',
      description: 'Calculate statistics per user',
      transform: (grouped: Record<string, typeof eventLogs>) => {
        const stats: Record<
          string,
          {
            userId: string;
            eventCount: number;
            loginCount: number;
            purchaseTotal: number;
          }
        > = {};
        for (const [userId, events] of Object.entries(grouped)) {
          stats[userId] = {
            userId,
            eventCount: events.length,
            loginCount: events.filter((e) => e.event === 'login').length,
            purchaseTotal: events
              .filter((e) => e.event === 'purchase')
              .reduce(
                (sum, e) => sum + ((e as (typeof eventLogs)[3]).amount || 0),
                0,
              ),
          };
        }
        return stats;
      },
    });

  const result = await pipeline.execute(eventLogs);

  if (result.success) {
    console.log('User statistics:');
    console.log(JSON.stringify(result.data, null, 2));

    console.log('\nValidation results:');
    Object.entries(result.metadata.validationResults).forEach(
      ([step, validation]) => {
        console.log(`  ${step}: ${validation.valid ? '✓ Passed' : '✗ Failed'}`);
        if (!validation.valid) {
          validation.errors.forEach((err) => {
            console.log(`    - ${err.message}`);
          });
        }
      },
    );
  } else {
    console.log('Pipeline failed:', result.errors);
  }
}

// ============================================================================
// Example 3: Analysis Orchestrator - Data Processing Workflow
// ============================================================================

export async function example3_DataProcessingWorkflow() {
  console.log('\n=== Example 3: Data Processing Workflow ===\n');

  const orchestrator = new AnalysisOrchestrator();

  const task: AnalysisTask = {
    type: AnalysisWorkflowType.DATA_PROCESSING,
    description: 'Process user activity logs and identify engagement patterns',
    sources: ['logs/user-activity.json'],
    options: {
      dataValidation: ['completeness', 'consistency', 'temporal_validity'],
    },
  };

  console.log('Executing data processing workflow...');
  console.log('Task:', task.description);
  console.log('Sources:', task.sources);

  const result = await orchestrator.executeAnalysis(task);

  console.log('\n=== Workflow Results ===');
  console.log('Summary:', result.summary);
  console.log('\nKey Findings:');
  result.keyFindings.forEach((finding, idx) => {
    console.log(`  ${idx + 1}. ${finding}`);
  });

  console.log('\nRecommendations:');
  result.recommendations.forEach((rec, idx) => {
    console.log(`  ${idx + 1}. ${rec}`);
  });

  console.log('\nMetadata:');
  console.log(
    '  Total execution time:',
    `${result.metadata.totalExecutionTime}ms`,
  );
  console.log('  Agents used:', result.metadata.agentsUsed.join(', '));
  console.log('  Context items used:', result.metadata.contextItemsUsed);
  console.log('  Memories created:', result.metadata.memoriesCreated);

  console.log('\nExecution steps:');
  result.steps.forEach((step, idx) => {
    console.log(`  ${idx + 1}. ${step.agent}`);
    console.log(`     Insights: ${step.insights.join(', ')}`);
  });
}

// ============================================================================
// Example 4: Literature Review Workflow
// ============================================================================

export async function example4_LiteratureReviewWorkflow() {
  console.log('\n=== Example 4: Literature Review Workflow ===\n');

  const orchestrator = new AnalysisOrchestrator();

  const task: AnalysisTask = {
    type: AnalysisWorkflowType.LITERATURE_REVIEW,
    description: 'Analyze recent ML optimization research papers',
    sources: [
      'papers/adam-optimizer.pdf',
      'papers/sgd-momentum.pdf',
      'papers/learning-rate-schedules.pdf',
    ],
    options: {
      depth: 'comprehensive',
      focus: ['methodology', 'results', 'future_work'],
    },
  };

  console.log('Executing literature review workflow...');
  console.log('Task:', task.description);
  console.log('Papers:', task.sources.length);
  console.log('Focus areas:', task.options?.focus?.join(', '));

  const result = await orchestrator.executeAnalysis(task);

  console.log('\n=== Literature Review Results ===');
  console.log('Summary:', result.summary);
  console.log('\nKey Findings:');
  result.keyFindings.forEach((finding, idx) => {
    console.log(`  ${idx + 1}. ${finding}`);
  });

  console.log('\nNext Steps:');
  result.recommendations.forEach((rec, idx) => {
    console.log(`  ${idx + 1}. ${rec}`);
  });

  console.log('\nAnalysis performed by:');
  result.metadata.agentsUsed.forEach((agent) => {
    console.log(`  - ${agent}`);
  });
}

// ============================================================================
// Example 5: Multi-Document Summary Workflow
// ============================================================================

export async function example5_MultiDocumentSummary() {
  console.log('\n=== Example 5: Multi-Document Summary ===\n');

  const orchestrator = new AnalysisOrchestrator();

  const task: AnalysisTask = {
    type: AnalysisWorkflowType.MULTI_DOCUMENT_SUMMARY,
    description: 'Create comprehensive summary of product documentation',
    sources: [
      'docs/getting-started.md',
      'docs/api-reference.md',
      'docs/architecture.md',
      'docs/best-practices.md',
      'docs/troubleshooting.md',
    ],
    options: {
      depth: 'moderate',
      includeActionItems: true,
    },
  };

  console.log('Executing multi-document summary workflow...');
  console.log('Documents to summarize:', task.sources.length);

  const result = await orchestrator.executeAnalysis(task);

  console.log('\n=== Summary Results ===');
  console.log('Executive Summary:', result.summary);
  console.log('\nKey Points:');
  result.keyFindings.forEach((point, idx) => {
    console.log(`  ${idx + 1}. ${point}`);
  });

  console.log('\nAction Items:');
  result.recommendations.forEach((action, idx) => {
    console.log(`  ${idx + 1}. ${action}`);
  });
}

// ============================================================================
// Example 6: Comprehensive Analysis (Data + Literature)
// ============================================================================

export async function example6_ComprehensiveAnalysis() {
  console.log('\n=== Example 6: Comprehensive Analysis ===\n');

  const orchestrator = new AnalysisOrchestrator();

  const task: AnalysisTask = {
    type: AnalysisWorkflowType.COMPREHENSIVE_ANALYSIS,
    description:
      'Analyze user retention data and compare with research best practices',
    sources: [
      'data/user-retention-2024.json',
      'research/retention-strategies.pdf',
      'research/engagement-patterns.pdf',
    ],
    options: {
      depth: 'comprehensive',
      includeActionItems: true,
    },
  };

  console.log('Executing comprehensive analysis workflow...');
  console.log('Combining data analysis and literature review...');

  const result = await orchestrator.executeAnalysis(task);

  console.log('\n=== Comprehensive Analysis Results ===');
  console.log('Summary:', result.summary);
  console.log('\nKey Insights:');
  result.keyFindings.forEach((insight, idx) => {
    console.log(`  ${idx + 1}. ${insight}`);
  });

  console.log('\nStrategic Recommendations:');
  result.recommendations.forEach((rec, idx) => {
    console.log(`  ${idx + 1}. ${rec}`);
  });

  console.log('\nWorkflow Metadata:');
  console.log('  Execution time:', `${result.metadata.totalExecutionTime}ms`);
  console.log('  Agents coordinated:', result.metadata.agentsUsed.join(' → '));
  console.log('  Knowledge items:', result.metadata.contextItemsUsed);
  console.log('  Memories preserved:', result.metadata.memoriesCreated);
}

// ============================================================================
// Example 7: Continuous Analysis with Memory
// ============================================================================

export async function example7_ContinuousAnalysis() {
  console.log('\n=== Example 7: Continuous Analysis with Memory ===\n');

  const orchestrator = new AnalysisOrchestrator();

  // Simulate analyzing daily reports
  const dailyReports = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday'];

  console.log('Analyzing daily reports...\n');

  for (const day of dailyReports) {
    console.log(`Processing ${day} report...`);

    await orchestrator.executeAnalysis({
      type: AnalysisWorkflowType.MULTI_DOCUMENT_SUMMARY,
      description: `Analyze ${day} team report`,
      sources: [`reports/${day}.md`],
      options: {
        depth: 'moderate',
      },
    });

    console.log(`  ✓ ${day} analysis complete`);
  }

  console.log('\nGenerating weekly summary...');

  // Weekly summary benefits from all daily analyses
  const weeklySummary = await orchestrator.executeAnalysis({
    type: AnalysisWorkflowType.MULTI_DOCUMENT_SUMMARY,
    description: 'Generate comprehensive weekly summary',
    sources: dailyReports.map((day) => `reports/${day}.md`),
    options: {
      depth: 'comprehensive',
      includeActionItems: true,
    },
  });

  console.log('\n=== Weekly Summary ===');
  console.log(weeklySummary.summary);

  // Consolidate important findings
  console.log('\nConsolidating memories...');
  const memoriesConsolidated = orchestrator.consolidateMemories(0.7);
  console.log(`Consolidated ${memoriesConsolidated} important memories`);

  // View analysis history
  const history = orchestrator.getAnalysisHistory();
  console.log(`\nTotal analyses performed: ${history.length}`);

  // Export state
  const state = orchestrator.exportAnalysisState();
  console.log('\nAnalysis State:');
  console.log('  Workflow types executed:', state.workflows.join(', '));
  console.log('  Total insights:', state.insights.length);
  console.log('  Patterns identified:', state.patterns.length);
}

// ============================================================================
// Example 8: Custom Transform and Validation
// ============================================================================

export async function example8_CustomTransformValidation() {
  console.log('\n=== Example 8: Custom Transform and Validation ===\n');

  // Custom transform: Calculate engagement score
  const calculateEngagementScore = (users: Array<Record<string, unknown>>) => {
    return users.map((user) => ({
      ...user,
      engagementScore:
        ((user.loginCount as number) || 0) * 2 +
        ((user.postsCount as number) || 0) * 5 +
        ((user.commentsCount as number) || 0) * 1,
    }));
  };

  // Custom validator: Check engagement threshold
  const validateEngagement = (users: Array<Record<string, unknown>>) => {
    const lowEngagement = users.filter(
      (u) => (u.engagementScore as number) < 10,
    );

    return {
      valid: lowEngagement.length === 0,
      errors:
        lowEngagement.length > 0
          ? [
              {
                message: `${lowEngagement.length} users below engagement threshold`,
                severity: 'warning' as const,
              },
            ]
          : [],
    };
  };

  const userData = [
    { id: 1, name: 'Alice', loginCount: 10, postsCount: 5, commentsCount: 20 },
    { id: 2, name: 'Bob', loginCount: 2, postsCount: 0, commentsCount: 1 },
    {
      id: 3,
      name: 'Charlie',
      loginCount: 15,
      postsCount: 8,
      commentsCount: 30,
    },
  ];

  const pipeline = new DataPipeline()
    .addStep({
      name: 'calculate-engagement',
      description: 'Calculate user engagement scores',
      transform: calculateEngagementScore,
      validate: validateEngagement,
    })
    .addStep({
      name: 'sort-by-engagement',
      description: 'Sort users by engagement',
      transform: Transforms.sort(
        (a: Record<string, unknown>, b: Record<string, unknown>) =>
          (b.engagementScore as number) - (a.engagementScore as number),
      ),
    });

  const result = await pipeline.execute(userData);

  if (result.success) {
    console.log('User engagement analysis:');
    console.log(JSON.stringify(result.data, null, 2));

    console.log('\nValidation warnings:');
    Object.entries(result.metadata.validationResults).forEach(
      ([step, validation]) => {
        validation.errors.forEach((err) => {
          if (err.severity === 'warning') {
            console.log(`  ⚠ ${step}: ${err.message}`);
          }
        });
      },
    );
  }
}

// ============================================================================
// Run All Examples
// ============================================================================

export async function runAllExamples() {
  console.log('╔══════════════════════════════════════════════════════════╗');
  console.log('║     Analysis System - Usage Examples                    ║');
  console.log('╚══════════════════════════════════════════════════════════╝');

  try {
    await example1_BasicDataPipeline();
    await example2_AdvancedPipeline();
    await example3_DataProcessingWorkflow();
    await example4_LiteratureReviewWorkflow();
    await example5_MultiDocumentSummary();
    await example6_ComprehensiveAnalysis();
    await example7_ContinuousAnalysis();
    await example8_CustomTransformValidation();

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
