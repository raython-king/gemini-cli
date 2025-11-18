/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 *
 * Algorithm Research System Examples
 *
 * Comprehensive examples demonstrating the algorithm research and development
 * system for LLM and CV research.
 */

import {
  type Config,
  AgentExecutor,
  LLMResearchAgent,
  CVResearchAgent,
  ExperimentAgent,
  ModelEvaluationAgent,
  AlgorithmWorkflowOrchestrator,
  AlgorithmWorkflowType,
  AlgorithmDomain,
  CollaborativeExecutor,
} from '@google/gemini-cli-core';

// ============================================================================
// Example 1: LLM Research - Analyze Flash Attention
// ============================================================================

export async function example1_LLMResearch(config: Config) {
  console.log('Example 1: LLM Research - Flash Attention Analysis\n');

  const executor = await AgentExecutor.create(LLMResearchAgent, config);

  const result = await executor.run({
    objective:
      'Analyze flash attention mechanism and provide implementation guidance',
    topic: 'attention_optimization',
    papers: ['./papers/flash_attention.pdf'],
    depth: 'comprehensive',
    focus: ['architecture', 'implementation', 'optimization'],
    includeImplementation: true,
  });

  const analysis = JSON.parse(result.result);

  console.log('=== Research Summary ===');
  console.log(analysis.Summary);
  console.log();

  console.log('=== Key Findings ===');
  analysis.KeyFindings.forEach((finding: any, i: number) => {
    console.log(`${i + 1}. ${finding.title}`);
    console.log(`   Significance: ${finding.significance}`);
    console.log(`   ${finding.description}`);
    console.log();
  });

  console.log('=== Technical Analysis ===');
  console.log('Architecture:', analysis.TechnicalAnalysis.architecture);
  console.log();

  console.log('=== Implementation Roadmap ===');
  analysis.ImplementationRoadmap?.forEach((phase: any, i: number) => {
    console.log(`Phase ${i + 1}: ${phase.phase}`);
    phase.tasks.forEach((task: string) => {
      console.log(`  - ${task}`);
    });
    console.log(`  Time: ${phase.estimatedTime}`);
    console.log();
  });
}

// ============================================================================
// Example 2: CV Research - YOLOv8 Architecture
// ============================================================================

export async function example2_CVResearch(config: Config) {
  console.log('Example 2: CV Research - YOLOv8 Architecture\n');

  const executor = await AgentExecutor.create(CVResearchAgent, config);

  const result = await executor.run({
    objective: 'Analyze YOLOv8 architecture improvements and optimizations',
    domain: 'object_detection',
    papers: ['./papers/yolov8.pdf'],
    depth: 'thorough',
    focus: ['architecture', 'optimization', 'inference'],
    includeImplementation: true,
  });

  const analysis = JSON.parse(result.result);

  console.log('=== Architecture Analysis ===');
  console.log(analysis.TechnicalAnalysis.architecture);
  console.log();

  console.log('=== SOTA Comparison ===');
  analysis.SOTAComparison?.forEach((model: any) => {
    console.log(`\n${model.model}:`);
    console.log(`  Accuracy: ${model.accuracy}`);
    console.log(`  Speed: ${model.speed}`);
    console.log(`  Parameters: ${model.parameters}`);
    console.log(`  Strengths: ${model.strengths.join(', ')}`);
  });

  console.log('\n=== Optimization Strategies ===');
  analysis.OptimizationStrategies?.forEach((strategy: any) => {
    console.log(`\n${strategy.strategy}:`);
    console.log(`  ${strategy.description}`);
    if (strategy.expectedSpeedup) {
      console.log(`  Expected speedup: ${strategy.expectedSpeedup}`);
    }
  });
}

// ============================================================================
// Example 3: Experiment Design
// ============================================================================

export async function example3_ExperimentDesign(config: Config) {
  console.log('Example 3: Experiment Design - Ablation Study\n');

  const executor = await AgentExecutor.create(ExperimentAgent, config);

  const result = await executor.run({
    objective: 'Design ablation study for transformer attention components',
    algorithmType: 'llm',
    experimentType: 'ablation',
    codebase: './transformer_model',
    dataset: './wikitext',
    constraints: {
      maxComputeBudget: '100 GPU hours',
      timeLimit: '1 week',
      hardwareAvailable: ['V100', 'A100'],
    },
  });

  const experiment = JSON.parse(result.result);

  console.log('=== Research Hypothesis ===');
  console.log('Null:', experiment.Hypothesis.nullHypothesis);
  console.log('Alternative:', experiment.Hypothesis.alternativeHypothesis);
  console.log();

  console.log('=== Experimental Design ===');
  console.log(experiment.ExperimentalDesign.overview);
  console.log();

  console.log('=== Configurations to Test ===');
  experiment.Configurations.forEach((config: any, i: number) => {
    console.log(`\n${i + 1}. ${config.name}`);
    console.log(`   ${config.description}`);
    console.log(`   Priority: ${config.priority}`);
    console.log(`   Expected: ${config.expectedOutcome}`);
  });

  console.log('\n=== Statistical Analysis Plan ===');
  console.log(
    'Significance level:',
    experiment.StatisticalAnalysis.significanceLevel,
  );
  console.log(
    'Tests:',
    experiment.StatisticalAnalysis.statisticalTests.join(', '),
  );
  console.log();

  console.log('=== Resource Requirements ===');
  console.log('Compute:', experiment.ResourceRequirements.computeResources);
  console.log('Time:', experiment.ResourceRequirements.estimatedTime);
  console.log('Storage:', experiment.ResourceRequirements.storageNeeded);
}

// ============================================================================
// Example 4: Model Evaluation
// ============================================================================

export async function example4_ModelEvaluation(config: Config) {
  console.log('Example 4: Model Evaluation - Comprehensive Assessment\n');

  const executor = await AgentExecutor.create(
    ModelEvaluationAgent,
    config,
  );

  const result = await executor.run({
    modelPath: './trained_model',
    objective: 'Comprehensive evaluation for production deployment',
    modelType: 'cv',
    testData: './test_data',
    focus: ['accuracy', 'efficiency', 'robustness'],
    baselines: ['./baseline_model'],
    analyzeOptimization: true,
  });

  const evaluation = JSON.parse(result.result);

  console.log('=== Model Information ===');
  console.log('Architecture:', evaluation.ModelInfo.architecture);
  console.log('Parameters:', evaluation.ModelInfo.parameters);
  console.log('Size:', evaluation.ModelInfo.modelSize);
  console.log();

  console.log('=== Performance Metrics ===');
  if (evaluation.PerformanceMetrics.accuracy) {
    console.log('Overall Accuracy:', evaluation.PerformanceMetrics.accuracy.overall);
    console.log('Top-1:', evaluation.PerformanceMetrics.accuracy.top1);
    console.log('Top-5:', evaluation.PerformanceMetrics.accuracy.top5);
  }
  console.log();

  console.log('=== Efficiency Metrics ===');
  console.log('Latency:', evaluation.EfficiencyMetrics.inferenceLatency.mean);
  console.log('Throughput:', evaluation.EfficiencyMetrics.throughput);
  console.log('Memory:', evaluation.EfficiencyMetrics.memoryUsage);
  console.log();

  console.log('=== Error Analysis ===');
  console.log('Error Rate:', evaluation.ErrorAnalysis.overallErrorRate);
  console.log('\nFailure Modes:');
  evaluation.ErrorAnalysis.failureModes.forEach((mode: any) => {
    console.log(`  - ${mode.mode} (${mode.severity}): ${mode.description}`);
  });
  console.log();

  console.log('=== Optimization Opportunities ===');
  evaluation.OptimizationOpportunities?.forEach((opp: any) => {
    console.log(`\n${opp.opportunity}:`);
    console.log(`  ${opp.description}`);
    console.log(`  Priority: ${opp.priority}`);
    if (opp.expectedSpeedup) {
      console.log(`  Expected speedup: ${opp.expectedSpeedup}`);
    }
  });
  console.log();

  console.log('=== Deployment Readiness ===');
  console.log('Status:', evaluation.DeploymentReadiness.overallReadiness);
  console.log('Strengths:', evaluation.DeploymentReadiness.strengths.join(', '));
  if (evaluation.DeploymentReadiness.blockers.length > 0) {
    console.log('Blockers:', evaluation.DeploymentReadiness.blockers.join(', '));
  }
}

// ============================================================================
// Example 5: Complete Research Workflow
// ============================================================================

export async function example5_CompleteWorkflow(config: Config) {
  console.log('Example 5: Complete Research Workflow\n');

  const orchestrator = new AlgorithmWorkflowOrchestrator();

  // Create a comprehensive research & implement workflow
  const workflow = orchestrator.createAlgorithmWorkflow({
    workflowType: AlgorithmWorkflowType.RESEARCH_AND_IMPLEMENT,
    domain: AlgorithmDomain.LLM,
    objective: 'Implement grouped-query attention for efficient inference',
    papers: ['./papers/gqa_paper.pdf'],
    dataset: './training_data',
    constraints: {
      computeBudget: '50 GPU hours',
      timeLimit: '2 weeks',
      hardware: ['V100', 'A100'],
    },
    requirements: {
      includeAblation: true,
      includeBaselines: true,
      targetMetric: 'latency',
    },
  });

  console.log('=== Workflow Created ===');
  console.log('Task:', workflow.originalTask);
  console.log('Complexity:', workflow.classification.complexity);
  console.log('Subtasks:', workflow.subtasks.length);
  console.log('Stages:', workflow.executionPlan.length);
  console.log('Duration:', workflow.estimatedDuration);
  console.log();

  console.log('=== Execution Plan ===');
  workflow.executionPlan.forEach((stage) => {
    const stageSubtasks = workflow.subtasks.filter((st) =>
      stage.subtasks.includes(st.id),
    );

    console.log(
      `\nStage ${stage.stage} (${stage.canRunInParallel ? 'Parallel' : 'Sequential'}):`,
    );
    stageSubtasks.forEach((st) => {
      console.log(`  - ${st.title}`);
      console.log(`    Agent: ${st.recommendedAgents[0]}`);
      console.log(`    Complexity: ${st.complexity}`);
    });
  });

  // Execute the workflow
  console.log('\n\n=== Executing Workflow ===\n');

  const executor = new CollaborativeExecutor(config);

  const result = await executor.execute(workflow.originalTask, {
    maxSubtasks: workflow.subtasks.length,
    maxConcurrency: 3,
    enableContextSharing: true,
    enableMemory: true,
    onEvent: (event) => {
      switch (event.type) {
        case 'decomposition_complete':
          console.log('✓ Task decomposed');
          break;
        case 'stage_start':
          console.log(
            `→ Starting stage ${(event.data as any).stage}/${(event.data as any).totalStages}`,
          );
          break;
        case 'subtask_complete':
          console.log(`  ✓ ${(event.data as any).agentName} completed`);
          break;
        case 'subtask_failed':
          console.log(`  ✗ ${(event.data as any).agentName} failed`);
          break;
        case 'execution_complete':
          console.log(
            `\n✓ Execution complete (${Math.round((event.data as any).duration / 1000)}s)`,
          );
          break;
      }
    },
  });

  console.log('\n=== Results ===');
  console.log('Success:', result.status.success);
  console.log('Progress:', result.status.progress + '%');
  console.log('Completed:', result.status.subtasks.filter((s) => s.status === 'completed').length);

  if (result.contextSummary) {
    console.log('\n=== Research Insights ===');
    console.log(result.contextSummary);
  }
}

// ============================================================================
// Example 6: Reproduce Paper Workflow
// ============================================================================

export async function example6_ReproducePaper(config: Config) {
  console.log('Example 6: Reproduce Paper Workflow\n');

  const orchestrator = new AlgorithmWorkflowOrchestrator();

  const workflow = orchestrator.createAlgorithmWorkflow({
    workflowType: AlgorithmWorkflowType.REPRODUCE_PAPER,
    domain: AlgorithmDomain.CV,
    objective: 'Reproduce Segment Anything Model results',
    papers: ['./papers/sam_paper.pdf'],
    dataset: './coco',
    constraints: {
      computeBudget: '200 GPU hours',
      hardware: ['A100'],
    },
  });

  console.log('Reproduce Paper Workflow:');
  console.log('Subtasks:', workflow.subtasks.length);

  workflow.subtasks.forEach((st, i) => {
    console.log(`\n${i + 1}. ${st.title}`);
    console.log(`   Agent: ${st.recommendedAgents[0]}`);
    console.log(`   ${st.description}`);
  });
}

// ============================================================================
// Example 7: Optimize Model Workflow
// ============================================================================

export async function example7_OptimizeModel(config: Config) {
  console.log('Example 7: Optimize Model Workflow\n');

  const orchestrator = new AlgorithmWorkflowOrchestrator();

  const workflow = orchestrator.createAlgorithmWorkflow({
    workflowType: AlgorithmWorkflowType.OPTIMIZE_MODEL,
    domain: AlgorithmDomain.CV,
    objective: 'Optimize YOLOv5 for edge deployment',
    codebase: './yolov5',
    constraints: {
      computeBudget: '20 GPU hours',
      hardware: ['edge', 'mobile'],
    },
    requirements: {
      targetMetric: 'inference_latency',
      deploymentTarget: 'mobile',
    },
  });

  console.log('Optimization Workflow:');
  console.log('Focus: Edge deployment');
  console.log('Target: < 50ms latency');
  console.log();

  workflow.subtasks.forEach((st, i) => {
    console.log(`${i + 1}. ${st.title}`);
  });
}

// ============================================================================
// Example 8: Benchmark Comparison
// ============================================================================

export async function example8_BenchmarkComparison(config: Config) {
  console.log('Example 8: Benchmark Comparison\n');

  const orchestrator = new AlgorithmWorkflowOrchestrator();

  const workflow = orchestrator.createAlgorithmWorkflow({
    workflowType: AlgorithmWorkflowType.BENCHMARK_COMPARISON,
    domain: AlgorithmDomain.LLM,
    objective:
      'Compare multi-head, multi-query, and grouped-query attention',
    codebase: './attention_variants',
    dataset: './wikitext',
    requirements: {
      includeBaselines: true,
    },
  });

  console.log('Benchmark Comparison Workflow:');
  console.log('Models to compare: 3');
  console.log('Metrics: Performance, Speed, Memory');
  console.log();

  workflow.subtasks.forEach((st, i) => {
    console.log(`${i + 1}. ${st.title}`);
    if (st.canParallelize) {
      console.log('   (Can run in parallel)');
    }
  });
}

// ============================================================================
// Example 9: Access Research Context
// ============================================================================

export async function example9_AccessContext(config: Config) {
  console.log('Example 9: Access Research Context\n');

  const executor = new CollaborativeExecutor(config);

  await executor.execute(
    'Research transformer optimizations',
    {
      maxSubtasks: 5,
      enableContextSharing: true,
      enableMemory: true,
    },
  );

  // Access accumulated knowledge
  const orchestrator = executor.getContextualOrchestrator();
  const context = orchestrator.getSharedContext();
  const memory = orchestrator.getMemory();

  console.log('=== Shared Context ===');
  const stats = context.getStats();
  console.log('Total items:', stats.totalItems);
  console.log('By type:');
  Object.entries(stats.byType).forEach(([type, count]) => {
    console.log(`  ${type}: ${count}`);
  });

  console.log('\n=== Agent Memory ===');
  const memoryStats = memory.getStats();
  console.log('Total memories:', memoryStats.totalMemories);
  console.log('Working memory:', memoryStats.workingMemoryCount);
  console.log('Long-term memory:', memoryStats.longTermMemoryCount);

  // Get relevant insights
  const insights = memory.getRelevant(
    ['transformer', 'optimization', 'attention'],
    5,
  );
  console.log('\n=== Relevant Insights ===');
  insights.forEach((mem, i) => {
    console.log(`${i + 1}. ${mem.content.slice(0, 100)}...`);
  });
}

// ============================================================================
// Example 10: Multi-Domain Research
// ============================================================================

export async function example10_MultiDomain(config: Config) {
  console.log('Example 10: Multi-Domain Research\n');

  // LLM + CV multimodal research
  const orchestrator = new AlgorithmWorkflowOrchestrator();

  const llmWorkflow = orchestrator.createAlgorithmWorkflow({
    workflowType: AlgorithmWorkflowType.RESEARCH_AND_IMPLEMENT,
    domain: AlgorithmDomain.LLM,
    objective: 'Implement text encoder for multimodal model',
  });

  const cvWorkflow = orchestrator.createAlgorithmWorkflow({
    workflowType: AlgorithmWorkflowType.RESEARCH_AND_IMPLEMENT,
    domain: AlgorithmDomain.CV,
    objective: 'Implement vision encoder for multimodal model',
  });

  console.log('=== Multimodal Research ===');
  console.log('LLM subtasks:', llmWorkflow.subtasks.length);
  console.log('CV subtasks:', cvWorkflow.subtasks.length);
  console.log('Total:', llmWorkflow.subtasks.length + cvWorkflow.subtasks.length);
  console.log();

  console.log('Agents involved:');
  const allAgents = new Set([
    ...llmWorkflow.subtasks.flatMap((st) => st.recommendedAgents),
    ...cvWorkflow.subtasks.flatMap((st) => st.recommendedAgents),
  ]);
  allAgents.forEach((agent) => {
    console.log(`  - ${agent}`);
  });
}

// ============================================================================
// Run All Examples
// ============================================================================

export async function runAllExamples(config: Config) {
  console.log('='.repeat(70));
  console.log('        Algorithm Research System Examples');
  console.log('='.repeat(70));
  console.log();

  try {
    await example1_LLMResearch(config);
    console.log('\n' + '='.repeat(70) + '\n');

    await example2_CVResearch(config);
    console.log('\n' + '='.repeat(70) + '\n');

    await example3_ExperimentDesign(config);
    console.log('\n' + '='.repeat(70) + '\n');

    await example4_ModelEvaluation(config);
    console.log('\n' + '='.repeat(70) + '\n');

    await example5_CompleteWorkflow(config);
    console.log('\n' + '='.repeat(70) + '\n');

    await example6_ReproducePaper(config);
    console.log('\n' + '='.repeat(70) + '\n');

    await example7_OptimizeModel(config);
    console.log('\n' + '='.repeat(70) + '\n');

    await example8_BenchmarkComparison(config);
    console.log('\n' + '='.repeat(70) + '\n');

    await example9_AccessContext(config);
    console.log('\n' + '='.repeat(70) + '\n');

    await example10_MultiDomain(config);
    console.log('\n' + '='.repeat(70) + '\n');

    console.log('All examples completed successfully!');
  } catch (error) {
    console.error('Error running examples:', error);
  }
}
