/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Example: Multi-Agent Hardware Optimization Workflow
 *
 * This example demonstrates how to use the hardware optimization agents
 * to automatically analyze hardware, generate optimal training strategy,
 * and execute distributed training.
 */

import { HardwareAnalyzer } from '../packages/core/src/hardware/analyzer.js';
import {
  TrainingStrategyOptimizer,
  type ModelConfig,
  type TrainingConfig,
} from '../packages/core/src/hardware/strategy-optimizer.js';

async function main() {
  console.log('=== Multi-Agent Hardware Optimization Example ===\n');

  // Step 1: Analyze Hardware
  console.log('Step 1: Analyzing hardware topology...');
  const hardwareAnalyzer = new HardwareAnalyzer();
  const hardwareAnalysis = await hardwareAnalyzer.analyze();

  console.log('\nHardware Summary:');
  console.log(`  CPUs: ${hardwareAnalysis.topology.cpu.cores} cores`);
  console.log(`  GPUs: ${hardwareAnalysis.topology.gpus.length} detected`);
  if (hardwareAnalysis.topology.gpus.length > 0) {
    hardwareAnalysis.topology.gpus.forEach((gpu, idx) => {
      console.log(
        `    GPU ${idx}: ${gpu.name} (${Math.round(gpu.memory / 1024)}GB)`,
      );
    });
  }
  console.log(
    `  Memory: ${Math.round(hardwareAnalysis.topology.memory.total / 1024)}GB`,
  );
  console.log(
    `  Interconnects: ${hardwareAnalysis.topology.interconnects.map((ic) => ic.type).join(', ')}`,
  );
  console.log(`  Hardware Score: ${hardwareAnalysis.score}/100`);

  console.log('\nRecommendations:');
  hardwareAnalysis.recommendations.forEach((rec, idx) => {
    console.log(`  ${idx + 1}. ${rec}`);
  });

  if (hardwareAnalysis.bottlenecks.length > 0) {
    console.log('\nBottlenecks:');
    hardwareAnalysis.bottlenecks.forEach((bottleneck, idx) => {
      console.log(`  ${idx + 1}. ${bottleneck}`);
    });
  }

  // Step 2: Generate Training Strategy
  console.log('\n\nStep 2: Generating optimal training strategy...');

  const modelConfig: ModelConfig = {
    name: 'LLaMA-7B',
    parameters: 7e9, // 7 billion parameters
    hiddenSize: 4096,
    numLayers: 32,
    sequenceLength: 2048,
  };

  const trainingConfig: TrainingConfig = {
    model: modelConfig,
    targetBatchSize: 32,
    datasetSize: 1e9,
    epochs: 3,
    precision: 'mixed',
  };

  const strategyOptimizer = new TrainingStrategyOptimizer();
  const trainingStrategy = strategyOptimizer.optimizeStrategy(
    hardwareAnalysis,
    trainingConfig,
  );

  console.log('\nTraining Strategy:');
  console.log(`  Parallelism Type: ${trainingStrategy.parallelismType}`);
  console.log(
    `  Data Parallel Degree: ${trainingStrategy.dataParallelDegree || 1}`,
  );
  if (
    trainingStrategy.modelParallelDegree &&
    trainingStrategy.modelParallelDegree > 1
  ) {
    console.log(
      `  Model Parallel Degree: ${trainingStrategy.modelParallelDegree}`,
    );
  }
  if (trainingStrategy.pipelineStages && trainingStrategy.pipelineStages > 1) {
    console.log(`  Pipeline Stages: ${trainingStrategy.pipelineStages}`);
  }

  console.log('\nBatch Configuration:');
  console.log(`  Micro Batch Size: ${trainingStrategy.microBatchSize}`);
  console.log(`  Global Batch Size: ${trainingStrategy.globalBatchSize}`);
  console.log(
    `  Gradient Accumulation Steps: ${trainingStrategy.gradientAccumulationSteps}`,
  );

  console.log('\nOptimizations:');
  console.log(
    `  Mixed Precision: ${trainingStrategy.mixedPrecision ? 'Yes' : 'No'}`,
  );
  console.log(
    `  Activation Checkpointing: ${trainingStrategy.activationCheckpointing ? 'Yes' : 'No'}`,
  );
  console.log(
    `  Optimizer Sharding: ${trainingStrategy.optimizerSharding ? 'Yes' : 'No'}`,
  );
  console.log(
    `  Tensor Parallel: ${trainingStrategy.tensorParallel ? 'Yes' : 'No'}`,
  );

  console.log('\nPerformance Estimates:');
  console.log(
    `  Throughput: ~${trainingStrategy.estimatedThroughput} samples/sec`,
  );
  console.log(
    `  Memory per GPU: ~${Math.round((trainingStrategy.estimatedMemoryUsage || 0) / 1024)}GB`,
  );

  console.log('\nStrategy Reasoning:');
  trainingStrategy.reasoning.forEach((reason, idx) => {
    console.log(`  ${idx + 1}. ${reason}`);
  });

  // Step 3: Generate Execution Plan
  console.log('\n\nStep 3: Generating execution plan...');
  const executionPlan = strategyOptimizer.generateExecutionPlan(
    trainingStrategy,
    './train_llama.py',
  );

  console.log('\nLaunch Command:');
  console.log(`  ${executionPlan.launchCommand}`);

  console.log('\nEnvironment Variables:');
  Object.entries(executionPlan.environmentVariables).forEach(([key, value]) => {
    console.log(`  ${key}=${value}`);
  });

  console.log('\nWorker Distribution:');
  executionPlan.workerDistribution.forEach((worker) => {
    console.log(
      `  ${worker.role}: ${worker.gpuIds.length} GPUs (${worker.gpuIds.join(', ')})`,
    );
  });

  // Summary
  console.log('\n\n=== Summary ===');
  console.log(
    `This ${hardwareAnalysis.topology.gpus.length}-GPU system is configured for ${trainingStrategy.parallelismType} parallelism.`,
  );
  console.log(
    `Expected to process ~${trainingStrategy.estimatedThroughput} samples/sec with ${trainingStrategy.globalBatchSize} global batch size.`,
  );
  console.log('\nTo execute training, run:');
  console.log(`  ${executionPlan.launchCommand}`);

  console.log('\n✓ Hardware optimization complete!');
}

// Run the example
main().catch((error) => {
  console.error('Error:', error);
  process.exit(1);
});
