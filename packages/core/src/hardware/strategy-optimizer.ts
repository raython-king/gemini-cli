/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Training strategy optimization based on hardware analysis
 */

import type {
  HardwareAnalysisResult,
  TrainingStrategy,
  TrainingExecutionPlan,
  GPUInfo,
  HardwareTopology,
} from './types.js';

export interface ModelConfig {
  name?: string;
  parameters: number; // Total model parameters
  hiddenSize?: number;
  numLayers?: number;
  vocabSize?: number;
  sequenceLength?: number;
}

export interface TrainingConfig {
  model: ModelConfig;
  datasetSize?: number;
  targetBatchSize?: number;
  epochs?: number;
  precision?: 'fp32' | 'fp16' | 'bf16' | 'mixed';
}

export class TrainingStrategyOptimizer {
  /**
   * Generate optimal training strategy based on hardware analysis
   */
  optimizeStrategy(
    hardwareAnalysis: HardwareAnalysisResult,
    trainingConfig: TrainingConfig,
  ): TrainingStrategy {
    const { topology } = hardwareAnalysis;
    const { model, targetBatchSize = 32, precision = 'mixed' } = trainingConfig;

    const reasoning: string[] = [];

    // Determine parallelism type
    const parallelismType = this.determineParallelismType(
      topology.gpus.length,
      model.parameters,
      reasoning,
    );

    // Calculate optimal degrees of parallelism
    const degrees = this.calculateParallelismDegrees(
      parallelismType,
      topology.gpus.length,
      model,
      reasoning,
    );

    // Calculate batch size and gradient accumulation
    const batchConfig = this.calculateBatchConfiguration(
      topology.gpus,
      targetBatchSize,
      degrees.dataParallelDegree,
      model,
      reasoning,
    );

    // Determine communication backend
    const communicationBackend = this.selectCommunicationBackend(
      topology,
      reasoning,
    );

    // Determine advanced optimizations
    const optimizations = this.selectOptimizations(
      topology.gpus,
      model,
      precision,
      reasoning,
    );

    // Estimate performance
    const performance = this.estimatePerformance(
      topology,
      batchConfig.globalBatchSize,
      model,
      degrees,
    );

    return {
      parallelismType,
      dataParallelDegree: degrees.dataParallelDegree,
      modelParallelDegree: degrees.modelParallelDegree,
      pipelineStages: degrees.pipelineStages,
      microBatchSize: batchConfig.microBatchSize,
      globalBatchSize: batchConfig.globalBatchSize,
      gradientAccumulationSteps: batchConfig.gradientAccumulationSteps,
      mixedPrecision:
        precision === 'mixed' || precision === 'fp16' || precision === 'bf16',
      activationCheckpointing: optimizations.activationCheckpointing,
      communicationBackend,
      optimizerSharding: optimizations.optimizerSharding,
      tensorParallel: optimizations.tensorParallel,
      sequenceParallel: optimizations.sequenceParallel,
      estimatedThroughput: performance.throughput,
      estimatedMemoryUsage: performance.memoryUsage,
      reasoning,
    };
  }

  /**
   * Determine the primary parallelism strategy
   */
  private determineParallelismType(
    numGPUs: number,
    modelParams: number,
    reasoning: string[],
  ): 'data' | 'model' | 'pipeline' | 'hybrid' {
    // Model size thresholds (in billions)
    const SMALL_MODEL = 1e9; // 1B parameters
    const MEDIUM_MODEL = 10e9; // 10B parameters
    const LARGE_MODEL = 50e9; // 50B parameters

    if (numGPUs === 0 || numGPUs === 1) {
      reasoning.push(
        'Single GPU/CPU setup: using standard training (no parallelism)',
      );
      return 'data';
    }

    if (modelParams < SMALL_MODEL) {
      // Small models: prefer data parallelism
      reasoning.push(
        `Model size (${(modelParams / 1e9).toFixed(1)}B params) is small: using data parallelism for optimal efficiency`,
      );
      return 'data';
    } else if (modelParams < MEDIUM_MODEL) {
      // Medium models: data parallelism or hybrid
      if (numGPUs >= 8) {
        reasoning.push(
          `Model size (${(modelParams / 1e9).toFixed(1)}B params) is medium with ${numGPUs} GPUs: using hybrid parallelism for better scaling`,
        );
        return 'hybrid';
      } else {
        reasoning.push(
          `Model size (${(modelParams / 1e9).toFixed(1)}B params) is medium: using data parallelism`,
        );
        return 'data';
      }
    } else if (modelParams < LARGE_MODEL) {
      // Large models: prefer hybrid
      reasoning.push(
        `Model size (${(modelParams / 1e9).toFixed(1)}B params) is large: using hybrid parallelism (data + model)`,
      );
      return 'hybrid';
    } else {
      // Very large models: pipeline + model parallelism
      reasoning.push(
        `Model size (${(modelParams / 1e9).toFixed(1)}B params) is very large: using hybrid parallelism with pipeline stages`,
      );
      return 'hybrid';
    }
  }

  /**
   * Calculate optimal degrees of parallelism
   */
  private calculateParallelismDegrees(
    parallelismType: string,
    numGPUs: number,
    model: ModelConfig,
    reasoning: string[],
  ): {
    dataParallelDegree: number;
    modelParallelDegree: number;
    pipelineStages: number;
  } {
    if (parallelismType === 'data') {
      reasoning.push(`Using all ${numGPUs} GPUs for data parallelism`);
      return {
        dataParallelDegree: numGPUs,
        modelParallelDegree: 1,
        pipelineStages: 1,
      };
    }

    if (parallelismType === 'model') {
      reasoning.push(`Using all ${numGPUs} GPUs for model parallelism`);
      return {
        dataParallelDegree: 1,
        modelParallelDegree: numGPUs,
        pipelineStages: 1,
      };
    }

    if (parallelismType === 'pipeline') {
      const stages = Math.min(numGPUs, model.numLayers || numGPUs);
      reasoning.push(`Using pipeline parallelism with ${stages} stages`);
      return {
        dataParallelDegree: 1,
        modelParallelDegree: 1,
        pipelineStages: stages,
      };
    }

    // Hybrid parallelism
    if (numGPUs <= 4) {
      reasoning.push(
        `Hybrid: 2-way model parallel, ${numGPUs / 2}-way data parallel`,
      );
      return {
        dataParallelDegree: numGPUs / 2,
        modelParallelDegree: 2,
        pipelineStages: 1,
      };
    } else if (numGPUs === 8) {
      reasoning.push('Hybrid: 2-way model parallel, 4-way data parallel');
      return {
        dataParallelDegree: 4,
        modelParallelDegree: 2,
        pipelineStages: 1,
      };
    } else {
      // Large scale: combine all strategies
      const modelParallel = 4;
      const pipeline = Math.min(4, Math.floor(numGPUs / modelParallel));
      const dataParallel = numGPUs / (modelParallel * pipeline);

      reasoning.push(
        `Hybrid: ${modelParallel}-way model parallel, ${pipeline} pipeline stages, ${dataParallel}-way data parallel`,
      );

      return {
        dataParallelDegree: dataParallel,
        modelParallelDegree: modelParallel,
        pipelineStages: pipeline,
      };
    }
  }

  /**
   * Calculate optimal batch configuration
   */
  private calculateBatchConfiguration(
    gpus: GPUInfo[],
    targetBatchSize: number,
    dataParallelDegree: number,
    model: ModelConfig,
    reasoning: string[],
  ): {
    microBatchSize: number;
    globalBatchSize: number;
    gradientAccumulationSteps: number;
  } {
    // Estimate memory per GPU (assuming 16GB default)
    const avgMemoryMB =
      gpus.reduce((sum, gpu) => sum + gpu.memory, 0) / (gpus.length || 1) ||
      16000;
    const memoryGB = avgMemoryMB / 1024;

    // Estimate memory per sample (very rough approximation)
    const modelSizeGB = (model.parameters * 4) / 1e9; // 4 bytes per param (fp32)
    const memoryPerSample = Math.max(0.5, modelSizeGB * 0.1); // Rough estimate

    // Calculate max micro batch size per GPU
    const maxMicroBatch = Math.floor((memoryGB * 0.7) / memoryPerSample);
    const microBatchSize = Math.max(
      1,
      Math.min(targetBatchSize / dataParallelDegree, maxMicroBatch),
    );

    // Calculate global batch size and gradient accumulation
    const naturalGlobalBatch = microBatchSize * dataParallelDegree;

    let gradientAccumulationSteps = 1;
    let globalBatchSize = naturalGlobalBatch;

    if (naturalGlobalBatch < targetBatchSize) {
      gradientAccumulationSteps = Math.ceil(
        targetBatchSize / naturalGlobalBatch,
      );
      globalBatchSize = naturalGlobalBatch * gradientAccumulationSteps;
      reasoning.push(
        `Using gradient accumulation (${gradientAccumulationSteps} steps) to reach target batch size of ${globalBatchSize}`,
      );
    } else {
      reasoning.push(
        `Micro batch size: ${microBatchSize}, global batch size: ${globalBatchSize}`,
      );
    }

    return {
      microBatchSize,
      globalBatchSize,
      gradientAccumulationSteps,
    };
  }

  /**
   * Select communication backend
   */
  private selectCommunicationBackend(
    topology: HardwareTopology,
    reasoning: string[],
  ): 'nccl' | 'gloo' | 'mpi' {
    if (topology.gpus.length > 0) {
      reasoning.push(
        'Using NCCL backend for GPU communication (optimal for NVIDIA GPUs)',
      );
      return 'nccl';
    } else {
      reasoning.push('Using Gloo backend for CPU-only training');
      return 'gloo';
    }
  }

  /**
   * Select optimization techniques
   */
  private selectOptimizations(
    gpus: GPUInfo[],
    model: ModelConfig,
    precision: string,
    reasoning: string[],
  ): {
    activationCheckpointing: boolean;
    optimizerSharding: boolean;
    tensorParallel: boolean;
    sequenceParallel: boolean;
  } {
    const avgMemoryMB =
      gpus.reduce((sum, gpu) => sum + gpu.memory, 0) / (gpus.length || 1) ||
      16000;
    const modelSizeGB = (model.parameters * 4) / 1e9;

    // Activation checkpointing for large models or limited memory
    const activationCheckpointing = modelSizeGB > 5 || avgMemoryMB < 20000;
    if (activationCheckpointing) {
      reasoning.push(
        'Enabling activation checkpointing to reduce memory usage (trades compute for memory)',
      );
    }

    // Optimizer sharding for models > 1B params with multiple GPUs
    const optimizerSharding = model.parameters > 1e9 && gpus.length > 1;
    if (optimizerSharding) {
      reasoning.push(
        'Enabling ZeRO optimizer sharding to distribute optimizer states across GPUs',
      );
    }

    // Tensor parallelism for very large models
    const tensorParallel = model.parameters > 10e9 && gpus.length >= 4;
    if (tensorParallel) {
      reasoning.push('Enabling tensor parallelism for very large model layers');
    }

    // Sequence parallelism for long sequences
    const sequenceParallel =
      (model.sequenceLength || 0) > 2048 && gpus.length >= 2;
    if (sequenceParallel) {
      reasoning.push(
        'Enabling sequence parallelism for long sequence processing',
      );
    }

    return {
      activationCheckpointing,
      optimizerSharding,
      tensorParallel,
      sequenceParallel,
    };
  }

  /**
   * Estimate training performance
   */
  private estimatePerformance(
    topology: HardwareTopology,
    globalBatchSize: number,
    model: ModelConfig,
    _degrees: {
      dataParallelDegree: number;
      modelParallelDegree: number;
      pipelineStages: number;
    },
  ): { throughput: number; memoryUsage: number } {
    // Very rough throughput estimation
    const gpuCount = topology.gpus.length || 1;
    const baselineThroughput = 100; // samples/second on single GPU (baseline)

    // Scale with GPUs (not linear due to communication overhead)
    const scalingEfficiency = Math.pow(gpuCount, 0.85);
    const throughput = baselineThroughput * scalingEfficiency;

    // Rough memory estimation
    const modelMemoryGB = (model.parameters * 4) / 1e9; // FP32
    const optimizerMemoryGB = modelMemoryGB * 2; // Adam uses 2x model params
    const activationMemoryGB = modelMemoryGB * 0.5; // Rough estimate

    const totalMemoryGB =
      modelMemoryGB + optimizerMemoryGB + activationMemoryGB;
    const memoryUsagePerGPU = (totalMemoryGB / gpuCount) * 1024; // Convert to MB

    return {
      throughput: Math.round(throughput),
      memoryUsage: Math.round(memoryUsagePerGPU),
    };
  }

  /**
   * Generate execution plan from strategy
   */
  generateExecutionPlan(
    strategy: TrainingStrategy,
    scriptPath: string = 'train.py',
  ): TrainingExecutionPlan {
    const envVars: Record<string, string> = {};
    const numGPUs =
      (strategy.dataParallelDegree || 1) *
      (strategy.modelParallelDegree || 1) *
      (strategy.pipelineStages || 1);

    // Set environment variables
    if (strategy.communicationBackend === 'nccl') {
      envVars['NCCL_DEBUG'] = 'INFO';
      envVars['NCCL_IB_DISABLE'] = '0';
    }

    // Build launch command
    let launchCommand = '';

    if (numGPUs > 1) {
      // Use torchrun for distributed training
      launchCommand = `torchrun --nproc_per_node=${numGPUs} --nnodes=1 ${scriptPath}`;
    } else {
      launchCommand = `python ${scriptPath}`;
    }

    // Add training arguments
    const args = [
      `--batch-size ${strategy.microBatchSize}`,
      `--gradient-accumulation-steps ${strategy.gradientAccumulationSteps || 1}`,
    ];

    if (strategy.mixedPrecision) {
      args.push('--mixed-precision');
    }

    if (strategy.activationCheckpointing) {
      args.push('--activation-checkpointing');
    }

    if (strategy.dataParallelDegree && strategy.dataParallelDegree > 1) {
      args.push(`--data-parallel ${strategy.dataParallelDegree}`);
    }

    if (strategy.modelParallelDegree && strategy.modelParallelDegree > 1) {
      args.push(`--model-parallel ${strategy.modelParallelDegree}`);
    }

    if (strategy.pipelineStages && strategy.pipelineStages > 1) {
      args.push(`--pipeline-stages ${strategy.pipelineStages}`);
    }

    launchCommand += ' ' + args.join(' ');

    // Worker distribution (single node for now)
    const workerDistribution = [
      {
        nodeId: 'node-0',
        gpuIds: Array.from({ length: numGPUs }, (_, i) => i),
        role: 'master',
      },
    ];

    return {
      strategy,
      environmentVariables: envVars,
      launchCommand,
      workerDistribution,
      monitoringConfig: {
        metricsInterval: 100,
        checkpointInterval: 1000,
        logLevel: 'INFO',
      },
    };
  }
}
