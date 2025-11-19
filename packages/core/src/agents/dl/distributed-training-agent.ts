/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { z } from 'zod';
import type { AgentDefinition } from '../types.js';

/**
 * Input schema for Distributed Training Agent.
 */
const DistributedTrainingInputSchema = z.object({
  /** Model specifications */
  model: z.object({
    architecture: z.string().describe('Model architecture name'),
    totalParameters: z.string().describe('Total parameter count (e.g., 7B, 70B)'),
    layerCount: z.number().describe('Number of layers'),
    hiddenSize: z.number().optional().describe('Hidden dimension size'),
    vocabSize: z.number().optional().describe('Vocabulary size for language models'),
    sequenceLength: z.number().optional().describe('Maximum sequence length'),
    attentionHeads: z.number().optional().describe('Number of attention heads'),
  }).describe('Model specifications'),

  /** Hardware configuration */
  hardware: z.object({
    nodeCount: z.number().describe('Number of compute nodes'),
    gpusPerNode: z.number().describe('GPUs per node'),
    gpuType: z.string().describe('GPU model'),
    gpuMemory: z.string().describe('Per-GPU memory'),
    interconnect: z.object({
      intraNode: z.enum(['NVLink', 'NVSwitch', 'PCIe']),
      interNode: z.enum(['InfiniBand', 'RoCE', 'Ethernet']),
      bandwidth: z.string().optional(),
    }),
    cpuMemoryPerNode: z.string().optional(),
    nvmePerNode: z.string().optional(),
  }).describe('Hardware configuration'),

  /** Training requirements */
  trainingRequirements: z.object({
    globalBatchSize: z.number().describe('Target global batch size'),
    microBatchSize: z.number().optional().describe('Preferred micro batch size'),
    sequenceLength: z.number().optional().describe('Training sequence length'),
    mixedPrecision: z.enum(['fp16', 'bf16', 'fp8', 'none']).optional(),
    checkpointFrequency: z.string().optional(),
  }).describe('Training requirements'),

  /** Parallelism preferences */
  preferences: z.object({
    preferredStrategies: z.array(z.enum([
      'data_parallel',
      'tensor_parallel',
      'pipeline_parallel',
      'sequence_parallel',
      'expert_parallel',
      'zero',
      'fsdp'
    ])).optional(),
    avoidStrategies: z.array(z.string()).optional(),
    memoryPriority: z.boolean().optional().describe('Prioritize memory efficiency over speed'),
    faultTolerance: z.boolean().optional().describe('Require fault tolerance'),
  }).optional().describe('Parallelism preferences'),

  /** Framework */
  framework: z.enum(['pytorch', 'megatron', 'deepspeed', 'fsdp', 'jax', 'tensorflow'])
    .describe('Deep learning framework'),

  /** Current configuration (if exists) */
  currentConfig: z.string().optional().describe('Current distributed configuration to analyze'),
});

/**
 * Output schema for Distributed Training Agent.
 */
const DistributedTrainingOutputSchema = z.object({
  /** Executive summary */
  Summary: z.string(),

  /** Memory analysis */
  MemoryAnalysis: z.object({
    modelStateMemory: z.object({
      parameters: z.string(),
      gradients: z.string(),
      optimizerStates: z.string(),
      total: z.string(),
    }),
    activationMemory: z.object({
      perLayer: z.string(),
      perMicroBatch: z.string(),
      total: z.string(),
      checkpointedTotal: z.string().optional(),
    }),
    totalPerGpu: z.string(),
    availablePerGpu: z.string(),
    requiresParallelism: z.boolean(),
    bottleneckAnalysis: z.string(),
  }),

  /** Recommended parallelism strategy */
  ParallelismStrategy: z.object({
    dataParallel: z.object({
      enabled: z.boolean(),
      degree: z.number(),
      implementation: z.string(),
      communicationPattern: z.string(),
    }),
    tensorParallel: z.object({
      enabled: z.boolean(),
      degree: z.number(),
      style: z.enum(['megatron', 'ulysses', 'ring_attention', 'none']).optional(),
      layersParallelized: z.array(z.string()).optional(),
      communicationOps: z.array(z.string()).optional(),
    }),
    pipelineParallel: z.object({
      enabled: z.boolean(),
      degree: z.number(),
      numMicroBatches: z.number().optional(),
      schedule: z.enum(['1F1B', 'interleaved', 'zero_bubble', 'chimera']).optional(),
      pipelineChunks: z.number().optional(),
      bubbleOverhead: z.string().optional(),
    }),
    sequenceParallel: z.object({
      enabled: z.boolean(),
      strategy: z.string().optional(),
      memoryReduction: z.string().optional(),
    }),
    expertParallel: z.object({
      enabled: z.boolean(),
      degree: z.number().optional(),
      routingStrategy: z.string().optional(),
    }),
    worldSize: z.number(),
    rationale: z.string(),
  }),

  /** Zero Redundancy Optimizer configuration */
  ZeroConfig: z.object({
    enabled: z.boolean(),
    stage: z.enum(['0', '1', '2', '3', 'infinity']),
    offloadOptimizer: z.object({
      enabled: z.boolean(),
      device: z.enum(['cpu', 'nvme']).optional(),
      pinMemory: z.boolean().optional(),
    }),
    offloadParams: z.object({
      enabled: z.boolean(),
      device: z.enum(['cpu', 'nvme']).optional(),
    }),
    contiguousGradients: z.boolean(),
    overlapComm: z.boolean(),
    reduceScatter: z.boolean(),
    reduceBucketSize: z.number().optional(),
    allgatherBucketSize: z.number().optional(),
    memoryEfficiency: z.string(),
    communicationOverhead: z.string(),
    implementation: z.string(),
  }),

  /** FSDP configuration (PyTorch alternative) */
  FSDPConfig: z.object({
    enabled: z.boolean(),
    shardingStrategy: z.enum([
      'FULL_SHARD',
      'SHARD_GRAD_OP',
      'NO_SHARD',
      'HYBRID_SHARD',
      '_HYBRID_SHARD_ZERO2'
    ]).optional(),
    backwardPrefetch: z.enum(['BACKWARD_PRE', 'BACKWARD_POST']).optional(),
    forwardPrefetch: z.boolean().optional(),
    mixedPrecision: z.object({
      paramDtype: z.string(),
      reduceDtype: z.string(),
      bufferDtype: z.string(),
    }).optional(),
    cpuOffload: z.boolean().optional(),
    wrapPolicy: z.string().optional(),
    stateCheckpointDtype: z.string().optional(),
    limitAllGathers: z.boolean().optional(),
    useOrigParams: z.boolean().optional(),
    implementation: z.string().optional(),
  }),

  /** Communication optimization */
  CommunicationOptimization: z.object({
    allReduceAlgorithm: z.enum(['ring', 'tree', 'recursive_halving', 'nccl_default']),
    gradientBucketing: z.object({
      enabled: z.boolean(),
      bucketSizeMb: z.number(),
    }),
    overlappedCommunication: z.object({
      enabled: z.boolean(),
      overlapType: z.array(z.string()),
    }),
    compressionConfig: z.object({
      enabled: z.boolean(),
      algorithm: z.enum(['fp16', 'bf16', 'powersgd', '1bit_adam', 'none']).optional(),
      compressionRatio: z.number().optional(),
    }),
    asyncCommunication: z.boolean(),
    hierarchicalAllReduce: z.boolean(),
    expectedOverhead: z.string(),
  }),

  /** Fault tolerance configuration */
  FaultTolerance: z.object({
    checkpointConfig: z.object({
      frequency: z.string(),
      asyncCheckpoint: z.boolean(),
      distributed: z.boolean(),
      format: z.string(),
      storageBackend: z.string(),
    }),
    elasticTraining: z.object({
      enabled: z.boolean(),
      minNodes: z.number().optional(),
      maxNodes: z.number().optional(),
      gracefulDegradation: z.boolean().optional(),
    }),
    errorRecovery: z.object({
      enableRetry: z.boolean(),
      maxRetries: z.number(),
      heartbeatInterval: z.number(),
      timeout: z.number(),
    }),
    preemptionHandling: z.object({
      enabled: z.boolean(),
      saveOnSignal: z.boolean(),
      signals: z.array(z.string()),
    }),
  }),

  /** Complete configuration files */
  ConfigurationFiles: z.object({
    deepspeedConfig: z.string().optional(),
    megatronConfig: z.string().optional(),
    torchrunCommand: z.string(),
    slurmScript: z.string().optional(),
    kubernetesManifest: z.string().optional(),
    modelCode: z.string(),
    trainingScript: z.string(),
  }),

  /** Performance projections */
  PerformanceProjections: z.object({
    theoreticalPeakTflops: z.number(),
    expectedTflops: z.number(),
    mfuProjection: z.string(),
    tokensPerSecond: z.number().optional(),
    samplesPerSecond: z.number().optional(),
    timePerStep: z.string(),
    communicationTime: z.string(),
    computeTime: z.string(),
    bubbleTime: z.string().optional(),
    scalingEfficiency: z.string(),
  }),

  /** Memory breakdown */
  MemoryBreakdown: z.object({
    perGpu: z.object({
      modelShards: z.string(),
      optimizerShards: z.string(),
      gradientShards: z.string(),
      activations: z.string(),
      temporaryBuffers: z.string(),
      cudaKernels: z.string(),
      total: z.string(),
    }),
    utilization: z.string(),
    headroom: z.string(),
  }),

  /** Debugging and profiling */
  DebuggingConfig: z.object({
    nccl: z.object({
      debug: z.string(),
      ibDisable: z.boolean().optional(),
      p2pDisable: z.boolean().optional(),
    }),
    cudaSettings: z.object({
      launchBlocking: z.boolean(),
      memoryStats: z.boolean(),
    }),
    profilingTools: z.array(z.object({
      tool: z.string(),
      usage: z.string(),
    })),
  }),

  /** Common issues and solutions */
  TroubleshootingGuide: z.array(z.object({
    issue: z.string(),
    symptoms: z.array(z.string()),
    cause: z.string(),
    solution: z.string(),
    preventiveMeasure: z.string(),
  })),

  /** Scaling recommendations */
  ScalingRecommendations: z.object({
    currentEfficiency: z.string(),
    strongScaling: z.array(z.object({
      nodeCount: z.number(),
      expectedEfficiency: z.string(),
      recommendation: z.string(),
    })),
    weakScaling: z.array(z.object({
      batchSize: z.number(),
      nodeCount: z.number(),
      expectedThroughput: z.string(),
    })),
    bottlenecksAtScale: z.array(z.string()),
  }),

  /** Next steps */
  NextSteps: z.array(z.object({
    priority: z.enum(['critical', 'high', 'medium', 'low']),
    action: z.string(),
    expectedBenefit: z.string(),
    complexity: z.enum(['simple', 'moderate', 'complex']),
  })),
});

/**
 * System prompt for Distributed Training Agent.
 */
const SYSTEM_PROMPT = `You are a world-class distributed deep learning expert with comprehensive knowledge of:

**Data Parallelism:**
- Distributed Data Parallel (DDP) with gradient synchronization
- Gradient bucketing and overlapped communication
- All-reduce algorithms: Ring, Tree, Recursive Halving-Doubling
- Large batch training with LARS/LAMB
- Local SGD and periodic averaging
- Asynchronous SGD and bounded staleness

**Tensor Parallelism:**
- Megatron-style column/row parallelism for linear layers
- Attention parallelism and sequence parallelism
- Cross-layer communication patterns (all-reduce, all-gather)
- Memory savings vs communication trade-offs
- Ulysses attention for long sequences
- Ring attention for extreme sequence lengths

**Pipeline Parallelism:**
- GPipe and PipeDream schedules
- 1F1B (one forward, one backward) scheduling
- Interleaved scheduling for reduced bubble
- Zero Bubble Pipeline Parallelism
- Micro-batch size optimization
- Memory-balanced stage partitioning
- Virtual pipeline stages
- Pipeline parallelism with tensor parallelism

**Zero Redundancy Optimizer (ZeRO):**
- ZeRO Stage 1: Optimizer state partitioning
- ZeRO Stage 2: Gradient partitioning
- ZeRO Stage 3: Parameter partitioning
- ZeRO-Infinity: NVMe offloading
- ZeRO-Offload: CPU offloading
- Communication volume analysis per stage
- When to use ZeRO vs 3D parallelism

**FSDP (Fully Sharded Data Parallel):**
- Sharding strategies: FULL_SHARD, SHARD_GRAD_OP, HYBRID_SHARD
- Backward/forward prefetching
- Mixed precision with FSDP
- Auto-wrap policies and module granularity
- CPU offloading configuration
- Rate limiting for memory pressure

**Communication Optimization:**
- NCCL tuning and configuration
- Hierarchical all-reduce for multi-node
- Gradient compression: PowerSGD, 1-bit Adam
- Asynchronous all-reduce with compute overlap
- Bucket sizing optimization
- InfiniBand vs Ethernet considerations
- NVLink and NVSwitch topology awareness

**Memory Optimization in Distributed Settings:**
- Activation checkpointing with tensor/pipeline parallelism
- Selective checkpointing based on recomputation cost
- Memory-efficient attention (Flash Attention) + TP
- Offloading strategies (CPU, NVMe)
- Parameter and gradient sharding

**Fault Tolerance:**
- Elastic training with dynamic membership
- Checkpointing strategies for distributed training
- Asynchronous distributed checkpointing
- Preemption handling for spot instances
- Automatic recovery and retry mechanisms
- Heartbeat and health monitoring

**Performance Analysis:**
- Model FLOPs Utilization (MFU)
- Hardware FLOPs Utilization (HFU)
- Communication/computation overlap analysis
- Bubble ratio analysis for pipeline parallelism
- Strong vs weak scaling efficiency
- Roofline model analysis

Your goal is to design optimal distributed training configurations that maximize training efficiency
while fitting within memory constraints and ensuring reliability.

Analysis Approach:
1. **Memory math first**: Calculate exact memory requirements
2. **Find the right parallelism mix**: Match strategy to model and hardware
3. **Optimize communication**: Minimize and overlap communication
4. **Ensure reliability**: Design for fault tolerance
5. **Provide complete configs**: Ready-to-use configuration files

Always provide:
- Exact memory breakdowns with calculations
- Complete configuration files (DeepSpeed, Megatron, FSDP)
- Launch commands (torchrun, srun)
- Performance projections with MFU estimates
- Debugging and profiling guidance
- Scaling analysis and recommendations

Use tools to:
- Calculate memory requirements precisely
- Generate complete configuration files
- Find optimal parallelism configurations
- Create launch scripts and manifests`;

/**
 * Distributed Training Agent - Expert distributed training configuration.
 *
 * Specializes in:
 * - Data, tensor, and pipeline parallelism
 * - ZeRO optimizer configuration
 * - Communication optimization
 * - Fault tolerance and checkpointing
 * - Memory-efficient distributed training
 * - Scaling analysis and optimization
 */
export const DistributedTrainingAgent: AgentDefinition<typeof DistributedTrainingOutputSchema> = {
  name: 'distributed_training_agent',
  description: 'Expert distributed training agent for configuring multi-GPU/multi-node training',
  version: '1.0.0',
  system_prompt: SYSTEM_PROMPT,
  input_schema: DistributedTrainingInputSchema,
  output_schema: DistributedTrainingOutputSchema,
  max_turns: 35,
  max_time_minutes: 35,
  thinking: {
    type: 'enabled',
    budget_tokens: 12000,
  },
  tool_config: {
    allowed_tools: [
      'read_file',
      'read_many_files',
      'grep',
      'glob',
      'write_file',
      'web_search',
    ],
    parallel_tool_calls: true,
  },
  model: 'gemini-2.0-flash-thinking-exp-01-21',
};

// Export utility types
export type DistributedTrainingInput = z.infer<typeof DistributedTrainingInputSchema>;
export type DistributedTrainingOutput = z.infer<typeof DistributedTrainingOutputSchema>;
