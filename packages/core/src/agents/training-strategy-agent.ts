/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type { AgentDefinition } from './types.js';
import { DEFAULT_GEMINI_MODEL } from '../config/models.js';
import { z } from 'zod';

// Define the output schema for training strategy
const TrainingStrategySchema = z.object({
  StrategySummary: z
    .string()
    .describe('A concise summary of the recommended training strategy'),
  ParallelismType: z
    .enum(['data', 'model', 'pipeline', 'hybrid'])
    .describe('Primary parallelism strategy'),
  Configuration: z.object({
    dataParallelDegree: z.number().describe('Number of data parallel replicas'),
    modelParallelDegree: z
      .number()
      .optional()
      .describe('Degree of model parallelism'),
    pipelineStages: z.number().optional().describe('Number of pipeline stages'),
    microBatchSize: z.number().describe('Batch size per GPU'),
    globalBatchSize: z.number().describe('Total effective batch size'),
    gradientAccumulationSteps: z
      .number()
      .describe('Number of gradient accumulation steps'),
  }),
  Optimizations: z.object({
    mixedPrecision: z
      .boolean()
      .describe('Use mixed precision training (FP16/BF16)'),
    activationCheckpointing: z
      .boolean()
      .describe('Use activation checkpointing to save memory'),
    optimizerSharding: z
      .boolean()
      .describe('Use ZeRO optimizer state sharding'),
    gradientCheckpointing: z.boolean().describe('Use gradient checkpointing'),
  }),
  CommunicationBackend: z
    .enum(['nccl', 'gloo', 'mpi'])
    .describe('Recommended communication backend'),
  EstimatedPerformance: z.object({
    throughput: z
      .number()
      .describe('Estimated training throughput (samples/sec)'),
    memoryUsagePerGPU: z
      .number()
      .describe('Estimated memory usage per GPU (MB)'),
    scalingEfficiency: z
      .number()
      .describe('Expected scaling efficiency (0-100%)'),
  }),
  LaunchCommand: z.string().describe('Recommended command to launch training'),
  EnvironmentVariables: z
    .record(z.string())
    .describe('Recommended environment variables'),
  Reasoning: z.array(z.string()).describe('Explanation of strategy decisions'),
});

/**
 * A specialized subagent for generating optimal training strategies based on hardware analysis
 */
export const TrainingStrategyAgent: AgentDefinition<
  typeof TrainingStrategySchema
> = {
  name: 'training_strategy',
  displayName: 'Training Strategy Agent',
  description: `Specialized agent for generating optimal distributed training strategies based on hardware topology and model characteristics.
    Analyzes hardware capabilities, model size, and training objectives to recommend the best parallelism strategy,
    batch sizes, optimizations, and execution configuration. Returns a complete training strategy with launch commands.`,

  inputConfig: {
    inputs: {
      hardwareReport: {
        description:
          'JSON string containing the hardware analysis report from HardwareAnalyzerAgent',
        type: 'string',
        required: true,
      },
      modelParameters: {
        description:
          'Total number of model parameters (e.g., 7000000000 for 7B model)',
        type: 'string',
        required: true,
      },
      targetBatchSize: {
        description:
          'Target global batch size for training. Default: 32. Optional.',
        type: 'string',
        required: false,
      },
      modelType: {
        description:
          'Type of model (e.g., "transformer", "cnn", "diffusion"). Optional.',
        type: 'string',
        required: false,
      },
      trainingScript: {
        description:
          'Path to the training script. Default: train.py. Optional.',
        type: 'string',
        required: false,
      },
    },
  },

  outputConfig: {
    outputName: 'trainingStrategy',
    description: 'The optimal training strategy as a JSON object.',
    schema: TrainingStrategySchema,
  },

  processOutput: (output) => JSON.stringify(output, null, 2),

  modelConfig: {
    model: DEFAULT_GEMINI_MODEL,
    temp: 0.2,
    top_p: 0.95,
    thinkingBudget: 8192,
  },

  runConfig: {
    max_time_minutes: 3,
    max_turns: 8,
  },

  toolConfig: {
    // Strategy generation is mostly analytical, minimal tool access needed
    tools: [],
  },

  promptConfig: {
    query: `Your task is to generate the optimal training strategy based on the hardware analysis and model characteristics.

<hardware_report>
\${hardwareReport}
</hardware_report>

<model_config>
Model parameters: \${modelParameters}
Target batch size: \${targetBatchSize || "32"}
Model type: \${modelType || "transformer"}
Training script: \${trainingScript || "train.py"}
</model_config>

Analyze the hardware capabilities and model requirements to generate:
1. Optimal parallelism strategy (data, model, pipeline, or hybrid)
2. Specific configuration (batch sizes, parallelism degrees)
3. Recommended optimizations (mixed precision, checkpointing, etc.)
4. Performance estimates
5. Exact launch command and environment setup`,

    systemPrompt: `You are **Training Strategy Optimizer**, an expert AI agent specialized in designing optimal distributed training strategies for deep learning models.

Your **PRIMARY OBJECTIVE**: Generate the most efficient training configuration based on:
- Available hardware (GPUs, memory, interconnects)
- Model characteristics (size, architecture)
- Training objectives (throughput, memory efficiency)

## Core Directives

<RULES>
1. **HARDWARE-AWARE**: Always consider hardware limitations (GPU memory, bandwidth, CPU cores)
2. **MEMORY OPTIMIZATION**: Ensure the strategy fits within available GPU memory
3. **COMMUNICATION EFFICIENCY**: Minimize communication overhead, especially for limited bandwidth
4. **PRACTICAL**: Generate strategies that are actually implementable with common frameworks (PyTorch, DeepSpeed, Megatron)
5. **PERFORMANCE ORIENTED**: Optimize for training throughput while maintaining stability
</RULES>

## Strategy Selection Guidelines

### Parallelism Type Selection:

1. **Data Parallelism** (recommended for):
   - Models < 1B parameters
   - 1-8 GPUs with good interconnect
   - When batch size can be effectively scaled

2. **Model Parallelism** (recommended for):
   - Models that don't fit in single GPU memory
   - 10B+ parameter models
   - Limited data parallelism scaling

3. **Pipeline Parallelism** (recommended for):
   - Very large models (50B+ parameters)
   - When model can be naturally divided into stages
   - Good for models with sequential layer structure

4. **Hybrid Parallelism** (recommended for):
   - 10B-100B+ parameter models
   - 8+ GPUs available
   - Need to maximize both throughput and model capacity

### Batch Size Calculation:

- **Micro batch size**: Must fit in GPU memory
- **Global batch size**: Should match training requirements
- **Gradient accumulation**: Use to bridge the gap

Formula: global_batch_size = micro_batch_size × data_parallel_degree × gradient_accumulation_steps

### Memory Estimation:

Per GPU memory usage ≈ model_size + optimizer_state + activations + gradients

- Model (FP32): parameters × 4 bytes
- Optimizer (Adam): parameters × 12 bytes (FP32) or 6 bytes (mixed precision)
- Activations: depends on batch size and model architecture
- Gradients: same as model size

With mixed precision: ~2x reduction
With activation checkpointing: ~50% activation memory reduction
With ZeRO Stage 2: optimizer + gradient sharding across GPUs
With ZeRO Stage 3: model + optimizer + gradient sharding

### Optimization Recommendations:

**Always enable for large models (>1B params)**:
- Mixed precision (FP16/BF16)
- Gradient checkpointing/activation checkpointing

**Enable for very large models (>10B params)**:
- ZeRO optimizer sharding
- Gradient accumulation

**Enable for extreme models (>50B params)**:
- Full ZeRO Stage 3
- CPU offloading (if necessary)

### Communication Backend:

- **NCCL**: For NVIDIA GPUs (best performance)
- **Gloo**: For CPU-only or mixed CPU/GPU
- **MPI**: For specialized HPC setups with InfiniBand

### Launch Command Format:

For PyTorch Distributed:
\`\`\`bash
torchrun --nproc_per_node=<num_gpus> --nnodes=1 train.py [args]
\`\`\`

For DeepSpeed:
\`\`\`bash
deepspeed --num_gpus=<num_gpus> train.py --deepspeed ds_config.json
\`\`\`

### Environment Variables:

Common useful variables:
- NCCL_DEBUG=INFO (for debugging)
- NCCL_IB_DISABLE=0 (enable InfiniBand)
- CUDA_VISIBLE_DEVICES (GPU selection)
- OMP_NUM_THREADS (CPU threads per process)

## Analysis Process

1. **Parse Hardware Report**: Extract GPU count, memory, interconnect type
2. **Calculate Model Memory**: Estimate memory requirements for the model
3. **Determine Parallelism**: Choose strategy based on model size and GPU count
4. **Calculate Batch Sizes**: Determine optimal micro and global batch sizes
5. **Select Optimizations**: Choose memory and performance optimizations
6. **Estimate Performance**: Calculate expected throughput and efficiency
7. **Generate Launch Command**: Create exact command to run training
8. **Provide Reasoning**: Explain each decision

## Output Format

You MUST call the complete_task tool with a JSON object matching the schema.

Example output:
\`\`\`json
{
  "StrategySummary": "8-way data parallelism with mixed precision and gradient checkpointing for 7B parameter model",
  "ParallelismType": "data",
  "Configuration": {
    "dataParallelDegree": 8,
    "modelParallelDegree": 1,
    "pipelineStages": 1,
    "microBatchSize": 4,
    "globalBatchSize": 32,
    "gradientAccumulationSteps": 1
  },
  "Optimizations": {
    "mixedPrecision": true,
    "activationCheckpointing": true,
    "optimizerSharding": true,
    "gradientCheckpointing": true
  },
  "CommunicationBackend": "nccl",
  "EstimatedPerformance": {
    "throughput": 850,
    "memoryUsagePerGPU": 45000,
    "scalingEfficiency": 85
  },
  "LaunchCommand": "torchrun --nproc_per_node=8 --nnodes=1 train.py --batch-size 4 --gradient-accumulation-steps 1 --mixed-precision",
  "EnvironmentVariables": {
    "NCCL_DEBUG": "INFO",
    "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"
  },
  "Reasoning": [
    "8 GPUs with NVLink detected - excellent for data parallelism",
    "7B parameters × 4 bytes ≈ 28GB model size - fits in 80GB GPU with headroom",
    "Data parallelism chosen: simpler, better utilization for this model size",
    "Micro batch size 4: conservative to ensure memory safety",
    "Mixed precision enabled: 2x memory reduction, faster computation",
    "Gradient checkpointing: further memory optimization for larger batch sizes"
  ]
}
\`\`\`

When you complete your analysis, call the \`complete_task\` tool with the complete strategy.`,
  },
};
