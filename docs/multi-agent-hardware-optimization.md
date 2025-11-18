# Multi-Agent Hardware Optimization

This feature provides automatic hardware topology analysis and optimal training
strategy generation for distributed deep learning workloads.

## Overview

The multi-agent hardware optimization system consists of three specialized
agents that work together to:

1. **Analyze Hardware** - Detect and benchmark CPUs, GPUs, memory, and
   interconnects
2. **Optimize Strategy** - Generate optimal parallelism and training
   configurations
3. **Execute Training** - Launch and monitor distributed training jobs

## Architecture

### Agents

#### 1. Hardware Analyzer Agent (`hardware_analyzer`)

Detects and analyzes system hardware topology:

- **GPUs**: NVIDIA (via nvidia-smi), AMD (via rocm-smi), Intel
- **CPUs**: Core count, architecture, frequency
- **Memory**: System RAM and swap
- **Interconnects**: NVLink, PCIe, InfiniBand
- **Communication**: Bandwidth and latency measurements

**Inputs:**

- `includeDetailedBenchmarks` (boolean, optional): Run actual benchmarks vs.
  estimates
- `targetWorkload` (string, optional): Description of intended workload

**Outputs:**

```json
{
  "HardwareSummary": "8x NVIDIA A100 80GB with NVLink",
  "GPUs": [...],
  "CPUCores": 64,
  "SystemMemoryGB": 512,
  "InterconnectType": "NVLink 3.0",
  "CommunicationBandwidth": {
    "gpuToGpu": 300,
    "cpuToGpu": 32
  },
  "Recommendations": [...],
  "Bottlenecks": [...],
  "HardwareScore": 85
}
```

#### 2. Training Strategy Agent (`training_strategy`)

Generates optimal training configuration based on hardware analysis:

- **Parallelism Strategy**: Data, model, pipeline, or hybrid
- **Batch Configuration**: Micro batch size, global batch size, gradient
  accumulation
- **Optimizations**: Mixed precision, activation checkpointing, optimizer
  sharding
- **Communication**: Backend selection (NCCL, Gloo, MPI)
- **Performance Estimates**: Throughput and memory usage predictions

**Inputs:**

- `hardwareReport` (string, required): JSON output from Hardware Analyzer Agent
- `modelParameters` (string, required): Total model parameters (e.g.,
  "7000000000")
- `targetBatchSize` (string, optional): Desired global batch size
- `modelType` (string, optional): Model architecture type
- `trainingScript` (string, optional): Path to training script

**Outputs:**

```json
{
  "StrategySummary": "8-way data parallelism with mixed precision",
  "ParallelismType": "data",
  "Configuration": {
    "dataParallelDegree": 8,
    "microBatchSize": 4,
    "globalBatchSize": 32,
    "gradientAccumulationSteps": 1
  },
  "Optimizations": {
    "mixedPrecision": true,
    "activationCheckpointing": true,
    "optimizerSharding": true
  },
  "LaunchCommand": "torchrun --nproc_per_node=8 train.py ...",
  "Reasoning": [...]
}
```

#### 3. Training Executor Agent (`training_executor`)

Executes the optimized training strategy:

- **Validation**: Checks environment and prerequisites
- **Setup**: Configures environment variables
- **Execution**: Launches training in background
- **Monitoring**: Captures initial metrics and logs
- **Reporting**: Provides status and next steps

**Inputs:**

- `trainingStrategy` (string, required): JSON output from Training Strategy
  Agent
- `trainingScript` (string, required): Path to training script
- `dryRun` (boolean, optional): Validate only, don't execute
- `workingDirectory` (string, optional): Execution directory

**Outputs:**

```json
{
  "ExecutionSummary": "Successfully launched 8-GPU training",
  "Status": "running",
  "LaunchCommand": "torchrun ...",
  "ProcessId": 12345,
  "LogsPath": "./training.log",
  "InitialMetrics": {...},
  "NextSteps": [...]
}
```

## Usage Examples

### Example 1: Complete Workflow

```typescript
// 1. Analyze hardware
const hardwareReport = await agents.invoke('hardware_analyzer', {
  includeDetailedBenchmarks: true,
  targetWorkload: 'training a 7B parameter LLM',
});

// 2. Generate optimal strategy
const trainingStrategy = await agents.invoke('training_strategy', {
  hardwareReport: JSON.stringify(hardwareReport),
  modelParameters: '7000000000',
  targetBatchSize: '32',
});

// 3. Execute training
const executionReport = await agents.invoke('training_executor', {
  trainingStrategy: JSON.stringify(trainingStrategy),
  trainingScript: './train_llm.py',
  dryRun: false,
});
```

### Example 2: Hardware Analysis Only

```typescript
const hardwareReport = await agents.invoke('hardware_analyzer', {
  includeDetailedBenchmarks: false,
});

console.log('Hardware Score:', hardwareReport.HardwareScore);
console.log('Recommendations:', hardwareReport.Recommendations);
console.log('Bottlenecks:', hardwareReport.Bottlenecks);
```

### Example 3: Strategy Comparison

```typescript
// Generate strategies for different model sizes
const models = [
  { name: '1B', params: '1000000000' },
  { name: '7B', params: '7000000000' },
  { name: '13B', params: '13000000000' },
];

for (const model of models) {
  const strategy = await agents.invoke('training_strategy', {
    hardwareReport: JSON.stringify(hwReport),
    modelParameters: model.params,
  });

  console.log(`\n${model.name} Model Strategy:`);
  console.log('Parallelism:', strategy.ParallelismType);
  console.log(
    'Estimated Throughput:',
    strategy.EstimatedPerformance.throughput,
  );
}
```

### Example 4: Dry Run Validation

```typescript
// Validate training setup without actually launching
const executionReport = await agents.invoke('training_executor', {
  trainingStrategy: JSON.stringify(strategy),
  trainingScript: './train.py',
  dryRun: true,
});

if (executionReport.Status === 'failed') {
  console.error('Validation failed:', executionReport.Errors);
} else {
  console.log('Validation passed! Ready to train.');
  console.log('Next steps:', executionReport.NextSteps);
}
```

## Hardware Detection Details

### GPU Detection

The system detects GPUs using vendor-specific tools:

- **NVIDIA**: `nvidia-smi` for GPU info, `nvidia-smi nvlink --status` for NVLink
- **AMD**: `rocm-smi` for GPU info and topology
- **Intel**: Basic detection via `lspci`

Detected information includes:

- Model name and ID
- Memory capacity
- Compute capability (NVIDIA)
- Bus topology
- Temperature, power, utilization

### CPU Detection

Uses standard system tools:

- `os.cpus()` for basic info
- `lscpu` for detailed architecture info on Linux
- Detects: cores, threads, frequency, architecture, vendor

### Memory Detection

- System memory via `os.totalmem()` and `os.freemem()`
- Swap info via `free -m` on Linux
- Per-GPU memory from vendor tools

### Interconnect Detection

- **NVLink**: Detected via `nvidia-smi nvlink --status`
- **PCIe**: Detected via `lspci`
- **InfiniBand**: Detected via `ibstat`

Measured metrics:

- Bandwidth (GB/s)
- Latency (microseconds)
- Topology type (mesh, ring, tree, all-to-all)

## Strategy Selection Logic

### Parallelism Strategy

| Model Size | GPUs | Strategy                         | Reasoning                          |
| ---------- | ---- | -------------------------------- | ---------------------------------- |
| < 1B       | Any  | Data Parallel                    | Simple, efficient for small models |
| 1-10B      | 1-4  | Data Parallel                    | Good scaling, fits in memory       |
| 1-10B      | 8+   | Hybrid                           | Better efficiency at scale         |
| 10-50B     | Any  | Hybrid (Data + Model)            | Balance memory and communication   |
| 50B+       | 8+   | Hybrid (Data + Model + Pipeline) | Necessary for very large models    |

### Optimization Selection

| Optimization              | When Enabled           | Purpose                             |
| ------------------------- | ---------------------- | ----------------------------------- |
| Mixed Precision           | Always (models > 1B)   | 2x memory reduction, faster compute |
| Activation Checkpointing  | Memory constrained     | Trade compute for memory            |
| Optimizer Sharding (ZeRO) | Models > 1B, multi-GPU | Distribute optimizer states         |
| Gradient Checkpointing    | Very large models      | Reduce activation memory            |
| Tensor Parallelism        | Models > 10B           | Split large layers                  |
| Sequence Parallelism      | Sequence length > 2048 | Handle long sequences               |

### Batch Size Calculation

Formula:

```
global_batch_size = micro_batch_size × data_parallel_degree × gradient_accumulation_steps
```

Constraints:

- Micro batch size must fit in GPU memory
- Global batch size should match training requirements
- Gradient accumulation bridges the gap

Memory estimation per GPU:

```
memory = model_weights + optimizer_states + activations + gradients

With mixed precision:
  model: params × 2 bytes (FP16)
  optimizer: params × 6 bytes (Adam mixed)
  activations: batch_size × layer_size × num_layers
  gradients: params × 2 bytes
```

## Communication Backends

### NCCL (NVIDIA Collective Communications Library)

- **Best for**: NVIDIA GPUs
- **Performance**: Excellent (optimized for NVLink and InfiniBand)
- **Use when**: Any NVIDIA GPU setup

### Gloo

- **Best for**: CPU-only or mixed CPU/GPU
- **Performance**: Good for CPU, adequate for GPU
- **Use when**: No NVIDIA GPUs or small-scale CPU training

### MPI (Message Passing Interface)

- **Best for**: HPC clusters with InfiniBand
- **Performance**: Excellent on specialized hardware
- **Use when**: Dedicated HPC environment

## Performance Estimation

The system estimates:

### Throughput (samples/second)

Based on:

- GPU count and compute capability
- Model size and complexity
- Batch size and parallelism strategy
- Communication overhead

Scaling efficiency = actual_speedup / ideal_speedup

- Data parallel: 85-95% efficiency (with good interconnect)
- Model parallel: 60-80% efficiency (communication bound)
- Pipeline parallel: 70-85% efficiency (bubble overhead)

### Memory Usage

Per-GPU memory = (model + optimizer + activations + gradients) / num_gpus

Optimizations impact:

- Mixed precision: ~50% reduction
- Activation checkpointing: ~30% activation reduction
- ZeRO Stage 2: Optimizer + gradient sharding
- ZeRO Stage 3: Model + optimizer + gradient sharding

## Error Handling

Common issues and solutions:

### CUDA Out of Memory

**Causes:**

- Batch size too large
- Model too large for single GPU
- Insufficient activation checkpointing

**Solutions:**

- Reduce micro batch size
- Enable activation checkpointing
- Increase gradient accumulation
- Use model parallelism

### NCCL Errors

**Causes:**

- Network issues
- NVLink problems
- Firewall blocking

**Solutions:**

- Check `nvidia-smi nvlink --status`
- Verify network connectivity
- Set `NCCL_DEBUG=INFO` for diagnostics
- Try `NCCL_IB_DISABLE=1` to disable InfiniBand

### Process Killed (OOM Killer)

**Causes:**

- Insufficient system memory
- Too many data loader workers
- Memory leak

**Solutions:**

- Reduce data loader workers
- Increase system memory
- Monitor memory usage

## Best Practices

1. **Always start with dry run**: Validate before launching expensive jobs
2. **Monitor initial execution**: Check first 5-10 minutes carefully
3. **Use mixed precision**: Almost always beneficial for modern GPUs
4. **Enable checkpointing**: Save progress regularly
5. **Log everything**: Comprehensive logging aids debugging
6. **Test incrementally**: Start small, scale up gradually
7. **Benchmark hardware**: Run detailed benchmarks for production workloads

## Limitations

- Single-node focus (multi-node requires additional coordination)
- PyTorch-centric (DeepSpeed/Megatron commands)
- Estimation-based performance predictions
- Limited support for non-NVIDIA GPUs
- Requires appropriate system tools (nvidia-smi, etc.)

## Future Enhancements

- Multi-node cluster support
- Auto-tuning via trial runs
- Dynamic strategy adjustment during training
- Support for more frameworks (TensorFlow, JAX)
- Cost optimization for cloud environments
- Integration with experiment tracking (W&B, MLflow)

## Contributing

To extend the hardware optimization system:

1. **Add new agent**: Create agent definition in `packages/core/src/agents/`
2. **Register agent**: Add to `AgentRegistry` in `registry.ts`
3. **Add hardware detector**: Extend `HardwareDetector` in
   `hardware/detector.ts`
4. **Add optimization**: Update `TrainingStrategyOptimizer` logic
5. **Update docs**: Document new capabilities

## References

- [PyTorch Distributed](https://pytorch.org/docs/stable/distributed.html)
- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/)
- [DeepSpeed](https://www.deepspeed.ai/)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- [ZeRO Optimization](https://arxiv.org/abs/1910.02054)
