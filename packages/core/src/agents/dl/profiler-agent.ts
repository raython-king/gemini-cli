/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type { AgentDefinition } from '../types.js';
import {
  GLOB_TOOL_NAME,
  GREP_TOOL_NAME,
  LS_TOOL_NAME,
  READ_FILE_TOOL_NAME,
  SHELL_TOOL_NAME,
  WRITE_FILE_TOOL_NAME,
} from '../../tools/tool-names.js';
import { DEFAULT_GEMINI_MODEL } from '../../config/models.js';
import { z } from 'zod';

/**
 * Schema for operation timing.
 */
const OperationTimingSchema = z.object({
  name: z.string().describe('Operation name.'),
  cpuTimeMs: z.number().describe('CPU time in milliseconds.'),
  cudaTimeMs: z.number().describe('CUDA time in milliseconds.'),
  callCount: z.number().describe('Number of times called.'),
  percentage: z.number().describe('Percentage of total time.'),
  memoryMB: z.number().optional().describe('Memory used in MB.'),
});

/**
 * Schema for GPU kernel analysis.
 */
const KernelAnalysisSchema = z.object({
  name: z.string().describe('Kernel name.'),
  gridSize: z.string().describe('Grid dimensions.'),
  blockSize: z.string().describe('Block dimensions.'),
  registers: z.number().optional().describe('Registers per thread.'),
  sharedMemory: z.number().optional().describe('Shared memory in bytes.'),
  occupancy: z.number().optional().describe('Theoretical occupancy percentage.'),
  durationUs: z.number().describe('Duration in microseconds.'),
  throughput: z.number().optional().describe('Memory throughput in GB/s.'),
});

/**
 * Schema for memory transfer analysis.
 */
const MemoryTransferSchema = z.object({
  direction: z.enum(['HOST_TO_DEVICE', 'DEVICE_TO_HOST', 'DEVICE_TO_DEVICE']),
  sizeBytes: z.number().describe('Transfer size in bytes.'),
  durationUs: z.number().describe('Duration in microseconds.'),
  bandwidthGBps: z.number().describe('Effective bandwidth in GB/s.'),
  isAsync: z.boolean().describe('Whether transfer was asynchronous.'),
  isPinned: z.boolean().describe('Whether host memory was pinned.'),
});

/**
 * Schema for roofline analysis.
 */
const RooflineAnalysisSchema = z.object({
  arithmeticIntensity: z.number().describe('FLOPs per byte of memory traffic.'),
  achievedTFLOPs: z.number().describe('Achieved compute throughput in TFLOP/s.'),
  peakTFLOPs: z.number().describe('Peak compute throughput for hardware.'),
  achievedBandwidthGBps: z.number().describe('Achieved memory bandwidth in GB/s.'),
  peakBandwidthGBps: z.number().describe('Peak memory bandwidth for hardware.'),
  limitingFactor: z.enum(['COMPUTE', 'MEMORY_BANDWIDTH']),
  distanceToRoofline: z.number().describe('Gap from theoretical maximum.'),
  optimizationPotential: z.string().describe('Estimated potential for improvement.'),
});

/**
 * Schema for bottleneck identification.
 */
const BottleneckSchema = z.object({
  type: z.enum([
    'DATA_LOADING',
    'CPU_PREPROCESSING',
    'HOST_DEVICE_TRANSFER',
    'GPU_COMPUTE',
    'MEMORY_BANDWIDTH',
    'KERNEL_LAUNCH_OVERHEAD',
    'SYNCHRONIZATION',
    'COMMUNICATION', // for distributed
  ]),
  severity: z.enum(['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']),
  description: z.string(),
  timeImpactMs: z.number().describe('Time impact per iteration in ms.'),
  suggestedFixes: z.array(z.string()),
  codeLocation: z.string().optional(),
});

/**
 * Schema for optimization recommendation.
 */
const OptimizationRecommendationSchema = z.object({
  category: z.enum([
    'DATA_PIPELINE',
    'MEMORY_OPTIMIZATION',
    'COMPUTE_OPTIMIZATION',
    'MIXED_PRECISION',
    'KERNEL_FUSION',
    'BATCHING',
    'DISTRIBUTED',
  ]),
  title: z.string(),
  description: z.string(),
  expectedSpeedup: z.string(),
  implementationEffort: z.enum(['LOW', 'MEDIUM', 'HIGH']),
  codeExample: z.string().optional(),
  priority: z.number().min(1).max(10),
});

/**
 * Schema for TensorBoard analysis.
 */
const TensorBoardAnalysisSchema = z.object({
  scalarSummary: z.object({
    trainLoss: z.object({
      initial: z.number(),
      final: z.number(),
      trend: z.enum(['DECREASING', 'STABLE', 'INCREASING', 'OSCILLATING']),
    }).optional(),
    learningRate: z.array(z.number()).optional(),
    gradientNorm: z.object({
      mean: z.number(),
      max: z.number(),
      hasSpikes: z.boolean(),
    }).optional(),
  }),
  histogramInsights: z.array(z.string()).optional(),
  imageInsights: z.array(z.string()).optional(),
  profilingInsights: z.array(z.string()).optional(),
});

/**
 * Complete profiling report schema.
 */
const ProfilingReportSchema = z.object({
  summary: z.string().describe('Executive summary of profiling analysis.'),
  framework: z.enum(['PYTORCH', 'TENSORFLOW', 'JAX', 'OTHER']),
  hardware: z.object({
    gpuModel: z.string(),
    gpuCount: z.number(),
    gpuMemoryGB: z.number(),
    cudaVersion: z.string().optional(),
    driverVersion: z.string().optional(),
  }),
  overallMetrics: z.object({
    totalTimeMs: z.number(),
    gpuUtilization: z.number(),
    gpuMemoryUtilization: z.number(),
    throughput: z.object({
      samplesPerSecond: z.number(),
      tokensPerSecond: z.number().optional(),
      imagesPerSecond: z.number().optional(),
    }),
    efficiency: z.number().describe('Overall efficiency score 0-100.'),
  }),
  topOperations: z.array(OperationTimingSchema),
  kernelAnalysis: z.array(KernelAnalysisSchema).optional(),
  memoryTransfers: z.array(MemoryTransferSchema).optional(),
  rooflineAnalysis: RooflineAnalysisSchema.optional(),
  bottlenecks: z.array(BottleneckSchema),
  tensorBoardAnalysis: TensorBoardAnalysisSchema.optional(),
  optimizationRecommendations: z.array(OptimizationRecommendationSchema),
  profilingScripts: z.array(z.object({
    name: z.string(),
    purpose: z.string(),
    code: z.string(),
  })).optional(),
  benchmarkComparison: z.object({
    currentPerformance: z.string(),
    targetPerformance: z.string(),
    industryBaseline: z.string().optional(),
    gapAnalysis: z.string(),
  }).optional(),
});

/**
 * DL Profiler Agent - Specialized in performance profiling and optimization.
 *
 * This agent provides expert-level profiling analysis for:
 * - PyTorch profiler integration
 * - TensorBoard log analysis
 * - GPU utilization analysis
 * - Memory profiling
 * - CPU/GPU transfer analysis
 * - Operation timing breakdown
 * - Roofline model analysis
 */
export const DLProfilerAgent: AgentDefinition<typeof ProfilingReportSchema> = {
  name: 'dl_profiler_agent',
  displayName: 'DL Profiler Agent',
  description: `A specialized agent for profiling and optimizing deep learning performance.
    Use this agent when you need to:
    - Analyze training/inference performance
    - Identify bottlenecks in data pipeline, compute, or memory
    - Optimize GPU utilization and throughput
    - Analyze TensorBoard logs for insights
    - Perform roofline analysis for hardware optimization

    Supports PyTorch, TensorFlow, and JAX with hardware-aware recommendations.`,

  inputConfig: {
    inputs: {
      profilingGoal: {
        description: `What you want to optimize:
          - Overall training speed
          - Inference latency
          - GPU memory usage
          - Throughput (samples/sec)
          - Specific operation performance`,
        type: 'string',
        required: true,
      },
      codebase: {
        description: `Path to the training/inference codebase`,
        type: 'string',
        required: false,
      },
      profileData: {
        description: `Path to existing profiling data:
          - PyTorch profiler traces (.json)
          - TensorBoard logs directory
          - nvprof/nsight profiles
          - Custom timing logs`,
        type: 'string',
        required: false,
      },
      hardwareSpec: {
        description: `Hardware specifications:
          - GPU model and count
          - Available memory
          - Interconnect type (NVLink, PCIe)`,
        type: 'string',
        required: false,
      },
      currentMetrics: {
        description: `Current performance metrics:
          - Training time per epoch
          - Samples per second
          - GPU utilization
          - Memory usage`,
        type: 'string',
        required: false,
      },
    },
  },

  outputConfig: {
    outputName: 'report',
    description: 'Comprehensive profiling analysis with optimization recommendations.',
    schema: ProfilingReportSchema,
  },

  processOutput: (output) => {
    let result = `# Deep Learning Profiling Report\n\n`;
    result += `## Summary\n${output.summary}\n\n`;
    result += `**Framework:** ${output.framework}\n`;
    result += `**Hardware:** ${output.hardware.gpuCount}x ${output.hardware.gpuModel} (${output.hardware.gpuMemoryGB}GB)\n\n`;

    result += `## Overall Metrics\n`;
    result += `| Metric | Value |\n|--------|-------|\n`;
    result += `| Total Time | ${output.overallMetrics.totalTimeMs.toFixed(2)} ms |\n`;
    result += `| GPU Utilization | ${output.overallMetrics.gpuUtilization.toFixed(1)}% |\n`;
    result += `| Memory Utilization | ${output.overallMetrics.gpuMemoryUtilization.toFixed(1)}% |\n`;
    result += `| Throughput | ${output.overallMetrics.throughput.samplesPerSecond.toFixed(2)} samples/sec |\n`;
    result += `| Efficiency Score | ${output.overallMetrics.efficiency}/100 |\n\n`;

    if (output.topOperations && output.topOperations.length > 0) {
      result += `## Top Operations by Time\n`;
      result += `| Operation | CUDA Time (ms) | CPU Time (ms) | Calls | % Total |\n`;
      result += `|-----------|----------------|---------------|-------|--------|\n`;
      output.topOperations.slice(0, 10).forEach((op) => {
        result += `| ${op.name} | ${op.cudaTimeMs.toFixed(2)} | ${op.cpuTimeMs.toFixed(2)} | ${op.callCount} | ${op.percentage.toFixed(1)}% |\n`;
      });
      result += `\n`;
    }

    if (output.bottlenecks && output.bottlenecks.length > 0) {
      result += `## Identified Bottlenecks\n\n`;
      output.bottlenecks.forEach((b, i) => {
        result += `### ${i + 1}. ${b.type} (${b.severity})\n`;
        result += `${b.description}\n`;
        result += `**Time Impact:** ${b.timeImpactMs.toFixed(2)} ms/iteration\n`;
        if (b.codeLocation) {
          result += `**Location:** ${b.codeLocation}\n`;
        }
        result += `**Suggested Fixes:**\n`;
        b.suggestedFixes.forEach((fix) => {
          result += `- ${fix}\n`;
        });
        result += `\n`;
      });
    }

    if (output.rooflineAnalysis) {
      result += `## Roofline Analysis\n`;
      result += `**Arithmetic Intensity:** ${output.rooflineAnalysis.arithmeticIntensity.toFixed(2)} FLOP/byte\n`;
      result += `**Achieved Performance:** ${output.rooflineAnalysis.achievedTFLOPs.toFixed(2)} TFLOP/s (Peak: ${output.rooflineAnalysis.peakTFLOPs.toFixed(2)})\n`;
      result += `**Achieved Bandwidth:** ${output.rooflineAnalysis.achievedBandwidthGBps.toFixed(2)} GB/s (Peak: ${output.rooflineAnalysis.peakBandwidthGBps.toFixed(2)})\n`;
      result += `**Limiting Factor:** ${output.rooflineAnalysis.limitingFactor}\n`;
      result += `**Optimization Potential:** ${output.rooflineAnalysis.optimizationPotential}\n\n`;
    }

    if (output.kernelAnalysis && output.kernelAnalysis.length > 0) {
      result += `## GPU Kernel Analysis\n`;
      result += `| Kernel | Grid | Block | Occupancy | Duration (us) |\n`;
      result += `|--------|------|-------|-----------|---------------|\n`;
      output.kernelAnalysis.slice(0, 10).forEach((k) => {
        result += `| ${k.name} | ${k.gridSize} | ${k.blockSize} | ${k.occupancy?.toFixed(1) || 'N/A'}% | ${k.durationUs.toFixed(2)} |\n`;
      });
      result += `\n`;
    }

    if (output.memoryTransfers && output.memoryTransfers.length > 0) {
      result += `## Memory Transfers\n`;
      const totalTransferTime = output.memoryTransfers.reduce((sum, t) => sum + t.durationUs, 0);
      result += `**Total Transfer Time:** ${(totalTransferTime / 1000).toFixed(2)} ms\n`;
      result += `**Host-to-Device:** ${output.memoryTransfers.filter(t => t.direction === 'HOST_TO_DEVICE').length} transfers\n`;
      result += `**Device-to-Host:** ${output.memoryTransfers.filter(t => t.direction === 'DEVICE_TO_HOST').length} transfers\n\n`;
    }

    if (output.tensorBoardAnalysis) {
      result += `## TensorBoard Analysis\n`;
      if (output.tensorBoardAnalysis.scalarSummary.trainLoss) {
        const loss = output.tensorBoardAnalysis.scalarSummary.trainLoss;
        result += `**Training Loss:** ${loss.initial.toFixed(4)} -> ${loss.final.toFixed(4)} (${loss.trend})\n`;
      }
      if (output.tensorBoardAnalysis.scalarSummary.gradientNorm) {
        const grad = output.tensorBoardAnalysis.scalarSummary.gradientNorm;
        result += `**Gradient Norm:** Mean ${grad.mean.toFixed(4)}, Max ${grad.max.toFixed(4)}${grad.hasSpikes ? ' (SPIKES DETECTED)' : ''}\n`;
      }
      if (output.tensorBoardAnalysis.profilingInsights) {
        result += `**Insights:**\n`;
        output.tensorBoardAnalysis.profilingInsights.forEach((insight) => {
          result += `- ${insight}\n`;
        });
      }
      result += `\n`;
    }

    result += `## Optimization Recommendations\n\n`;
    const sortedRecs = [...output.optimizationRecommendations].sort((a, b) => b.priority - a.priority);
    sortedRecs.forEach((rec, i) => {
      result += `### ${i + 1}. ${rec.title} (Priority: ${rec.priority}/10)\n`;
      result += `**Category:** ${rec.category}\n`;
      result += `**Effort:** ${rec.implementationEffort}\n`;
      result += `**Expected Speedup:** ${rec.expectedSpeedup}\n\n`;
      result += `${rec.description}\n`;
      if (rec.codeExample) {
        result += `\`\`\`python\n${rec.codeExample}\n\`\`\`\n`;
      }
      result += `\n`;
    });

    if (output.profilingScripts && output.profilingScripts.length > 0) {
      result += `## Profiling Scripts\n`;
      output.profilingScripts.forEach((script) => {
        result += `### ${script.name}\n`;
        result += `**Purpose:** ${script.purpose}\n`;
        result += `\`\`\`python\n${script.code}\n\`\`\`\n\n`;
      });
    }

    if (output.benchmarkComparison) {
      result += `## Benchmark Comparison\n`;
      result += `**Current:** ${output.benchmarkComparison.currentPerformance}\n`;
      result += `**Target:** ${output.benchmarkComparison.targetPerformance}\n`;
      if (output.benchmarkComparison.industryBaseline) {
        result += `**Industry Baseline:** ${output.benchmarkComparison.industryBaseline}\n`;
      }
      result += `**Gap Analysis:** ${output.benchmarkComparison.gapAnalysis}\n`;
    }

    return result;
  },

  modelConfig: {
    model: DEFAULT_GEMINI_MODEL,
    temp: 0.2,
    top_p: 0.95,
    thinkingBudget: -1,
  },

  runConfig: {
    max_time_minutes: 15,
    max_turns: 30,
  },

  toolConfig: {
    tools: [
      LS_TOOL_NAME,
      READ_FILE_TOOL_NAME,
      GLOB_TOOL_NAME,
      GREP_TOOL_NAME,
      SHELL_TOOL_NAME,
      WRITE_FILE_TOOL_NAME,
    ],
  },

  promptConfig: {
    query: `Analyze and optimize the following deep learning workload:

Profiling Goal:
<goal>
\${profilingGoal}
</goal>

Codebase Location:
<codebase>
\${codebase}
</codebase>

Profile Data:
<profile_data>
\${profileData}
</profile_data>

Hardware Specification:
<hardware>
\${hardwareSpec}
</hardware>

Current Metrics:
<metrics>
\${currentMetrics}
</metrics>

Perform comprehensive profiling analysis and provide optimization recommendations.`,

    systemPrompt: `You are the **DL Profiler Agent**, an expert AI for profiling and optimizing deep learning performance.

Your **CORE MISSION** is to analyze performance profiles, identify bottlenecks, and provide actionable optimization recommendations with hardware-aware insights.

## Expert Knowledge Base

### PyTorch Profiler Integration

**Basic Profiling Setup:**
\`\`\`python
import torch
from torch.profiler import profile, record_function, ProfilerActivity, schedule

# Define schedule for warm-up
my_schedule = schedule(
    skip_first=5,      # Skip first 5 steps (warm-up)
    wait=5,            # Wait 5 steps
    warmup=3,          # Warmup 3 steps
    active=5,          # Profile 5 steps
    repeat=2           # Repeat cycle 2 times
)

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=my_schedule,
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/profiler'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
    with_flops=True,
    with_modules=True,
) as prof:
    for step, (data, target) in enumerate(dataloader):
        if step >= 30:  # Stop after profiling
            break

        with record_function("data_transfer"):
            data = data.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        with record_function("forward"):
            output = model(data)
            loss = criterion(output, target)

        with record_function("backward"):
            optimizer.zero_grad()
            loss.backward()

        with record_function("optimizer_step"):
            optimizer.step()

        prof.step()

# Analyze results
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=20))
\`\`\`

**Advanced Analysis:**
\`\`\`python
# Export for visualization
prof.export_chrome_trace("trace.json")  # For Chrome tracing
prof.export_stacks("stacks.txt", "self_cuda_time_total")  # For flame graphs

# Memory timeline
from torch.cuda import memory_stats
stats = memory_stats()
print(f"Peak allocated: {stats['allocated_bytes.all.peak'] / 1e9:.2f} GB")
print(f"Peak reserved: {stats['reserved_bytes.all.peak'] / 1e9:.2f} GB")

# FLOPS analysis
from torch.profiler import FlopCounterMode
with FlopCounterMode(display=True) as fcm:
    output = model(input)
print(f"Total FLOPs: {fcm.get_total_flops() / 1e9:.2f} GFLOPs")
\`\`\`

### TensorBoard Analysis

**Interpreting Training Curves:**
1. **Loss Curve Analysis**
   - Smooth decrease: Good convergence
   - High variance: Learning rate too high or batch size too small
   - Plateaus: Learning rate decay needed or local minimum
   - Sudden spikes: Gradient explosion or data issues

2. **Gradient Histograms**
   - Check for vanishing (clustered near 0)
   - Check for exploding (spreading out)
   - Layer-by-layer analysis for problematic layers

3. **Weight Distributions**
   - Should evolve slowly during training
   - Sudden changes indicate instability
   - Dead neurons show as fixed distributions

**Reading TensorBoard Profiler:**
\`\`\`python
# Launch TensorBoard with profiler plugin
# tensorboard --logdir=./log/profiler --port=6006

# Key tabs to analyze:
# 1. Overview: GPU utilization, step time breakdown
# 2. Input Pipeline: Data loading efficiency
# 3. TensorFlow Stats / PyTorch Ops: Operation breakdown
# 4. GPU Kernel: Kernel-level analysis
# 5. Memory Profile: Memory timeline
\`\`\`

### GPU Utilization Analysis

**Understanding GPU Metrics:**
\`\`\`python
import subprocess
import time

def monitor_gpu_utilization(duration_sec=10, interval=0.5):
    """Monitor GPU utilization over time."""
    readings = []
    start = time.time()

    while time.time() - start < duration_sec:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )
        values = result.stdout.strip().split(', ')
        readings.append({
            'gpu_util': float(values[0]),
            'mem_util': float(values[1]),
            'mem_used': float(values[2]),
            'mem_total': float(values[3]),
            'temp': float(values[4]),
            'power': float(values[5]),
        })
        time.sleep(interval)

    # Analyze
    avg_gpu = sum(r['gpu_util'] for r in readings) / len(readings)
    avg_mem = sum(r['mem_util'] for r in readings) / len(readings)

    return {
        'avg_gpu_util': avg_gpu,
        'avg_mem_util': avg_mem,
        'readings': readings,
    }
\`\`\`

**Interpreting Utilization:**
- **< 30% GPU**: Major bottleneck elsewhere (data loading, CPU)
- **30-70% GPU**: Some inefficiency, optimization possible
- **> 90% GPU**: Well-optimized for current workload
- **100% Memory**: At limit, may need optimization

### Memory Profiling

**Detailed Memory Breakdown:**
\`\`\`python
import torch

def analyze_model_memory(model, input_shape, batch_size=1):
    """Analyze memory breakdown for a model."""

    # Parameter memory
    param_mem = sum(p.numel() * p.element_size() for p in model.parameters())

    # Gradient memory (same as parameters)
    grad_mem = param_mem

    # Optimizer memory (for Adam: momentum + variance = 2x params)
    optimizer_mem = param_mem * 2  # Adjust based on optimizer

    # Activation memory (estimated via forward pass)
    torch.cuda.reset_peak_memory_stats()
    model.cuda()

    with torch.no_grad():
        dummy_input = torch.randn(batch_size, *input_shape).cuda()
        _ = model(dummy_input)

    activation_mem = torch.cuda.max_memory_allocated() - param_mem

    return {
        'parameters_MB': param_mem / 1e6,
        'gradients_MB': grad_mem / 1e6,
        'optimizer_MB': optimizer_mem / 1e6,
        'activations_MB': activation_mem / 1e6,
        'total_MB': (param_mem + grad_mem + optimizer_mem + activation_mem) / 1e6,
    }

# Memory profiling during training
torch.cuda.memory._record_memory_history(
    enabled='all',
    context='all',
    stacks='python'
)

# ... run training ...

torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
\`\`\`

### CPU/GPU Transfer Analysis

**Identifying Transfer Bottlenecks:**
\`\`\`python
# In profiler output, look for:
# - cudaMemcpyAsync, cudaMemcpy operations
# - High CPU time with low CUDA time
# - Data loading operations

# Optimization techniques:
# 1. Use pinned memory
dataloader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,
    pin_memory=True,      # Critical for fast transfers
    prefetch_factor=2,    # Prefetch batches
    persistent_workers=True,  # Keep workers alive
)

# 2. Non-blocking transfers
data = data.cuda(non_blocking=True)
target = target.cuda(non_blocking=True)

# 3. Use CUDA streams
stream = torch.cuda.Stream()
with torch.cuda.stream(stream):
    data = data.cuda(non_blocking=True)
torch.cuda.current_stream().wait_stream(stream)
\`\`\`

### Operation Timing Breakdown

**Key Operations to Monitor:**

1. **Forward Pass Operations**
   - Matrix multiplications (gemm, mm)
   - Convolutions (cudnn_convolution)
   - Attention (scaled_dot_product_attention)
   - Activation functions (relu, gelu)
   - Normalization (batch_norm, layer_norm)

2. **Backward Pass Operations**
   - Usually 2-3x forward time
   - Gradient computation
   - Weight gradient computation

3. **Optimizer Operations**
   - Adam: moment updates
   - Memory-bound operations

**Identifying Slow Operations:**
\`\`\`python
# Parse profiler output
import pandas as pd

def analyze_operations(prof):
    """Extract and analyze operation timings."""
    events = []
    for event in prof.key_averages():
        events.append({
            'name': event.key,
            'cpu_time': event.cpu_time_total / 1000,  # ms
            'cuda_time': event.cuda_time_total / 1000,  # ms
            'calls': event.count,
            'cpu_memory': event.cpu_memory_usage / 1e6,  # MB
            'cuda_memory': event.cuda_memory_usage / 1e6,  # MB
        })

    df = pd.DataFrame(events)

    # Find operations taking >5% of total time
    total_cuda = df['cuda_time'].sum()
    df['percentage'] = df['cuda_time'] / total_cuda * 100

    significant = df[df['percentage'] > 5].sort_values('percentage', ascending=False)
    return significant
\`\`\`

### Roofline Model Analysis

**Understanding the Roofline Model:**
\`\`\`python
def roofline_analysis(achieved_flops, memory_bytes, duration_sec, hardware):
    """
    Perform roofline analysis.

    Args:
        achieved_flops: Total floating-point operations performed
        memory_bytes: Total bytes moved to/from memory
        duration_sec: Time taken
        hardware: Dict with peak_tflops and peak_bandwidth_gbps
    """
    # Calculate metrics
    arithmetic_intensity = achieved_flops / memory_bytes  # FLOP/byte
    achieved_throughput = achieved_flops / duration_sec / 1e12  # TFLOP/s
    achieved_bandwidth = memory_bytes / duration_sec / 1e9  # GB/s

    # Determine ridge point
    ridge_point = hardware['peak_tflops'] / hardware['peak_bandwidth_gbps']

    # Determine limiting factor
    if arithmetic_intensity < ridge_point:
        limiting_factor = 'MEMORY_BANDWIDTH'
        theoretical_max = arithmetic_intensity * hardware['peak_bandwidth_gbps']
    else:
        limiting_factor = 'COMPUTE'
        theoretical_max = hardware['peak_tflops']

    efficiency = achieved_throughput / theoretical_max * 100

    return {
        'arithmetic_intensity': arithmetic_intensity,
        'achieved_tflops': achieved_throughput,
        'theoretical_max_tflops': theoretical_max,
        'efficiency': efficiency,
        'limiting_factor': limiting_factor,
        'ridge_point': ridge_point,
    }

# Hardware specs (example for A100)
a100_spec = {
    'peak_tflops': 19.5,  # FP32
    'peak_bandwidth_gbps': 2039,  # HBM2e
}
\`\`\`

### Hardware-Specific Optimizations

**NVIDIA GPU Optimization:**

*Tensor Cores (V100, A100, H100):*
\`\`\`python
# Enable tensor cores via mixed precision
from torch.cuda.amp import autocast

# Dimensions should be multiples of 8 for optimal tensor core usage
# For A100: multiples of 64 for best performance

# Example: Pad embedding dimension
embedding_dim = 768  # Multiple of 64 for A100
hidden_dim = 3072    # Multiple of 64

# Use TF32 on A100
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
\`\`\`

*cuDNN Optimizations:*
\`\`\`python
# Enable cuDNN autotuner (benchmark mode)
torch.backends.cudnn.benchmark = True

# Warning: Only use with fixed input sizes
# Disable for variable-length sequences
\`\`\`

*Memory Efficiency:*
\`\`\`python
# Use memory-efficient attention (PyTorch 2.0+)
from torch.nn.functional import scaled_dot_product_attention

# This automatically selects the best algorithm:
# - Flash Attention
# - Memory-efficient attention
# - Standard attention
output = scaled_dot_product_attention(query, key, value)
\`\`\`

### Common Bottleneck Patterns

1. **Data Loading Bottleneck**
   - Symptoms: Low GPU utilization, high data loading time
   - Solutions: More workers, prefetching, memory-mapped data, caching

2. **Small Batch Operations**
   - Symptoms: Many small CUDA kernels, high launch overhead
   - Solutions: torch.compile, operator fusion, increase batch size

3. **Synchronization Overhead**
   - Symptoms: Many cudaStreamSynchronize calls
   - Solutions: Async operations, reduce .item() calls, batch logging

4. **Memory Bandwidth Bound**
   - Symptoms: Low compute utilization, high memory throughput
   - Solutions: Mixed precision, better memory access patterns

5. **Distributed Communication**
   - Symptoms: High time in NCCL operations
   - Solutions: Gradient compression, async communication, better topology

## Systematic Profiling Process

### Phase 1: Baseline Measurement
1. Run profiler with default settings
2. Collect overall metrics
3. Identify major time consumers

### Phase 2: Deep Dive
1. Profile specific operations
2. Analyze kernel-level performance
3. Check memory patterns

### Phase 3: Roofline Analysis
1. Calculate arithmetic intensity
2. Determine limiting factor
3. Estimate optimization potential

### Phase 4: Optimization
1. Prioritize by impact and effort
2. Apply optimizations incrementally
3. Re-profile after each change

### Phase 5: Validation
1. Ensure correctness preserved
2. Measure actual speedup
3. Compare against targets

## Tool Usage Strategy

- **shell:** Run profiling commands, nvidia-smi
- **read_file:** Examine code, profile outputs, TensorBoard logs
- **write_file:** Create profiling scripts, optimization code
- **grep/glob:** Find performance-critical code sections

## Best Practices

**DO:**
- Always profile before optimizing
- Focus on the biggest bottlenecks first
- Measure impact of each optimization
- Consider hardware constraints
- Use hardware-specific features

**DON'T:**
- Optimize without measuring
- Assume CPU patterns apply to GPU
- Ignore memory bandwidth limits
- Over-optimize non-critical paths
- Forget to verify correctness

Remember: Profiling is measurement, not guessing. Let the data guide your optimization efforts.`,
  },
};
