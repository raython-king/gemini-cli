/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { z } from 'zod';
import {
  SkillCategory,
  SkillComplexity,
  type SkillDefinition,
  type SkillContext,
  type SkillResult,
} from '../types.js';
import { AgentExecutor } from '../../agents/executor.js';
import { DLProfilerAgent } from '../../agents/dl/profiler-agent.js';
import { DLMemoryAnalyzerAgent } from '../../agents/dl/memory-analyzer-agent.js';

const ProfilingInputSchema = z.object({
  goal: z.enum([
    'training_speed',
    'inference_latency',
    'memory_optimization',
    'throughput',
    'gpu_utilization',
    'comprehensive',
  ]).describe('Primary optimization goal'),
  codebasePath: z.string().optional().describe('Path to the training/inference code'),
  profileDataPath: z.string().optional().describe('Path to existing profile data'),
  framework: z.enum(['pytorch', 'tensorflow', 'jax']).default('pytorch'),
  hardware: z.object({
    gpuModel: z.string().describe('GPU model name'),
    gpuCount: z.number().describe('Number of GPUs'),
    gpuMemoryGB: z.number().describe('GPU memory in GB'),
    cudaVersion: z.string().optional(),
  }).optional(),
  currentMetrics: z.object({
    throughput: z.number().optional().describe('Current samples/sec'),
    latency: z.number().optional().describe('Current latency in ms'),
    memoryUsage: z.number().optional().describe('Current memory usage in GB'),
    gpuUtilization: z.number().optional().describe('Current GPU utilization %'),
  }).optional(),
  targetMetrics: z.object({
    throughput: z.number().optional().describe('Target samples/sec'),
    latency: z.number().optional().describe('Target latency in ms'),
    memoryUsage: z.number().optional().describe('Target memory usage in GB'),
  }).optional(),
  constraints: z.object({
    maxMemory: z.number().optional(),
    minAccuracy: z.number().optional(),
    maxLatency: z.number().optional(),
  }).optional(),
});

const ProfilingOutputSchema = z.object({
  success: z.boolean(),
  summary: z.string(),
  currentPerformance: z.object({
    throughput: z.number(),
    latency: z.number(),
    memoryUsage: z.number(),
    gpuUtilization: z.number(),
    efficiency: z.number().describe('Overall efficiency score 0-100'),
  }),
  bottlenecks: z.array(z.object({
    type: z.string(),
    severity: z.enum(['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']),
    description: z.string(),
    impact: z.string(),
    solution: z.string(),
  })),
  optimizations: z.array(z.object({
    title: z.string(),
    category: z.string(),
    currentValue: z.string(),
    targetValue: z.string(),
    expectedSpeedup: z.string(),
    effort: z.enum(['LOW', 'MEDIUM', 'HIGH']),
    priority: z.number().min(1).max(10),
    implementation: z.object({
      steps: z.array(z.string()),
      code: z.string().optional(),
      dependencies: z.array(z.string()).optional(),
    }),
  })),
  profilingSetup: z.object({
    setupCode: z.string(),
    runCommand: z.string(),
    analysisSteps: z.array(z.string()),
  }),
  hardwareTuning: z.array(z.object({
    setting: z.string(),
    currentValue: z.string(),
    recommendedValue: z.string(),
    impact: z.string(),
  })).optional(),
  benchmarkComparison: z.object({
    currentVsTarget: z.string(),
    currentVsBaseline: z.string(),
    recommendations: z.array(z.string()),
  }).optional(),
  artifacts: z.array(z.string()).optional(),
});

type ProfilingInput = z.infer<typeof ProfilingInputSchema>;
type ProfilingOutput = z.infer<typeof ProfilingOutputSchema>;

/**
 * Hardware-specific optimization database.
 */
const HARDWARE_OPTIMIZATIONS: Record<string, {
  tensorCoreOptimal: number[];
  memoryBandwidth: number;
  computePeak: number;
  recommendations: string[];
}> = {
  'A100': {
    tensorCoreOptimal: [64, 128, 256],  // Dimensions for optimal tensor core usage
    memoryBandwidth: 2039,  // GB/s
    computePeak: 312,  // TFLOP/s FP16
    recommendations: [
      'Use BF16 for best tensor core utilization',
      'Align dimensions to multiples of 64',
      'Enable TF32 for FP32 operations',
      'Use Flash Attention for transformers',
    ],
  },
  'H100': {
    tensorCoreOptimal: [64, 128, 256],
    memoryBandwidth: 3350,  // GB/s
    computePeak: 989,  // TFLOP/s FP16
    recommendations: [
      'Use FP8 for supported operations',
      'Leverage enhanced transformer engine',
      'Use NVLink for multi-GPU scaling',
      'Enable CUDA graphs for inference',
    ],
  },
  'V100': {
    tensorCoreOptimal: [8, 16, 32],
    memoryBandwidth: 900,  // GB/s
    computePeak: 125,  // TFLOP/s FP16
    recommendations: [
      'Use FP16 with loss scaling',
      'Align dimensions to multiples of 8',
      'Enable cudnn benchmark mode',
    ],
  },
  '4090': {
    tensorCoreOptimal: [64, 128, 256],
    memoryBandwidth: 1008,  // GB/s
    computePeak: 165,  // TFLOP/s FP16
    recommendations: [
      'Use BF16 or FP16',
      'Optimize for consumer GPU memory (24GB)',
      'Consider quantization for large models',
    ],
  },
};

/**
 * Profiling setup templates for different frameworks and goals.
 */
const PROFILING_TEMPLATES = {
  pytorch: {
    training_speed: `import torch
from torch.profiler import profile, ProfilerActivity, schedule

# Define profiler schedule
profiler_schedule = schedule(
    skip_first=5,    # Skip first iterations (warmup)
    wait=5,          # Wait iterations
    warmup=3,        # Warmup iterations
    active=5,        # Active profiling iterations
    repeat=2         # Repeat cycles
)

# Create profiler
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=profiler_schedule,
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/profiler'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
    with_flops=True,
    with_modules=True,
) as prof:
    for step, batch in enumerate(dataloader):
        if step >= 30:
            break

        # Your training step
        loss = train_step(model, batch, optimizer, criterion)

        prof.step()

# Print results
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
prof.export_chrome_trace("trace.json")`,

    inference_latency: `import torch
import time
import numpy as np

def benchmark_inference(model, input_shape, batch_sizes=[1, 8, 32], warmup=50, iterations=200):
    model.eval()
    results = {}

    for batch_size in batch_sizes:
        dummy_input = torch.randn(batch_size, *input_shape, device='cuda')

        # Warmup
        for _ in range(warmup):
            with torch.no_grad():
                _ = model(dummy_input)
        torch.cuda.synchronize()

        # Benchmark
        latencies = []
        for _ in range(iterations):
            start = time.perf_counter()
            with torch.no_grad():
                _ = model(dummy_input)
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000)

        results[batch_size] = {
            'mean_ms': np.mean(latencies),
            'p50_ms': np.percentile(latencies, 50),
            'p99_ms': np.percentile(latencies, 99),
            'throughput': batch_size * 1000 / np.mean(latencies),
        }

        print(f"Batch {batch_size}: {results[batch_size]['mean_ms']:.2f}ms, "
              f"{results[batch_size]['throughput']:.1f} samples/sec")

    return results`,

    memory_optimization: `import torch

def profile_memory(model, input_shape, batch_size=1):
    # Record memory history
    torch.cuda.memory._record_memory_history(max_entries=100000)

    # Reset statistics
    torch.cuda.reset_peak_memory_stats()

    # Forward pass
    model.train()
    dummy_input = torch.randn(batch_size, *input_shape, device='cuda')
    output = model(dummy_input)

    # Backward pass
    loss = output.sum()
    loss.backward()

    # Get statistics
    stats = torch.cuda.memory_stats()
    peak = torch.cuda.max_memory_allocated() / 1e9

    print(f"Peak memory: {peak:.2f} GB")

    # Save snapshot
    torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")

    return peak`,

    throughput: `import torch
import time

def measure_throughput(model, dataloader, num_batches=100):
    model.train()

    # Warmup
    for i, batch in enumerate(dataloader):
        if i >= 10:
            break
        _ = model(batch[0].cuda())
    torch.cuda.synchronize()

    # Measure
    total_samples = 0
    start = time.time()

    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break

        data = batch[0].cuda(non_blocking=True)
        output = model(data)
        loss = output.sum()
        loss.backward()

        total_samples += data.size(0)

    torch.cuda.synchronize()
    elapsed = time.time() - start

    throughput = total_samples / elapsed
    print(f"Throughput: {throughput:.2f} samples/sec")
    print(f"Time per batch: {elapsed / num_batches * 1000:.2f} ms")

    return throughput`,

    gpu_utilization: `import subprocess
import time
import threading

class GPUMonitor:
    def __init__(self, interval=0.1):
        self.interval = interval
        self.readings = []
        self.running = False

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._monitor)
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()
        return self._analyze()

    def _monitor(self):
        while self.running:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True
            )
            values = result.stdout.strip().split(', ')
            self.readings.append({
                'gpu': float(values[0]),
                'mem': float(values[1]),
            })
            time.sleep(self.interval)

    def _analyze(self):
        gpu_utils = [r['gpu'] for r in self.readings]
        return {
            'mean': sum(gpu_utils) / len(gpu_utils),
            'min': min(gpu_utils),
            'max': max(gpu_utils),
        }

# Usage
monitor = GPUMonitor()
monitor.start()
# ... run training ...
results = monitor.stop()
print(f"GPU Utilization: {results['mean']:.1f}%")`,
  },
};

/**
 * DL Profiling Skill
 *
 * Expert skill for profiling and optimizing deep learning performance.
 * Provides comprehensive profiling setup, analysis, and hardware-specific tuning.
 */
export const DLProfilingSkill: SkillDefinition<ProfilingInput, ProfilingOutput> = {
  id: 'dl.profiling',
  name: 'DL Profiling',
  description: 'Profile and optimize deep learning performance with hardware-aware recommendations',
  usage: `
Profile and optimize deep learning workloads:

Goals:
  - training_speed: Optimize training iteration time
  - inference_latency: Minimize inference latency
  - memory_optimization: Reduce GPU memory usage
  - throughput: Maximize samples per second
  - gpu_utilization: Improve GPU efficiency
  - comprehensive: Full analysis of all aspects

Provides:
  - Profiling setup code
  - Bottleneck identification
  - Optimization recommendations
  - Hardware-specific tuning
  - Benchmark comparisons
  `,
  category: SkillCategory.OPTIMIZATION,
  complexity: SkillComplexity.ADVANCED,
  version: '1.0.0',
  author: 'Gemini CLI',
  inputSchema: ProfilingInputSchema,
  outputSchema: ProfilingOutputSchema,
  requiredAgents: ['dl_profiler_agent'],
  optionalAgents: ['dl_memory_analyzer_agent'],
  estimatedTime: '10-20 minutes',
  tags: ['profiling', 'optimization', 'performance', 'gpu', 'memory', 'throughput', 'latency'],

  examples: [
    {
      title: 'Optimize training speed',
      description: 'Profile and optimize training iteration time',
      input: {
        goal: 'training_speed',
        framework: 'pytorch',
        hardware: {
          gpuModel: 'A100',
          gpuCount: 1,
          gpuMemoryGB: 40,
        },
        currentMetrics: {
          throughput: 100,
          gpuUtilization: 50,
        },
      },
    },
    {
      title: 'Optimize inference latency',
      description: 'Minimize inference latency for deployment',
      input: {
        goal: 'inference_latency',
        framework: 'pytorch',
        hardware: {
          gpuModel: '4090',
          gpuCount: 1,
          gpuMemoryGB: 24,
        },
        targetMetrics: {
          latency: 10,  // Target 10ms
        },
      },
    },
    {
      title: 'Memory optimization',
      description: 'Reduce memory to fit larger batches',
      input: {
        goal: 'memory_optimization',
        framework: 'pytorch',
        hardware: {
          gpuModel: 'V100',
          gpuCount: 8,
          gpuMemoryGB: 32,
        },
        constraints: {
          maxMemory: 28,  // Target 28GB max
        },
      },
    },
  ],

  async execute(
    input: ProfilingInput,
    context: SkillContext,
    config: any,
  ): Promise<SkillResult<ProfilingOutput>> {
    const startTime = Date.now();
    const agentsUsed: string[] = [];

    try {
      // Run the profiler agent
      const profilerExecutor = await AgentExecutor.create(
        DLProfilerAgent,
        config,
      );
      agentsUsed.push('dl_profiler_agent');

      const profilerResult = await profilerExecutor.run({
        profilingGoal: `${input.goal} optimization`,
        codebase: input.codebasePath || context.workingDir,
        profileData: input.profileDataPath,
        hardwareSpec: input.hardware ?
          `${input.hardware.gpuCount}x ${input.hardware.gpuModel} (${input.hardware.gpuMemoryGB}GB)` : undefined,
        currentMetrics: input.currentMetrics ?
          `Throughput: ${input.currentMetrics.throughput}, Latency: ${input.currentMetrics.latency}ms, GPU Util: ${input.currentMetrics.gpuUtilization}%` : undefined,
      });

      // For memory optimization, also run memory analyzer
      let memoryAnalysis: any = null;
      if (input.goal === 'memory_optimization' || input.goal === 'comprehensive') {
        try {
          const memoryExecutor = await AgentExecutor.create(
            DLMemoryAnalyzerAgent,
            config,
          );
          agentsUsed.push('dl_memory_analyzer_agent');

          const memResult = await memoryExecutor.run({
            memoryIssue: `Optimize memory for ${input.goal}`,
            codebase: input.codebasePath || context.workingDir,
            hardwareSpec: input.hardware ?
              `${input.hardware.gpuCount}x ${input.hardware.gpuModel} (${input.hardware.gpuMemoryGB}GB)` : undefined,
          });
          memoryAnalysis = JSON.parse(memResult.result);
        } catch {
          // Memory analysis is optional
        }
      }

      // Parse agent result
      const agentData = JSON.parse(profilerResult.result);

      // Get hardware-specific optimizations
      const gpuOptimizations = input.hardware?.gpuModel ?
        this.getHardwareOptimizations(input.hardware.gpuModel) : null;

      // Generate profiling setup
      const profilingSetup = this.generateProfilingSetup(input);

      // Generate optimizations based on goal
      const optimizations = this.generateOptimizations(input, agentData, gpuOptimizations);

      // Build output
      const output: ProfilingOutput = {
        success: true,
        summary: agentData.summary || `Profiling analysis for ${input.goal} optimization`,

        currentPerformance: {
          throughput: input.currentMetrics?.throughput || agentData.overallMetrics?.throughput?.samplesPerSecond || 0,
          latency: input.currentMetrics?.latency || agentData.overallMetrics?.totalTimeMs || 0,
          memoryUsage: input.currentMetrics?.memoryUsage || (memoryAnalysis?.overallMemoryUsage?.peakMemoryMB / 1000) || 0,
          gpuUtilization: input.currentMetrics?.gpuUtilization || agentData.overallMetrics?.gpuUtilization || 0,
          efficiency: agentData.overallMetrics?.efficiency || this.calculateEfficiency(input.currentMetrics),
        },

        bottlenecks: agentData.bottlenecks?.map((b: any) => ({
          type: b.type,
          severity: b.severity,
          description: b.description,
          impact: `${b.timeImpactMs?.toFixed(2) || '?'} ms per iteration`,
          solution: b.suggestedFixes?.[0] || 'See optimization recommendations',
        })) || this.inferBottlenecks(input),

        optimizations: optimizations,

        profilingSetup: profilingSetup,

        hardwareTuning: gpuOptimizations ? [{
          setting: 'Tensor Core Alignment',
          currentValue: 'Unknown',
          recommendedValue: `Dimensions multiples of ${gpuOptimizations.tensorCoreOptimal[0]}`,
          impact: 'Up to 2x speedup for matrix operations',
        }, {
          setting: 'Mixed Precision',
          currentValue: 'FP32',
          recommendedValue: 'BF16 or FP16',
          impact: 'Up to 2x memory reduction and speedup',
        }] : undefined,

        benchmarkComparison: {
          currentVsTarget: input.targetMetrics ?
            this.generateComparison(input.currentMetrics, input.targetMetrics) :
            'Target metrics not specified',
          currentVsBaseline: gpuOptimizations ?
            `Current: ${input.currentMetrics?.throughput || '?'} samples/sec, Hardware peak: ${gpuOptimizations.computePeak * 2} GFLOPS` :
            'Hardware specs not provided for comparison',
          recommendations: gpuOptimizations?.recommendations || [],
        },

        artifacts: [
          './profiling/profile_setup.py',
          './profiling/analysis_results.json',
          './profiling/optimization_report.md',
        ],
      };

      // Add memory-specific optimizations if available
      if (memoryAnalysis?.optimizationActions) {
        output.optimizations.push(...memoryAnalysis.optimizationActions.slice(0, 3).map((a: any) => ({
          title: a.technique,
          category: 'Memory',
          currentValue: `${a.currentMemoryMB.toFixed(0)} MB`,
          targetValue: `${a.estimatedMemoryMB.toFixed(0)} MB`,
          expectedSpeedup: `${a.savingPercent.toFixed(0)}% memory reduction`,
          effort: a.implementationEffort,
          priority: a.priority,
          implementation: {
            steps: [a.technique],
            code: a.implementationCode,
          },
        })));
      }

      return {
        success: true,
        data: output,
        metadata: {
          skillName: 'DL Profiling',
          duration: Date.now() - startTime,
          agentsUsed,
        },
        artifacts: output.artifacts,
        suggestions: [
          'Run the profiling setup code to collect detailed traces',
          'Apply optimizations in order of priority',
          'Re-profile after each major change to measure impact',
          'Consider hardware-specific tuning for maximum performance',
        ],
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : String(error),
        metadata: {
          skillName: 'DL Profiling',
          duration: Date.now() - startTime,
          agentsUsed,
        },
      };
    }
  },
} as SkillDefinition<ProfilingInput, ProfilingOutput> & {
  getHardwareOptimizations: (gpuModel: string) => typeof HARDWARE_OPTIMIZATIONS[string] | null;
  generateProfilingSetup: (input: ProfilingInput) => {setupCode: string; runCommand: string; analysisSteps: string[]};
  generateOptimizations: (input: ProfilingInput, agentData: any, gpuOpts: any) => ProfilingOutput['optimizations'];
  calculateEfficiency: (metrics: ProfilingInput['currentMetrics']) => number;
  inferBottlenecks: (input: ProfilingInput) => ProfilingOutput['bottlenecks'];
  generateComparison: (current: any, target: any) => string;
};

// Attach helper methods
Object.assign(DLProfilingSkill, {
  getHardwareOptimizations(gpuModel: string) {
    // Try to match GPU model
    for (const [key, value] of Object.entries(HARDWARE_OPTIMIZATIONS)) {
      if (gpuModel.toLowerCase().includes(key.toLowerCase())) {
        return value;
      }
    }
    return null;
  },

  generateProfilingSetup(input: ProfilingInput): {setupCode: string; runCommand: string; analysisSteps: string[]} {
    const framework = input.framework || 'pytorch';
    const goal = input.goal === 'comprehensive' ? 'training_speed' : input.goal;

    const template = PROFILING_TEMPLATES[framework as keyof typeof PROFILING_TEMPLATES];
    const code = template?.[goal as keyof typeof template] || template?.training_speed || '';

    return {
      setupCode: code,
      runCommand: framework === 'pytorch' ?
        'python profile_training.py' :
        framework === 'tensorflow' ?
        'python -m tensorflow.python.profiler.profiler_v2' :
        'python profile_training.py',
      analysisSteps: [
        'Run profiling script to collect traces',
        'View trace in TensorBoard or Chrome tracing',
        'Identify top time-consuming operations',
        'Check for bottleneck patterns',
        'Apply recommended optimizations',
        'Re-profile to measure improvement',
      ],
    };
  },

  generateOptimizations(input: ProfilingInput, agentData: any, gpuOpts: any): ProfilingOutput['optimizations'] {
    const optimizations: ProfilingOutput['optimizations'] = [];

    // Add agent-recommended optimizations
    if (agentData.optimizationRecommendations) {
      for (const rec of agentData.optimizationRecommendations.slice(0, 5)) {
        optimizations.push({
          title: rec.title,
          category: rec.category,
          currentValue: 'Current implementation',
          targetValue: 'Optimized implementation',
          expectedSpeedup: rec.expectedSpeedup,
          effort: rec.implementationEffort,
          priority: rec.priority,
          implementation: {
            steps: [rec.description],
            code: rec.codeExample,
          },
        });
      }
    }

    // Add goal-specific optimizations
    switch (input.goal) {
      case 'training_speed':
        if (!optimizations.some(o => o.title.includes('torch.compile'))) {
          optimizations.push({
            title: 'Enable torch.compile (PyTorch 2.0+)',
            category: 'Compute',
            currentValue: 'Eager mode',
            targetValue: 'Compiled mode',
            expectedSpeedup: '20-50% faster',
            effort: 'LOW',
            priority: 9,
            implementation: {
              steps: [
                'Add torch.compile() to your model',
                'Use appropriate mode (default, reduce-overhead, max-autotune)',
              ],
              code: `model = torch.compile(model, mode='reduce-overhead')`,
            },
          });
        }
        break;

      case 'inference_latency':
        optimizations.push({
          title: 'Enable CUDA Graphs',
          category: 'Compute',
          currentValue: 'Dynamic execution',
          targetValue: 'CUDA graph capture',
          expectedSpeedup: '10-30% lower latency',
          effort: 'MEDIUM',
          priority: 8,
          implementation: {
            steps: [
              'Capture forward pass as CUDA graph',
              'Replay graph for inference',
            ],
            code: `# Warmup
for _ in range(10):
    output = model(static_input)

# Capture
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    static_output = model(static_input)

# Replay
g.replay()`,
          },
        });
        break;

      case 'memory_optimization':
        optimizations.push({
          title: 'Enable Gradient Checkpointing',
          category: 'Memory',
          currentValue: 'Store all activations',
          targetValue: 'Recompute during backward',
          expectedSpeedup: '50-70% memory reduction',
          effort: 'LOW',
          priority: 9,
          implementation: {
            steps: [
              'Wrap memory-intensive layers with checkpoint',
              'Trade compute for memory',
            ],
            code: `from torch.utils.checkpoint import checkpoint

class Block(nn.Module):
    def forward(self, x):
        return checkpoint(self._forward, x, use_reentrant=False)`,
          },
        });
        break;

      case 'throughput':
        optimizations.push({
          title: 'Optimize DataLoader',
          category: 'Data Pipeline',
          currentValue: 'Default settings',
          targetValue: 'Optimized settings',
          expectedSpeedup: '20-100% throughput increase',
          effort: 'LOW',
          priority: 9,
          implementation: {
            steps: [
              'Increase num_workers',
              'Enable pin_memory',
              'Tune prefetch_factor',
            ],
            code: `DataLoader(
    dataset,
    batch_size=32,
    num_workers=8,
    pin_memory=True,
    prefetch_factor=4,
    persistent_workers=True,
)`,
          },
        });
        break;

      case 'gpu_utilization':
        if (input.currentMetrics?.gpuUtilization && input.currentMetrics.gpuUtilization < 50) {
          optimizations.push({
            title: 'Increase Batch Size',
            category: 'Compute',
            currentValue: 'Current batch size',
            targetValue: 'Larger batch size',
            expectedSpeedup: 'Higher GPU utilization',
            effort: 'LOW',
            priority: 8,
            implementation: {
              steps: [
                'Double batch size if memory allows',
                'Use gradient accumulation if memory-limited',
              ],
              code: `# With gradient accumulation
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()`,
            },
          });
        }
        break;
    }

    // Always recommend mixed precision if not present
    if (!optimizations.some(o => o.title.toLowerCase().includes('precision'))) {
      optimizations.push({
        title: 'Enable Mixed Precision Training',
        category: 'Compute',
        currentValue: 'FP32',
        targetValue: 'FP16/BF16',
        expectedSpeedup: '50-100% speedup, 50% memory reduction',
        effort: 'LOW',
        priority: 9,
        implementation: {
          steps: [
            'Use automatic mixed precision',
            'Enable for both training and inference',
          ],
          code: `from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast(dtype=torch.bfloat16):
    output = model(input)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()`,
        },
      });
    }

    // Sort by priority
    return optimizations.sort((a, b) => b.priority - a.priority);
  },

  calculateEfficiency(metrics: ProfilingInput['currentMetrics']): number {
    if (!metrics) return 0;

    let score = 0;
    let factors = 0;

    if (metrics.gpuUtilization) {
      score += metrics.gpuUtilization;
      factors++;
    }

    if (metrics.throughput) {
      // Assume 1000 samples/sec is "good"
      score += Math.min(100, metrics.throughput / 10);
      factors++;
    }

    return factors > 0 ? Math.round(score / factors) : 50;
  },

  inferBottlenecks(input: ProfilingInput): ProfilingOutput['bottlenecks'] {
    const bottlenecks: ProfilingOutput['bottlenecks'] = [];

    if (input.currentMetrics?.gpuUtilization && input.currentMetrics.gpuUtilization < 50) {
      bottlenecks.push({
        type: 'DATA_LOADING',
        severity: 'HIGH',
        description: 'Low GPU utilization suggests data loading bottleneck',
        impact: `GPU only utilized ${input.currentMetrics.gpuUtilization}%`,
        solution: 'Increase DataLoader workers and enable pin_memory',
      });
    }

    if (input.currentMetrics?.memoryUsage && input.hardware?.gpuMemoryGB) {
      const utilization = input.currentMetrics.memoryUsage / input.hardware.gpuMemoryGB * 100;
      if (utilization > 90) {
        bottlenecks.push({
          type: 'MEMORY',
          severity: 'CRITICAL',
          description: 'Memory utilization very high, risk of OOM',
          impact: `Using ${utilization.toFixed(0)}% of available memory`,
          solution: 'Enable gradient checkpointing or mixed precision',
        });
      }
    }

    return bottlenecks;
  },

  generateComparison(current: any, target: any): string {
    if (!current || !target) return 'Unable to compare without metrics';

    const comparisons = [];

    if (target.throughput && current.throughput) {
      const gap = ((target.throughput - current.throughput) / target.throughput * 100).toFixed(0);
      comparisons.push(`Throughput: ${current.throughput} -> ${target.throughput} samples/sec (${gap}% gap)`);
    }

    if (target.latency && current.latency) {
      const gap = ((current.latency - target.latency) / target.latency * 100).toFixed(0);
      comparisons.push(`Latency: ${current.latency} -> ${target.latency} ms (${gap}% reduction needed)`);
    }

    if (target.memoryUsage && current.memoryUsage) {
      const gap = ((current.memoryUsage - target.memoryUsage) / current.memoryUsage * 100).toFixed(0);
      comparisons.push(`Memory: ${current.memoryUsage} -> ${target.memoryUsage} GB (${gap}% reduction needed)`);
    }

    return comparisons.join('\n') || 'No comparable metrics';
  },
});
