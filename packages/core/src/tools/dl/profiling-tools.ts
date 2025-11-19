/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  BaseDeclarativeTool,
  BaseToolInvocation,
  Kind,
  type ToolInvocation,
  type ToolResult,
} from '../tools.js';
import type { MessageBus } from '../../confirmation-bus/message-bus.js';

// ============================================================================
// Profile Analyzer Tool
// ============================================================================

interface ProfileAnalyzerParams {
  profile_path: string;
  profile_type: 'pytorch' | 'tensorboard' | 'nvprof' | 'nsight';
  analysis_focus?: 'time' | 'memory' | 'gpu' | 'all';
  top_k?: number;
}

class ProfileAnalyzerInvocation extends BaseToolInvocation<
  ProfileAnalyzerParams,
  ToolResult
> {
  getDescription(): string {
    return `Analyze DL profile from ${this.params.profile_path} (${this.params.profile_type})`;
  }

  async execute(): Promise<ToolResult> {
    const { profile_path, profile_type, analysis_focus = 'all', top_k = 20 } = this.params;

    // Generate analysis script based on profile type
    let analysisScript = '';
    let analysisGuide = '';

    switch (profile_type) {
      case 'pytorch':
        analysisScript = this.generatePyTorchAnalysisScript(profile_path, analysis_focus, top_k);
        analysisGuide = `
## PyTorch Profile Analysis

**Profile Location:** ${profile_path}
**Focus:** ${analysis_focus}

### How to Analyze

1. **Run the generated script** to parse the profile data
2. **Key metrics to check:**
   - Operations sorted by CUDA time
   - Operations sorted by CPU time
   - Memory allocation patterns
   - Kernel launch overhead

### Common Issues to Look For

- **High CPU time with low CUDA time:** Data loading bottleneck
- **Many small CUDA operations:** Kernel launch overhead
- **Large memory allocations:** Activation memory spikes
- **Sync operations:** Unnecessary CPU-GPU synchronization

### Generated Analysis Script

\`\`\`python
${analysisScript}
\`\`\`
`;
        break;

      case 'tensorboard':
        analysisScript = this.generateTensorBoardAnalysisScript(profile_path);
        analysisGuide = `
## TensorBoard Profile Analysis

**Log Directory:** ${profile_path}

### How to Analyze

1. **Launch TensorBoard:**
   \`\`\`bash
   tensorboard --logdir=${profile_path} --port=6006
   \`\`\`

2. **Navigate to Profile tab** and check:
   - Overview page for GPU utilization
   - Trace viewer for timeline
   - Memory profile for allocations
   - Op profile for operation breakdown

### Key Metrics

- **Step Time:** Total time per training step
- **GPU Utilization:** Should be >80% for efficiency
- **Memory Bandwidth Utilization:** Check for memory-bound ops
- **Input Pipeline Analysis:** Data loading efficiency

### Automated Analysis Script

\`\`\`python
${analysisScript}
\`\`\`
`;
        break;

      case 'nvprof':
      case 'nsight':
        analysisScript = this.generateNvprofAnalysisScript(profile_path, profile_type);
        analysisGuide = `
## ${profile_type === 'nvprof' ? 'nvprof' : 'Nsight Systems'} Profile Analysis

**Profile Location:** ${profile_path}

### How to Analyze

${profile_type === 'nvprof' ? `
1. **Import to NVIDIA Visual Profiler:**
   \`\`\`bash
   nvvp ${profile_path}
   \`\`\`

2. **Check Timeline View** for:
   - Kernel execution patterns
   - Memory transfers
   - Stream synchronization
` : `
1. **Open in Nsight Systems:**
   \`\`\`bash
   nsys-ui ${profile_path}
   \`\`\`

2. **Analyze Timeline** for:
   - GPU kernel execution
   - CPU-GPU transfers
   - CUDA API calls
`}

### Key Metrics

- **Kernel Occupancy:** Theoretical vs achieved
- **Memory Throughput:** GB/s achieved
- **SM Efficiency:** Compute utilization
- **Warp Execution Efficiency:** Thread divergence

### Command-line Analysis

\`\`\`bash
${analysisScript}
\`\`\`
`;
        break;
    }

    const result = `# Deep Learning Profile Analysis

${analysisGuide}

## Next Steps

1. Run the provided analysis script/commands
2. Identify the top time-consuming operations
3. Check for common bottleneck patterns
4. Apply optimizations based on findings

## Optimization Recommendations Based on Profile Type

### For Compute-Bound Profiles
- Enable mixed precision training (FP16/BF16)
- Use torch.compile() for kernel fusion
- Optimize batch size for GPU utilization

### For Memory-Bound Profiles
- Implement gradient checkpointing
- Use memory-efficient attention
- Reduce activation memory with smaller micro-batches

### For Data Loading Bottlenecks
- Increase DataLoader workers
- Enable pin_memory and prefetching
- Use memory-mapped datasets
`;

    return {
      llmContent: result,
      returnDisplay: result,
    };
  }

  private generatePyTorchAnalysisScript(path: string, focus: string, topK: number): string {
    return `import torch
import json
from pathlib import Path

def analyze_pytorch_profile(profile_path: str, top_k: int = ${topK}):
    """Analyze PyTorch profiler output."""

    # Load profile data
    with open(profile_path, 'r') as f:
        profile_data = json.load(f)

    # Extract events
    events = profile_data.get('traceEvents', [])

    # Categorize operations
    cuda_ops = []
    cpu_ops = []
    memory_ops = []

    for event in events:
        if event.get('cat') == 'kernel':
            cuda_ops.append(event)
        elif event.get('cat') == 'cpu_op':
            cpu_ops.append(event)
        elif 'memory' in event.get('name', '').lower():
            memory_ops.append(event)

    # Sort by duration
    cuda_ops.sort(key=lambda x: x.get('dur', 0), reverse=True)
    cpu_ops.sort(key=lambda x: x.get('dur', 0), reverse=True)

    print("=" * 60)
    print("TOP ${topK} CUDA OPERATIONS BY TIME")
    print("=" * 60)
    for i, op in enumerate(cuda_ops[:top_k]):
        name = op.get('name', 'Unknown')
        dur = op.get('dur', 0) / 1000  # Convert to ms
        print(f"{i+1:3}. {name[:50]:50} {dur:10.3f} ms")

    print("\\n" + "=" * 60)
    print("TOP ${topK} CPU OPERATIONS BY TIME")
    print("=" * 60)
    for i, op in enumerate(cpu_ops[:top_k]):
        name = op.get('name', 'Unknown')
        dur = op.get('dur', 0) / 1000  # Convert to ms
        print(f"{i+1:3}. {name[:50]:50} {dur:10.3f} ms")

    # Calculate totals
    total_cuda = sum(op.get('dur', 0) for op in cuda_ops) / 1000
    total_cpu = sum(op.get('dur', 0) for op in cpu_ops) / 1000

    print("\\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total CUDA time: {total_cuda:.2f} ms")
    print(f"Total CPU time: {total_cpu:.2f} ms")
    print(f"CUDA/CPU ratio: {total_cuda/max(total_cpu, 0.001):.2f}")

if __name__ == '__main__':
    analyze_pytorch_profile('${path}')`;
  }

  private generateTensorBoardAnalysisScript(path: string): string {
    return `from tensorboard.backend.event_processing import event_accumulator
import os

def analyze_tensorboard_logs(log_dir: str):
    """Analyze TensorBoard logs for training insights."""

    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()

    print("Available tags:")
    print(f"  Scalars: {ea.Tags()['scalars']}")

    # Analyze training loss
    if 'train/loss' in ea.Tags()['scalars']:
        losses = ea.Scalars('train/loss')
        initial_loss = losses[0].value
        final_loss = losses[-1].value
        min_loss = min(l.value for l in losses)

        print(f"\\nTraining Loss Analysis:")
        print(f"  Initial: {initial_loss:.4f}")
        print(f"  Final: {final_loss:.4f}")
        print(f"  Minimum: {min_loss:.4f}")
        print(f"  Reduction: {(1 - final_loss/initial_loss) * 100:.1f}%")

    # Check for gradient norms
    if 'train/grad_norm' in ea.Tags()['scalars']:
        grads = ea.Scalars('train/grad_norm')
        grad_values = [g.value for g in grads]
        print(f"\\nGradient Norm Analysis:")
        print(f"  Mean: {sum(grad_values)/len(grad_values):.4f}")
        print(f"  Max: {max(grad_values):.4f}")
        print(f"  Min: {min(grad_values):.4f}")

        # Check for spikes (>10x mean)
        mean_grad = sum(grad_values)/len(grad_values)
        spikes = [g for g in grad_values if g > 10 * mean_grad]
        if spikes:
            print(f"  WARNING: {len(spikes)} gradient spikes detected!")

    # Check for learning rate schedule
    if 'train/lr' in ea.Tags()['scalars']:
        lrs = ea.Scalars('train/lr')
        print(f"\\nLearning Rate:")
        print(f"  Initial: {lrs[0].value:.2e}")
        print(f"  Final: {lrs[-1].value:.2e}")

if __name__ == '__main__':
    analyze_tensorboard_logs('${path}')`;
  }

  private generateNvprofAnalysisScript(path: string, type: string): string {
    if (type === 'nvprof') {
      return `# nvprof analysis commands

# Summary of GPU activities
nvprof --print-gpu-summary -i ${path}

# Top kernels by time
nvprof --print-gpu-trace -i ${path} | head -50

# Memory transfers
nvprof --print-gpu-trace --print-api-trace -i ${path} | grep -E "memcpy|memset"

# Occupancy analysis
nvprof --metrics achieved_occupancy,sm_efficiency -i ${path}`;
    } else {
      return `# Nsight Systems analysis commands

# Generate summary report
nsys stats ${path}

# Export to SQLite for custom analysis
nsys export --type=sqlite ${path}

# View in GUI
nsys-ui ${path}`;
    }
  }
}

/**
 * ProfileAnalyzer tool for analyzing DL profiling data.
 */
export class ProfileAnalyzerTool extends BaseDeclarativeTool<
  ProfileAnalyzerParams,
  ToolResult
> {
  constructor(messageBus?: MessageBus) {
    super(
      'dl_profile_analyzer',
      'DL Profile Analyzer',
      'Analyze deep learning profiling data from PyTorch, TensorBoard, or NVIDIA tools',
      Kind.Read,
      {
        type: 'object',
        properties: {
          profile_path: {
            type: 'string',
            description: 'Path to the profile data file or directory',
          },
          profile_type: {
            type: 'string',
            enum: ['pytorch', 'tensorboard', 'nvprof', 'nsight'],
            description: 'Type of profiling data',
          },
          analysis_focus: {
            type: 'string',
            enum: ['time', 'memory', 'gpu', 'all'],
            description: 'Focus area for analysis',
          },
          top_k: {
            type: 'number',
            description: 'Number of top operations to show',
          },
        },
        required: ['profile_path', 'profile_type'],
      },
      true, // isOutputMarkdown
      false, // canUpdateOutput
      messageBus,
    );
  }

  protected createInvocation(
    params: ProfileAnalyzerParams,
    messageBus?: MessageBus,
    toolName?: string,
    toolDisplayName?: string,
  ): ToolInvocation<ProfileAnalyzerParams, ToolResult> {
    return new ProfileAnalyzerInvocation(
      params,
      messageBus,
      toolName,
      toolDisplayName,
    );
  }
}

// ============================================================================
// Memory Tracker Tool
// ============================================================================

interface MemoryTrackerParams {
  action: 'snapshot' | 'timeline' | 'breakdown' | 'fragmentation';
  output_path?: string;
  include_tensors?: boolean;
}

class MemoryTrackerInvocation extends BaseToolInvocation<
  MemoryTrackerParams,
  ToolResult
> {
  getDescription(): string {
    return `Track GPU memory: ${this.params.action}`;
  }

  async execute(): Promise<ToolResult> {
    const { action, output_path, include_tensors = false } = this.params;

    let script = '';
    let description = '';

    switch (action) {
      case 'snapshot':
        script = this.generateSnapshotScript(output_path, include_tensors);
        description = 'Capture a memory snapshot for detailed analysis';
        break;
      case 'timeline':
        script = this.generateTimelineScript(output_path);
        description = 'Track memory usage over time during training';
        break;
      case 'breakdown':
        script = this.generateBreakdownScript();
        description = 'Get detailed breakdown of current memory usage';
        break;
      case 'fragmentation':
        script = this.generateFragmentationScript();
        description = 'Analyze memory fragmentation';
        break;
    }

    const result = `# GPU Memory Tracking: ${action}

## Description
${description}

## Implementation

\`\`\`python
${script}
\`\`\`

## Usage Instructions

${this.getUsageInstructions(action, output_path)}

## Interpretation Guide

${this.getInterpretationGuide(action)}
`;

    return {
      llmContent: result,
      returnDisplay: result,
    };
  }

  private generateSnapshotScript(outputPath?: string, includeTensors?: boolean): string {
    const path = outputPath || 'memory_snapshot.pickle';
    return `import torch

def capture_memory_snapshot(output_path='${path}'):
    """Capture detailed memory snapshot for analysis."""

    # Start recording memory history
    torch.cuda.memory._record_memory_history(
        max_entries=100000,
        ${includeTensors ? "context='all'," : ''}
    )

    # ... Your training code here ...
    # Example:
    # for batch in dataloader:
    #     output = model(batch)
    #     loss = criterion(output, target)
    #     loss.backward()
    #     optimizer.step()
    #     optimizer.zero_grad()

    # Save snapshot
    torch.cuda.memory._dump_snapshot(output_path)
    torch.cuda.memory._record_memory_history(enabled=None)

    print(f"Memory snapshot saved to {output_path}")
    print("Analyze with: python -m torch.cuda.memory._viz " + output_path)

    return output_path

# To visualize:
# 1. Run: python -m torch.cuda.memory._viz memory_snapshot.pickle
# 2. Open the generated HTML in a browser`;
  }

  private generateTimelineScript(outputPath?: string): string {
    const path = outputPath || 'memory_timeline.csv';
    return `import torch
import time
import csv

class MemoryTimeline:
    def __init__(self, output_path='${path}'):
        self.output_path = output_path
        self.timeline = []
        self.start_time = time.time()

    def record(self, label=''):
        """Record current memory state."""
        self.timeline.append({
            'timestamp': time.time() - self.start_time,
            'label': label,
            'allocated_MB': torch.cuda.memory_allocated() / 1e6,
            'reserved_MB': torch.cuda.memory_reserved() / 1e6,
            'max_allocated_MB': torch.cuda.max_memory_allocated() / 1e6,
        })

    def save(self):
        """Save timeline to CSV."""
        with open(self.output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.timeline[0].keys())
            writer.writeheader()
            writer.writerows(self.timeline)
        print(f"Memory timeline saved to {self.output_path}")

    def print_summary(self):
        """Print summary statistics."""
        peak = max(r['allocated_MB'] for r in self.timeline)
        avg = sum(r['allocated_MB'] for r in self.timeline) / len(self.timeline)

        print(f"Peak allocated: {peak:.2f} MB")
        print(f"Average allocated: {avg:.2f} MB")

        # Find peak moment
        peak_entry = max(self.timeline, key=lambda x: x['allocated_MB'])
        print(f"Peak at: {peak_entry['label']} ({peak_entry['timestamp']:.2f}s)")

# Usage:
# timeline = MemoryTimeline()
# timeline.record('start')
# output = model(input)
# timeline.record('after_forward')
# loss.backward()
# timeline.record('after_backward')
# timeline.save()
# timeline.print_summary()`;
  }

  private generateBreakdownScript(): string {
    return `import torch

def memory_breakdown():
    """Get detailed breakdown of GPU memory usage."""

    stats = torch.cuda.memory_stats()

    print("=" * 50)
    print("GPU MEMORY BREAKDOWN")
    print("=" * 50)

    # Current allocation
    allocated = stats['allocated_bytes.all.current'] / 1e9
    reserved = stats['reserved_bytes.all.current'] / 1e9

    print(f"\\nCurrent State:")
    print(f"  Allocated: {allocated:.3f} GB")
    print(f"  Reserved:  {reserved:.3f} GB")
    print(f"  Fragmentation: {(reserved - allocated) / reserved * 100:.1f}%")

    # Peak usage
    peak_allocated = stats['allocated_bytes.all.peak'] / 1e9
    peak_reserved = stats['reserved_bytes.all.peak'] / 1e9

    print(f"\\nPeak Usage:")
    print(f"  Peak Allocated: {peak_allocated:.3f} GB")
    print(f"  Peak Reserved:  {peak_reserved:.3f} GB")

    # Allocation statistics
    num_allocs = stats['allocation.all.current']
    num_frees = stats['allocation.all.freed']

    print(f"\\nAllocation Statistics:")
    print(f"  Active allocations: {num_allocs - num_frees}")
    print(f"  Total allocations: {num_allocs}")
    print(f"  Total frees: {num_frees}")

    # Large allocations
    large_allocs = stats.get('large_pool.large_pool_allocated', 0) / 1e9
    small_allocs = stats.get('small_pool.small_pool_allocated', 0) / 1e9

    print(f"\\nPool Breakdown:")
    print(f"  Large pool: {large_allocs:.3f} GB")
    print(f"  Small pool: {small_allocs:.3f} GB")

    return stats

# Get breakdown for model
def model_memory_breakdown(model):
    """Analyze memory usage by model component."""

    param_mem = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_mem = sum(b.numel() * b.element_size() for b in model.buffers())
    grad_mem = sum(p.grad.numel() * p.grad.element_size()
                   for p in model.parameters() if p.grad is not None)

    print(f"\\nModel Memory:")
    print(f"  Parameters: {param_mem / 1e6:.2f} MB")
    print(f"  Buffers: {buffer_mem / 1e6:.2f} MB")
    print(f"  Gradients: {grad_mem / 1e6:.2f} MB")

    # Per-layer breakdown
    print(f"\\nPer-Layer Parameters:")
    for name, param in model.named_parameters():
        mem = param.numel() * param.element_size() / 1e6
        if mem > 1:  # Only show layers > 1MB
            print(f"  {name}: {mem:.2f} MB")

memory_breakdown()`;
  }

  private generateFragmentationScript(): string {
    return `import torch

def analyze_fragmentation():
    """Analyze GPU memory fragmentation."""

    stats = torch.cuda.memory_stats()

    allocated = stats['allocated_bytes.all.current']
    reserved = stats['reserved_bytes.all.current']

    fragmentation = (reserved - allocated) / reserved * 100 if reserved > 0 else 0

    print("=" * 50)
    print("MEMORY FRAGMENTATION ANALYSIS")
    print("=" * 50)

    print(f"\\nFragmentation: {fragmentation:.1f}%")

    # Interpretation
    if fragmentation < 10:
        print("Status: GOOD - Low fragmentation")
    elif fragmentation < 30:
        print("Status: MODERATE - Some fragmentation")
    else:
        print("Status: HIGH - Significant fragmentation")

    # Check inactive split blocks (indicator of fragmentation)
    inactive_split = stats.get('inactive_split.all.current', 0)
    print(f"\\nInactive split blocks: {inactive_split}")

    # Recommendations
    print("\\nRecommendations:")
    if fragmentation > 20:
        print("  1. Call torch.cuda.empty_cache() between batches")
        print("  2. Use PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
        print("  3. Pre-allocate tensors instead of dynamic allocation")
        print("  4. Reduce tensor size variability")

    # Check allocation patterns
    num_allocs = stats['allocation.all.current']
    num_frees = stats['allocation.all.freed']

    print(f"\\nAllocation Patterns:")
    print(f"  Total allocations: {num_allocs}")
    print(f"  Total frees: {num_frees}")
    print(f"  Allocation rate: {num_allocs / max(num_frees, 1):.2f}")

    return {
        'fragmentation_percent': fragmentation,
        'allocated_bytes': allocated,
        'reserved_bytes': reserved,
        'inactive_split_blocks': inactive_split,
    }

result = analyze_fragmentation()`;
  }

  private getUsageInstructions(action: string, outputPath?: string): string {
    switch (action) {
      case 'snapshot':
        return `
1. Insert the \`capture_memory_snapshot()\` call in your training code
2. Run your training script
3. The snapshot will be saved to \`${outputPath || 'memory_snapshot.pickle'}\`
4. Visualize with: \`python -m torch.cuda.memory._viz ${outputPath || 'memory_snapshot.pickle'}\`
`;
      case 'timeline':
        return `
1. Create a \`MemoryTimeline()\` instance before training
2. Call \`timeline.record('label')\` at key points
3. Call \`timeline.save()\` after training
4. Analyze the CSV or plot the data
`;
      case 'breakdown':
        return `
1. Run the script after loading your model and during training
2. Call \`memory_breakdown()\` to see current state
3. Call \`model_memory_breakdown(model)\` for per-component analysis
`;
      case 'fragmentation':
        return `
1. Run during training when you suspect fragmentation
2. Call multiple times to track fragmentation over time
3. Follow recommendations to reduce fragmentation
`;
      default:
        return '';
    }
  }

  private getInterpretationGuide(action: string): string {
    switch (action) {
      case 'snapshot':
        return `
- **Dark colors**: High memory usage
- **Tall blocks**: Large allocations
- **Many small blocks**: Potential fragmentation
- **Look for**: Memory spikes, leaked tensors, large activations
`;
      case 'timeline':
        return `
- **Upward trends**: Memory accumulation (potential leak)
- **Spikes**: Large temporary allocations
- **Steady state**: Normal operation
- **Look for**: Patterns in memory usage, peak moments
`;
      case 'breakdown':
        return `
- **High reserved vs allocated**: Fragmentation
- **Large parameter memory**: Consider model parallelism
- **Large gradient memory**: Consider gradient checkpointing
- **Many allocations**: Consider tensor reuse
`;
      case 'fragmentation':
        return `
- **<10%**: Healthy memory state
- **10-30%**: Moderate fragmentation, monitor
- **>30%**: High fragmentation, take action
- **Many inactive split blocks**: Severe fragmentation
`;
      default:
        return '';
    }
  }
}

/**
 * MemoryTracker tool for tracking GPU memory usage.
 */
export class MemoryTrackerTool extends BaseDeclarativeTool<
  MemoryTrackerParams,
  ToolResult
> {
  constructor(messageBus?: MessageBus) {
    super(
      'dl_memory_tracker',
      'DL Memory Tracker',
      'Track and analyze GPU memory usage with snapshots, timelines, and breakdown analysis',
      Kind.Read,
      {
        type: 'object',
        properties: {
          action: {
            type: 'string',
            enum: ['snapshot', 'timeline', 'breakdown', 'fragmentation'],
            description: 'Type of memory tracking to perform',
          },
          output_path: {
            type: 'string',
            description: 'Path to save output data',
          },
          include_tensors: {
            type: 'boolean',
            description: 'Include tensor-level details in snapshot',
          },
        },
        required: ['action'],
      },
      true,
      false,
      messageBus,
    );
  }

  protected createInvocation(
    params: MemoryTrackerParams,
    messageBus?: MessageBus,
    toolName?: string,
    toolDisplayName?: string,
  ): ToolInvocation<MemoryTrackerParams, ToolResult> {
    return new MemoryTrackerInvocation(
      params,
      messageBus,
      toolName,
      toolDisplayName,
    );
  }
}

// ============================================================================
// Bottleneck Detector Tool
// ============================================================================

interface BottleneckDetectorParams {
  target: 'training' | 'inference' | 'data_loading';
  metrics?: string[];
}

class BottleneckDetectorInvocation extends BaseToolInvocation<
  BottleneckDetectorParams,
  ToolResult
> {
  getDescription(): string {
    return `Detect performance bottlenecks in ${this.params.target}`;
  }

  async execute(): Promise<ToolResult> {
    const { target, metrics = [] } = this.params;

    const detectionScript = this.generateDetectionScript(target, metrics);
    const bottleneckPatterns = this.getBottleneckPatterns(target);

    const result = `# Performance Bottleneck Detection: ${target}

## Detection Script

\`\`\`python
${detectionScript}
\`\`\`

## Common Bottleneck Patterns

${bottleneckPatterns}

## Diagnostic Checklist

${this.getDiagnosticChecklist(target)}

## Resolution Strategies

${this.getResolutionStrategies(target)}
`;

    return {
      llmContent: result,
      returnDisplay: result,
    };
  }

  private generateDetectionScript(target: string, metrics: string[]): string {
    if (target === 'data_loading') {
      return `import torch
import time
from torch.utils.data import DataLoader

def detect_data_loading_bottleneck(dataloader: DataLoader, num_batches: int = 100):
    """Detect if data loading is a bottleneck."""

    # Measure pure data loading time
    load_times = []
    start = time.time()

    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        load_times.append(time.time() - start)
        start = time.time()

    avg_load_time = sum(load_times) / len(load_times) * 1000  # ms

    # Measure GPU processing time (forward pass only)
    model.eval()
    process_times = []

    for i, (data, _) in enumerate(dataloader):
        if i >= num_batches:
            break

        data = data.cuda(non_blocking=True)
        torch.cuda.synchronize()

        start = time.time()
        with torch.no_grad():
            _ = model(data)
        torch.cuda.synchronize()

        process_times.append(time.time() - start)

    avg_process_time = sum(process_times) / len(process_times) * 1000  # ms

    # Analysis
    print("=" * 50)
    print("DATA LOADING BOTTLENECK ANALYSIS")
    print("=" * 50)
    print(f"Average data loading time: {avg_load_time:.2f} ms")
    print(f"Average GPU processing time: {avg_process_time:.2f} ms")
    print(f"Loading / Processing ratio: {avg_load_time / avg_process_time:.2f}")

    if avg_load_time > avg_process_time:
        print("\\nSTATUS: DATA LOADING BOTTLENECK DETECTED")
        print("\\nRecommendations:")
        print("  1. Increase num_workers in DataLoader")
        print("  2. Enable pin_memory=True")
        print("  3. Increase prefetch_factor")
        print("  4. Use faster storage (SSD/NVMe)")
        print("  5. Consider caching dataset in memory")
    else:
        print("\\nSTATUS: No data loading bottleneck")

    return {
        'load_time_ms': avg_load_time,
        'process_time_ms': avg_process_time,
        'is_bottleneck': avg_load_time > avg_process_time,
    }`;
    } else if (target === 'training') {
      return `import torch
import time
from torch.profiler import profile, ProfilerActivity

def detect_training_bottleneck(model, dataloader, criterion, optimizer, num_steps=20):
    """Detect bottlenecks in training loop."""

    model.train()

    # Timing breakdown
    timings = {
        'data_load': [],
        'forward': [],
        'backward': [],
        'optimizer': [],
        'total': [],
    }

    data_iter = iter(dataloader)

    for step in range(num_steps):
        # Data loading
        start = time.time()
        data, target = next(data_iter)
        data = data.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        torch.cuda.synchronize()
        timings['data_load'].append(time.time() - start)

        # Forward
        start = time.time()
        output = model(data)
        loss = criterion(output, target)
        torch.cuda.synchronize()
        timings['forward'].append(time.time() - start)

        # Backward
        start = time.time()
        optimizer.zero_grad()
        loss.backward()
        torch.cuda.synchronize()
        timings['backward'].append(time.time() - start)

        # Optimizer
        start = time.time()
        optimizer.step()
        torch.cuda.synchronize()
        timings['optimizer'].append(time.time() - start)

        timings['total'].append(sum(timings[k][-1] for k in ['data_load', 'forward', 'backward', 'optimizer']))

    # Analysis
    print("=" * 50)
    print("TRAINING BOTTLENECK ANALYSIS")
    print("=" * 50)

    avg_timings = {k: sum(v) / len(v) * 1000 for k, v in timings.items()}
    total = avg_timings['total']

    for key in ['data_load', 'forward', 'backward', 'optimizer']:
        pct = avg_timings[key] / total * 100
        print(f"{key:15}: {avg_timings[key]:8.2f} ms ({pct:5.1f}%)")

    print(f"{'total':15}: {total:8.2f} ms")

    # Identify bottleneck
    bottleneck = max(['data_load', 'forward', 'backward', 'optimizer'],
                     key=lambda k: avg_timings[k])

    print(f"\\nPrimary bottleneck: {bottleneck}")

    return avg_timings, bottleneck`;
    } else {  // inference
      return `import torch
import time
import numpy as np

def detect_inference_bottleneck(model, input_shape, batch_sizes=[1, 8, 32], warmup=10, iterations=100):
    """Detect inference bottlenecks across batch sizes."""

    model.eval()
    model.cuda()

    results = {}

    for batch_size in batch_sizes:
        # Create input
        dummy_input = torch.randn(batch_size, *input_shape).cuda()

        # Warmup
        for _ in range(warmup):
            with torch.no_grad():
                _ = model(dummy_input)
        torch.cuda.synchronize()

        # Benchmark
        latencies = []
        for _ in range(iterations):
            start = time.time()
            with torch.no_grad():
                _ = model(dummy_input)
            torch.cuda.synchronize()
            latencies.append(time.time() - start)

        results[batch_size] = {
            'mean_ms': np.mean(latencies) * 1000,
            'std_ms': np.std(latencies) * 1000,
            'p50_ms': np.percentile(latencies, 50) * 1000,
            'p99_ms': np.percentile(latencies, 99) * 1000,
            'throughput': batch_size / np.mean(latencies),
        }

    # Analysis
    print("=" * 50)
    print("INFERENCE BOTTLENECK ANALYSIS")
    print("=" * 50)

    print("\\nLatency by Batch Size:")
    for bs, stats in results.items():
        print(f"  Batch {bs:3}: {stats['mean_ms']:.2f} +/- {stats['std_ms']:.2f} ms, "
              f"throughput: {stats['throughput']:.1f} samples/sec")

    # Check scaling efficiency
    if len(batch_sizes) >= 2:
        b1, b2 = batch_sizes[0], batch_sizes[-1]
        expected_speedup = b2 / b1
        actual_speedup = results[b2]['throughput'] / results[b1]['throughput']
        efficiency = actual_speedup / expected_speedup * 100

        print(f"\\nScaling Efficiency: {efficiency:.1f}%")
        if efficiency < 50:
            print("WARNING: Poor scaling - likely memory bandwidth bound")
        elif efficiency < 80:
            print("NOTE: Moderate scaling - some overhead")
        else:
            print("GOOD: Efficient scaling")

    return results`;
    }
  }

  private getBottleneckPatterns(target: string): string {
    const common = `
### Common Patterns

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| Low GPU utilization | Data loading slow | More workers, prefetching |
| High CPU usage | Data preprocessing | Move to GPU, optimize |
| Memory keeps growing | Memory leak | Find and fix leak |
| Variable step times | GC or allocation | Pre-allocate tensors |
`;

    const specific = {
      training: `
### Training-Specific Patterns

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| Long backward pass | Large activations | Gradient checkpointing |
| Slow optimizer step | Large model | Fused optimizer, 8-bit Adam |
| GPU stalls | Synchronization | Async operations |
| Memory spikes | Gradient accumulation | Clear gradients properly |
`,
      inference: `
### Inference-Specific Patterns

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| High latency, low throughput | Batch size too small | Increase batch |
| First inference slow | JIT compilation | Warmup runs |
| Memory bound | Large model | Quantization, pruning |
| Poor scaling | Memory bandwidth | Mixed precision |
`,
      data_loading: `
### Data Loading-Specific Patterns

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| CPU at 100% | Too few workers | Increase num_workers |
| Disk I/O wait | Slow storage | SSD, caching, LMDB |
| Memory pressure | Large batches | Prefetch tuning |
| Variable load times | Complex transforms | Pre-process, cache |
`,
    };

    return common + (specific[target as keyof typeof specific] || '');
  }

  private getDiagnosticChecklist(target: string): string {
    const checks = {
      training: `
- [ ] GPU utilization (nvidia-smi) > 80%?
- [ ] Data loading time < forward pass time?
- [ ] Gradient norms stable?
- [ ] Memory usage stable across steps?
- [ ] No unnecessary CPU-GPU sync?
`,
      inference: `
- [ ] Throughput scales with batch size?
- [ ] Latency meets requirements?
- [ ] Memory fits in GPU?
- [ ] Model in eval mode?
- [ ] torch.no_grad() enabled?
`,
      data_loading: `
- [ ] num_workers > 0?
- [ ] pin_memory = True?
- [ ] prefetch_factor tuned?
- [ ] persistent_workers = True?
- [ ] Data on fast storage?
`,
    };

    return checks[target as keyof typeof checks] || '';
  }

  private getResolutionStrategies(target: string): string {
    const strategies = {
      training: `
### Priority Order for Training Optimization

1. **Fix data loading** if GPU utilization < 70%
2. **Enable mixed precision** for 2x speedup on modern GPUs
3. **Use torch.compile()** for kernel fusion (PyTorch 2.0+)
4. **Optimize batch size** for GPU memory utilization
5. **Gradient checkpointing** if memory-constrained
`,
      inference: `
### Priority Order for Inference Optimization

1. **Quantization** (INT8) for 2-4x speedup
2. **TensorRT/ONNX** for optimized deployment
3. **Batching** for throughput optimization
4. **Mixed precision** (FP16) for memory and speed
5. **Model pruning** for size reduction
`,
      data_loading: `
### Priority Order for Data Loading Optimization

1. **Increase num_workers** (typically 4-8 per GPU)
2. **Enable pin_memory** for faster CPU-GPU transfer
3. **Tune prefetch_factor** (default 2, try 4-8)
4. **Use persistent_workers** to avoid worker restart
5. **Cache or memory-map** frequently accessed data
`,
    };

    return strategies[target as keyof typeof strategies] || '';
  }
}

/**
 * BottleneckDetector tool for identifying performance bottlenecks.
 */
export class BottleneckDetectorTool extends BaseDeclarativeTool<
  BottleneckDetectorParams,
  ToolResult
> {
  constructor(messageBus?: MessageBus) {
    super(
      'dl_bottleneck_detector',
      'DL Bottleneck Detector',
      'Detect and diagnose performance bottlenecks in training, inference, or data loading',
      Kind.Read,
      {
        type: 'object',
        properties: {
          target: {
            type: 'string',
            enum: ['training', 'inference', 'data_loading'],
            description: 'What to analyze for bottlenecks',
          },
          metrics: {
            type: 'array',
            items: { type: 'string' },
            description: 'Specific metrics to collect',
          },
        },
        required: ['target'],
      },
      true,
      false,
      messageBus,
    );
  }

  protected createInvocation(
    params: BottleneckDetectorParams,
    messageBus?: MessageBus,
    toolName?: string,
    toolDisplayName?: string,
  ): ToolInvocation<BottleneckDetectorParams, ToolResult> {
    return new BottleneckDetectorInvocation(
      params,
      messageBus,
      toolName,
      toolDisplayName,
    );
  }
}

// ============================================================================
// GPU Utilization Analyzer Tool
// ============================================================================

interface GPUUtilizationParams {
  duration_seconds?: number;
  sample_interval_ms?: number;
  gpus?: number[];
}

class GPUUtilizationInvocation extends BaseToolInvocation<
  GPUUtilizationParams,
  ToolResult
> {
  getDescription(): string {
    return `Analyze GPU utilization for ${this.params.duration_seconds || 10} seconds`;
  }

  async execute(): Promise<ToolResult> {
    const {
      duration_seconds = 10,
      sample_interval_ms = 100,
      gpus,
    } = this.params;

    const script = this.generateUtilizationScript(duration_seconds, sample_interval_ms, gpus);

    const result = `# GPU Utilization Analysis

## Monitoring Script

\`\`\`python
${script}
\`\`\`

## Interpretation Guide

### GPU Utilization Levels

| Utilization | Status | Action |
|-------------|--------|--------|
| < 30% | Low | Check for data loading bottleneck |
| 30-60% | Moderate | Optimize batch size or operations |
| 60-80% | Good | Minor optimizations possible |
| 80-95% | Excellent | Well-optimized |
| > 95% | Max | Consider if this is sustainable |

### Memory Utilization

| Utilization | Status | Action |
|-------------|--------|--------|
| < 50% | Underutilized | Increase batch size |
| 50-80% | Good | Optimal range |
| 80-90% | High | Monitor for OOM |
| > 90% | Critical | Reduce batch size or optimize |

### Common Issues

1. **Low GPU, High CPU**: Data loading bottleneck
2. **High GPU, Low Memory**: Compute-bound (good)
3. **High Memory, Low GPU**: Memory bandwidth bound
4. **Fluctuating**: Irregular workload or sync issues

## Command Line Alternative

\`\`\`bash
# Real-time monitoring
watch -n 0.5 nvidia-smi

# Log to file
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw \\
  --format=csv -l 1 > gpu_log.csv
\`\`\`
`;

    return {
      llmContent: result,
      returnDisplay: result,
    };
  }

  private generateUtilizationScript(duration: number, interval: number, gpus?: number[]): string {
    const gpuFilter = gpus ? `gpus = ${JSON.stringify(gpus)}` : 'gpus = None  # Monitor all GPUs';

    return `import subprocess
import time
import statistics

def monitor_gpu_utilization(duration_sec=${duration}, interval_ms=${interval}):
    """Monitor GPU utilization over time."""

    ${gpuFilter}

    readings = []
    start_time = time.time()

    while time.time() - start_time < duration_sec:
        # Query nvidia-smi
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )

        for line in result.stdout.strip().split('\\n'):
            values = [v.strip() for v in line.split(',')]
            gpu_id = int(values[0])

            if gpus is None or gpu_id in gpus:
                readings.append({
                    'timestamp': time.time() - start_time,
                    'gpu_id': gpu_id,
                    'gpu_util': float(values[1]),
                    'mem_util': float(values[2]),
                    'mem_used': float(values[3]),
                    'mem_total': float(values[4]),
                    'temperature': float(values[5]),
                    'power': float(values[6]) if values[6] != '[N/A]' else 0,
                })

        time.sleep(interval_ms / 1000)

    # Analyze results
    analyze_results(readings)
    return readings

def analyze_results(readings):
    """Analyze and print GPU utilization statistics."""

    # Group by GPU
    gpu_ids = set(r['gpu_id'] for r in readings)

    print("=" * 60)
    print("GPU UTILIZATION ANALYSIS")
    print("=" * 60)

    for gpu_id in sorted(gpu_ids):
        gpu_readings = [r for r in readings if r['gpu_id'] == gpu_id]

        gpu_utils = [r['gpu_util'] for r in gpu_readings]
        mem_utils = [r['mem_util'] for r in gpu_readings]
        temps = [r['temperature'] for r in gpu_readings]

        print(f"\\nGPU {gpu_id}:")
        print(f"  GPU Utilization:")
        print(f"    Mean: {statistics.mean(gpu_utils):.1f}%")
        print(f"    Std:  {statistics.stdev(gpu_utils) if len(gpu_utils) > 1 else 0:.1f}%")
        print(f"    Min:  {min(gpu_utils):.1f}%")
        print(f"    Max:  {max(gpu_utils):.1f}%")

        print(f"  Memory Utilization:")
        print(f"    Mean: {statistics.mean(mem_utils):.1f}%")
        print(f"    Max:  {max(mem_utils):.1f}%")

        print(f"  Temperature:")
        print(f"    Mean: {statistics.mean(temps):.1f}C")
        print(f"    Max:  {max(temps):.1f}C")

        # Diagnosis
        avg_gpu = statistics.mean(gpu_utils)
        if avg_gpu < 30:
            print(f"  Status: LOW UTILIZATION - Check for bottlenecks")
        elif avg_gpu < 60:
            print(f"  Status: MODERATE - Room for optimization")
        elif avg_gpu < 80:
            print(f"  Status: GOOD")
        else:
            print(f"  Status: EXCELLENT")

if __name__ == '__main__':
    monitor_gpu_utilization()`;
  }
}

/**
 * GPUUtilizationAnalyzer tool for monitoring GPU utilization.
 */
export class GPUUtilizationAnalyzerTool extends BaseDeclarativeTool<
  GPUUtilizationParams,
  ToolResult
> {
  constructor(messageBus?: MessageBus) {
    super(
      'dl_gpu_utilization',
      'DL GPU Utilization Analyzer',
      'Monitor and analyze GPU utilization, memory, temperature, and power',
      Kind.Read,
      {
        type: 'object',
        properties: {
          duration_seconds: {
            type: 'number',
            description: 'Duration to monitor in seconds',
          },
          sample_interval_ms: {
            type: 'number',
            description: 'Sampling interval in milliseconds',
          },
          gpus: {
            type: 'array',
            items: { type: 'number' },
            description: 'Specific GPU indices to monitor',
          },
        },
        required: [],
      },
      true,
      false,
      messageBus,
    );
  }

  protected createInvocation(
    params: GPUUtilizationParams,
    messageBus?: MessageBus,
    toolName?: string,
    toolDisplayName?: string,
  ): ToolInvocation<GPUUtilizationParams, ToolResult> {
    return new GPUUtilizationInvocation(
      params,
      messageBus,
      toolName,
      toolDisplayName,
    );
  }
}

// ============================================================================
// Exports
// ============================================================================

export const DL_PROFILING_TOOLS = [
  ProfileAnalyzerTool,
  MemoryTrackerTool,
  BottleneckDetectorTool,
  GPUUtilizationAnalyzerTool,
];
