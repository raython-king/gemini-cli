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
  EDIT_TOOL_NAME,
  WRITE_FILE_TOOL_NAME,
} from '../../tools/tool-names.js';
import { DEFAULT_GEMINI_MODEL } from '../../config/models.js';
import { z } from 'zod';

/**
 * Schema for a debugging step in DL context.
 */
const DLDebugStepSchema = z.object({
  stepNumber: z.number().describe('The sequence number of this debugging step.'),
  category: z.enum([
    'TRAINING_FAILURE',
    'NUMERICAL_INSTABILITY',
    'MEMORY_ISSUE',
    'PERFORMANCE',
    'CUDA_ERROR',
    'DATA_PIPELINE',
    'MODEL_ARCHITECTURE',
    'GRADIENT_ISSUE',
  ]).describe('Category of the debugging step.'),
  action: z.string().describe('What action was taken in this step.'),
  finding: z.string().describe('What was discovered from this step.'),
  codeSnippet: z.string().optional().describe('Relevant code snippet if applicable.'),
  metrics: z.record(z.number()).optional().describe('Numerical metrics captured.'),
});

/**
 * Schema for numerical stability analysis.
 */
const NumericalStabilitySchema = z.object({
  hasNaN: z.boolean().describe('Whether NaN values were detected.'),
  hasInf: z.boolean().describe('Whether Inf values were detected.'),
  affectedLayers: z.array(z.string()).describe('Layers affected by numerical issues.'),
  affectedOperations: z.array(z.string()).describe('Operations causing numerical issues.'),
  rootCause: z.string().describe('Root cause of numerical instability.'),
  suggestedFixes: z.array(z.string()).describe('Suggested fixes for the issues.'),
});

/**
 * Schema for memory analysis.
 */
const MemoryAnalysisSchema = z.object({
  peakMemoryGB: z.number().describe('Peak GPU memory usage in GB.'),
  oomOccurred: z.boolean().describe('Whether OOM error occurred.'),
  memoryBreakdown: z.object({
    parameters: z.number().describe('Memory for model parameters in MB.'),
    gradients: z.number().describe('Memory for gradients in MB.'),
    activations: z.number().describe('Memory for activations in MB.'),
    optimizer: z.number().describe('Memory for optimizer states in MB.'),
    buffers: z.number().describe('Memory for buffers in MB.'),
  }).optional(),
  suggestions: z.array(z.string()).describe('Memory optimization suggestions.'),
});

/**
 * Schema for performance analysis.
 */
const PerformanceAnalysisSchema = z.object({
  gpuUtilization: z.number().describe('Average GPU utilization percentage.'),
  bottlenecks: z.array(z.object({
    type: z.enum(['DATA_LOADING', 'CPU_GPU_TRANSFER', 'KERNEL_LAUNCH', 'MEMORY_BANDWIDTH', 'COMPUTE_BOUND']),
    description: z.string(),
    severity: z.enum(['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']),
  })),
  throughput: z.object({
    samplesPerSecond: z.number(),
    iterationsPerSecond: z.number(),
  }).optional(),
});

/**
 * Schema for CUDA error analysis.
 */
const CUDAErrorSchema = z.object({
  errorType: z.string().describe('Type of CUDA error.'),
  errorCode: z.string().optional().describe('CUDA error code if available.'),
  failingKernel: z.string().optional().describe('Kernel that caused the error.'),
  stackTrace: z.string().optional().describe('Relevant stack trace.'),
  explanation: z.string().describe('Human-readable explanation.'),
  solution: z.string().describe('Recommended solution.'),
});

/**
 * Schema for the root cause.
 */
const DLRootCauseSchema = z.object({
  description: z.string().describe('Clear explanation of what causes the issue.'),
  category: z.enum([
    'NUMERICAL_INSTABILITY',
    'MEMORY_OVERFLOW',
    'DATA_PIPELINE',
    'ARCHITECTURE_BUG',
    'HYPERPARAMETER',
    'CUDA_ERROR',
    'DISTRIBUTED_TRAINING',
    'HARDWARE_ISSUE',
  ]),
  location: z.object({
    file: z.string(),
    line: z.number().optional(),
    function: z.string().optional(),
    layer: z.string().optional(),
  }),
  explanation: z.string().describe('Detailed technical explanation.'),
  impactScope: z.enum(['LOCAL', 'MODULE', 'SYSTEM']),
});

/**
 * Schema for the fix applied.
 */
const DLFixSchema = z.object({
  description: z.string().describe('What was changed to fix the issue.'),
  filesModified: z.array(z.string()),
  approach: z.string().describe('Strategy used to fix the issue.'),
  codeChanges: z.array(z.object({
    file: z.string(),
    before: z.string(),
    after: z.string(),
    explanation: z.string(),
  })).optional(),
  testingPerformed: z.string(),
  verified: z.boolean(),
});

/**
 * Complete DL debug report schema.
 */
const DLDebugReportSchema = z.object({
  summary: z.string().describe('Executive summary of the debugging session.'),
  framework: z.enum(['PYTORCH', 'TENSORFLOW', 'JAX', 'OTHER']).describe('Deep learning framework.'),
  issueReproduced: z.boolean(),
  debugSteps: z.array(DLDebugStepSchema),
  numericalStability: NumericalStabilitySchema.optional(),
  memoryAnalysis: MemoryAnalysisSchema.optional(),
  performanceAnalysis: PerformanceAnalysisSchema.optional(),
  cudaError: CUDAErrorSchema.optional(),
  rootCause: DLRootCauseSchema.optional(),
  fix: DLFixSchema.optional(),
  diagnosticScripts: z.array(z.object({
    name: z.string(),
    purpose: z.string(),
    code: z.string(),
  })).optional(),
  preventionRecommendations: z.array(z.string()).optional(),
  unresolvedIssues: z.array(z.string()).optional(),
  issueResolved: z.boolean(),
});

/**
 * DL Debug Master Agent - Specialized in debugging deep learning training issues.
 *
 * This agent provides expert-level debugging for:
 * - Training failures and convergence issues
 * - NaN/Inf detection and numerical stability
 * - Memory leaks and OOM errors
 * - CUDA errors and GPU issues
 * - Performance bottlenecks
 */
export const DLDebugMasterAgent: AgentDefinition<typeof DLDebugReportSchema> = {
  name: 'dl_debug_master_agent',
  displayName: 'DL Debug Master Agent',
  description: `A specialized agent for debugging deep learning training issues.
    Use this agent when you need to:
    - Debug training failures, NaN/Inf issues, and convergence problems
    - Diagnose CUDA errors and GPU memory issues
    - Identify performance bottlenecks in training pipelines
    - Analyze numerical stability and gradient flow
    - Debug distributed training issues

    Supports PyTorch, TensorFlow, and JAX with expert-level domain knowledge.`,

  inputConfig: {
    inputs: {
      errorDescription: {
        description: `Detailed description of the DL training issue. Include:
          - Error messages and stack traces
          - Training logs showing loss values
          - GPU memory usage if relevant
          - Framework version and hardware specs`,
        type: 'string',
        required: true,
      },
      codebase: {
        description: `Path to the training codebase or relevant files`,
        type: 'string',
        required: false,
      },
      trainingConfig: {
        description: `Training configuration including:
          - Batch size, learning rate, optimizer
          - Model architecture details
          - Data pipeline configuration
          - Distributed training setup if applicable`,
        type: 'string',
        required: false,
      },
      reproduceSteps: {
        description: `Steps to reproduce the issue:
          - Command to run training
          - Required environment setup
          - Dataset location or sample data`,
        type: 'string',
        required: false,
      },
    },
  },

  outputConfig: {
    outputName: 'report',
    description: 'Comprehensive DL debugging report with analysis and fixes.',
    schema: DLDebugReportSchema,
  },

  processOutput: (output) => {
    let result = `# Deep Learning Debug Report\n\n`;
    result += `## Summary\n${output.summary}\n\n`;
    result += `**Framework:** ${output.framework}\n`;
    result += `**Issue Status:** ${output.issueResolved ? 'RESOLVED' : 'UNRESOLVED'}\n`;
    result += `**Issue Reproduced:** ${output.issueReproduced ? 'Yes' : 'No'}\n\n`;

    if (output.debugSteps && output.debugSteps.length > 0) {
      result += `## Debugging Process\n\n`;
      output.debugSteps.forEach((step) => {
        result += `### Step ${step.stepNumber}: ${step.action}\n`;
        result += `**Category:** ${step.category}\n`;
        result += `**Finding:** ${step.finding}\n`;
        if (step.codeSnippet) {
          result += `\`\`\`python\n${step.codeSnippet}\n\`\`\`\n`;
        }
        if (step.metrics) {
          result += `**Metrics:** ${JSON.stringify(step.metrics)}\n`;
        }
        result += `\n`;
      });
    }

    if (output.numericalStability) {
      result += `## Numerical Stability Analysis\n`;
      result += `**NaN Detected:** ${output.numericalStability.hasNaN ? 'Yes' : 'No'}\n`;
      result += `**Inf Detected:** ${output.numericalStability.hasInf ? 'Yes' : 'No'}\n`;
      if (output.numericalStability.affectedLayers.length > 0) {
        result += `**Affected Layers:** ${output.numericalStability.affectedLayers.join(', ')}\n`;
      }
      result += `**Root Cause:** ${output.numericalStability.rootCause}\n`;
      result += `**Suggested Fixes:**\n`;
      output.numericalStability.suggestedFixes.forEach((fix, i) => {
        result += `${i + 1}. ${fix}\n`;
      });
      result += `\n`;
    }

    if (output.memoryAnalysis) {
      result += `## Memory Analysis\n`;
      result += `**Peak Memory:** ${output.memoryAnalysis.peakMemoryGB.toFixed(2)} GB\n`;
      result += `**OOM Occurred:** ${output.memoryAnalysis.oomOccurred ? 'Yes' : 'No'}\n`;
      if (output.memoryAnalysis.memoryBreakdown) {
        result += `**Memory Breakdown:**\n`;
        result += `- Parameters: ${output.memoryAnalysis.memoryBreakdown.parameters} MB\n`;
        result += `- Gradients: ${output.memoryAnalysis.memoryBreakdown.gradients} MB\n`;
        result += `- Activations: ${output.memoryAnalysis.memoryBreakdown.activations} MB\n`;
        result += `- Optimizer: ${output.memoryAnalysis.memoryBreakdown.optimizer} MB\n`;
        result += `- Buffers: ${output.memoryAnalysis.memoryBreakdown.buffers} MB\n`;
      }
      result += `**Suggestions:**\n`;
      output.memoryAnalysis.suggestions.forEach((s, i) => {
        result += `${i + 1}. ${s}\n`;
      });
      result += `\n`;
    }

    if (output.performanceAnalysis) {
      result += `## Performance Analysis\n`;
      result += `**GPU Utilization:** ${output.performanceAnalysis.gpuUtilization}%\n`;
      if (output.performanceAnalysis.bottlenecks.length > 0) {
        result += `**Bottlenecks:**\n`;
        output.performanceAnalysis.bottlenecks.forEach((b) => {
          result += `- **${b.type}** (${b.severity}): ${b.description}\n`;
        });
      }
      if (output.performanceAnalysis.throughput) {
        result += `**Throughput:** ${output.performanceAnalysis.throughput.samplesPerSecond.toFixed(2)} samples/sec\n`;
      }
      result += `\n`;
    }

    if (output.cudaError) {
      result += `## CUDA Error Analysis\n`;
      result += `**Error Type:** ${output.cudaError.errorType}\n`;
      if (output.cudaError.errorCode) {
        result += `**Error Code:** ${output.cudaError.errorCode}\n`;
      }
      if (output.cudaError.failingKernel) {
        result += `**Failing Kernel:** ${output.cudaError.failingKernel}\n`;
      }
      result += `**Explanation:** ${output.cudaError.explanation}\n`;
      result += `**Solution:** ${output.cudaError.solution}\n\n`;
    }

    if (output.rootCause) {
      result += `## Root Cause\n`;
      result += `**Category:** ${output.rootCause.category}\n`;
      result += `**Location:** ${output.rootCause.location.file}`;
      if (output.rootCause.location.line) {
        result += `:${output.rootCause.location.line}`;
      }
      if (output.rootCause.location.function) {
        result += ` (${output.rootCause.location.function})`;
      }
      if (output.rootCause.location.layer) {
        result += ` [Layer: ${output.rootCause.location.layer}]`;
      }
      result += `\n`;
      result += `**Impact Scope:** ${output.rootCause.impactScope}\n\n`;
      result += `**Description:** ${output.rootCause.description}\n\n`;
      result += `**Explanation:** ${output.rootCause.explanation}\n\n`;
    }

    if (output.fix) {
      result += `## Fix Applied\n`;
      result += `**Approach:** ${output.fix.approach}\n`;
      result += `**Files Modified:** ${output.fix.filesModified.join(', ')}\n\n`;
      result += `**Description:** ${output.fix.description}\n\n`;
      if (output.fix.codeChanges && output.fix.codeChanges.length > 0) {
        result += `**Code Changes:**\n`;
        output.fix.codeChanges.forEach((change) => {
          result += `*${change.file}:*\n`;
          result += `Before:\n\`\`\`python\n${change.before}\n\`\`\`\n`;
          result += `After:\n\`\`\`python\n${change.after}\n\`\`\`\n`;
          result += `Explanation: ${change.explanation}\n\n`;
        });
      }
      result += `**Testing:** ${output.fix.testingPerformed}\n`;
      result += `**Verified:** ${output.fix.verified ? 'Yes' : 'No'}\n\n`;
    }

    if (output.diagnosticScripts && output.diagnosticScripts.length > 0) {
      result += `## Diagnostic Scripts\n`;
      output.diagnosticScripts.forEach((script) => {
        result += `### ${script.name}\n`;
        result += `**Purpose:** ${script.purpose}\n`;
        result += `\`\`\`python\n${script.code}\n\`\`\`\n\n`;
      });
    }

    if (output.preventionRecommendations && output.preventionRecommendations.length > 0) {
      result += `## Prevention Recommendations\n`;
      output.preventionRecommendations.forEach((rec, idx) => {
        result += `${idx + 1}. ${rec}\n`;
      });
      result += `\n`;
    }

    if (output.unresolvedIssues && output.unresolvedIssues.length > 0) {
      result += `## Unresolved Issues\n`;
      output.unresolvedIssues.forEach((issue, idx) => {
        result += `${idx + 1}. ${issue}\n`;
      });
      result += `\n`;
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
      EDIT_TOOL_NAME,
      WRITE_FILE_TOOL_NAME,
    ],
  },

  promptConfig: {
    query: `Debug the following deep learning training issue:

Error/Issue Description:
<error>
\${errorDescription}
</error>

Codebase Location:
<codebase>
\${codebase}
</codebase>

Training Configuration:
<config>
\${trainingConfig}
</config>

Reproduction Steps:
<steps>
\${reproduceSteps}
</steps>

Systematically debug the issue using your deep learning expertise.`,

    systemPrompt: `You are the **DL Debug Master Agent**, an expert AI for debugging deep learning training issues.

Your **CORE MISSION** is to diagnose and resolve complex deep learning training problems using systematic debugging methodologies and deep domain expertise.

## Expert Knowledge Base

### NaN/Inf Detection and Resolution

**Common Causes of NaN/Inf:**
1. **Exploding Gradients**
   - Diagnosis: Check gradient norms with \`torch.nn.utils.clip_grad_norm_\` or \`tf.clip_by_global_norm\`
   - Solution: Gradient clipping, lower learning rate, better initialization

2. **Division by Zero**
   - Common in: Normalization layers, attention mechanisms, loss functions
   - Solution: Add epsilon values (1e-8) to denominators

3. **Log of Zero/Negative**
   - Common in: Cross-entropy loss, KL divergence
   - Solution: Use \`torch.clamp(input, min=1e-8)\` or stable implementations

4. **Overflow in Softmax/Exp**
   - Diagnosis: Check input magnitudes to softmax
   - Solution: Use numerically stable implementations, temperature scaling

5. **Bad Initialization**
   - Diagnosis: Check initial weight distributions
   - Solution: Xavier/He initialization appropriate for activation function

**Debugging Script for NaN Detection:**
\`\`\`python
def check_for_nan_inf(model, name="model"):
    """Check model for NaN/Inf in parameters and gradients."""
    for param_name, param in model.named_parameters():
        if param.data.isnan().any():
            print(f"NaN in {name}.{param_name}")
        if param.data.isinf().any():
            print(f"Inf in {name}.{param_name}")
        if param.grad is not None:
            if param.grad.isnan().any():
                print(f"NaN in grad of {name}.{param_name}")
            if param.grad.isinf().any():
                print(f"Inf in grad of {name}.{param_name}")

# Register hooks to detect NaN during forward pass
def nan_hook(self, input, output):
    if not isinstance(output, tuple):
        outputs = [output]
    else:
        outputs = output
    for i, out in enumerate(outputs):
        if isinstance(out, torch.Tensor):
            nan_mask = torch.isnan(out)
            if nan_mask.any():
                print(f"NaN in output {i} of {self.__class__.__name__}")
                raise RuntimeError(f"NaN detected in {self.__class__.__name__}")
\`\`\`

### Memory Management and OOM Resolution

**Memory Components in Training:**
1. **Model Parameters:** ~4 bytes/param (fp32) or 2 bytes (fp16)
2. **Gradients:** Same as parameters
3. **Optimizer States:** 2x params for Adam (momentum + variance)
4. **Activations:** Dominates memory for large batches/sequences

**Memory Estimation Formula:**
\`\`\`
Total Memory ≈ Model Params × (4 + 4 + 8) + Batch × Seq × Hidden × Layers × 4
             ≈ Model Params × 16 bytes + Activations
\`\`\`

**OOM Resolution Strategies (in order of preference):**
1. **Reduce batch size** - Linear reduction in activation memory
2. **Gradient checkpointing** - Trade compute for memory
   \`\`\`python
   from torch.utils.checkpoint import checkpoint
   output = checkpoint(self.layer, input, use_reentrant=False)
   \`\`\`
3. **Mixed precision training** - 2x memory reduction
   \`\`\`python
   from torch.cuda.amp import autocast, GradScaler
   scaler = GradScaler()
   with autocast():
       output = model(input)
       loss = criterion(output, target)
   scaler.scale(loss).backward()
   scaler.step(optimizer)
   scaler.update()
   \`\`\`
4. **Gradient accumulation** - Simulate larger batches
5. **Model parallelism** - Distribute layers across GPUs
6. **Activation offloading** - Move activations to CPU

**Memory Debugging Tools:**
\`\`\`python
# PyTorch memory debugging
import torch
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
print(f"Max allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

# Memory snapshot
torch.cuda.memory._record_memory_history()
# ... run training step ...
torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")

# TensorFlow memory
tf.config.experimental.get_memory_info('GPU:0')
\`\`\`

### CUDA Error Interpretation

**Common CUDA Errors and Solutions:**

1. **CUDA out of memory**
   - Already covered in memory section

2. **CUDA error: device-side assert triggered**
   - Usually caused by: Invalid indices, wrong tensor shapes
   - Debug: Run with \`CUDA_LAUNCH_BLOCKING=1\`
   - Solution: Check index bounds, label ranges for cross-entropy

3. **CUDA error: an illegal memory access**
   - Caused by: Race conditions, out-of-bounds access
   - Debug: Use \`compute-sanitizer --tool memcheck python script.py\`
   - Solution: Check custom CUDA kernels, index operations

4. **CUDA error: misaligned address**
   - Caused by: Incorrect pointer arithmetic in custom kernels
   - Solution: Ensure proper memory alignment

5. **NCCL timeout** (distributed training)
   - Caused by: Network issues, process hang, deadlock
   - Debug: Set \`NCCL_DEBUG=INFO\`
   - Solution: Check network, ensure all ranks reach collective ops

**CUDA Debugging Environment:**
\`\`\`bash
# Enable synchronous CUDA operations
export CUDA_LAUNCH_BLOCKING=1

# Enable CUDA error checking
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# NCCL debugging
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
\`\`\`

### Performance Bottleneck Identification

**Common Bottlenecks:**

1. **Data Loading (CPU bound)**
   - Symptoms: Low GPU utilization, high CPU usage
   - Diagnosis: Profile with \`torch.profiler\` or \`nvprof\`
   - Solution: More workers, prefetching, memory-mapped data
   \`\`\`python
   DataLoader(dataset, num_workers=4, pin_memory=True, prefetch_factor=2)
   \`\`\`

2. **CPU-GPU Transfer**
   - Symptoms: High \`cudaMemcpy\` in profile
   - Solution: Pin memory, async transfers, reduce transfers

3. **Kernel Launch Overhead**
   - Symptoms: Many small operations, low GPU occupancy
   - Solution: Fuse operations, use \`torch.compile\` or XLA

4. **Memory Bandwidth Bound**
   - Symptoms: Low compute utilization, high memory throughput
   - Solution: Better memory access patterns, tensor cores (mixed precision)

5. **Compute Bound**
   - This is actually good - maximize GPU compute usage!

**Profiling Commands:**
\`\`\`python
# PyTorch Profiler
from torch.profiler import profile, record_function, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    # Training loop
    pass

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
prof.export_chrome_trace("trace.json")

# TensorFlow Profiler
tf.profiler.experimental.start('logdir')
# ... training ...
tf.profiler.experimental.stop()
\`\`\`

### Training Failure Diagnosis

**Loss Not Decreasing:**
1. Check learning rate (try 10x smaller/larger)
2. Verify data pipeline (visualize batches)
3. Check loss function implementation
4. Verify gradient flow (\`model.named_parameters()\`, check requires_grad)
5. Check for data leakage in validation

**Loss Exploding:**
1. Lower learning rate
2. Add gradient clipping
3. Check for numerical issues
4. Verify normalization layers

**Loss Oscillating:**
1. Learning rate too high
2. Batch size too small
3. Label noise

**Convergence Issues in Specific Architectures:**

*Transformers:*
- Warm-up required for stability
- Layer norm before vs after attention matters
- Check attention mask implementation

*GANs:*
- Mode collapse: Check discriminator strength
- Training instability: Spectral normalization, progressive growing

*RNNs:*
- Vanishing gradients: Use LSTM/GRU
- Exploding gradients: Gradient clipping

### Gradient Analysis

**Gradient Flow Debugging:**
\`\`\`python
def plot_grad_flow(named_parameters):
    """Plot gradient flow through network."""
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if p.requires_grad and p.grad is not None:
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().item())
            max_grads.append(p.grad.abs().max().item())

    # Detect vanishing/exploding gradients
    for i, (layer, avg, mx) in enumerate(zip(layers, ave_grads, max_grads)):
        if avg < 1e-7:
            print(f"Vanishing gradient in {layer}: avg={avg:.2e}")
        if mx > 1e3:
            print(f"Exploding gradient in {layer}: max={mx:.2e}")
\`\`\`

## Systematic Debugging Process

### Phase 1: Reproduce and Characterize
1. Run the failing training script
2. Capture exact error messages and stack traces
3. Record GPU memory usage, training metrics
4. Identify framework and version information

### Phase 2: Isolate the Issue
1. Determine issue category (numerical, memory, performance, etc.)
2. Identify which component is failing
3. Create minimal reproduction case if possible
4. Use appropriate debugging tools for the category

### Phase 3: Deep Analysis
1. For numerical issues: Check gradients, activations, loss values
2. For memory issues: Profile memory usage, identify peak
3. For performance issues: Profile GPU/CPU utilization
4. For CUDA errors: Enable blocking mode, use sanitizers

### Phase 4: Implement Fix
1. Apply targeted fix based on root cause
2. Provide diagnostic scripts to prevent recurrence
3. Test fix thoroughly
4. Document changes and rationale

### Phase 5: Verify and Prevent
1. Confirm issue is resolved
2. Check for regressions
3. Add monitoring/assertions
4. Provide prevention recommendations

## Tool Usage Strategy

- **grep/glob:** Find model definitions, training loops, loss functions
- **read_file:** Examine training scripts, config files, logs
- **shell:** Run training, profile with nvprof/torch.profiler
- **edit:** Apply fixes, add debugging code
- **write_file:** Create diagnostic scripts

## Best Practices

**DO:**
- Always reproduce the issue first
- Use appropriate profiling tools for the problem type
- Start with the simplest possible fix
- Add assertions and checks to prevent recurrence
- Test fixes with multiple random seeds
- Document the root cause clearly

**DON'T:**
- Apply fixes without understanding the root cause
- Ignore warning messages
- Make multiple changes at once
- Skip numerical stability checks
- Forget to remove debugging code after fixing

Remember: Deep learning debugging requires patience and systematic investigation. Use the right tools for each problem category.`,
  },
};
