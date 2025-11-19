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
import { DLDebugMasterAgent } from '../../agents/dl/debug-master-agent.js';
import { DLMemoryAnalyzerAgent } from '../../agents/dl/memory-analyzer-agent.js';

const DebuggingInputSchema = z.object({
  errorType: z.enum([
    'nan_inf',
    'oom',
    'cuda_error',
    'training_failure',
    'performance',
    'convergence',
    'distributed',
    'general',
  ]).describe('Type of error to debug'),
  errorMessage: z.string().describe('Error message or description'),
  stackTrace: z.string().optional().describe('Stack trace if available'),
  codebasePath: z.string().optional().describe('Path to the training codebase'),
  trainingConfig: z.object({
    framework: z.enum(['pytorch', 'tensorflow', 'jax']).default('pytorch'),
    batchSize: z.number().optional(),
    learningRate: z.number().optional(),
    optimizer: z.string().optional(),
    modelType: z.string().optional(),
  }).optional().describe('Training configuration'),
  hardwareInfo: z.object({
    gpuModel: z.string().optional(),
    gpuCount: z.number().optional(),
    gpuMemoryGB: z.number().optional(),
  }).optional().describe('Hardware information'),
  additionalContext: z.string().optional().describe('Any additional context'),
});

const DebuggingOutputSchema = z.object({
  success: z.boolean(),
  summary: z.string(),
  errorAnalysis: z.object({
    category: z.string(),
    rootCause: z.string(),
    affectedComponents: z.array(z.string()),
    severity: z.enum(['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']),
  }),
  solution: z.object({
    description: z.string(),
    steps: z.array(z.string()),
    codeChanges: z.array(z.object({
      file: z.string(),
      change: z.string(),
      explanation: z.string(),
    })).optional(),
    estimatedEffort: z.string(),
  }),
  diagnosticScripts: z.array(z.object({
    name: z.string(),
    purpose: z.string(),
    code: z.string(),
  })),
  preventionMeasures: z.array(z.string()),
  relatedPatterns: z.array(z.object({
    pattern: z.string(),
    description: z.string(),
    solution: z.string(),
  })),
  artifacts: z.array(z.string()).optional(),
});

type DebuggingInput = z.infer<typeof DebuggingInputSchema>;
type DebuggingOutput = z.infer<typeof DebuggingOutputSchema>;

/**
 * Common error patterns database for deep learning debugging.
 */
const ERROR_PATTERNS_DATABASE = {
  nan_inf: [
    {
      pattern: 'NaN in loss after few iterations',
      description: 'Loss becomes NaN early in training',
      causes: ['Learning rate too high', 'Bad initialization', 'Exploding gradients'],
      solution: 'Reduce learning rate by 10x, use gradient clipping, check initialization',
      diagnosticCode: `
# Check for NaN in model
def check_nan(model, name="model"):
    for n, p in model.named_parameters():
        if p.data.isnan().any():
            print(f"NaN in {name}.{n}")
        if p.grad is not None and p.grad.isnan().any():
            print(f"NaN in grad of {name}.{n}")
`,
    },
    {
      pattern: 'NaN in attention/softmax',
      description: 'NaN appears specifically in attention layers',
      causes: ['Overflow before softmax', 'Div by zero in normalization'],
      solution: 'Use scaled dot product attention, add epsilon to softmax denominator',
      diagnosticCode: `
# Check attention scores before softmax
def attention_hook(module, input, output):
    attn_weights = input[0]  # Adjust based on your model
    if attn_weights.abs().max() > 1e4:
        print(f"Large attention scores: {attn_weights.abs().max()}")
`,
    },
    {
      pattern: 'NaN in batch normalization',
      description: 'NaN in batch norm running statistics',
      causes: ['Very small batch size', 'Extreme feature values'],
      solution: 'Use layer norm or group norm, increase batch size, clamp running stats',
      diagnosticCode: `
# Check batch norm statistics
for name, module in model.named_modules():
    if isinstance(module, nn.BatchNorm2d):
        if module.running_var.min() < 1e-10:
            print(f"Very small variance in {name}")
`,
    },
  ],
  oom: [
    {
      pattern: 'CUDA out of memory on first batch',
      description: 'OOM immediately when training starts',
      causes: ['Model too large', 'Batch size too large', 'Multiple models loaded'],
      solution: 'Reduce batch size, use gradient checkpointing, use mixed precision',
      diagnosticCode: `
# Estimate memory before training
import torch
params = sum(p.numel() * p.element_size() for p in model.parameters())
print(f"Model parameters: {params / 1e9:.2f} GB")
print(f"Estimated total (with Adam): {params * 16 / 1e9:.2f} GB")
`,
    },
    {
      pattern: 'OOM during backward pass',
      description: 'OOM when computing gradients',
      causes: ['Activation memory too high', 'Long sequences', 'Deep networks'],
      solution: 'Use gradient checkpointing, reduce sequence length, use FSDP',
      diagnosticCode: `
# Enable gradient checkpointing
from torch.utils.checkpoint import checkpoint_sequential
# Wrap your sequential layers
output = checkpoint_sequential(self.layers, segments=4, input=x)
`,
    },
    {
      pattern: 'Gradual memory increase',
      description: 'Memory grows over training steps',
      causes: ['Memory leak', 'Accumulating computation graph', 'Storing tensors'],
      solution: 'Detach tensors for logging, use .item() for scalars, clear cache',
      diagnosticCode: `
# Find memory leaks
import gc
gc.collect()
torch.cuda.empty_cache()

# Check for tensors with grad_fn
for obj in gc.get_objects():
    if torch.is_tensor(obj) and obj.grad_fn is not None:
        print(f"Tensor with grad_fn: {obj.shape}")
`,
    },
  ],
  cuda_error: [
    {
      pattern: 'CUDA error: device-side assert triggered',
      description: 'Assert failure on GPU',
      causes: ['Invalid index', 'Label out of range', 'Shape mismatch'],
      solution: 'Run with CUDA_LAUNCH_BLOCKING=1 to get exact location',
      diagnosticCode: `
# Run with blocking mode
# export CUDA_LAUNCH_BLOCKING=1
# python train.py

# Check common issues
assert labels.max() < num_classes, f"Label {labels.max()} >= num_classes {num_classes}"
assert indices.max() < tensor.size(0), "Index out of bounds"
`,
    },
    {
      pattern: 'CUDA error: an illegal memory access',
      description: 'Invalid memory access on GPU',
      causes: ['Out-of-bounds indexing', 'Use after free', 'Race condition'],
      solution: 'Use compute-sanitizer for detailed analysis',
      diagnosticCode: `
# Use compute-sanitizer
# compute-sanitizer --tool memcheck python train.py

# Or enable anomaly detection
torch.autograd.set_detect_anomaly(True)
`,
    },
    {
      pattern: 'NCCL timeout',
      description: 'Distributed training timeout',
      causes: ['Network issues', 'Process hang', 'Deadlock'],
      solution: 'Check network, ensure all ranks reach collective ops',
      diagnosticCode: `
# Debug NCCL
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL

# Barrier for debugging
import torch.distributed as dist
print(f"Rank {dist.get_rank()} before barrier")
dist.barrier()
print(f"Rank {dist.get_rank()} after barrier")
`,
    },
  ],
  training_failure: [
    {
      pattern: 'Loss not decreasing',
      description: 'Training loss stays flat',
      causes: ['Learning rate issue', 'Data problem', 'Architecture bug'],
      solution: 'Overfit on single batch first, check gradients, visualize data',
      diagnosticCode: `
# Overfit test
single_batch = next(iter(dataloader))
for i in range(100):
    loss = train_step(model, single_batch)
    if i % 10 == 0:
        print(f"Step {i}: Loss = {loss:.4f}")
# Loss should decrease significantly
`,
    },
    {
      pattern: 'Mode collapse (GAN)',
      description: 'Generator produces same output',
      causes: ['Discriminator too strong', 'Gradient vanishing'],
      solution: 'Use spectral norm, WGAN-GP, feature matching',
      diagnosticCode: `
# Check generator output diversity
outputs = [model.generate() for _ in range(10)]
variance = torch.stack(outputs).var(dim=0).mean()
print(f"Output variance: {variance:.4f}")  # Should be > 0
`,
    },
    {
      pattern: 'Gradient explosion',
      description: 'Gradients become very large',
      causes: ['Learning rate too high', 'Deep networks', 'Recurrent connections'],
      solution: 'Gradient clipping, better initialization, residual connections',
      diagnosticCode: `
# Monitor gradient norms
total_norm = 0
for p in model.parameters():
    if p.grad is not None:
        total_norm += p.grad.data.norm(2).item() ** 2
total_norm = total_norm ** 0.5
print(f"Total gradient norm: {total_norm:.4f}")
`,
    },
  ],
  convergence: [
    {
      pattern: 'Training-validation gap',
      description: 'Training loss low but validation high',
      causes: ['Overfitting', 'Data leakage', 'Insufficient regularization'],
      solution: 'Add dropout, weight decay, data augmentation, early stopping',
      diagnosticCode: `
# Check for data leakage
train_samples = set(train_dataset.samples)
val_samples = set(val_dataset.samples)
overlap = train_samples & val_samples
if overlap:
    print(f"WARNING: {len(overlap)} overlapping samples!")
`,
    },
    {
      pattern: 'Oscillating loss',
      description: 'Loss goes up and down',
      causes: ['Learning rate too high', 'Batch size too small', 'Noisy gradients'],
      solution: 'Reduce learning rate, increase batch size, use gradient accumulation',
      diagnosticCode: `
# Compute gradient noise
grads_1 = compute_gradients(batch_1)
grads_2 = compute_gradients(batch_2)
noise = (grads_1 - grads_2).norm() / grads_1.norm()
print(f"Gradient noise ratio: {noise:.4f}")
`,
    },
  ],
};

/**
 * DL Debugging Skill
 *
 * Expert skill for debugging deep learning training issues.
 * Provides systematic debugging workflows, common error patterns,
 * and ready-to-use diagnostic scripts.
 */
export const DLDebuggingSkill: SkillDefinition<DebuggingInput, DebuggingOutput> = {
  id: 'dl.debugging',
  name: 'DL Debugging',
  description: 'Debug deep learning training issues with expert analysis and solution templates',
  usage: `
Debug common deep learning training problems:

Error Types:
  - nan_inf: NaN or Inf values in training
  - oom: Out of memory errors
  - cuda_error: CUDA/GPU errors
  - training_failure: Training not progressing
  - performance: Performance issues
  - convergence: Convergence problems
  - distributed: Multi-GPU/distributed issues
  - general: Other issues

Provides:
  - Root cause analysis
  - Step-by-step solutions
  - Diagnostic scripts
  - Prevention measures
  `,
  category: SkillCategory.GENERAL,
  complexity: SkillComplexity.ADVANCED,
  version: '1.0.0',
  author: 'Gemini CLI',
  inputSchema: DebuggingInputSchema,
  outputSchema: DebuggingOutputSchema,
  requiredAgents: ['dl_debug_master_agent'],
  optionalAgents: ['dl_memory_analyzer_agent'],
  estimatedTime: '5-15 minutes',
  tags: ['debugging', 'deep-learning', 'pytorch', 'tensorflow', 'cuda', 'training', 'troubleshooting'],

  examples: [
    {
      title: 'Debug NaN in loss',
      description: 'Diagnose and fix NaN values appearing during training',
      input: {
        errorType: 'nan_inf',
        errorMessage: 'Loss is NaN after 100 iterations',
        trainingConfig: {
          framework: 'pytorch',
          batchSize: 32,
          learningRate: 0.001,
          optimizer: 'Adam',
        },
      },
    },
    {
      title: 'Debug CUDA OOM',
      description: 'Solve out-of-memory error on GPU',
      input: {
        errorType: 'oom',
        errorMessage: 'CUDA out of memory. Tried to allocate 2.00 GiB',
        hardwareInfo: {
          gpuModel: 'NVIDIA A100',
          gpuCount: 1,
          gpuMemoryGB: 40,
        },
        trainingConfig: {
          framework: 'pytorch',
          batchSize: 64,
          modelType: 'transformer',
        },
      },
    },
    {
      title: 'Debug training not converging',
      description: 'Investigate why training loss is not decreasing',
      input: {
        errorType: 'training_failure',
        errorMessage: 'Loss stuck at 2.3 after 1000 iterations',
        trainingConfig: {
          framework: 'pytorch',
          learningRate: 0.0001,
          optimizer: 'SGD',
        },
      },
    },
  ],

  async execute(
    input: DebuggingInput,
    context: SkillContext,
    config: any,
  ): Promise<SkillResult<DebuggingOutput>> {
    const startTime = Date.now();
    const agentsUsed: string[] = [];

    try {
      // Get relevant error patterns
      const patterns = ERROR_PATTERNS_DATABASE[input.errorType] || [];

      // Run the debug master agent
      const debugExecutor = await AgentExecutor.create(
        DLDebugMasterAgent,
        config,
      );
      agentsUsed.push('dl_debug_master_agent');

      const debugResult = await debugExecutor.run({
        errorDescription: `${input.errorMessage}\n\nStack Trace:\n${input.stackTrace || 'Not provided'}`,
        codebase: input.codebasePath || context.workingDir,
        trainingConfig: input.trainingConfig ?
          `Framework: ${input.trainingConfig.framework}
Batch Size: ${input.trainingConfig.batchSize}
Learning Rate: ${input.trainingConfig.learningRate}
Optimizer: ${input.trainingConfig.optimizer}
Model Type: ${input.trainingConfig.modelType}` : undefined,
        reproduceSteps: input.additionalContext,
      });

      // For memory-related issues, also run memory analyzer
      let memoryAnalysis: any = null;
      if (input.errorType === 'oom') {
        try {
          const memoryExecutor = await AgentExecutor.create(
            DLMemoryAnalyzerAgent,
            config,
          );
          agentsUsed.push('dl_memory_analyzer_agent');

          const memResult = await memoryExecutor.run({
            memoryIssue: input.errorMessage,
            codebase: input.codebasePath || context.workingDir,
            modelConfig: input.trainingConfig ?
              `Batch Size: ${input.trainingConfig.batchSize}, Model: ${input.trainingConfig.modelType}` : undefined,
            hardwareSpec: input.hardwareInfo ?
              `${input.hardwareInfo.gpuCount}x ${input.hardwareInfo.gpuModel} (${input.hardwareInfo.gpuMemoryGB}GB)` : undefined,
          });
          memoryAnalysis = JSON.parse(memResult.result);
        } catch {
          // Memory analysis is optional
        }
      }

      // Parse agent result
      const agentData = JSON.parse(debugResult.result);

      // Generate diagnostic scripts based on error type
      const diagnosticScripts = this.generateDiagnosticScripts(input.errorType, input.trainingConfig?.framework || 'pytorch');

      // Map related patterns from our database
      const relatedPatterns = patterns.map(p => ({
        pattern: p.pattern,
        description: p.description,
        solution: p.solution,
      }));

      // Build output
      const output: DebuggingOutput = {
        success: true,
        summary: agentData.summary || `Debugging ${input.errorType} issue: ${input.errorMessage.substring(0, 100)}...`,

        errorAnalysis: {
          category: input.errorType.toUpperCase(),
          rootCause: agentData.rootCause?.description || this.inferRootCause(input),
          affectedComponents: agentData.rootCause?.location ?
            [agentData.rootCause.location.file] :
            this.inferAffectedComponents(input.errorType),
          severity: this.determineSeverity(input.errorType),
        },

        solution: {
          description: agentData.fix?.description || this.getDefaultSolution(input.errorType),
          steps: agentData.fix ?
            [
              `Apply fix: ${agentData.fix.approach}`,
              `Modified files: ${agentData.fix.filesModified?.join(', ')}`,
              `Testing: ${agentData.fix.testingPerformed}`,
            ] :
            this.getDefaultSteps(input.errorType),
          codeChanges: agentData.fix?.codeChanges,
          estimatedEffort: this.estimateEffort(input.errorType),
        },

        diagnosticScripts: diagnosticScripts,

        preventionMeasures: agentData.preventionRecommendations || this.getPreventionMeasures(input.errorType),

        relatedPatterns: relatedPatterns,

        artifacts: [
          './debug_scripts/check_nan.py',
          './debug_scripts/memory_analysis.py',
          './debug_scripts/gradient_monitor.py',
        ],
      };

      // Add memory-specific information if available
      if (memoryAnalysis) {
        output.solution.steps.push(
          `Memory optimization: Estimated ${memoryAnalysis.implementationPlan?.estimatedTotalSaving || 'significant'} memory reduction possible`
        );
      }

      return {
        success: true,
        data: output,
        metadata: {
          skillName: 'DL Debugging',
          duration: Date.now() - startTime,
          agentsUsed,
        },
        artifacts: output.artifacts,
        suggestions: [
          'Run diagnostic scripts to gather more information',
          'Apply the suggested fixes incrementally',
          'Monitor training metrics after each change',
          'Add assertions to catch issues early in future runs',
        ],
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : String(error),
        metadata: {
          skillName: 'DL Debugging',
          duration: Date.now() - startTime,
          agentsUsed,
        },
      };
    }
  },

  // Helper methods attached to the skill
} as SkillDefinition<DebuggingInput, DebuggingOutput> & {
  generateDiagnosticScripts: (errorType: string, framework: string) => Array<{name: string; purpose: string; code: string}>;
  inferRootCause: (input: DebuggingInput) => string;
  inferAffectedComponents: (errorType: string) => string[];
  determineSeverity: (errorType: string) => 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  getDefaultSolution: (errorType: string) => string;
  getDefaultSteps: (errorType: string) => string[];
  estimateEffort: (errorType: string) => string;
  getPreventionMeasures: (errorType: string) => string[];
};

// Attach helper methods
Object.assign(DLDebuggingSkill, {
  generateDiagnosticScripts(errorType: string, framework: string): Array<{name: string; purpose: string; code: string}> {
    const scripts: Array<{name: string; purpose: string; code: string}> = [];

    // Common diagnostic script
    scripts.push({
      name: 'check_environment.py',
      purpose: 'Verify environment and dependencies',
      code: `import torch
import sys

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
`,
    });

    // Error-specific scripts
    switch (errorType) {
      case 'nan_inf':
        scripts.push({
          name: 'check_nan_inf.py',
          purpose: 'Detect NaN/Inf in model parameters and gradients',
          code: `import torch

def check_model_for_nan_inf(model):
    issues = []
    for name, param in model.named_parameters():
        if param.data.isnan().any():
            issues.append(f"NaN in {name}")
        if param.data.isinf().any():
            issues.append(f"Inf in {name}")
        if param.grad is not None:
            if param.grad.isnan().any():
                issues.append(f"NaN in gradient of {name}")
            if param.grad.isinf().any():
                issues.append(f"Inf in gradient of {name}")
    return issues

# Add forward hook to detect NaN during forward pass
def register_nan_hooks(model):
    def nan_hook(module, input, output):
        if isinstance(output, torch.Tensor) and output.isnan().any():
            raise RuntimeError(f"NaN detected in {module.__class__.__name__}")

    for module in model.modules():
        module.register_forward_hook(nan_hook)
`,
        });
        break;

      case 'oom':
        scripts.push({
          name: 'memory_analysis.py',
          purpose: 'Analyze GPU memory usage',
          code: `import torch

def analyze_memory():
    print("=== GPU Memory Analysis ===")
    print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    print(f"Max allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

def estimate_model_memory(model):
    params = sum(p.numel() * p.element_size() for p in model.parameters())
    buffers = sum(b.numel() * b.element_size() for b in model.buffers())

    print(f"Parameters: {params / 1e6:.2f} MB")
    print(f"Buffers: {buffers / 1e6:.2f} MB")
    print(f"With gradients: {params * 2 / 1e6:.2f} MB")
    print(f"With Adam optimizer: {params * 4 / 1e6:.2f} MB")
`,
        });
        break;

      case 'cuda_error':
        scripts.push({
          name: 'cuda_debug.py',
          purpose: 'Debug CUDA errors',
          code: `import os
import torch

# Enable synchronous CUDA execution for better error messages
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Enable anomaly detection for autograd
torch.autograd.set_detect_anomaly(True)

print("CUDA debugging enabled")
print("Re-run your training script with these settings")
`,
        });
        break;

      case 'training_failure':
        scripts.push({
          name: 'gradient_analysis.py',
          purpose: 'Analyze gradient flow',
          code: `import torch

def analyze_gradients(model):
    total_norm = 0.0
    max_norm = 0.0
    zero_grads = []

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm(2).item()
            total_norm += grad_norm ** 2
            max_norm = max(max_norm, grad_norm)

            if grad_norm < 1e-7:
                zero_grads.append(name)

    total_norm = total_norm ** 0.5

    print(f"Total gradient norm: {total_norm:.6f}")
    print(f"Max gradient norm: {max_norm:.6f}")

    if zero_grads:
        print(f"\\nZero gradients in: {', '.join(zero_grads)}")

    if total_norm < 1e-6:
        print("WARNING: Vanishing gradients!")
    elif total_norm > 1e3:
        print("WARNING: Exploding gradients!")
`,
        });
        break;
    }

    return scripts;
  },

  inferRootCause(input: DebuggingInput): string {
    const causes: Record<string, string> = {
      nan_inf: 'Numerical instability causing NaN/Inf values, likely due to learning rate, initialization, or numerical operations',
      oom: 'GPU memory exhausted due to model size, batch size, or activation memory',
      cuda_error: 'CUDA runtime error, often caused by invalid operations or memory access',
      training_failure: 'Training process not progressing, possibly due to hyperparameters or data issues',
      performance: 'Training slower than expected due to bottlenecks',
      convergence: 'Model not converging to good solution',
      distributed: 'Issues with multi-GPU or distributed training setup',
      general: 'General training issue requiring investigation',
    };
    return causes[input.errorType] || 'Unknown root cause';
  },

  inferAffectedComponents(errorType: string): string[] {
    const components: Record<string, string[]> = {
      nan_inf: ['Model parameters', 'Loss function', 'Optimizer'],
      oom: ['GPU memory', 'Model layers', 'Data pipeline'],
      cuda_error: ['CUDA runtime', 'GPU operations'],
      training_failure: ['Training loop', 'Model architecture', 'Data loading'],
      performance: ['Data loading', 'GPU utilization', 'CPU-GPU transfer'],
      convergence: ['Loss function', 'Optimizer', 'Learning rate schedule'],
      distributed: ['NCCL', 'Process group', 'Network'],
      general: ['Training pipeline'],
    };
    return components[errorType] || ['Unknown'];
  },

  determineSeverity(errorType: string): 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL' {
    const severity: Record<string, 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL'> = {
      nan_inf: 'HIGH',
      oom: 'CRITICAL',
      cuda_error: 'CRITICAL',
      training_failure: 'HIGH',
      performance: 'MEDIUM',
      convergence: 'MEDIUM',
      distributed: 'HIGH',
      general: 'MEDIUM',
    };
    return severity[errorType] || 'MEDIUM';
  },

  getDefaultSolution(errorType: string): string {
    const solutions: Record<string, string> = {
      nan_inf: 'Reduce learning rate, add gradient clipping, check for numerical operations',
      oom: 'Reduce batch size, enable gradient checkpointing, use mixed precision',
      cuda_error: 'Run with CUDA_LAUNCH_BLOCKING=1, check index bounds and tensor shapes',
      training_failure: 'Check data pipeline, verify gradients flow, try overfitting on single batch',
      performance: 'Increase DataLoader workers, use pin_memory, profile to find bottleneck',
      convergence: 'Tune learning rate, add regularization, check for data leakage',
      distributed: 'Verify network connectivity, check NCCL configuration, ensure synchronized operations',
      general: 'Review code, check logs, add debugging statements',
    };
    return solutions[errorType] || 'Investigate and apply appropriate fix';
  },

  getDefaultSteps(errorType: string): string[] {
    const steps: Record<string, string[]> = {
      nan_inf: [
        'Enable anomaly detection: torch.autograd.set_detect_anomaly(True)',
        'Add NaN checks after each layer',
        'Reduce learning rate by 10x',
        'Add gradient clipping: torch.nn.utils.clip_grad_norm_(params, 1.0)',
        'Check for log(0) or div/0 in loss computation',
      ],
      oom: [
        'Check current memory usage with torch.cuda.memory_summary()',
        'Reduce batch size by half',
        'Enable mixed precision training',
        'Enable gradient checkpointing',
        'Clear cache between batches: torch.cuda.empty_cache()',
      ],
      cuda_error: [
        'Set CUDA_LAUNCH_BLOCKING=1 for better error messages',
        'Check tensor shapes before operations',
        'Verify label values are within valid range',
        'Run with compute-sanitizer for detailed analysis',
      ],
      training_failure: [
        'Overfit on a single batch to verify model can learn',
        'Check gradient norms for vanishing/exploding gradients',
        'Visualize training data to verify correctness',
        'Try different learning rates',
        'Check loss function implementation',
      ],
      performance: [
        'Profile with PyTorch profiler',
        'Check GPU utilization with nvidia-smi',
        'Increase DataLoader num_workers',
        'Enable pin_memory in DataLoader',
        'Use torch.compile() for kernel fusion',
      ],
      convergence: [
        'Check for train/val data overlap',
        'Add early stopping',
        'Tune learning rate schedule',
        'Add regularization (dropout, weight decay)',
        'Check data augmentation',
      ],
      distributed: [
        'Verify all processes can reach each other',
        'Check NCCL_DEBUG logs',
        'Ensure all ranks execute same operations',
        'Add barriers for debugging',
        'Check network bandwidth',
      ],
      general: [
        'Review recent code changes',
        'Check logs for warnings',
        'Add print statements/breakpoints',
        'Compare with known working configuration',
      ],
    };
    return steps[errorType] || ['Investigate the issue'];
  },

  estimateEffort(errorType: string): string {
    const effort: Record<string, string> = {
      nan_inf: '30 minutes - 2 hours',
      oom: '1-4 hours',
      cuda_error: '1-3 hours',
      training_failure: '2-6 hours',
      performance: '2-8 hours',
      convergence: '4-12 hours',
      distributed: '4-8 hours',
      general: '1-4 hours',
    };
    return effort[errorType] || '1-4 hours';
  },

  getPreventionMeasures(errorType: string): string[] {
    const measures: Record<string, string[]> = {
      nan_inf: [
        'Always use gradient clipping',
        'Add NaN assertions in forward pass',
        'Use numerically stable implementations',
        'Start with conservative learning rate',
        'Log gradient norms during training',
      ],
      oom: [
        'Profile memory before full training',
        'Start with small batch and increase',
        'Use mixed precision by default',
        'Implement early OOM detection',
        'Use memory-efficient attention',
      ],
      cuda_error: [
        'Always validate tensor shapes',
        'Check label ranges before loss computation',
        'Use type hints and assertions',
        'Test with small data first',
      ],
      training_failure: [
        'Always test overfitting on single batch first',
        'Log comprehensive metrics',
        'Use learning rate finder',
        'Implement sanity checks',
      ],
      performance: [
        'Profile regularly during development',
        'Benchmark data loading separately',
        'Monitor GPU utilization',
        'Use efficient data formats',
      ],
      convergence: [
        'Use proper train/val/test splits',
        'Implement early stopping',
        'Log learning curves',
        'Use learning rate schedules',
      ],
      distributed: [
        'Test single GPU before multi-GPU',
        'Use robust communication libraries',
        'Implement proper cleanup/error handling',
        'Monitor all processes',
      ],
      general: [
        'Write comprehensive tests',
        'Use logging extensively',
        'Version control experiments',
        'Document configurations',
      ],
    };
    return measures[errorType] || ['Follow best practices'];
  },
});
