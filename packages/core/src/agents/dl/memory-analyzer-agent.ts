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
  EDIT_TOOL_NAME,
} from '../../tools/tool-names.js';
import { DEFAULT_GEMINI_MODEL } from '../../config/models.js';
import { z } from 'zod';

/**
 * Schema for layer-level memory analysis.
 */
const LayerMemorySchema = z.object({
  layerName: z.string().describe('Name of the layer.'),
  layerType: z.string().describe('Type of layer (Linear, Conv2d, etc.).'),
  parameterMemoryMB: z.number().describe('Memory for parameters in MB.'),
  gradientMemoryMB: z.number().describe('Memory for gradients in MB.'),
  activationMemoryMB: z.number().describe('Memory for activations in MB.'),
  bufferMemoryMB: z.number().describe('Memory for buffers in MB.'),
  totalMemoryMB: z.number().describe('Total memory in MB.'),
  inputShape: z.string().describe('Input tensor shape.'),
  outputShape: z.string().describe('Output tensor shape.'),
  recommendations: z.array(z.string()).optional(),
});

/**
 * Schema for memory timeline event.
 */
const MemoryTimelineEventSchema = z.object({
  timestamp: z.number().describe('Timestamp in milliseconds.'),
  operation: z.string().describe('Operation being performed.'),
  allocatedMB: z.number().describe('Currently allocated memory.'),
  reservedMB: z.number().describe('Reserved memory.'),
  delta: z.number().describe('Change from previous event.'),
  isActivation: z.boolean().describe('Whether this is activation memory.'),
  tensorShape: z.string().optional().describe('Shape of tensor involved.'),
});

/**
 * Schema for activation checkpointing analysis.
 */
const CheckpointingAnalysisSchema = z.object({
  currentActivationMemoryMB: z.number(),
  estimatedSavingsPercent: z.number(),
  estimatedMemoryAfterMB: z.number(),
  computeOverheadPercent: z.number(),
  recommendedLayers: z.array(z.object({
    layerName: z.string(),
    memorySavingMB: z.number(),
    computeCost: z.string(),
    priority: z.number().min(1).max(10),
  })),
  implementationCode: z.string(),
});

/**
 * Schema for gradient accumulation analysis.
 */
const GradientAccumulationSchema = z.object({
  currentBatchSize: z.number(),
  currentMemoryMB: z.number(),
  recommendations: z.array(z.object({
    microBatchSize: z.number(),
    accumulationSteps: z.number(),
    effectiveBatchSize: z.number(),
    estimatedMemoryMB: z.number(),
    memorySavingPercent: z.number(),
  })),
  implementationCode: z.string(),
});

/**
 * Schema for memory fragmentation analysis.
 */
const FragmentationAnalysisSchema = z.object({
  fragmentationPercent: z.number().describe('Percentage of memory fragmented.'),
  largestFreeBlockMB: z.number(),
  totalFreeMemoryMB: z.number(),
  fragmentCount: z.number(),
  causes: z.array(z.string()),
  solutions: z.array(z.object({
    technique: z.string(),
    description: z.string(),
    expectedImprovement: z.string(),
    implementationCode: z.string().optional(),
  })),
});

/**
 * Schema for mixed precision analysis.
 */
const MixedPrecisionAnalysisSchema = z.object({
  currentPrecision: z.string(),
  currentMemoryMB: z.number(),
  fp16MemoryMB: z.number(),
  bf16MemoryMB: z.number(),
  memorySavingPercent: z.number(),
  compatibleLayers: z.array(z.string()),
  incompatibleLayers: z.array(z.object({
    layer: z.string(),
    reason: z.string(),
    workaround: z.string().optional(),
  })),
  implementationCode: z.string(),
  lossScalingRecommendation: z.string(),
});

/**
 * Schema for distributed memory analysis.
 */
const DistributedMemorySchema = z.object({
  strategy: z.enum(['DATA_PARALLEL', 'MODEL_PARALLEL', 'PIPELINE_PARALLEL', 'FSDP', 'DEEPSPEED']),
  perGPUMemoryMB: z.number(),
  totalMemoryMB: z.number(),
  communicationOverheadMB: z.number(),
  shardingRecommendations: z.array(z.object({
    technique: z.string(),
    description: z.string(),
    memorySaving: z.string(),
    throughputImpact: z.string(),
  })),
  optimalConfiguration: z.object({
    worldSize: z.number(),
    strategy: z.string(),
    shardingConfig: z.string(),
  }),
});

/**
 * Schema for memory optimization action.
 */
const MemoryOptimizationActionSchema = z.object({
  technique: z.string().describe('Optimization technique name.'),
  category: z.enum([
    'CHECKPOINTING',
    'MIXED_PRECISION',
    'GRADIENT_ACCUMULATION',
    'MODEL_PARALLELISM',
    'MEMORY_EFFICIENT_ATTENTION',
    'OFFLOADING',
    'QUANTIZATION',
  ]),
  currentMemoryMB: z.number(),
  estimatedMemoryMB: z.number(),
  savingPercent: z.number(),
  implementationEffort: z.enum(['LOW', 'MEDIUM', 'HIGH']),
  accuracyImpact: z.enum(['NONE', 'MINIMAL', 'MODERATE', 'SIGNIFICANT']),
  speedImpact: z.string(),
  priority: z.number().min(1).max(10),
  implementationCode: z.string(),
  prerequisites: z.array(z.string()).optional(),
});

/**
 * Complete memory analysis report schema.
 */
const MemoryAnalysisReportSchema = z.object({
  summary: z.string().describe('Executive summary of memory analysis.'),
  framework: z.enum(['PYTORCH', 'TENSORFLOW', 'JAX', 'OTHER']),
  hardware: z.object({
    gpuModel: z.string(),
    gpuMemoryGB: z.number(),
    gpuCount: z.number(),
  }),
  overallMemoryUsage: z.object({
    peakMemoryMB: z.number(),
    availableMemoryMB: z.number(),
    utilizationPercent: z.number(),
    headroomMB: z.number(),
    oomRisk: z.enum(['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']),
  }),
  memoryBreakdown: z.object({
    parametersMB: z.number(),
    gradientsMB: z.number(),
    optimizerStatesMB: z.number(),
    activationsMB: z.number(),
    buffersMB: z.number(),
    temporaryMB: z.number(),
    frameworkOverheadMB: z.number(),
  }),
  layerAnalysis: z.array(LayerMemorySchema),
  memoryTimeline: z.array(MemoryTimelineEventSchema).optional(),
  peakAnalysis: z.object({
    peakTimestamp: z.number(),
    peakOperation: z.string(),
    contributingFactors: z.array(z.string()),
    reductionStrategies: z.array(z.string()),
  }),
  checkpointingAnalysis: CheckpointingAnalysisSchema.optional(),
  gradientAccumulation: GradientAccumulationSchema.optional(),
  fragmentationAnalysis: FragmentationAnalysisSchema.optional(),
  mixedPrecisionAnalysis: MixedPrecisionAnalysisSchema.optional(),
  distributedAnalysis: DistributedMemorySchema.optional(),
  optimizationActions: z.array(MemoryOptimizationActionSchema),
  diagnosticScripts: z.array(z.object({
    name: z.string(),
    purpose: z.string(),
    code: z.string(),
  })).optional(),
  implementationPlan: z.object({
    phase1: z.string().describe('Quick wins with minimal code changes.'),
    phase2: z.string().describe('Medium effort optimizations.'),
    phase3: z.string().describe('Major architecture changes if needed.'),
    estimatedTotalSaving: z.string(),
  }),
});

/**
 * DL Memory Analyzer Agent - Specialized in GPU memory analysis and optimization.
 *
 * This agent provides expert-level memory analysis for:
 * - Activation memory tracking
 * - Gradient memory analysis
 * - Buffer/parameter memory
 * - Peak memory detection
 * - Memory fragmentation
 * - Checkpointing suggestions
 */
export const DLMemoryAnalyzerAgent: AgentDefinition<typeof MemoryAnalysisReportSchema> = {
  name: 'dl_memory_analyzer_agent',
  displayName: 'DL Memory Analyzer Agent',
  description: `A specialized agent for deep learning GPU memory analysis and optimization.
    Use this agent when you need to:
    - Analyze memory usage breakdown by component
    - Identify memory-intensive layers and operations
    - Get checkpointing and memory optimization recommendations
    - Debug out-of-memory errors
    - Plan memory-efficient training strategies

    Provides detailed memory breakdown and actionable optimization strategies.`,

  inputConfig: {
    inputs: {
      memoryIssue: {
        description: `Description of the memory issue or optimization goal:
          - OOM error message and context
          - Current memory usage if known
          - Target batch size or model size
          - Memory reduction target`,
        type: 'string',
        required: true,
      },
      codebase: {
        description: `Path to the model and training code`,
        type: 'string',
        required: false,
      },
      modelConfig: {
        description: `Model configuration:
          - Model architecture (e.g., transformer, CNN)
          - Number of parameters
          - Sequence length / image size
          - Batch size
          - Optimizer type`,
        type: 'string',
        required: false,
      },
      hardwareSpec: {
        description: `Hardware specifications:
          - GPU model (e.g., A100-80GB)
          - Number of GPUs
          - Available memory per GPU`,
        type: 'string',
        required: false,
      },
      memoryProfile: {
        description: `Path to existing memory profile data:
          - PyTorch memory snapshot
          - Memory timeline logs
          - nvidia-smi outputs`,
        type: 'string',
        required: false,
      },
    },
  },

  outputConfig: {
    outputName: 'report',
    description: 'Comprehensive memory analysis with optimization strategies.',
    schema: MemoryAnalysisReportSchema,
  },

  processOutput: (output) => {
    let result = `# Deep Learning Memory Analysis Report\n\n`;
    result += `## Summary\n${output.summary}\n\n`;
    result += `**Framework:** ${output.framework}\n`;
    result += `**Hardware:** ${output.hardware.gpuCount}x ${output.hardware.gpuModel} (${output.hardware.gpuMemoryGB}GB)\n\n`;

    result += `## Overall Memory Usage\n`;
    result += `| Metric | Value |\n|--------|-------|\n`;
    result += `| Peak Memory | ${output.overallMemoryUsage.peakMemoryMB.toFixed(2)} MB |\n`;
    result += `| Available Memory | ${output.overallMemoryUsage.availableMemoryMB.toFixed(2)} MB |\n`;
    result += `| Utilization | ${output.overallMemoryUsage.utilizationPercent.toFixed(1)}% |\n`;
    result += `| Headroom | ${output.overallMemoryUsage.headroomMB.toFixed(2)} MB |\n`;
    result += `| OOM Risk | **${output.overallMemoryUsage.oomRisk}** |\n\n`;

    result += `## Memory Breakdown\n`;
    const breakdown = output.memoryBreakdown;
    const total = breakdown.parametersMB + breakdown.gradientsMB + breakdown.optimizerStatesMB +
                  breakdown.activationsMB + breakdown.buffersMB + breakdown.temporaryMB + breakdown.frameworkOverheadMB;

    result += `| Component | Memory (MB) | Percentage |\n`;
    result += `|-----------|-------------|------------|\n`;
    result += `| Parameters | ${breakdown.parametersMB.toFixed(2)} | ${(breakdown.parametersMB/total*100).toFixed(1)}% |\n`;
    result += `| Gradients | ${breakdown.gradientsMB.toFixed(2)} | ${(breakdown.gradientsMB/total*100).toFixed(1)}% |\n`;
    result += `| Optimizer States | ${breakdown.optimizerStatesMB.toFixed(2)} | ${(breakdown.optimizerStatesMB/total*100).toFixed(1)}% |\n`;
    result += `| Activations | ${breakdown.activationsMB.toFixed(2)} | ${(breakdown.activationsMB/total*100).toFixed(1)}% |\n`;
    result += `| Buffers | ${breakdown.buffersMB.toFixed(2)} | ${(breakdown.buffersMB/total*100).toFixed(1)}% |\n`;
    result += `| Temporary | ${breakdown.temporaryMB.toFixed(2)} | ${(breakdown.temporaryMB/total*100).toFixed(1)}% |\n`;
    result += `| Framework Overhead | ${breakdown.frameworkOverheadMB.toFixed(2)} | ${(breakdown.frameworkOverheadMB/total*100).toFixed(1)}% |\n`;
    result += `| **Total** | **${total.toFixed(2)}** | **100%** |\n\n`;

    if (output.layerAnalysis && output.layerAnalysis.length > 0) {
      result += `## Top Memory-Consuming Layers\n`;
      const sortedLayers = [...output.layerAnalysis].sort((a, b) => b.totalMemoryMB - a.totalMemoryMB).slice(0, 10);
      result += `| Layer | Type | Params (MB) | Activations (MB) | Total (MB) |\n`;
      result += `|-------|------|-------------|------------------|------------|\n`;
      sortedLayers.forEach((layer) => {
        result += `| ${layer.layerName} | ${layer.layerType} | ${layer.parameterMemoryMB.toFixed(2)} | ${layer.activationMemoryMB.toFixed(2)} | ${layer.totalMemoryMB.toFixed(2)} |\n`;
      });
      result += `\n`;
    }

    result += `## Peak Memory Analysis\n`;
    result += `**Peak Operation:** ${output.peakAnalysis.peakOperation}\n`;
    result += `**Contributing Factors:**\n`;
    output.peakAnalysis.contributingFactors.forEach((factor) => {
      result += `- ${factor}\n`;
    });
    result += `**Reduction Strategies:**\n`;
    output.peakAnalysis.reductionStrategies.forEach((strategy) => {
      result += `- ${strategy}\n`;
    });
    result += `\n`;

    if (output.checkpointingAnalysis) {
      result += `## Activation Checkpointing Analysis\n`;
      result += `**Current Activation Memory:** ${output.checkpointingAnalysis.currentActivationMemoryMB.toFixed(2)} MB\n`;
      result += `**Estimated Savings:** ${output.checkpointingAnalysis.estimatedSavingsPercent.toFixed(1)}%\n`;
      result += `**Memory After Checkpointing:** ${output.checkpointingAnalysis.estimatedMemoryAfterMB.toFixed(2)} MB\n`;
      result += `**Compute Overhead:** ${output.checkpointingAnalysis.computeOverheadPercent.toFixed(1)}%\n\n`;
      result += `**Recommended Layers to Checkpoint:**\n`;
      output.checkpointingAnalysis.recommendedLayers.forEach((layer) => {
        result += `- ${layer.layerName}: Save ${layer.memorySavingMB.toFixed(2)} MB (Priority: ${layer.priority}/10)\n`;
      });
      result += `\n**Implementation:**\n\`\`\`python\n${output.checkpointingAnalysis.implementationCode}\n\`\`\`\n\n`;
    }

    if (output.mixedPrecisionAnalysis) {
      result += `## Mixed Precision Analysis\n`;
      result += `**Current Precision:** ${output.mixedPrecisionAnalysis.currentPrecision}\n`;
      result += `**Current Memory:** ${output.mixedPrecisionAnalysis.currentMemoryMB.toFixed(2)} MB\n`;
      result += `**FP16 Memory:** ${output.mixedPrecisionAnalysis.fp16MemoryMB.toFixed(2)} MB\n`;
      result += `**BF16 Memory:** ${output.mixedPrecisionAnalysis.bf16MemoryMB.toFixed(2)} MB\n`;
      result += `**Potential Savings:** ${output.mixedPrecisionAnalysis.memorySavingPercent.toFixed(1)}%\n\n`;
      if (output.mixedPrecisionAnalysis.incompatibleLayers.length > 0) {
        result += `**Incompatible Layers:**\n`;
        output.mixedPrecisionAnalysis.incompatibleLayers.forEach((layer) => {
          result += `- ${layer.layer}: ${layer.reason}`;
          if (layer.workaround) {
            result += ` (Workaround: ${layer.workaround})`;
          }
          result += `\n`;
        });
      }
      result += `\n**Implementation:**\n\`\`\`python\n${output.mixedPrecisionAnalysis.implementationCode}\n\`\`\`\n\n`;
    }

    if (output.gradientAccumulation) {
      result += `## Gradient Accumulation Analysis\n`;
      result += `**Current Batch Size:** ${output.gradientAccumulation.currentBatchSize}\n`;
      result += `**Current Memory:** ${output.gradientAccumulation.currentMemoryMB.toFixed(2)} MB\n\n`;
      result += `**Recommendations:**\n`;
      result += `| Micro Batch | Accum Steps | Effective Batch | Memory (MB) | Saving |\n`;
      result += `|-------------|-------------|-----------------|-------------|--------|\n`;
      output.gradientAccumulation.recommendations.forEach((rec) => {
        result += `| ${rec.microBatchSize} | ${rec.accumulationSteps} | ${rec.effectiveBatchSize} | ${rec.estimatedMemoryMB.toFixed(2)} | ${rec.memorySavingPercent.toFixed(1)}% |\n`;
      });
      result += `\n**Implementation:**\n\`\`\`python\n${output.gradientAccumulation.implementationCode}\n\`\`\`\n\n`;
    }

    if (output.fragmentationAnalysis) {
      result += `## Memory Fragmentation Analysis\n`;
      result += `**Fragmentation:** ${output.fragmentationAnalysis.fragmentationPercent.toFixed(1)}%\n`;
      result += `**Largest Free Block:** ${output.fragmentationAnalysis.largestFreeBlockMB.toFixed(2)} MB\n`;
      result += `**Total Free Memory:** ${output.fragmentationAnalysis.totalFreeMemoryMB.toFixed(2)} MB\n`;
      result += `**Causes:**\n`;
      output.fragmentationAnalysis.causes.forEach((cause) => {
        result += `- ${cause}\n`;
      });
      result += `**Solutions:**\n`;
      output.fragmentationAnalysis.solutions.forEach((sol) => {
        result += `- **${sol.technique}:** ${sol.description} (${sol.expectedImprovement})\n`;
      });
      result += `\n`;
    }

    if (output.distributedAnalysis) {
      result += `## Distributed Memory Analysis\n`;
      result += `**Strategy:** ${output.distributedAnalysis.strategy}\n`;
      result += `**Per-GPU Memory:** ${output.distributedAnalysis.perGPUMemoryMB.toFixed(2)} MB\n`;
      result += `**Total Memory:** ${output.distributedAnalysis.totalMemoryMB.toFixed(2)} MB\n`;
      result += `**Communication Overhead:** ${output.distributedAnalysis.communicationOverheadMB.toFixed(2)} MB\n\n`;
      result += `**Optimal Configuration:**\n`;
      result += `- World Size: ${output.distributedAnalysis.optimalConfiguration.worldSize}\n`;
      result += `- Strategy: ${output.distributedAnalysis.optimalConfiguration.strategy}\n`;
      result += `- Sharding: ${output.distributedAnalysis.optimalConfiguration.shardingConfig}\n\n`;
    }

    result += `## Optimization Actions\n\n`;
    const sortedActions = [...output.optimizationActions].sort((a, b) => b.priority - a.priority);
    sortedActions.forEach((action, i) => {
      result += `### ${i + 1}. ${action.technique} (Priority: ${action.priority}/10)\n`;
      result += `**Category:** ${action.category}\n`;
      result += `**Memory:** ${action.currentMemoryMB.toFixed(2)} MB -> ${action.estimatedMemoryMB.toFixed(2)} MB (${action.savingPercent.toFixed(1)}% saving)\n`;
      result += `**Effort:** ${action.implementationEffort} | **Accuracy Impact:** ${action.accuracyImpact} | **Speed Impact:** ${action.speedImpact}\n`;
      if (action.prerequisites && action.prerequisites.length > 0) {
        result += `**Prerequisites:** ${action.prerequisites.join(', ')}\n`;
      }
      result += `\n\`\`\`python\n${action.implementationCode}\n\`\`\`\n\n`;
    });

    if (output.diagnosticScripts && output.diagnosticScripts.length > 0) {
      result += `## Diagnostic Scripts\n`;
      output.diagnosticScripts.forEach((script) => {
        result += `### ${script.name}\n`;
        result += `**Purpose:** ${script.purpose}\n`;
        result += `\`\`\`python\n${script.code}\n\`\`\`\n\n`;
      });
    }

    result += `## Implementation Plan\n`;
    result += `**Phase 1 (Quick Wins):** ${output.implementationPlan.phase1}\n\n`;
    result += `**Phase 2 (Medium Effort):** ${output.implementationPlan.phase2}\n\n`;
    result += `**Phase 3 (Major Changes):** ${output.implementationPlan.phase3}\n\n`;
    result += `**Estimated Total Saving:** ${output.implementationPlan.estimatedTotalSaving}\n`;

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
      EDIT_TOOL_NAME,
    ],
  },

  promptConfig: {
    query: `Analyze GPU memory usage for the following deep learning workload:

Memory Issue/Goal:
<issue>
\${memoryIssue}
</issue>

Codebase Location:
<codebase>
\${codebase}
</codebase>

Model Configuration:
<config>
\${modelConfig}
</config>

Hardware Specification:
<hardware>
\${hardwareSpec}
</hardware>

Memory Profile Data:
<profile>
\${memoryProfile}
</profile>

Perform comprehensive memory analysis and provide optimization recommendations.`,

    systemPrompt: `You are the **DL Memory Analyzer Agent**, an expert AI for GPU memory analysis and optimization in deep learning.

Your **CORE MISSION** is to provide detailed memory breakdown analysis, identify memory-intensive components, and deliver actionable optimization strategies.

## Expert Knowledge Base

### Memory Breakdown Components

**1. Parameter Memory**
\`\`\`python
def calculate_parameter_memory(model):
    """Calculate memory for model parameters."""
    total_params = 0
    param_memory = 0

    for name, param in model.named_parameters():
        total_params += param.numel()
        param_memory += param.numel() * param.element_size()

    return {
        'total_params': total_params,
        'memory_bytes': param_memory,
        'memory_MB': param_memory / 1e6,
        # FP32: 4 bytes, FP16: 2 bytes, BF16: 2 bytes
    }
\`\`\`

**2. Gradient Memory**
- Equal to parameter memory during training
- Can be reduced with:
  - Gradient checkpointing
  - Mixed precision (FP16/BF16 gradients)
  - Gradient compression for distributed training

**3. Optimizer State Memory**
\`\`\`python
def optimizer_memory(num_params, optimizer_type):
    """Calculate optimizer state memory."""
    bytes_per_param = 4  # FP32

    if optimizer_type == 'SGD':
        # Momentum: 1x params
        return num_params * bytes_per_param * 1
    elif optimizer_type == 'SGD_momentum':
        # Momentum buffer
        return num_params * bytes_per_param * 1
    elif optimizer_type in ['Adam', 'AdamW']:
        # First moment (m) + Second moment (v): 2x params
        return num_params * bytes_per_param * 2
    elif optimizer_type == 'Adafactor':
        # Factorized second moment: ~1.5x params
        return num_params * bytes_per_param * 1.5
    elif optimizer_type == '8bit_Adam':
        # 8-bit states: 0.5x params
        return num_params * 1  # 1 byte per param
\`\`\`

**4. Activation Memory**
\`\`\`python
def estimate_activation_memory(model, input_shape, batch_size):
    """Estimate activation memory during forward pass."""
    # This is the most variable component
    # Depends on:
    # - Batch size (linear scaling)
    # - Sequence length (quadratic for attention)
    # - Hidden dimensions
    # - Number of layers

    # For transformers:
    # activation_per_layer = batch * seq * hidden * 4 bytes * factor
    # factor ~= 10-34 depending on architecture

    # Use hooks to measure actual activation memory
    activation_sizes = []

    def hook_fn(module, input, output):
        if isinstance(output, torch.Tensor):
            activation_sizes.append(output.numel() * output.element_size())

    hooks = []
    for module in model.modules():
        hooks.append(module.register_forward_hook(hook_fn))

    # Forward pass
    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(batch_size, *input_shape)
        model(dummy_input)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return sum(activation_sizes)
\`\`\`

### Memory Estimation Formulas

**Transformer Memory:**
\`\`\`
Parameters = 12 * L * H^2 (approximate)
where L = layers, H = hidden dim

Activation Memory (per layer) ≈ B * S * H * (34 + 5*S/H)
where B = batch, S = sequence, H = hidden

For GPT-style:
Total Memory ≈ Params * 16 + B * S * H * L * 34 * 4
(16 = params + grads + optimizer for Adam in FP32)
\`\`\`

**CNN Memory:**
\`\`\`
Activation per layer = B * C * H * W * 4 bytes

Conv layer activation = input_size + output_size + intermediate
                      ≈ 2 * B * C_out * H_out * W_out * 4
\`\`\`

### Activation Checkpointing

**How It Works:**
- Don't store all activations during forward pass
- Recompute them during backward pass
- Trade-off: ~30% more compute for ~60-70% less activation memory

**Implementation:**
\`\`\`python
import torch
from torch.utils.checkpoint import checkpoint, checkpoint_sequential

# Method 1: Checkpoint specific layers
class TransformerBlock(nn.Module):
    def forward(self, x):
        # Checkpoint the attention and FFN
        x = x + checkpoint(self.attention, x, use_reentrant=False)
        x = x + checkpoint(self.ffn, x, use_reentrant=False)
        return x

# Method 2: Checkpoint sequential layers
class Model(nn.Module):
    def __init__(self):
        self.layers = nn.ModuleList([...])

    def forward(self, x):
        # Checkpoint every N layers
        return checkpoint_sequential(
            self.layers,
            segments=4,  # Checkpoint every 4 layers
            input=x,
            use_reentrant=False
        )

# Method 3: Selective checkpointing
def selective_checkpoint(model):
    """Checkpoint only the largest activations."""
    for name, module in model.named_modules():
        if isinstance(module, (nn.MultiheadAttention, nn.TransformerEncoderLayer)):
            module.forward = checkpoint(
                module.forward,
                use_reentrant=False
            )
\`\`\`

### Mixed Precision Training

**Memory Savings:**
- FP16/BF16: 2 bytes vs FP32: 4 bytes = 50% reduction for activations
- Parameters and optimizer states can remain in FP32

**Implementation:**
\`\`\`python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for data, target in dataloader:
    optimizer.zero_grad()

    # Automatic mixed precision
    with autocast(dtype=torch.float16):  # or torch.bfloat16
        output = model(data)
        loss = criterion(output, target)

    # Scaled backward pass
    scaler.scale(loss).backward()

    # Unscale and clip gradients
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # Optimizer step with scaler
    scaler.step(optimizer)
    scaler.update()
\`\`\`

**BF16 vs FP16:**
- BF16: Same range as FP32, less precision
- FP16: More precision, smaller range (needs loss scaling)
- Recommendation: Use BF16 on Ampere+ GPUs (A100, H100)

### Gradient Accumulation

**Purpose:** Simulate larger batches without memory increase

**Implementation:**
\`\`\`python
accumulation_steps = 4
effective_batch_size = micro_batch_size * accumulation_steps

for i, (data, target) in enumerate(dataloader):
    # Forward pass
    output = model(data)
    loss = criterion(output, target) / accumulation_steps

    # Backward pass (accumulate gradients)
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        # Update weights
        optimizer.step()
        optimizer.zero_grad()

# Handle remaining batches
if (i + 1) % accumulation_steps != 0:
    optimizer.step()
    optimizer.zero_grad()
\`\`\`

### Memory-Efficient Attention

**Flash Attention:**
\`\`\`python
# PyTorch 2.0+ automatically uses Flash Attention
from torch.nn.functional import scaled_dot_product_attention

# Memory: O(N) instead of O(N^2) for sequence length N
output = scaled_dot_product_attention(
    query, key, value,
    attn_mask=mask,
    dropout_p=0.0 if not training else dropout,
    is_causal=True,  # For autoregressive models
)

# For custom models, use xformers
from xformers.ops import memory_efficient_attention

output = memory_efficient_attention(query, key, value)
\`\`\`

### Memory Fragmentation

**Causes:**
1. Variable-size allocations
2. Frequent allocation/deallocation
3. Long-lived tensors interspersed with short-lived

**Detection:**
\`\`\`python
import torch

def check_fragmentation():
    stats = torch.cuda.memory_stats()
    allocated = stats['allocated_bytes.all.current']
    reserved = stats['reserved_bytes.all.current']

    fragmentation = (reserved - allocated) / reserved * 100
    print(f"Fragmentation: {fragmentation:.2f}%")

    # Check largest free block
    free_blocks = stats.get('inactive_split.all.current', 0)
    print(f"Inactive split blocks: {free_blocks}")
\`\`\`

**Solutions:**
\`\`\`python
# 1. Clear cache periodically
torch.cuda.empty_cache()

# 2. Use memory pools
torch.cuda.set_per_process_memory_fraction(0.8)  # Reserve 20% headroom

# 3. Pre-allocate tensors
# Instead of creating new tensors, reuse buffers
buffer = torch.empty(max_size, device='cuda')
output = buffer[:actual_size]

# 4. Use expandable memory segments (PyTorch 2.0+)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
\`\`\`

### Distributed Memory Optimization

**Fully Sharded Data Parallel (FSDP):**
\`\`\`python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision

# Configure sharding
sharding_strategy = ShardingStrategy.FULL_SHARD  # Maximum memory saving
# or ShardingStrategy.SHARD_GRAD_OP  # Better throughput

# Configure mixed precision
mixed_precision = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,
    buffer_dtype=torch.bfloat16,
)

# Wrap model
model = FSDP(
    model,
    sharding_strategy=sharding_strategy,
    mixed_precision=mixed_precision,
    auto_wrap_policy=transformer_auto_wrap_policy,
    device_id=torch.cuda.current_device(),
)
\`\`\`

**DeepSpeed ZeRO:**
\`\`\`python
# ZeRO Stage 1: Optimizer state sharding
# ZeRO Stage 2: + Gradient sharding
# ZeRO Stage 3: + Parameter sharding (most memory efficient)

# deepspeed_config.json
{
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu"  # Offload optimizer to CPU
        },
        "offload_param": {
            "device": "cpu"  # Offload parameters to CPU
        }
    }
}
\`\`\`

### Memory Profiling Tools

**PyTorch Memory Snapshot:**
\`\`\`python
# Start recording
torch.cuda.memory._record_memory_history(
    max_entries=100000
)

# ... your training code ...

# Save snapshot
torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")

# Analyze with:
# python -m torch.cuda.memory._viz memory_snapshot.pickle
\`\`\`

**Custom Memory Tracker:**
\`\`\`python
class MemoryTracker:
    def __init__(self):
        self.peak_memory = 0
        self.timeline = []

    def track(self, label=""):
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()

        self.peak_memory = max(self.peak_memory, allocated)
        self.timeline.append({
            'label': label,
            'allocated_MB': allocated / 1e6,
            'reserved_MB': reserved / 1e6,
        })

        return allocated / 1e6

    def report(self):
        print(f"Peak memory: {self.peak_memory / 1e6:.2f} MB")
        for entry in self.timeline:
            print(f"{entry['label']}: {entry['allocated_MB']:.2f} MB")

# Usage
tracker = MemoryTracker()
tracker.track("Initial")
output = model(input)
tracker.track("After forward")
loss.backward()
tracker.track("After backward")
tracker.report()
\`\`\`

### Advanced Optimization Techniques

**1. Offloading to CPU:**
\`\`\`python
# Activation offloading
from deepspeed.runtime.activation_checkpointing import checkpointing

# CPU offloading with FSDP
model = FSDP(
    model,
    cpu_offload=CPUOffload(offload_params=True)
)
\`\`\`

**2. Model Parallelism:**
\`\`\`python
# Pipeline parallelism
from torch.distributed.pipeline.sync import Pipe

model = nn.Sequential(layer1, layer2, layer3, layer4)
model = Pipe(model, chunks=8)

# Tensor parallelism (requires specialized implementation)
# Split large layers across GPUs
\`\`\`

**3. Selective Precision:**
\`\`\`python
# Keep critical layers in FP32
model.classifier = model.classifier.float()

# Or use autocast exclusions
with autocast():
    # Most operations in FP16
    output = model.encoder(input)

    with autocast(enabled=False):
        # Critical operations in FP32
        loss = F.cross_entropy(output.float(), target)
\`\`\`

**4. Dynamic Batching:**
\`\`\`python
def dynamic_batch(data, max_memory_mb):
    """Dynamically adjust batch size based on available memory."""
    torch.cuda.reset_peak_memory_stats()

    # Start with small batch
    batch_size = 1
    while True:
        try:
            batch = data[:batch_size]
            output = model(batch)
            loss = criterion(output)
            loss.backward()

            current_memory = torch.cuda.max_memory_allocated() / 1e6
            if current_memory > max_memory_mb * 0.8:
                break

            batch_size *= 2
            torch.cuda.empty_cache()

        except RuntimeError:  # OOM
            batch_size //= 2
            break

    return batch_size
\`\`\`

## Systematic Memory Analysis Process

### Phase 1: Measure Current Usage
1. Profile peak memory during training step
2. Break down by component type
3. Identify memory timeline

### Phase 2: Identify Bottlenecks
1. Find largest memory consumers
2. Analyze layer-by-layer usage
3. Detect fragmentation issues

### Phase 3: Analyze Optimization Potential
1. Calculate checkpointing savings
2. Estimate mixed precision benefits
3. Evaluate distributed strategies

### Phase 4: Implement Optimizations
1. Apply quick wins first
2. Progress to more complex changes
3. Verify correctness at each step

### Phase 5: Validate and Iterate
1. Re-measure memory usage
2. Check for regressions
3. Fine-tune configurations

## Best Practices

**DO:**
- Profile before optimizing
- Start with highest-impact, lowest-effort optimizations
- Verify numerical correctness after changes
- Monitor memory throughout training, not just peak
- Use appropriate precision for each component

**DON'T:**
- Apply all optimizations at once
- Ignore fragmentation issues
- Forget about optimizer state memory
- Skip validation of mixed precision training
- Overlook communication overhead in distributed training

Remember: Memory optimization often involves trade-offs with speed or accuracy. Make informed decisions based on your constraints.`,
  },
};
