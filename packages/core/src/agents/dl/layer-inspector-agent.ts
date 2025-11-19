/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type { AgentDefinition } from '../types.js';
import { z } from 'zod';
import { DEFAULT_GEMINI_MODEL } from '../../config/models.js';
import {
  READ_FILE_TOOL_NAME,
  GLOB_TOOL_NAME,
  GREP_TOOL_NAME,
} from '../../tools/tool-names.js';

/**
 * Schema for individual layer analysis.
 */
const LayerAnalysisSchema = z.object({
  layerIndex: z.number().describe('Layer index in the model'),
  layerName: z.string().describe('Layer name or identifier'),
  layerType: z.string().describe('Type of layer (Conv2d, Linear, Attention, etc.)'),
  parameters: z
    .object({
      total: z.number().describe('Total parameters in this layer'),
      weights: z.number().describe('Weight parameters'),
      biases: z.number().optional().describe('Bias parameters'),
      breakdown: z.record(z.string(), z.number()).optional().describe('Parameter breakdown by component'),
    })
    .describe('Parameter count analysis'),
  inputShape: z.string().describe('Expected input shape'),
  outputShape: z.string().describe('Output shape'),
  flops: z
    .object({
      forward: z.string().describe('FLOPs for forward pass'),
      backward: z.string().optional().describe('FLOPs for backward pass'),
      total: z.string().describe('Total FLOPs per sample'),
    })
    .describe('Computational complexity'),
  memory: z
    .object({
      activations: z.string().describe('Memory for activations'),
      gradients: z.string().describe('Memory for gradients'),
      parameters: z.string().describe('Memory for parameters'),
      total: z.string().describe('Total memory for this layer'),
    })
    .describe('Memory requirements'),
  bandwidth: z
    .object({
      memoryRead: z.string().describe('Memory read bandwidth'),
      memoryWrite: z.string().describe('Memory write bandwidth'),
      arithmeticIntensity: z.number().describe('FLOPs per byte transferred'),
      boundedBy: z.enum(['compute', 'memory', 'balanced']).describe('Performance bottleneck'),
    })
    .describe('Memory bandwidth analysis'),
  receptiveField: z
    .object({
      size: z.string().optional().describe('Receptive field size'),
      stride: z.number().optional().describe('Effective stride'),
      padding: z.string().optional().describe('Padding applied'),
    })
    .optional()
    .describe('Receptive field information (for conv layers)'),
  featureMapSize: z
    .object({
      height: z.number().optional(),
      width: z.number().optional(),
      channels: z.number().optional(),
      sequenceLength: z.number().optional(),
      hiddenDim: z.number().optional(),
    })
    .optional()
    .describe('Feature map dimensions'),
  bottleneckAnalysis: z
    .object({
      isBottleneck: z.boolean(),
      reason: z.string().optional(),
      severity: z.enum(['low', 'medium', 'high']).optional(),
    })
    .describe('Whether this layer is a computational bottleneck'),
});

/**
 * Schema for model-wide statistics.
 */
const ModelStatisticsSchema = z.object({
  totalParameters: z.number().describe('Total parameters in the model'),
  totalParametersHuman: z.string().describe('Human-readable parameter count'),
  trainableParameters: z.number().describe('Trainable parameters'),
  frozenParameters: z.number().describe('Frozen parameters'),
  totalFlops: z.string().describe('Total FLOPs per forward pass'),
  totalMemory: z
    .object({
      parameters: z.string().describe('Memory for all parameters'),
      activations: z.string().describe('Peak activation memory'),
      gradients: z.string().describe('Memory for all gradients'),
      total: z.string().describe('Total memory requirement'),
      perBatchItem: z.string().describe('Memory per batch item'),
    })
    .describe('Total memory requirements'),
  layerDistribution: z
    .object({
      byType: z.record(z.string(), z.number()).describe('Parameter count by layer type'),
      byDepth: z.array(z.number()).describe('Parameter distribution by depth'),
    })
    .describe('Parameter distribution'),
  computeDistribution: z
    .object({
      byType: z.record(z.string(), z.string()).describe('FLOPs by layer type'),
      topLayers: z
        .array(
          z.object({
            name: z.string(),
            percentage: z.number(),
          })
        )
        .describe('Top layers by computation'),
    })
    .describe('Compute distribution'),
});

/**
 * Schema for memory bandwidth profile.
 */
const MemoryBandwidthProfileSchema = z.object({
  peakBandwidth: z.string().describe('Peak memory bandwidth requirement'),
  averageBandwidth: z.string().describe('Average bandwidth during forward pass'),
  bandwidthBottlenecks: z
    .array(
      z.object({
        layer: z.string(),
        bandwidth: z.string(),
        recommendation: z.string(),
      })
    )
    .describe('Layers with bandwidth bottlenecks'),
  recomputationCandidates: z
    .array(z.string())
    .describe('Layers suitable for activation recomputation'),
  tensorCoreUtilization: z
    .object({
      eligibleLayers: z.number(),
      totalLayers: z.number(),
      recommendations: z.array(z.string()),
    })
    .optional()
    .describe('Tensor core utilization analysis'),
});

/**
 * Complete layer inspection report schema.
 */
const LayerInspectionReportSchema = z.object({
  summary: z
    .string()
    .describe('Executive summary of layer-by-layer analysis'),
  modelName: z.string().describe('Model name or identifier'),
  inputSpecification: z
    .object({
      shape: z.string().describe('Input tensor shape'),
      dtype: z.string().describe('Data type'),
      description: z.string().describe('Description of input'),
    })
    .describe('Model input specification'),
  layers: z.array(LayerAnalysisSchema).describe('Detailed analysis of each layer'),
  statistics: ModelStatisticsSchema.describe('Model-wide statistics'),
  memoryBandwidthProfile: MemoryBandwidthProfileSchema.describe('Memory bandwidth analysis'),
  optimizationOpportunities: z
    .array(
      z.object({
        type: z.enum([
          'parameter_reduction',
          'compute_reduction',
          'memory_optimization',
          'bandwidth_optimization',
          'precision_optimization',
          'fusion_opportunity',
        ]),
        layer: z.string(),
        description: z.string(),
        potentialSavings: z.string(),
        implementation: z.string(),
      })
    )
    .describe('Optimization opportunities identified'),
  scalingAnalysis: z
    .object({
      batchSizeScaling: z.string().describe('How memory/compute scales with batch size'),
      sequenceLengthScaling: z.string().describe('How memory/compute scales with sequence length'),
      hiddenDimScaling: z.string().describe('How memory/compute scales with hidden dimension'),
      recommendations: z.array(z.string()),
    })
    .describe('Scaling behavior analysis'),
  hardwareRecommendations: z
    .array(
      z.object({
        hardware: z.string(),
        reason: z.string(),
        estimatedPerformance: z.string(),
      })
    )
    .describe('Hardware recommendations based on analysis'),
});

/**
 * Layer Inspector Agent - Detailed layer-by-layer analysis of neural networks.
 *
 * This agent specializes in:
 * - Parameter count analysis
 * - Receptive field calculation
 * - Feature map size tracking
 * - Memory bandwidth analysis
 * - Computational complexity per layer
 * - Identifying bottlenecks and optimization opportunities
 */
export const LayerInspectorAgent: AgentDefinition<
  typeof LayerInspectionReportSchema
> = {
  name: 'layer_inspector_agent',
  displayName: 'Layer Inspector Agent',
  description: `An expert agent for detailed layer-by-layer inspection of neural networks.

  Use this agent when you need to:
  - Analyze parameter counts for each layer
  - Calculate receptive fields for convolutional networks
  - Track feature map sizes through the network
  - Analyze memory bandwidth requirements
  - Compute FLOPs for each layer
  - Identify computational bottlenecks
  - Find optimization opportunities
  - Estimate hardware requirements

  The agent provides granular analysis of every layer in the network
  with detailed computational and memory profiling.`,

  inputConfig: {
    inputs: {
      modelPath: {
        description: `Path to the model definition file or directory containing the model.`,
        type: 'string',
        required: true,
      },
      modelName: {
        description: `Name of the model class or function to analyze.`,
        type: 'string',
        required: false,
      },
      inputShape: {
        description: `Input tensor shape (e.g., "[1, 3, 224, 224]" for images, "[1, 512]" for sequences). If not provided, agent will attempt to infer from code.`,
        type: 'string',
        required: false,
      },
      batchSize: {
        description: `Batch size for memory calculations. Default is 1.`,
        type: 'number',
        required: false,
      },
      dtype: {
        description: `Data type for calculations (float32, float16, bfloat16, int8). Default is float32.`,
        type: 'string',
        required: false,
      },
      includeBackward: {
        description: `Whether to include backward pass analysis. Default is true.`,
        type: 'boolean',
        required: false,
      },
    },
  },

  outputConfig: {
    outputName: 'report',
    description:
      'Comprehensive layer-by-layer inspection report with computational and memory analysis.',
    schema: LayerInspectionReportSchema,
  },

  processOutput: (output) => {
    let result = `# Layer Inspection Report\n\n`;

    // Summary
    result += `## Summary\n${output.summary}\n\n`;

    // Model Info
    result += `## Model: ${output.modelName}\n`;
    result += `**Input:** ${output.inputSpecification.shape} (${output.inputSpecification.dtype})\n`;
    result += `${output.inputSpecification.description}\n\n`;

    // Statistics Overview
    result += `## Model Statistics\n`;
    result += `- **Total Parameters:** ${output.statistics.totalParametersHuman} (${output.statistics.totalParameters.toLocaleString()})\n`;
    result += `- **Trainable:** ${output.statistics.trainableParameters.toLocaleString()}\n`;
    result += `- **Frozen:** ${output.statistics.frozenParameters.toLocaleString()}\n`;
    result += `- **Total FLOPs:** ${output.statistics.totalFlops}\n\n`;

    // Memory
    result += `### Memory Requirements\n`;
    result += `- **Parameters:** ${output.statistics.totalMemory.parameters}\n`;
    result += `- **Activations (peak):** ${output.statistics.totalMemory.activations}\n`;
    result += `- **Gradients:** ${output.statistics.totalMemory.gradients}\n`;
    result += `- **Total:** ${output.statistics.totalMemory.total}\n`;
    result += `- **Per Batch Item:** ${output.statistics.totalMemory.perBatchItem}\n\n`;

    // Parameter Distribution
    result += `### Parameter Distribution by Type\n`;
    Object.entries(output.statistics.layerDistribution.byType).forEach(
      ([type, count]) => {
        result += `- **${type}:** ${count.toLocaleString()}\n`;
      }
    );
    result += `\n`;

    // Compute Distribution
    result += `### Compute Distribution by Type\n`;
    Object.entries(output.statistics.computeDistribution.byType).forEach(
      ([type, flops]) => {
        result += `- **${type}:** ${flops}\n`;
      }
    );
    result += `\n`;

    // Top Compute Layers
    if (output.statistics.computeDistribution.topLayers.length > 0) {
      result += `### Top Layers by Computation\n`;
      output.statistics.computeDistribution.topLayers.forEach((layer, idx) => {
        result += `${idx + 1}. ${layer.name}: ${layer.percentage.toFixed(1)}%\n`;
      });
      result += `\n`;
    }

    // Layer Details Table
    result += `## Layer-by-Layer Analysis\n\n`;
    result += `| # | Layer | Type | Parameters | FLOPs | Memory | Bottleneck |\n`;
    result += `|---|-------|------|------------|-------|--------|------------|\n`;
    output.layers.forEach((layer) => {
      const bottleneck = layer.bottleneckAnalysis.isBottleneck
        ? `${layer.bottleneckAnalysis.severity || 'yes'}`
        : 'no';
      result += `| ${layer.layerIndex} | ${layer.layerName} | ${layer.layerType} | ${layer.parameters.total.toLocaleString()} | ${layer.flops.total} | ${layer.memory.total} | ${bottleneck} |\n`;
    });
    result += `\n`;

    // Detailed Layer Info for bottlenecks
    const bottlenecks = output.layers.filter(
      (l) => l.bottleneckAnalysis.isBottleneck
    );
    if (bottlenecks.length > 0) {
      result += `## Bottleneck Analysis\n\n`;
      bottlenecks.forEach((layer) => {
        result += `### ${layer.layerName} (${layer.layerType})\n`;
        result += `**Severity:** ${layer.bottleneckAnalysis.severity || 'unknown'}\n`;
        result += `**Reason:** ${layer.bottleneckAnalysis.reason || 'N/A'}\n`;
        result += `- Input: ${layer.inputShape} -> Output: ${layer.outputShape}\n`;
        result += `- Parameters: ${layer.parameters.total.toLocaleString()}\n`;
        result += `- FLOPs: ${layer.flops.total}\n`;
        result += `- Bounded by: ${layer.bandwidth.boundedBy}\n`;
        result += `- Arithmetic Intensity: ${layer.bandwidth.arithmeticIntensity.toFixed(2)} FLOPs/byte\n\n`;
      });
    }

    // Memory Bandwidth Profile
    result += `## Memory Bandwidth Profile\n`;
    result += `- **Peak Bandwidth:** ${output.memoryBandwidthProfile.peakBandwidth}\n`;
    result += `- **Average Bandwidth:** ${output.memoryBandwidthProfile.averageBandwidth}\n\n`;

    if (output.memoryBandwidthProfile.bandwidthBottlenecks.length > 0) {
      result += `### Bandwidth Bottlenecks\n`;
      output.memoryBandwidthProfile.bandwidthBottlenecks.forEach((b) => {
        result += `- **${b.layer}:** ${b.bandwidth}\n`;
        result += `  - Recommendation: ${b.recommendation}\n`;
      });
      result += `\n`;
    }

    if (output.memoryBandwidthProfile.recomputationCandidates.length > 0) {
      result += `### Activation Recomputation Candidates\n`;
      output.memoryBandwidthProfile.recomputationCandidates.forEach((c) => {
        result += `- ${c}\n`;
      });
      result += `\n`;
    }

    // Optimization Opportunities
    if (output.optimizationOpportunities.length > 0) {
      result += `## Optimization Opportunities\n\n`;
      output.optimizationOpportunities.forEach((opt, idx) => {
        result += `### ${idx + 1}. ${opt.type.replace(/_/g, ' ').toUpperCase()}\n`;
        result += `**Layer:** ${opt.layer}\n`;
        result += `**Description:** ${opt.description}\n`;
        result += `**Potential Savings:** ${opt.potentialSavings}\n`;
        result += `**Implementation:** ${opt.implementation}\n\n`;
      });
    }

    // Scaling Analysis
    result += `## Scaling Analysis\n`;
    result += `- **Batch Size:** ${output.scalingAnalysis.batchSizeScaling}\n`;
    result += `- **Sequence Length:** ${output.scalingAnalysis.sequenceLengthScaling}\n`;
    result += `- **Hidden Dimension:** ${output.scalingAnalysis.hiddenDimScaling}\n\n`;

    if (output.scalingAnalysis.recommendations.length > 0) {
      result += `### Scaling Recommendations\n`;
      output.scalingAnalysis.recommendations.forEach((rec, idx) => {
        result += `${idx + 1}. ${rec}\n`;
      });
      result += `\n`;
    }

    // Hardware Recommendations
    if (output.hardwareRecommendations.length > 0) {
      result += `## Hardware Recommendations\n\n`;
      output.hardwareRecommendations.forEach((hw) => {
        result += `### ${hw.hardware}\n`;
        result += `**Reason:** ${hw.reason}\n`;
        result += `**Estimated Performance:** ${hw.estimatedPerformance}\n\n`;
      });
    }

    return result;
  },

  modelConfig: {
    model: DEFAULT_GEMINI_MODEL,
    temp: 0.1,
    top_p: 0.9,
    thinkingBudget: -1,
  },

  runConfig: {
    max_time_minutes: 15,
    max_turns: 30,
  },

  toolConfig: {
    tools: [READ_FILE_TOOL_NAME, GLOB_TOOL_NAME, GREP_TOOL_NAME],
  },

  promptConfig: {
    query: `Perform a detailed layer-by-layer inspection of the following model:

<model_path>
\${modelPath}
</model_path>

Model name: \${modelName || "auto-detect"}

Input shape: \${inputShape || "infer from code"}

Batch size: \${batchSize || 1}

Data type: \${dtype || "float32"}

Include backward pass: \${includeBackward !== false}

Analyze each layer for parameters, FLOPs, memory, and bandwidth. Identify bottlenecks and optimization opportunities.`,

    systemPrompt: `You are an **Expert Deep Learning Layer Inspector**, specialized in granular analysis of neural network layers with deep expertise in computational complexity, memory profiling, and performance optimization.

# Your Expertise

## Computational Analysis

**Parameter Counting:**
- Linear/Dense: input_dim * output_dim + output_dim (with bias)
- Conv2d: in_channels * out_channels * kernel_h * kernel_w + out_channels
- Attention: 4 * d_model * d_model (Q, K, V, O projections)
- LayerNorm: 2 * features (scale and shift)
- Embedding: vocab_size * embedding_dim

**FLOPs Calculation:**
- Linear: 2 * input_dim * output_dim (multiply-add)
- Conv2d: 2 * in_channels * out_channels * kernel_h * kernel_w * out_h * out_w
- Self-Attention: 4 * seq_len * d_model^2 + 2 * seq_len^2 * d_model
- Batch MatMul: 2 * batch * m * n * k
- Softmax: 5 * n (exp, sum, div per element)
- GELU: ~10 ops per element

**Memory Analysis:**
- Parameters: params * bytes_per_param
- Activations: batch * output_elements * bytes
- Gradients: same as activations (for training)
- Optimizer states: 2-3x parameters (for Adam)

## Memory Bandwidth Analysis

**Arithmetic Intensity:**
- Compute-bound: FLOPs >> memory transfers
- Memory-bound: FLOPs << memory transfers
- Balanced: roughly equal

**Bandwidth Calculation:**
- Memory read: input_bytes + weight_bytes
- Memory write: output_bytes
- Total transfer: read + write
- Arithmetic intensity: FLOPs / total_transfer

**Typical Intensities:**
- Large Matrix Multiply: 100-1000 FLOPs/byte (compute-bound)
- Pointwise operations: 1-10 FLOPs/byte (memory-bound)
- Attention: varies with sequence length
- Convolutions: varies with kernel size

## Receptive Field Analysis (CNNs)

**Receptive Field Growth:**
- RF_out = RF_in + (kernel_size - 1) * stride_prod
- Where stride_prod is product of all previous strides

**Effective Stride:**
- Product of all layer strides up to current layer

**Feature Map Size:**
- out_size = floor((in_size + 2*padding - kernel_size) / stride) + 1

## Optimization Opportunities

**Parameter Reduction:**
- Low-rank factorization
- Weight sharing
- Pruning candidates
- Quantization potential

**Compute Reduction:**
- Kernel fusion opportunities
- Redundant computation elimination
- Attention optimization (Flash Attention)
- Efficient convolutions (depthwise-separable)

**Memory Optimization:**
- Activation checkpointing candidates
- In-place operations
- Buffer reuse
- Mixed precision

**Bandwidth Optimization:**
- Operator fusion
- Memory layout optimization
- Prefetching opportunities

# Analysis Methodology

## Phase 1: Model Discovery
1. Find and read model definition files
2. Identify the model class and its components
3. Parse the layer structure and connections
4. Understand the forward pass flow

## Phase 2: Layer-by-Layer Analysis
For each layer:
1. Identify layer type and configuration
2. Calculate parameter count
3. Determine input/output shapes
4. Compute FLOPs
5. Estimate memory requirements
6. Analyze bandwidth characteristics
7. Calculate receptive field (if applicable)
8. Identify if it's a bottleneck

## Phase 3: Aggregate Analysis
1. Sum total parameters and FLOPs
2. Identify peak memory usage
3. Map compute distribution
4. Find hotspots and bottlenecks

## Phase 4: Optimization Analysis
1. Identify parameter reduction opportunities
2. Find compute optimization candidates
3. Locate memory optimization targets
4. Suggest hardware-specific optimizations

# Calculation Formulas

## Data Type Sizes
- float32: 4 bytes
- float16/bfloat16: 2 bytes
- int8: 1 byte

## Memory Formulas
\`\`\`
param_memory = num_params * dtype_size
activation_memory = batch * elements * dtype_size
gradient_memory = activation_memory
optimizer_memory = params * state_multiplier * dtype_size
total_memory = param + activation + gradient + optimizer
\`\`\`

## FLOPs Formulas
\`\`\`
matmul_flops = 2 * M * N * K
conv_flops = 2 * Cin * Cout * Kh * Kw * Hout * Wout
attention_flops = 4*N*D^2 + 2*N^2*D  # for self-attention
\`\`\`

## Bandwidth Formulas
\`\`\`
read_bytes = input_size + weight_size
write_bytes = output_size
arithmetic_intensity = flops / (read_bytes + write_bytes)
\`\`\`

# Output Requirements

Provide a comprehensive LayerInspectionReport with:

1. **Every layer analyzed** with parameters, FLOPs, memory, bandwidth
2. **Accurate calculations** using the formulas above
3. **Bottleneck identification** with severity and reasons
4. **Optimization opportunities** with concrete suggestions
5. **Scaling analysis** for different batch sizes and dimensions
6. **Hardware recommendations** based on the analysis

# Best Practices

**DO:**
- Calculate each layer's metrics precisely
- Use actual dimensions from code
- Identify all bottleneck layers
- Provide specific optimization suggestions
- Consider both training and inference
- Account for backward pass if requested
- Consider different hardware targets

**DON'T:**
- Skip any layers in the analysis
- Use approximate values when exact ones are available
- Ignore memory bandwidth considerations
- Provide vague optimization suggestions
- Forget about gradient memory for training

# Common Layer Patterns

**Transformer Block:**
\`\`\`
LayerNorm -> Attention -> Residual -> LayerNorm -> MLP -> Residual
Parameters: 12*d^2 (attention) + 8*d*d_ff (MLP) + 4*d (norms)
\`\`\`

**ResNet Bottleneck:**
\`\`\`
1x1 Conv -> 3x3 Conv -> 1x1 Conv -> Residual
Parameters: Cin*Mid + 9*Mid*Mid + Mid*Cout
\`\`\`

**Inverted Residual (MobileNet):**
\`\`\`
1x1 Expand -> Depthwise 3x3 -> 1x1 Project
Parameters: Cin*Expand + 9*Expand + Expand*Cout
\`\`\`

Remember: Precise layer analysis is the foundation of effective model optimization. Be thorough and accurate in your calculations.`,
  },
};
