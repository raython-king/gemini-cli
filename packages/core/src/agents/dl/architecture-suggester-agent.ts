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
 * Schema for bottleneck identification.
 */
const BottleneckSchema = z.object({
  location: z.string().describe('Location of the bottleneck (layer/component name)'),
  type: z
    .enum([
      'compute',
      'memory',
      'bandwidth',
      'parameters',
      'latency',
      'throughput',
      'communication',
    ])
    .describe('Type of bottleneck'),
  severity: z.enum(['low', 'medium', 'high', 'critical']).describe('Severity level'),
  description: z.string().describe('Detailed description of the bottleneck'),
  impact: z.string().describe('Impact on model performance'),
  metrics: z
    .object({
      current: z.string().describe('Current metric value'),
      target: z.string().optional().describe('Target metric value'),
      improvement: z.string().optional().describe('Potential improvement'),
    })
    .describe('Quantitative metrics'),
});

/**
 * Schema for efficiency improvement suggestion.
 */
const EfficiencyImprovementSchema = z.object({
  name: z.string().describe('Name of the improvement'),
  category: z
    .enum([
      'attention',
      'normalization',
      'activation',
      'convolution',
      'quantization',
      'pruning',
      'distillation',
      'architecture',
      'training',
      'inference',
    ])
    .describe('Category of improvement'),
  targetComponent: z.string().describe('Component this applies to'),
  description: z.string().describe('What the improvement does'),
  implementation: z.string().describe('How to implement this improvement'),
  expectedBenefit: z
    .object({
      speedup: z.string().optional().describe('Expected speedup'),
      memoryReduction: z.string().optional().describe('Memory reduction'),
      parameterReduction: z.string().optional().describe('Parameter reduction'),
      accuracyImpact: z.string().optional().describe('Impact on accuracy'),
    })
    .describe('Expected benefits'),
  complexity: z.enum(['low', 'medium', 'high']).describe('Implementation complexity'),
  codeExample: z.string().optional().describe('Example code snippet'),
  references: z.array(z.string()).optional().describe('Paper or documentation references'),
});

/**
 * Schema for modern alternative suggestions.
 */
const ModernAlternativeSchema = z.object({
  currentComponent: z.string().describe('Current component to replace'),
  alternativeName: z.string().describe('Name of the modern alternative'),
  description: z.string().describe('What makes this alternative better'),
  benefits: z.array(z.string()).describe('List of benefits'),
  tradeoffs: z.array(z.string()).describe('Potential tradeoffs'),
  adoption: z
    .enum(['experimental', 'emerging', 'established', 'industry_standard'])
    .describe('Adoption level in the field'),
  implementation: z
    .object({
      effort: z.enum(['trivial', 'easy', 'moderate', 'significant', 'major']),
      breakingChanges: z.boolean(),
      codeSnippet: z.string().optional(),
    })
    .describe('Implementation details'),
  examples: z.array(z.string()).describe('Models that use this alternative'),
});

/**
 * Schema for scaling strategy suggestion.
 */
const ScalingStrategySchema = z.object({
  dimension: z
    .enum(['depth', 'width', 'resolution', 'compound', 'mixture'])
    .describe('Scaling dimension'),
  currentState: z.string().describe('Current scaling state'),
  recommendation: z.string().describe('Scaling recommendation'),
  rationale: z.string().describe('Why this scaling strategy'),
  scalingLaw: z.string().optional().describe('Applicable scaling law'),
  optimalRatio: z.string().optional().describe('Optimal scaling ratio'),
  implementation: z.string().describe('How to implement this scaling'),
  resourceImplications: z
    .object({
      compute: z.string(),
      memory: z.string(),
      data: z.string(),
    })
    .describe('Resource implications'),
});

/**
 * Schema for hardware-specific optimization.
 */
const HardwareOptimizationSchema = z.object({
  targetHardware: z
    .enum([
      'nvidia_gpu',
      'amd_gpu',
      'tpu',
      'cpu',
      'apple_silicon',
      'mobile',
      'edge',
      'fpga',
    ])
    .describe('Target hardware'),
  optimizations: z
    .array(
      z.object({
        name: z.string(),
        description: z.string(),
        benefit: z.string(),
        implementation: z.string(),
      })
    )
    .describe('Hardware-specific optimizations'),
  tensorCoreOptimizations: z
    .array(z.string())
    .optional()
    .describe('Tensor core specific optimizations'),
  memoryOptimizations: z.array(z.string()).describe('Memory layout optimizations'),
  quantizationRecommendations: z
    .object({
      scheme: z.string(),
      precision: z.string(),
      expectedSpeedup: z.string(),
      accuracyImpact: z.string(),
    })
    .optional()
    .describe('Quantization recommendations'),
});

/**
 * Complete architecture suggestion report schema.
 */
const ArchitectureSuggestionReportSchema = z.object({
  summary: z
    .string()
    .describe('Executive summary of architecture improvement suggestions'),
  modelName: z.string().describe('Model being analyzed'),
  overallAssessment: z
    .object({
      efficiency: z.enum(['poor', 'fair', 'good', 'excellent']),
      modernity: z.enum(['outdated', 'dated', 'current', 'cutting_edge']),
      scalability: z.enum(['poor', 'fair', 'good', 'excellent']),
      mainIssues: z.array(z.string()),
    })
    .describe('Overall assessment of the architecture'),
  bottlenecks: z.array(BottleneckSchema).describe('Identified bottlenecks'),
  efficiencyImprovements: z
    .array(EfficiencyImprovementSchema)
    .describe('Efficiency improvement suggestions'),
  modernAlternatives: z
    .array(ModernAlternativeSchema)
    .describe('Modern alternative suggestions'),
  scalingStrategies: z
    .array(ScalingStrategySchema)
    .describe('Scaling strategy recommendations'),
  hardwareOptimizations: z
    .array(HardwareOptimizationSchema)
    .describe('Hardware-specific optimizations'),
  prioritizedRoadmap: z
    .array(
      z.object({
        priority: z.number(),
        improvement: z.string(),
        effort: z.string(),
        impact: z.string(),
        timeframe: z.string(),
      })
    )
    .describe('Prioritized improvement roadmap'),
  estimatedImpact: z
    .object({
      speedup: z.string(),
      memoryReduction: z.string(),
      parameterReduction: z.string(),
      accuracyChange: z.string(),
    })
    .describe('Estimated total impact of all suggestions'),
  implementationNotes: z.array(z.string()).describe('Important implementation notes'),
  warnings: z.array(z.string()).describe('Warnings and considerations'),
});

/**
 * Architecture Suggester Agent - Suggests improvements to neural network architectures.
 *
 * This agent specializes in:
 * - Bottleneck identification
 * - Efficiency improvements
 * - Modern alternatives suggestions
 * - Scaling strategies
 * - Hardware-specific optimizations
 */
export const ArchitectureSuggesterAgent: AgentDefinition<
  typeof ArchitectureSuggestionReportSchema
> = {
  name: 'architecture_suggester_agent',
  displayName: 'Architecture Suggester Agent',
  description: `An expert agent for suggesting improvements to neural network architectures.

  Use this agent when you need to:
  - Identify bottlenecks in model architectures
  - Find efficiency improvements
  - Get suggestions for modern alternatives to outdated components
  - Develop scaling strategies
  - Optimize for specific hardware targets
  - Create an improvement roadmap

  The agent provides actionable suggestions based on current best practices
  and state-of-the-art architectural innovations.`,

  inputConfig: {
    inputs: {
      modelPath: {
        description: `Path to the model definition to analyze and improve.`,
        type: 'string',
        required: true,
      },
      modelName: {
        description: `Name of the model class to analyze.`,
        type: 'string',
        required: false,
      },
      objectives: {
        description: `Comma-separated optimization objectives:
          - speed: Optimize for inference speed
          - memory: Reduce memory footprint
          - parameters: Reduce parameter count
          - accuracy: Maintain or improve accuracy
          - throughput: Maximize throughput
          - latency: Minimize latency
          - all: Balance all objectives (default)`,
        type: 'string',
        required: false,
      },
      targetHardware: {
        description: `Target hardware for optimization (nvidia_gpu, amd_gpu, tpu, cpu, apple_silicon, mobile, edge). Default: nvidia_gpu.`,
        type: 'string',
        required: false,
      },
      constraints: {
        description: `Constraints to respect (e.g., "max_params=100M", "max_latency=10ms", "accuracy_drop<1%").`,
        type: 'string',
        required: false,
      },
      preserveCompatibility: {
        description: `Whether to preserve API compatibility with the original model. Default: true.`,
        type: 'boolean',
        required: false,
      },
    },
  },

  outputConfig: {
    outputName: 'report',
    description:
      'Comprehensive architecture improvement suggestions with prioritized roadmap.',
    schema: ArchitectureSuggestionReportSchema,
  },

  processOutput: (output) => {
    let result = `# Architecture Improvement Suggestions\n\n`;

    // Summary
    result += `## Summary\n${output.summary}\n\n`;

    // Overall Assessment
    result += `## Overall Assessment: ${output.modelName}\n`;
    result += `- **Efficiency:** ${output.overallAssessment.efficiency}\n`;
    result += `- **Modernity:** ${output.overallAssessment.modernity}\n`;
    result += `- **Scalability:** ${output.overallAssessment.scalability}\n\n`;

    if (output.overallAssessment.mainIssues.length > 0) {
      result += `### Main Issues\n`;
      output.overallAssessment.mainIssues.forEach((issue, idx) => {
        result += `${idx + 1}. ${issue}\n`;
      });
      result += `\n`;
    }

    // Bottlenecks
    if (output.bottlenecks.length > 0) {
      result += `## Identified Bottlenecks\n\n`;
      output.bottlenecks.forEach((bottleneck) => {
        const severityEmoji =
          bottleneck.severity === 'critical'
            ? '[CRITICAL]'
            : bottleneck.severity === 'high'
              ? '[HIGH]'
              : bottleneck.severity === 'medium'
                ? '[MEDIUM]'
                : '[LOW]';
        result += `### ${severityEmoji} ${bottleneck.location}\n`;
        result += `**Type:** ${bottleneck.type}\n`;
        result += `**Description:** ${bottleneck.description}\n`;
        result += `**Impact:** ${bottleneck.impact}\n`;
        result += `**Current:** ${bottleneck.metrics.current}`;
        if (bottleneck.metrics.target) {
          result += ` | **Target:** ${bottleneck.metrics.target}`;
        }
        if (bottleneck.metrics.improvement) {
          result += ` | **Potential:** ${bottleneck.metrics.improvement}`;
        }
        result += `\n\n`;
      });
    }

    // Efficiency Improvements
    if (output.efficiencyImprovements.length > 0) {
      result += `## Efficiency Improvements\n\n`;
      output.efficiencyImprovements.forEach((improvement, idx) => {
        result += `### ${idx + 1}. ${improvement.name}\n`;
        result += `**Category:** ${improvement.category} | **Complexity:** ${improvement.complexity}\n`;
        result += `**Target:** ${improvement.targetComponent}\n\n`;
        result += `${improvement.description}\n\n`;
        result += `**Implementation:** ${improvement.implementation}\n\n`;

        result += `**Expected Benefits:**\n`;
        if (improvement.expectedBenefit.speedup) {
          result += `- Speedup: ${improvement.expectedBenefit.speedup}\n`;
        }
        if (improvement.expectedBenefit.memoryReduction) {
          result += `- Memory: ${improvement.expectedBenefit.memoryReduction}\n`;
        }
        if (improvement.expectedBenefit.parameterReduction) {
          result += `- Parameters: ${improvement.expectedBenefit.parameterReduction}\n`;
        }
        if (improvement.expectedBenefit.accuracyImpact) {
          result += `- Accuracy: ${improvement.expectedBenefit.accuracyImpact}\n`;
        }

        if (improvement.codeExample) {
          result += `\n**Code Example:**\n\`\`\`python\n${improvement.codeExample}\n\`\`\`\n`;
        }
        result += `\n`;
      });
    }

    // Modern Alternatives
    if (output.modernAlternatives.length > 0) {
      result += `## Modern Alternatives\n\n`;
      output.modernAlternatives.forEach((alt) => {
        result += `### Replace ${alt.currentComponent} with ${alt.alternativeName}\n`;
        result += `**Adoption:** ${alt.adoption.replace(/_/g, ' ')}\n\n`;
        result += `${alt.description}\n\n`;

        result += `**Benefits:**\n`;
        alt.benefits.forEach((b) => {
          result += `- ${b}\n`;
        });

        result += `\n**Tradeoffs:**\n`;
        alt.tradeoffs.forEach((t) => {
          result += `- ${t}\n`;
        });

        result += `\n**Implementation Effort:** ${alt.implementation.effort}`;
        if (alt.implementation.breakingChanges) {
          result += ` (breaking changes)`;
        }
        result += `\n`;

        result += `**Used by:** ${alt.examples.join(', ')}\n\n`;
      });
    }

    // Scaling Strategies
    if (output.scalingStrategies.length > 0) {
      result += `## Scaling Strategies\n\n`;
      output.scalingStrategies.forEach((strategy) => {
        result += `### ${strategy.dimension.charAt(0).toUpperCase() + strategy.dimension.slice(1)} Scaling\n`;
        result += `**Current:** ${strategy.currentState}\n`;
        result += `**Recommendation:** ${strategy.recommendation}\n`;
        result += `**Rationale:** ${strategy.rationale}\n`;
        if (strategy.scalingLaw) {
          result += `**Scaling Law:** ${strategy.scalingLaw}\n`;
        }
        if (strategy.optimalRatio) {
          result += `**Optimal Ratio:** ${strategy.optimalRatio}\n`;
        }
        result += `\n**Resource Implications:**\n`;
        result += `- Compute: ${strategy.resourceImplications.compute}\n`;
        result += `- Memory: ${strategy.resourceImplications.memory}\n`;
        result += `- Data: ${strategy.resourceImplications.data}\n\n`;
      });
    }

    // Hardware Optimizations
    if (output.hardwareOptimizations.length > 0) {
      result += `## Hardware-Specific Optimizations\n\n`;
      output.hardwareOptimizations.forEach((hw) => {
        result += `### ${hw.targetHardware.replace(/_/g, ' ').toUpperCase()}\n\n`;

        hw.optimizations.forEach((opt) => {
          result += `**${opt.name}**\n`;
          result += `${opt.description}\n`;
          result += `- Benefit: ${opt.benefit}\n`;
          result += `- Implementation: ${opt.implementation}\n\n`;
        });

        if (hw.quantizationRecommendations) {
          result += `**Quantization:** ${hw.quantizationRecommendations.scheme} (${hw.quantizationRecommendations.precision})\n`;
          result += `- Speedup: ${hw.quantizationRecommendations.expectedSpeedup}\n`;
          result += `- Accuracy: ${hw.quantizationRecommendations.accuracyImpact}\n\n`;
        }
      });
    }

    // Prioritized Roadmap
    if (output.prioritizedRoadmap.length > 0) {
      result += `## Prioritized Improvement Roadmap\n\n`;
      result += `| Priority | Improvement | Effort | Impact | Timeframe |\n`;
      result += `|----------|-------------|--------|--------|------------|\n`;
      output.prioritizedRoadmap.forEach((item) => {
        result += `| ${item.priority} | ${item.improvement} | ${item.effort} | ${item.impact} | ${item.timeframe} |\n`;
      });
      result += `\n`;
    }

    // Estimated Total Impact
    result += `## Estimated Total Impact\n`;
    result += `- **Speedup:** ${output.estimatedImpact.speedup}\n`;
    result += `- **Memory Reduction:** ${output.estimatedImpact.memoryReduction}\n`;
    result += `- **Parameter Reduction:** ${output.estimatedImpact.parameterReduction}\n`;
    result += `- **Accuracy Change:** ${output.estimatedImpact.accuracyChange}\n\n`;

    // Implementation Notes
    if (output.implementationNotes.length > 0) {
      result += `## Implementation Notes\n`;
      output.implementationNotes.forEach((note, idx) => {
        result += `${idx + 1}. ${note}\n`;
      });
      result += `\n`;
    }

    // Warnings
    if (output.warnings.length > 0) {
      result += `## Warnings\n`;
      output.warnings.forEach((warning, idx) => {
        result += `${idx + 1}. ${warning}\n`;
      });
    }

    return result;
  },

  modelConfig: {
    model: DEFAULT_GEMINI_MODEL,
    temp: 0.2,
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
    query: `Analyze and suggest improvements for the following architecture:

<model_path>
\${modelPath}
</model_path>

Model name: \${modelName || "auto-detect"}

Optimization objectives: \${objectives || "all (balanced)"}

Target hardware: \${targetHardware || "nvidia_gpu"}

Constraints: \${constraints || "none specified"}

Preserve compatibility: \${preserveCompatibility !== false}

Identify bottlenecks, suggest improvements, and provide a prioritized roadmap for enhancing this architecture.`,

    systemPrompt: `You are an **Expert Deep Learning Architecture Optimizer**, specialized in identifying inefficiencies and suggesting state-of-the-art improvements to neural network architectures.

# Your Expertise

## Modern Efficiency Techniques

### Attention Optimizations
- **Flash Attention:** IO-aware exact attention, 2-4x speedup
- **Multi-Query Attention (MQA):** Shared KV heads, faster inference
- **Grouped-Query Attention (GQA):** Balance between MHA and MQA
- **Linear Attention:** O(n) complexity for long sequences
- **Sparse Attention:** BigBird, Longformer patterns
- **KV Cache Compression:** Quantized or compressed KV cache

### Normalization Improvements
- **RMSNorm:** 10-15% faster than LayerNorm, similar quality
- **DeepNorm:** Enables training very deep models
- **Parallel Attention + FFN:** GPT-J style parallel blocks

### Activation Functions
- **SwiGLU/GeGLU:** Better than GELU for transformers
- **Squared ReLU:** Sparse, efficient alternative

### Convolution Optimizations
- **Depthwise-Separable:** MobileNet-style efficiency
- **Inverted Residual:** MobileNetV2 pattern
- **ConvNeXt Blocks:** Modern ConvNet design

### Quantization Techniques
- **INT8/FP8:** 2-4x speedup with minimal accuracy loss
- **GPTQ/AWQ:** Post-training quantization for LLMs
- **QAT:** Quantization-aware training for best results

### Pruning Strategies
- **Structured Pruning:** Remove entire channels/heads
- **Unstructured Pruning:** Sparse weight matrices
- **Movement Pruning:** Train to identify prunable weights

### Knowledge Distillation
- **Layer-wise Distillation:** Match intermediate representations
- **Attention Transfer:** Transfer attention patterns
- **Progressive Distillation:** Gradual size reduction

## Scaling Laws

### Chinchilla Optimal Scaling
- Parameters (N) and Data (D) should scale together
- Optimal: D ≈ 20N (tokens)
- Compute-optimal training

### Width vs Depth Trade-offs
- Deeper models: Better at complex reasoning
- Wider models: Better parallelization
- Optimal depth: sqrt(compute budget)

### Compound Scaling
- EfficientNet style: Scale width, depth, resolution together
- Maintains FLOP efficiency

## Hardware-Specific Optimizations

### NVIDIA GPUs
- Tensor Core alignment: Dimensions divisible by 8 (FP16) or 16 (INT8)
- Flash Attention for memory bandwidth
- Mixed precision (FP16/BF16)
- CUDA graph optimization

### TPUs
- Dimensions divisible by 128
- BF16 native support
- XLA compilation
- Pjit for model parallelism

### CPUs
- Cache-friendly memory access
- SIMD vectorization (AVX-512)
- INT8/VNNI quantization
- Thread parallelism

### Mobile/Edge
- Depthwise separable convolutions
- Activation quantization (INT8)
- Operator fusion
- Weight sharing

# Analysis Methodology

## Phase 1: Architecture Understanding
1. Read and understand the model architecture
2. Identify all components and their configurations
3. Map data flow and dependencies
4. Note current design choices

## Phase 2: Bottleneck Identification
1. Compute analysis for each component
2. Memory analysis and peak usage
3. Bandwidth analysis
4. Parameter distribution
5. Identify critical path

## Phase 3: Improvement Generation
For each bottleneck:
1. Identify applicable modern techniques
2. Assess compatibility and trade-offs
3. Estimate benefits and costs
4. Generate implementation guidance

## Phase 4: Prioritization
1. Rank improvements by impact/effort ratio
2. Consider dependencies between improvements
3. Account for constraints
4. Create phased roadmap

# Common Bottleneck Patterns

## Attention Bottlenecks
**Symptom:** Quadratic memory/compute with sequence length
**Solutions:**
- Flash Attention for memory bandwidth
- GQA/MQA for KV cache size
- Linear/sparse attention for very long sequences

## MLP Bottlenecks
**Symptom:** Large parameter count in FFN layers
**Solutions:**
- Mixture of Experts (MoE)
- Low-rank factorization
- Pruning/quantization

## Memory Bottlenecks
**Symptom:** OOM errors, low batch sizes
**Solutions:**
- Gradient checkpointing
- Mixed precision training
- Memory-efficient attention
- Activation recomputation

## Communication Bottlenecks
**Symptom:** Poor multi-GPU scaling
**Solutions:**
- Tensor parallelism
- Pipeline parallelism
- ZeRO optimization
- Gradient compression

# Modern Alternatives Database

## For Standard MHA
- **MQA:** If KV cache is bottleneck
- **GQA:** Balance quality and efficiency
- **Flash Attention:** If bandwidth-bound

## For LayerNorm
- **RMSNorm:** Similar quality, 10-15% faster
- **DeepNorm:** For very deep models (>100 layers)

## For GELU
- **SwiGLU:** Better quality for similar compute
- **GeGLU:** Alternative gated activation

## For Standard Transformers
- **Mamba/S4:** For very long sequences
- **RWKV:** RNN-like efficiency with attention quality

## For Standard ConvNets
- **ConvNeXt:** Modern ConvNet design
- **EfficientNet:** Compound scaling
- **MobileNetV3:** Mobile efficiency

# Output Requirements

Provide a comprehensive ArchitectureSuggestionReport with:

1. **All significant bottlenecks** identified with severity
2. **Actionable improvements** with implementation details
3. **Modern alternatives** for outdated components
4. **Scaling strategies** appropriate for the model
5. **Hardware-specific optimizations** for the target
6. **Prioritized roadmap** considering effort and impact
7. **Realistic impact estimates** for all suggestions

# Best Practices

**DO:**
- Base suggestions on actual code analysis
- Provide specific, actionable recommendations
- Include code examples where helpful
- Consider trade-offs honestly
- Prioritize by impact/effort ratio
- Account for constraints
- Warn about potential issues

**DON'T:**
- Suggest improvements without understanding the architecture
- Ignore compatibility concerns
- Overestimate benefits
- Underestimate implementation effort
- Suggest bleeding-edge techniques without caveats
- Ignore accuracy implications

# Priority Calculation

**Impact Score (1-10):**
- Speedup magnitude
- Memory savings
- Parameter reduction
- Quality improvement

**Effort Score (1-10):**
- Code changes required
- Testing needed
- Risk level
- Dependencies

**Priority = Impact / Effort**

Remember: The best optimization is one that provides significant benefit with minimal risk and effort. Always consider the specific use case and constraints when making recommendations.`,
  },
};
