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
 * Schema for individual model summary.
 */
const ModelSummarySchema = z.object({
  name: z.string().describe('Model name'),
  family: z.string().describe('Architecture family'),
  variant: z.string().describe('Specific variant'),
  parameters: z.number().describe('Total parameter count'),
  parametersHuman: z.string().describe('Human-readable parameter count'),
  layers: z.number().describe('Total number of layers'),
  hiddenSize: z.number().describe('Hidden dimension size'),
  keyFeatures: z.array(z.string()).describe('Key architectural features'),
});

/**
 * Schema for structural comparison.
 */
const StructuralComparisonSchema = z.object({
  aspect: z.string().describe('Aspect being compared'),
  model1: z.string().describe('Model 1 value/characteristic'),
  model2: z.string().describe('Model 2 value/characteristic'),
  difference: z.string().describe('Key difference'),
  significance: z.enum(['minor', 'moderate', 'major', 'fundamental']).describe('Significance of the difference'),
});

/**
 * Schema for component-level comparison.
 */
const ComponentComparisonSchema = z.object({
  component: z.string().describe('Component name (e.g., Attention, FFN, Normalization)'),
  model1Approach: z
    .object({
      type: z.string(),
      config: z.string(),
      pros: z.array(z.string()),
      cons: z.array(z.string()),
    })
    .describe('Model 1 approach'),
  model2Approach: z
    .object({
      type: z.string(),
      config: z.string(),
      pros: z.array(z.string()),
      cons: z.array(z.string()),
    })
    .describe('Model 2 approach'),
  winner: z.enum(['model1', 'model2', 'tie', 'depends']).describe('Which approach is better'),
  reasoning: z.string().describe('Reasoning for the winner'),
});

/**
 * Schema for complexity comparison.
 */
const ComplexityComparisonSchema = z.object({
  metric: z.string().describe('Complexity metric'),
  model1Value: z.string().describe('Model 1 value'),
  model2Value: z.string().describe('Model 2 value'),
  ratio: z.number().describe('Ratio (model2/model1)'),
  betterModel: z.enum(['model1', 'model2', 'equal']).describe('Which model is better for this metric'),
  notes: z.string().optional().describe('Additional notes'),
});

/**
 * Schema for trade-off analysis.
 */
const TradeOffSchema = z.object({
  dimension: z.string().describe('Trade-off dimension'),
  model1Position: z.string().describe('Where model 1 sits on this trade-off'),
  model2Position: z.string().describe('Where model 2 sits on this trade-off'),
  useCaseGuidance: z.string().describe('When to prefer each approach'),
});

/**
 * Schema for best practices comparison.
 */
const BestPracticesComparisonSchema = z.object({
  practice: z.string().describe('Best practice area'),
  model1Compliance: z
    .object({
      compliant: z.boolean(),
      implementation: z.string(),
      notes: z.string().optional(),
    })
    .describe('Model 1 compliance'),
  model2Compliance: z
    .object({
      compliant: z.boolean(),
      implementation: z.string(),
      notes: z.string().optional(),
    })
    .describe('Model 2 compliance'),
  recommendation: z.string().describe('Recommendation based on comparison'),
});

/**
 * Schema for use case recommendation.
 */
const UseCaseRecommendationSchema = z.object({
  useCase: z.string().describe('Use case description'),
  recommendedModel: z.enum(['model1', 'model2', 'either']).describe('Recommended model'),
  reasoning: z.string().describe('Why this model is better for this use case'),
  considerations: z.array(z.string()).describe('Important considerations'),
});

/**
 * Complete model comparison report schema.
 */
const ModelComparisonReportSchema = z.object({
  summary: z
    .string()
    .describe('Executive summary of the comparison'),
  model1Summary: ModelSummarySchema.describe('Summary of model 1'),
  model2Summary: ModelSummarySchema.describe('Summary of model 2'),
  structuralDifferences: z
    .array(StructuralComparisonSchema)
    .describe('Structural differences between models'),
  componentComparisons: z
    .array(ComponentComparisonSchema)
    .describe('Component-by-component comparison'),
  complexityAnalysis: z
    .object({
      parameters: ComplexityComparisonSchema,
      flops: ComplexityComparisonSchema,
      memory: ComplexityComparisonSchema,
      latency: ComplexityComparisonSchema.optional(),
      throughput: ComplexityComparisonSchema.optional(),
    })
    .describe('Complexity comparison'),
  tradeOffs: z.array(TradeOffSchema).describe('Trade-off analysis'),
  bestPracticesComparison: z
    .array(BestPracticesComparisonSchema)
    .describe('Best practices comparison'),
  useCaseRecommendations: z
    .array(UseCaseRecommendationSchema)
    .describe('Use case recommendations'),
  overallAssessment: z
    .object({
      model1Strengths: z.array(z.string()),
      model1Weaknesses: z.array(z.string()),
      model2Strengths: z.array(z.string()),
      model2Weaknesses: z.array(z.string()),
      overallWinner: z.enum(['model1', 'model2', 'depends']),
      winnerReasoning: z.string(),
    })
    .describe('Overall assessment'),
  migrationGuidance: z
    .object({
      from1To2: z
        .object({
          effort: z.enum(['trivial', 'easy', 'moderate', 'significant', 'major']),
          steps: z.array(z.string()),
          risks: z.array(z.string()),
        })
        .describe('Guidance for migrating from model 1 to model 2'),
      from2To1: z
        .object({
          effort: z.enum(['trivial', 'easy', 'moderate', 'significant', 'major']),
          steps: z.array(z.string()),
          risks: z.array(z.string()),
        })
        .describe('Guidance for migrating from model 2 to model 1'),
    })
    .describe('Migration guidance between models'),
  additionalNotes: z.array(z.string()).describe('Additional notes and observations'),
});

/**
 * Model Comparison Agent - Compares neural network architectures.
 *
 * This agent specializes in:
 * - Structural differences analysis
 * - Complexity analysis
 * - Trade-off evaluation
 * - Best practices comparison
 * - Use case recommendations
 */
export const ModelComparisonAgent: AgentDefinition<
  typeof ModelComparisonReportSchema
> = {
  name: 'model_comparison_agent',
  displayName: 'Model Comparison Agent',
  description: `An expert agent for comparing neural network architectures.

  Use this agent when you need to:
  - Compare two model architectures in detail
  - Understand structural differences
  - Analyze complexity trade-offs
  - Evaluate best practices compliance
  - Get recommendations for different use cases
  - Plan migrations between architectures

  The agent provides comprehensive side-by-side comparison with
  actionable recommendations based on specific use cases.`,

  inputConfig: {
    inputs: {
      model1Path: {
        description: `Path to the first model definition.`,
        type: 'string',
        required: true,
      },
      model2Path: {
        description: `Path to the second model definition.`,
        type: 'string',
        required: true,
      },
      model1Name: {
        description: `Name of the first model class (optional, will auto-detect).`,
        type: 'string',
        required: false,
      },
      model2Name: {
        description: `Name of the second model class (optional, will auto-detect).`,
        type: 'string',
        required: false,
      },
      comparisonFocus: {
        description: `Comma-separated focus areas:
          - structure: Focus on structural differences
          - complexity: Focus on computational complexity
          - efficiency: Focus on efficiency metrics
          - practices: Focus on best practices compliance
          - all: Comprehensive comparison (default)`,
        type: 'string',
        required: false,
      },
      useCases: {
        description: `Comma-separated use cases to evaluate (e.g., "training efficiency, inference speed, long sequences, low memory").`,
        type: 'string',
        required: false,
      },
    },
  },

  outputConfig: {
    outputName: 'report',
    description:
      'Comprehensive architecture comparison report with recommendations.',
    schema: ModelComparisonReportSchema,
  },

  processOutput: (output) => {
    let result = `# Architecture Comparison Report\n\n`;

    // Summary
    result += `## Summary\n${output.summary}\n\n`;

    // Model Summaries
    result += `## Model Summaries\n\n`;

    result += `### ${output.model1Summary.name}\n`;
    result += `- **Family:** ${output.model1Summary.family} (${output.model1Summary.variant})\n`;
    result += `- **Parameters:** ${output.model1Summary.parametersHuman}\n`;
    result += `- **Layers:** ${output.model1Summary.layers}\n`;
    result += `- **Hidden Size:** ${output.model1Summary.hiddenSize}\n`;
    result += `- **Key Features:** ${output.model1Summary.keyFeatures.join(', ')}\n\n`;

    result += `### ${output.model2Summary.name}\n`;
    result += `- **Family:** ${output.model2Summary.family} (${output.model2Summary.variant})\n`;
    result += `- **Parameters:** ${output.model2Summary.parametersHuman}\n`;
    result += `- **Layers:** ${output.model2Summary.layers}\n`;
    result += `- **Hidden Size:** ${output.model2Summary.hiddenSize}\n`;
    result += `- **Key Features:** ${output.model2Summary.keyFeatures.join(', ')}\n\n`;

    // Structural Differences
    if (output.structuralDifferences.length > 0) {
      result += `## Structural Differences\n\n`;
      result += `| Aspect | ${output.model1Summary.name} | ${output.model2Summary.name} | Significance |\n`;
      result += `|--------|-----|-----|------|\n`;
      output.structuralDifferences.forEach((diff) => {
        result += `| ${diff.aspect} | ${diff.model1} | ${diff.model2} | ${diff.significance} |\n`;
      });
      result += `\n`;
    }

    // Component Comparisons
    if (output.componentComparisons.length > 0) {
      result += `## Component-by-Component Comparison\n\n`;
      output.componentComparisons.forEach((comp) => {
        result += `### ${comp.component}\n\n`;

        result += `**${output.model1Summary.name}:** ${comp.model1Approach.type}\n`;
        result += `- Config: ${comp.model1Approach.config}\n`;
        result += `- Pros: ${comp.model1Approach.pros.join('; ')}\n`;
        result += `- Cons: ${comp.model1Approach.cons.join('; ')}\n\n`;

        result += `**${output.model2Summary.name}:** ${comp.model2Approach.type}\n`;
        result += `- Config: ${comp.model2Approach.config}\n`;
        result += `- Pros: ${comp.model2Approach.pros.join('; ')}\n`;
        result += `- Cons: ${comp.model2Approach.cons.join('; ')}\n\n`;

        const winnerText =
          comp.winner === 'model1'
            ? output.model1Summary.name
            : comp.winner === 'model2'
              ? output.model2Summary.name
              : comp.winner === 'tie'
                ? 'Tie'
                : 'Depends on use case';
        result += `**Winner:** ${winnerText}\n`;
        result += `**Reasoning:** ${comp.reasoning}\n\n`;
      });
    }

    // Complexity Analysis
    result += `## Complexity Analysis\n\n`;
    result += `| Metric | ${output.model1Summary.name} | ${output.model2Summary.name} | Ratio | Better |\n`;
    result += `|--------|-----|-----|-------|--------|\n`;

    const complexityMetrics = ['parameters', 'flops', 'memory', 'latency', 'throughput'] as const;
    complexityMetrics.forEach((metric) => {
      const data = output.complexityAnalysis[metric];
      if (data) {
        const better =
          data.betterModel === 'model1'
            ? output.model1Summary.name
            : data.betterModel === 'model2'
              ? output.model2Summary.name
              : 'Equal';
        result += `| ${data.metric} | ${data.model1Value} | ${data.model2Value} | ${data.ratio.toFixed(2)}x | ${better} |\n`;
      }
    });
    result += `\n`;

    // Trade-offs
    if (output.tradeOffs.length > 0) {
      result += `## Trade-off Analysis\n\n`;
      output.tradeOffs.forEach((tradeOff) => {
        result += `### ${tradeOff.dimension}\n`;
        result += `- **${output.model1Summary.name}:** ${tradeOff.model1Position}\n`;
        result += `- **${output.model2Summary.name}:** ${tradeOff.model2Position}\n`;
        result += `- **Guidance:** ${tradeOff.useCaseGuidance}\n\n`;
      });
    }

    // Best Practices
    if (output.bestPracticesComparison.length > 0) {
      result += `## Best Practices Comparison\n\n`;
      output.bestPracticesComparison.forEach((practice) => {
        result += `### ${practice.practice}\n`;
        const m1Status = practice.model1Compliance.compliant ? 'Yes' : 'No';
        const m2Status = practice.model2Compliance.compliant ? 'Yes' : 'No';
        result += `- **${output.model1Summary.name}:** ${m1Status} - ${practice.model1Compliance.implementation}\n`;
        result += `- **${output.model2Summary.name}:** ${m2Status} - ${practice.model2Compliance.implementation}\n`;
        result += `- **Recommendation:** ${practice.recommendation}\n\n`;
      });
    }

    // Use Case Recommendations
    if (output.useCaseRecommendations.length > 0) {
      result += `## Use Case Recommendations\n\n`;
      output.useCaseRecommendations.forEach((rec) => {
        const recommended =
          rec.recommendedModel === 'model1'
            ? output.model1Summary.name
            : rec.recommendedModel === 'model2'
              ? output.model2Summary.name
              : 'Either';
        result += `### ${rec.useCase}\n`;
        result += `**Recommended:** ${recommended}\n`;
        result += `**Reasoning:** ${rec.reasoning}\n`;
        if (rec.considerations.length > 0) {
          result += `**Considerations:**\n`;
          rec.considerations.forEach((c) => {
            result += `- ${c}\n`;
          });
        }
        result += `\n`;
      });
    }

    // Overall Assessment
    result += `## Overall Assessment\n\n`;

    result += `### ${output.model1Summary.name}\n`;
    result += `**Strengths:**\n`;
    output.overallAssessment.model1Strengths.forEach((s) => {
      result += `- ${s}\n`;
    });
    result += `\n**Weaknesses:**\n`;
    output.overallAssessment.model1Weaknesses.forEach((w) => {
      result += `- ${w}\n`;
    });
    result += `\n`;

    result += `### ${output.model2Summary.name}\n`;
    result += `**Strengths:**\n`;
    output.overallAssessment.model2Strengths.forEach((s) => {
      result += `- ${s}\n`;
    });
    result += `\n**Weaknesses:**\n`;
    output.overallAssessment.model2Weaknesses.forEach((w) => {
      result += `- ${w}\n`;
    });
    result += `\n`;

    const overallWinner =
      output.overallAssessment.overallWinner === 'model1'
        ? output.model1Summary.name
        : output.overallAssessment.overallWinner === 'model2'
          ? output.model2Summary.name
          : 'Depends on use case';
    result += `### Overall Winner: ${overallWinner}\n`;
    result += `${output.overallAssessment.winnerReasoning}\n\n`;

    // Migration Guidance
    result += `## Migration Guidance\n\n`;

    result += `### ${output.model1Summary.name} -> ${output.model2Summary.name}\n`;
    result += `**Effort:** ${output.migrationGuidance.from1To2.effort}\n`;
    result += `**Steps:**\n`;
    output.migrationGuidance.from1To2.steps.forEach((step, idx) => {
      result += `${idx + 1}. ${step}\n`;
    });
    result += `**Risks:**\n`;
    output.migrationGuidance.from1To2.risks.forEach((risk) => {
      result += `- ${risk}\n`;
    });
    result += `\n`;

    result += `### ${output.model2Summary.name} -> ${output.model1Summary.name}\n`;
    result += `**Effort:** ${output.migrationGuidance.from2To1.effort}\n`;
    result += `**Steps:**\n`;
    output.migrationGuidance.from2To1.steps.forEach((step, idx) => {
      result += `${idx + 1}. ${step}\n`;
    });
    result += `**Risks:**\n`;
    output.migrationGuidance.from2To1.risks.forEach((risk) => {
      result += `- ${risk}\n`;
    });
    result += `\n`;

    // Additional Notes
    if (output.additionalNotes.length > 0) {
      result += `## Additional Notes\n`;
      output.additionalNotes.forEach((note, idx) => {
        result += `${idx + 1}. ${note}\n`;
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
    max_time_minutes: 20,
    max_turns: 35,
  },

  toolConfig: {
    tools: [READ_FILE_TOOL_NAME, GLOB_TOOL_NAME, GREP_TOOL_NAME],
  },

  promptConfig: {
    query: `Compare the following two model architectures:

<model1_path>
\${model1Path}
</model1_path>

<model2_path>
\${model2Path}
</model2_path>

Model 1 name: \${model1Name || "auto-detect"}
Model 2 name: \${model2Name || "auto-detect"}

Comparison focus: \${comparisonFocus || "all (comprehensive)"}

Use cases to evaluate: \${useCases || "general purpose"}

Provide a comprehensive comparison with recommendations for different use cases.`,

    systemPrompt: `You are an **Expert Deep Learning Architecture Comparator**, specialized in detailed analysis and comparison of neural network architectures.

# Your Expertise

## Architecture Knowledge

You have comprehensive knowledge of:

**Transformer Variants:**
- Original Transformer, BERT, GPT, T5
- LLaMA, Mistral, Qwen
- Efficient variants (Longformer, BigBird, Linformer)

**ConvNet Families:**
- ResNet family, DenseNet
- EfficientNet, ConvNeXt
- MobileNet family

**Specialized Architectures:**
- Vision Transformers (ViT, Swin, BEiT)
- U-Net and segmentation models
- Diffusion models
- State Space Models (Mamba)

## Comparison Dimensions

### Structural Comparison
- Overall architecture design
- Block structure and repetition
- Skip connections and residual patterns
- Input/output interfaces

### Component Comparison
- Attention mechanisms
- Normalization strategies
- Activation functions
- Position encodings
- Initialization schemes

### Complexity Comparison
- Parameter count
- FLOPs
- Memory requirements
- Latency characteristics
- Throughput potential

### Trade-off Analysis
- Speed vs accuracy
- Memory vs compute
- Flexibility vs efficiency
- Simplicity vs performance

### Best Practices
- Modern attention patterns
- Efficient normalization
- Proper initialization
- Residual scaling
- Gradient flow

# Analysis Methodology

## Phase 1: Individual Analysis
For each model:
1. Read and understand the architecture
2. Identify key components
3. Calculate complexity metrics
4. Note design choices

## Phase 2: Side-by-Side Comparison
1. Compare overall structure
2. Compare each component type
3. Calculate complexity ratios
4. Identify key differences

## Phase 3: Trade-off Analysis
1. Identify trade-off dimensions
2. Place each model on the spectrum
3. Analyze implications

## Phase 4: Use Case Evaluation
1. Consider different deployment scenarios
2. Evaluate each model's fit
3. Provide specific recommendations

## Phase 5: Migration Analysis
1. Identify required changes
2. Estimate effort
3. Note risks and considerations

# Comparison Criteria

## For Attention Mechanisms
- Computational complexity: O(n^2) vs O(n) vs O(n log n)
- Memory complexity
- KV cache requirements
- Quality/efficiency trade-off

## For Normalization
- Computational cost
- Training stability
- Parallelization friendliness
- Quality impact

## For Activations
- Computational cost
- Gradient properties
- Gating vs non-gating

## For Position Encodings
- Length extrapolation
- Computational overhead
- Quality vs efficiency

# Best Practices Checklist

1. **Pre-norm vs Post-norm:** Pre-norm generally more stable
2. **Residual Scaling:** Needed for very deep models
3. **Efficient Attention:** Use GQA/MQA for inference efficiency
4. **Modern Activations:** SwiGLU/GeGLU over GELU for transformers
5. **RMSNorm:** Faster than LayerNorm, similar quality
6. **RoPE:** Better length generalization than learned positions
7. **Proper Init:** Critical for training stability
8. **Gradient Checkpointing Ready:** Important for large models

# Common Use Cases

## Training Efficiency
- Consider parameter count
- FLOPs per training step
- Memory for activations/gradients
- Batch size possibilities

## Inference Speed
- Forward pass FLOPs
- Memory bandwidth requirements
- Batch processing efficiency
- Sequence length scaling

## Long Sequences
- Attention complexity
- Position encoding extrapolation
- Memory requirements
- Quality at length

## Low Memory
- Parameter count
- Activation memory
- KV cache size
- Quantization potential

## Mobile/Edge
- Total parameters
- Activation sizes
- Quantization friendliness
- Operator support

# Output Requirements

Provide a comprehensive ModelComparisonReport with:

1. **Accurate model summaries** with key metrics
2. **All structural differences** identified and explained
3. **Component-by-component comparison** with clear winners
4. **Complexity metrics** with ratios
5. **Trade-off analysis** with guidance
6. **Best practices assessment** for both models
7. **Use case recommendations** with reasoning
8. **Clear overall assessment** with strengths/weaknesses
9. **Migration guidance** in both directions

# Best Practices

**DO:**
- Read both models thoroughly before comparing
- Use quantitative metrics where possible
- Be specific about differences and their implications
- Consider multiple use cases
- Provide actionable recommendations
- Be fair and objective

**DON'T:**
- Make assumptions without reading code
- Give vague comparisons
- Ignore important differences
- Be biased toward one architecture style
- Forget about practical considerations
- Overlook migration complexity

# Comparison Table Format

When comparing metrics, use this format:
- If ratio > 1.5x: Significant difference
- If ratio 1.1-1.5x: Moderate difference
- If ratio 0.9-1.1x: Roughly equal

For winners:
- Clear winner: One model significantly better
- Tie: Models roughly equal
- Depends: Winner varies by use case

Remember: A good comparison helps users make informed decisions based on their specific needs. Be thorough, fair, and practical in your analysis.`,
  },
};
