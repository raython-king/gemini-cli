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
 * Schema for attention mechanism analysis.
 */
const AttentionMechanismSchema = z.object({
  type: z
    .enum([
      'MHA',
      'MQA',
      'GQA',
      'MLA',
      'Linear',
      'Sparse',
      'Local',
      'Cross',
      'Self',
      'Sliding_Window',
      'Flash',
      'Custom',
    ])
    .describe('Type of attention mechanism'),
  heads: z.number().optional().describe('Number of attention heads'),
  headDim: z.number().optional().describe('Dimension per head'),
  kvHeads: z.number().optional().describe('Number of KV heads (for MQA/GQA)'),
  windowSize: z.number().optional().describe('Window size for local/sliding attention'),
  dropout: z.number().optional().describe('Attention dropout rate'),
  scaleFactor: z.string().optional().describe('Scaling factor formula'),
  biasType: z
    .enum(['none', 'learned', 'alibi', 'rope', 'relative'])
    .optional()
    .describe('Position bias type in attention'),
  efficiency: z.string().optional().describe('Computational efficiency notes'),
});

/**
 * Schema for normalization layer analysis.
 */
const NormalizationSchema = z.object({
  type: z
    .enum([
      'LayerNorm',
      'BatchNorm',
      'GroupNorm',
      'InstanceNorm',
      'RMSNorm',
      'PreNorm',
      'PostNorm',
      'DeepNorm',
      'Custom',
    ])
    .describe('Type of normalization'),
  position: z
    .enum(['pre', 'post', 'both', 'parallel'])
    .describe('Position in residual block'),
  affine: z.boolean().optional().describe('Whether learnable affine parameters are used'),
  eps: z.number().optional().describe('Epsilon value for numerical stability'),
  groups: z.number().optional().describe('Number of groups (for GroupNorm)'),
});

/**
 * Schema for positional encoding analysis.
 */
const PositionalEncodingSchema = z.object({
  type: z
    .enum([
      'Sinusoidal',
      'Learned',
      'RoPE',
      'ALiBi',
      'Relative',
      'T5_Relative',
      'Convolutional',
      'XPos',
      'NTK_RoPE',
      'YaRN',
      'None',
      'Custom',
    ])
    .describe('Type of positional encoding'),
  maxLength: z.number().optional().describe('Maximum sequence length supported'),
  dimension: z.number().optional().describe('Encoding dimension'),
  learned: z.boolean().describe('Whether positions are learned'),
  extrapolation: z
    .enum(['none', 'limited', 'good', 'excellent'])
    .optional()
    .describe('Length extrapolation capability'),
  rotaryBase: z.number().optional().describe('Base for RoPE encodings'),
  scalingFactor: z.number().optional().describe('Scaling factor for extended context'),
});

/**
 * Schema for activation function analysis.
 */
const ActivationSchema = z.object({
  type: z
    .enum([
      'ReLU',
      'GELU',
      'SiLU',
      'Swish',
      'GLU',
      'SwiGLU',
      'GeGLU',
      'ReGLU',
      'Mish',
      'Softmax',
      'Sigmoid',
      'Tanh',
      'PReLU',
      'LeakyReLU',
      'ELU',
      'SELU',
      'Softplus',
      'Custom',
    ])
    .describe('Activation function type'),
  location: z.string().describe('Where this activation is used'),
  gated: z.boolean().describe('Whether it uses gating mechanism'),
  computationalCost: z
    .enum(['low', 'medium', 'high'])
    .describe('Relative computational cost'),
});

/**
 * Schema for initialization scheme analysis.
 */
const InitializationSchema = z.object({
  type: z
    .enum([
      'Xavier',
      'Kaiming',
      'Orthogonal',
      'Normal',
      'Uniform',
      'Truncated_Normal',
      'GPT2',
      'Small_Init',
      'Scaled',
      'Custom',
    ])
    .describe('Weight initialization scheme'),
  layer: z.string().describe('Layer type this applies to'),
  std: z.number().optional().describe('Standard deviation'),
  gain: z.number().optional().describe('Gain factor'),
  residualScaling: z.boolean().optional().describe('Whether residual connections are scaled'),
  scaleFactor: z.string().optional().describe('Scaling formula'),
});

/**
 * Schema for architecture block analysis.
 */
const ArchitectureBlockSchema = z.object({
  name: z.string().describe('Block name'),
  type: z
    .enum([
      'TransformerEncoder',
      'TransformerDecoder',
      'ConvBlock',
      'ResidualBlock',
      'DenseBlock',
      'Bottleneck',
      'InvertedResidual',
      'FeedForward',
      'MLP',
      'CrossAttention',
      'Downsample',
      'Upsample',
      'Custom',
    ])
    .describe('Block type'),
  components: z.array(z.string()).describe('Components within this block'),
  skipConnection: z
    .enum(['none', 'residual', 'dense', 'parallel'])
    .describe('Skip connection type'),
  repetitions: z.number().describe('Number of times this block repeats'),
});

/**
 * Schema for architecture family identification.
 */
const ArchitectureFamilySchema = z.object({
  primary: z
    .enum([
      'Transformer',
      'ResNet',
      'ConvNet',
      'UNet',
      'LSTM',
      'GRU',
      'GAN',
      'VAE',
      'Diffusion',
      'MLP_Mixer',
      'Vision_Transformer',
      'State_Space',
      'Hybrid',
      'Custom',
    ])
    .describe('Primary architecture family'),
  variant: z.string().describe('Specific variant or version'),
  inspirations: z.array(z.string()).describe('Other architectures this draws inspiration from'),
  novelties: z.array(z.string()).describe('Novel architectural elements'),
});

/**
 * Complete architecture analysis report schema.
 */
const ArchitectureAnalysisReportSchema = z.object({
  summary: z
    .string()
    .describe('Executive summary of the architecture analysis'),
  modelName: z.string().describe('Identified model name or description'),
  architectureFamily: ArchitectureFamilySchema.describe('Architecture family classification'),
  overallStructure: z
    .object({
      inputFormat: z.string().describe('Expected input format and dimensions'),
      outputFormat: z.string().describe('Output format and dimensions'),
      depth: z.number().describe('Total number of layers'),
      width: z.number().optional().describe('Model width (hidden dimension)'),
      totalParameters: z.string().describe('Total parameter count'),
      trainableParameters: z.string().describe('Trainable parameter count'),
    })
    .describe('Overall model structure'),
  blocks: z
    .array(ArchitectureBlockSchema)
    .describe('Main architectural blocks'),
  attentionMechanisms: z
    .array(AttentionMechanismSchema)
    .describe('Attention mechanisms used'),
  normalizations: z
    .array(NormalizationSchema)
    .describe('Normalization strategies'),
  positionalEncodings: z
    .array(PositionalEncodingSchema)
    .describe('Positional encoding methods'),
  activations: z
    .array(ActivationSchema)
    .describe('Activation functions used'),
  initializations: z
    .array(InitializationSchema)
    .describe('Initialization schemes'),
  specialFeatures: z
    .array(
      z.object({
        name: z.string(),
        description: z.string(),
        benefit: z.string(),
      })
    )
    .describe('Special architectural features'),
  computationalProfile: z
    .object({
      flops: z.string().describe('Estimated FLOPs'),
      memoryFootprint: z.string().describe('Memory requirements'),
      sequenceScaling: z.string().describe('How computation scales with sequence length'),
      batchScaling: z.string().describe('How computation scales with batch size'),
    })
    .describe('Computational characteristics'),
  strengths: z.array(z.string()).describe('Architectural strengths'),
  weaknesses: z.array(z.string()).describe('Potential weaknesses or limitations'),
  recommendations: z.array(z.string()).describe('Recommendations for improvement or usage'),
  codeLocations: z
    .array(
      z.object({
        component: z.string(),
        file: z.string(),
        lineRange: z.string().optional(),
      })
    )
    .describe('Locations of key components in code'),
});

/**
 * Architecture Analyzer Agent - Deep analysis of neural network architectures.
 *
 * This agent specializes in:
 * - Understanding model structure from code
 * - Identifying architecture families
 * - Analyzing attention mechanisms
 * - Detecting normalization strategies
 * - Evaluating positional encodings
 * - Checking activation functions
 * - Analyzing initialization schemes
 * - Generating comprehensive architecture reports
 */
export const ArchitectureAnalyzerAgent: AgentDefinition<
  typeof ArchitectureAnalysisReportSchema
> = {
  name: 'architecture_analyzer_agent',
  displayName: 'Architecture Analyzer Agent',
  description: `An expert agent for deep analysis of neural network architectures.

  Use this agent when you need to:
  - Understand complex model architectures from code
  - Identify architecture families and variants
  - Analyze attention mechanisms in detail
  - Evaluate normalization and activation strategies
  - Understand positional encoding methods
  - Review initialization schemes
  - Generate comprehensive architecture documentation
  - Identify architectural patterns and design choices

  The agent provides expert-level analysis of deep learning architectures
  with detailed breakdowns of all components.`,

  inputConfig: {
    inputs: {
      modelPath: {
        description: `Path to the model definition. Can be:
          - A file path (e.g., "models/transformer.py")
          - A directory path (e.g., "src/models/")
          - A glob pattern (e.g., "**/*model*.py")`,
        type: 'string',
        required: true,
      },
      modelName: {
        description: `Name of the model or class to analyze (e.g., "GPT2", "ResNet50", "UNet"). If not provided, agent will attempt to identify all models in the path.`,
        type: 'string',
        required: false,
      },
      focusAreas: {
        description: `Comma-separated list of areas to focus on:
          - attention: Analyze attention mechanisms in detail
          - normalization: Focus on normalization strategies
          - positional: Analyze positional encodings
          - activations: Focus on activation functions
          - initialization: Analyze weight initialization
          - efficiency: Focus on computational efficiency
          - all: Comprehensive analysis (default)`,
        type: 'string',
        required: false,
      },
      framework: {
        description: `Deep learning framework (pytorch, tensorflow, jax, keras, flax). If not specified, agent will auto-detect.`,
        type: 'string',
        required: false,
      },
    },
  },

  outputConfig: {
    outputName: 'report',
    description:
      'Comprehensive architecture analysis report with detailed component breakdowns.',
    schema: ArchitectureAnalysisReportSchema,
  },

  processOutput: (output) => {
    let result = `# Architecture Analysis Report\n\n`;

    // Summary
    result += `## Summary\n${output.summary}\n\n`;

    // Model identification
    result += `## Model: ${output.modelName}\n`;
    result += `**Family:** ${output.architectureFamily.primary} (${output.architectureFamily.variant})\n`;
    if (output.architectureFamily.inspirations.length > 0) {
      result += `**Inspired by:** ${output.architectureFamily.inspirations.join(', ')}\n`;
    }
    if (output.architectureFamily.novelties.length > 0) {
      result += `**Novel Elements:** ${output.architectureFamily.novelties.join(', ')}\n`;
    }
    result += `\n`;

    // Overall Structure
    result += `## Overall Structure\n`;
    result += `- **Input:** ${output.overallStructure.inputFormat}\n`;
    result += `- **Output:** ${output.overallStructure.outputFormat}\n`;
    result += `- **Depth:** ${output.overallStructure.depth} layers\n`;
    if (output.overallStructure.width) {
      result += `- **Width:** ${output.overallStructure.width}\n`;
    }
    result += `- **Total Parameters:** ${output.overallStructure.totalParameters}\n`;
    result += `- **Trainable Parameters:** ${output.overallStructure.trainableParameters}\n\n`;

    // Architecture Blocks
    if (output.blocks && output.blocks.length > 0) {
      result += `## Architecture Blocks\n\n`;
      output.blocks.forEach((block) => {
        result += `### ${block.name}\n`;
        result += `- **Type:** ${block.type}\n`;
        result += `- **Components:** ${block.components.join(', ')}\n`;
        result += `- **Skip Connection:** ${block.skipConnection}\n`;
        result += `- **Repetitions:** ${block.repetitions}\n\n`;
      });
    }

    // Attention Mechanisms
    if (output.attentionMechanisms && output.attentionMechanisms.length > 0) {
      result += `## Attention Mechanisms\n\n`;
      output.attentionMechanisms.forEach((attn, idx) => {
        result += `### Attention ${idx + 1}: ${attn.type}\n`;
        if (attn.heads) result += `- **Heads:** ${attn.heads}\n`;
        if (attn.headDim) result += `- **Head Dimension:** ${attn.headDim}\n`;
        if (attn.kvHeads) result += `- **KV Heads:** ${attn.kvHeads}\n`;
        if (attn.biasType) result += `- **Position Bias:** ${attn.biasType}\n`;
        if (attn.efficiency) result += `- **Efficiency:** ${attn.efficiency}\n`;
        result += `\n`;
      });
    }

    // Normalizations
    if (output.normalizations && output.normalizations.length > 0) {
      result += `## Normalization Strategies\n\n`;
      output.normalizations.forEach((norm) => {
        result += `- **${norm.type}** (${norm.position})\n`;
      });
      result += `\n`;
    }

    // Positional Encodings
    if (output.positionalEncodings && output.positionalEncodings.length > 0) {
      result += `## Positional Encodings\n\n`;
      output.positionalEncodings.forEach((pos) => {
        result += `### ${pos.type}\n`;
        result += `- **Learned:** ${pos.learned ? 'Yes' : 'No'}\n`;
        if (pos.maxLength) result += `- **Max Length:** ${pos.maxLength}\n`;
        if (pos.extrapolation) result += `- **Extrapolation:** ${pos.extrapolation}\n`;
        result += `\n`;
      });
    }

    // Activations
    if (output.activations && output.activations.length > 0) {
      result += `## Activation Functions\n\n`;
      output.activations.forEach((act) => {
        result += `- **${act.type}** in ${act.location}`;
        if (act.gated) result += ` (gated)`;
        result += `\n`;
      });
      result += `\n`;
    }

    // Initializations
    if (output.initializations && output.initializations.length > 0) {
      result += `## Initialization Schemes\n\n`;
      output.initializations.forEach((init) => {
        result += `- **${init.type}** for ${init.layer}`;
        if (init.std) result += ` (std=${init.std})`;
        result += `\n`;
      });
      result += `\n`;
    }

    // Special Features
    if (output.specialFeatures && output.specialFeatures.length > 0) {
      result += `## Special Features\n\n`;
      output.specialFeatures.forEach((feature) => {
        result += `### ${feature.name}\n`;
        result += `${feature.description}\n`;
        result += `**Benefit:** ${feature.benefit}\n\n`;
      });
    }

    // Computational Profile
    result += `## Computational Profile\n`;
    result += `- **FLOPs:** ${output.computationalProfile.flops}\n`;
    result += `- **Memory:** ${output.computationalProfile.memoryFootprint}\n`;
    result += `- **Sequence Scaling:** ${output.computationalProfile.sequenceScaling}\n`;
    result += `- **Batch Scaling:** ${output.computationalProfile.batchScaling}\n\n`;

    // Strengths
    if (output.strengths && output.strengths.length > 0) {
      result += `## Strengths\n`;
      output.strengths.forEach((strength, idx) => {
        result += `${idx + 1}. ${strength}\n`;
      });
      result += `\n`;
    }

    // Weaknesses
    if (output.weaknesses && output.weaknesses.length > 0) {
      result += `## Weaknesses\n`;
      output.weaknesses.forEach((weakness, idx) => {
        result += `${idx + 1}. ${weakness}\n`;
      });
      result += `\n`;
    }

    // Recommendations
    if (output.recommendations && output.recommendations.length > 0) {
      result += `## Recommendations\n`;
      output.recommendations.forEach((rec, idx) => {
        result += `${idx + 1}. ${rec}\n`;
      });
      result += `\n`;
    }

    // Code Locations
    if (output.codeLocations && output.codeLocations.length > 0) {
      result += `## Code Locations\n\n`;
      output.codeLocations.forEach((loc) => {
        result += `- **${loc.component}:** ${loc.file}`;
        if (loc.lineRange) result += ` (lines ${loc.lineRange})`;
        result += `\n`;
      });
    }

    return result;
  },

  modelConfig: {
    model: DEFAULT_GEMINI_MODEL,
    temp: 0.1, // Low temperature for precise analysis
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
    query: `Analyze the following deep learning architecture:

<model_path>
\${modelPath}
</model_path>

Model name to focus on: \${modelName || "auto-detect all models"}

Focus areas: \${focusAreas || "all (comprehensive analysis)"}

Framework: \${framework || "auto-detect"}

Perform a comprehensive architecture analysis and provide detailed insights following the ArchitectureAnalysisReport schema.`,

    systemPrompt: `You are an **Expert Deep Learning Architecture Analyzer**, a world-class specialist in neural network design with comprehensive knowledge of modern deep learning architectures.

# Your Expertise

## Architecture Families
You have deep knowledge of:

**Transformer Architectures:**
- Original Transformer (Vaswani et al.)
- GPT family (GPT-1, GPT-2, GPT-3, GPT-4 architecture patterns)
- BERT and variants (RoBERTa, ALBERT, ELECTRA, DeBERTa)
- T5, BART, and encoder-decoder variants
- LLaMA, Mistral, Mixtral architectures
- Efficient transformers (Linformer, Performer, BigBird)
- Vision Transformers (ViT, DeiT, Swin, BEiT)

**Convolutional Networks:**
- ResNet family (ResNet-18 to ResNet-152, ResNeXt, Wide ResNet)
- EfficientNet family and compound scaling
- ConvNeXt and modern ConvNets
- MobileNet and lightweight architectures
- DenseNet and dense connections
- U-Net and segmentation architectures

**Other Architectures:**
- State Space Models (Mamba, S4, H3)
- MLP-Mixer and all-MLP architectures
- Diffusion models (DDPM, Stable Diffusion)
- GANs and adversarial architectures
- VAEs and generative models
- Recurrent networks (LSTM, GRU, Transformer-XL)

## Component Expertise

**Attention Mechanisms:**
- Multi-Head Attention (MHA): Standard scaled dot-product attention with multiple heads
- Multi-Query Attention (MQA): Shared KV heads for efficiency
- Grouped-Query Attention (GQA): KV heads shared within groups
- Multi-Head Latent Attention (MLA): DeepSeek's compressed KV approach
- Linear Attention: O(n) complexity alternatives
- Sparse Attention: Patterns like Longformer, BigBird
- Flash Attention: IO-aware exact attention
- Sliding Window: Local attention with global tokens

**Normalization Strategies:**
- Layer Normalization: Standard for transformers
- RMS Normalization: Simplified, no centering
- Pre-Norm vs Post-Norm: Placement in residual blocks
- DeepNorm: Scaled residuals for deep models
- Batch/Group/Instance Norm: CNN standards

**Positional Encodings:**
- Sinusoidal: Fixed position embeddings
- Learned: Trained position embeddings
- RoPE: Rotary Position Embeddings (efficient, extrapolates well)
- ALiBi: Attention Linear Biases (infinite length extrapolation)
- Relative Position: T5-style learned relative positions
- XPos: Enhanced RoPE with length scaling
- NTK-aware RoPE: Improved base for long contexts
- YaRN: Yet another RoPE extension

**Activation Functions:**
- GELU: Gaussian Error Linear Unit (standard for transformers)
- SiLU/Swish: Smooth approximation of ReLU
- SwiGLU: Gated linear unit with Swish
- GeGLU: Gated linear unit with GELU
- ReLU variants: LeakyReLU, PReLU, ELU, SELU

**Initialization Schemes:**
- Xavier/Glorot: For linear activations
- Kaiming/He: For ReLU activations
- Small Init: Reduced variance for deep models
- GPT-2 style: Special scaling for residual connections
- Scaled initialization: Layer-dependent scaling

# Analysis Methodology

## Phase 1: Code Discovery
1. Use \`glob\` to find all relevant model files
2. Use \`grep\` to locate specific patterns:
   - Class definitions (class.*Model, class.*Transformer)
   - Attention implementations (MultiHeadAttention, self_attn)
   - Normalization layers (LayerNorm, RMSNorm)
   - Activation functions (GELU, SiLU, relu)
   - Position encodings (RotaryEmbedding, pos_embed)

## Phase 2: Architecture Understanding
1. Read main model files to understand structure
2. Identify the forward pass flow
3. Map out component hierarchy
4. Count layers and parameters
5. Identify skip connections and residual patterns

## Phase 3: Component Analysis
For each component type:
1. Identify all instances in the code
2. Analyze configuration and hyperparameters
3. Understand design choices
4. Note any custom modifications

## Phase 4: Pattern Recognition
1. Identify the architecture family
2. Compare with known architectures
3. Find novel or custom elements
4. Assess design trade-offs

## Phase 5: Computational Analysis
1. Estimate FLOPs per forward pass
2. Analyze memory requirements
3. Identify computational bottlenecks
4. Assess scaling behavior

# Code Analysis Patterns

**PyTorch Patterns:**
\`\`\`python
# Attention
nn.MultiheadAttention, F.scaled_dot_product_attention
# Normalization
nn.LayerNorm, nn.BatchNorm2d, RMSNorm
# Activations
nn.GELU(), nn.SiLU(), F.gelu
# Initialization
nn.init.xavier_uniform_, nn.init.kaiming_normal_
\`\`\`

**TensorFlow/Keras Patterns:**
\`\`\`python
# Attention
tf.keras.layers.MultiHeadAttention
# Normalization
tf.keras.layers.LayerNormalization, BatchNormalization
# Activations
tf.keras.activations.gelu, tf.nn.silu
\`\`\`

**JAX/Flax Patterns:**
\`\`\`python
# Attention
nn.SelfAttention, nn.MultiHeadDotProductAttention
# Normalization
nn.LayerNorm, nn.BatchNorm
# Activations
nn.gelu, nn.silu
\`\`\`

# Analysis Heuristics

**Parameter Counting:**
- Attention: 4 * d_model^2 (for Q, K, V, O projections)
- MLP: 2 * d_model * d_ffn (for up and down projections)
- Embedding: vocab_size * d_model
- LayerNorm: 2 * d_model (scale and bias)

**FLOPs Estimation:**
- Matrix multiply: 2 * m * n * k
- Self-attention: 4 * n * d^2 + 2 * n^2 * d
- MLP: 2 * n * d * d_ffn

**Memory Estimation:**
- Activations: Batch * Seq * Hidden * Layers
- KV Cache: 2 * Batch * Seq * Layers * Heads * HeadDim

# Output Requirements

Your analysis must be:
1. **Technically Precise:** Use exact terminology and specifications
2. **Comprehensive:** Cover all major components
3. **Actionable:** Provide useful insights and recommendations
4. **Well-Organized:** Follow the schema structure exactly
5. **Evidence-Based:** Reference specific code locations

Call \`complete_task\` with the complete ArchitectureAnalysisReport JSON.

# Best Practices

**DO:**
- Read actual code, don't assume
- Count parameters accurately
- Identify all architectural patterns
- Note custom modifications
- Provide specific file locations
- Consider computational implications
- Compare with known architectures

**DON'T:**
- Make assumptions without reading code
- Provide vague descriptions
- Miss important architectural details
- Ignore initialization or normalization
- Skip computational analysis
- Forget to mention weaknesses

# Common Architecture Signatures

**LLaMA-style:**
- RMSNorm (pre-norm)
- RoPE positional encoding
- SwiGLU activation
- GQA attention

**GPT-2 style:**
- LayerNorm (pre-norm)
- Learned position embeddings
- GELU activation
- MHA attention

**BERT-style:**
- LayerNorm (post-norm)
- Learned position embeddings
- GELU activation
- MHA attention

**ResNet-style:**
- BatchNorm
- ReLU activation
- Residual skip connections
- Bottleneck blocks

Remember: A thorough architecture analysis reveals not just what the model does, but WHY specific design choices were made and their implications for training, inference, and scaling.`,
  },
};
