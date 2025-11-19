/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type { MessageBus } from '../../confirmation-bus/message-bus.js';
import path from 'node:path';
import fs from 'node:fs/promises';
import type { ToolInvocation, ToolResult } from '../tools.js';
import { BaseDeclarativeTool, BaseToolInvocation, Kind } from '../tools.js';
import { ToolErrorType } from '../tool-error.js';
import type { Config } from '../../config/config.js';

/**
 * Parameters for the Neural Network Analyzer tool
 */
export interface NeuralNetworkAnalyzerParams {
  /**
   * Path to the model definition file or directory
   */
  model_path: string;

  /**
   * Framework to analyze (pytorch, tensorflow, jax, auto)
   */
  framework?: 'pytorch' | 'tensorflow' | 'jax' | 'auto';

  /**
   * Input shape for FLOPs calculation (e.g., [1, 3, 224, 224])
   */
  input_shape?: number[];

  /**
   * Whether to generate architecture visualization
   */
  generate_visualization?: boolean;

  /**
   * Visualization format (ascii, mermaid)
   */
  visualization_format?: 'ascii' | 'mermaid';

  /**
   * Whether to analyze memory footprint
   */
  analyze_memory?: boolean;

  /**
   * Whether to detect architecture patterns
   */
  detect_patterns?: boolean;
}

/**
 * Layer information structure
 */
interface LayerInfo {
  name: string;
  type: string;
  inputShape: string;
  outputShape: string;
  params: number;
  trainableParams: number;
  flops: number;
  memoryMB: number;
}

/**
 * Architecture pattern structure
 */
interface ArchitecturePattern {
  name: string;
  description: string;
  locations: string[];
  confidence: number;
}

/**
 * Analysis result structure
 */
interface AnalysisResult {
  modelName: string;
  framework: string;
  layers: LayerInfo[];
  totalParams: number;
  trainableParams: number;
  totalFlops: number;
  totalMemoryMB: number;
  patterns: ArchitecturePattern[];
  visualization?: string;
  summary: string;
}

class NeuralNetworkAnalyzerInvocation extends BaseToolInvocation<
  NeuralNetworkAnalyzerParams,
  ToolResult
> {
  constructor(
    private readonly config: Config,
    params: NeuralNetworkAnalyzerParams,
    messageBus?: MessageBus,
    _toolName?: string,
    _toolDisplayName?: string,
  ) {
    super(params, messageBus, _toolName, _toolDisplayName);
  }

  getDescription(): string {
    return `Analyzing neural network architecture: ${this.params.model_path}`;
  }

  async execute(): Promise<ToolResult> {
    try {
      const resolvedPath = path.resolve(
        this.config.getTargetDir(),
        this.params.model_path,
      );

      // Read the model file
      let content: string;
      try {
        content = await fs.readFile(resolvedPath, 'utf-8');
      } catch (error) {
        return {
          llmContent: `Error: Could not read model file at ${resolvedPath}`,
          returnDisplay: 'Failed to read model file',
          error: {
            message: `Could not read model file: ${error}`,
            type: ToolErrorType.FILE_NOT_FOUND,
          },
        };
      }

      // Detect framework if auto
      const framework = this.params.framework === 'auto' || !this.params.framework
        ? this.detectFramework(content)
        : this.params.framework;

      // Parse and analyze the model
      const analysis = await this.analyzeModel(content, framework, resolvedPath);

      // Generate visualization if requested
      if (this.params.generate_visualization) {
        analysis.visualization = this.generateVisualization(
          analysis,
          this.params.visualization_format || 'ascii',
        );
      }

      // Format the output
      const output = this.formatOutput(analysis);

      return {
        llmContent: output,
        returnDisplay: `Analyzed ${analysis.modelName}: ${analysis.totalParams.toLocaleString()} params, ${this.formatFlops(analysis.totalFlops)}`,
      };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      return {
        llmContent: `Error analyzing neural network: ${errorMessage}`,
        returnDisplay: `Error: ${errorMessage}`,
        error: {
          message: errorMessage,
          type: ToolErrorType.EXECUTION_FAILED,
        },
      };
    }
  }

  private detectFramework(content: string): 'pytorch' | 'tensorflow' | 'jax' {
    if (content.includes('torch.nn') || content.includes('nn.Module') || content.includes('import torch')) {
      return 'pytorch';
    } else if (content.includes('tensorflow') || content.includes('tf.keras') || content.includes('keras.layers')) {
      return 'tensorflow';
    } else if (content.includes('import jax') || content.includes('flax.linen') || content.includes('haiku')) {
      return 'jax';
    }
    return 'pytorch'; // Default
  }

  private async analyzeModel(
    content: string,
    framework: string,
    filePath: string,
  ): Promise<AnalysisResult> {
    const layers: LayerInfo[] = [];
    const patterns: ArchitecturePattern[] = [];

    // Extract model name from file
    const modelName = path.basename(filePath, path.extname(filePath));

    // Parse layers based on framework
    if (framework === 'pytorch') {
      this.parsePyTorchLayers(content, layers);
    } else if (framework === 'tensorflow') {
      this.parseTensorFlowLayers(content, layers);
    } else if (framework === 'jax') {
      this.parseJAXLayers(content, layers);
    }

    // Detect architecture patterns if requested
    if (this.params.detect_patterns !== false) {
      this.detectPatterns(content, layers, patterns);
    }

    // Calculate totals
    const totalParams = layers.reduce((sum, l) => sum + l.params, 0);
    const trainableParams = layers.reduce((sum, l) => sum + l.trainableParams, 0);
    const totalFlops = layers.reduce((sum, l) => sum + l.flops, 0);
    const totalMemoryMB = layers.reduce((sum, l) => sum + l.memoryMB, 0);

    // Generate summary
    const summary = this.generateSummary(layers, totalParams, totalFlops, patterns);

    return {
      modelName,
      framework,
      layers,
      totalParams,
      trainableParams,
      totalFlops,
      totalMemoryMB,
      patterns,
      summary,
    };
  }

  private parsePyTorchLayers(content: string, layers: LayerInfo[]): void {
    // Parse nn.Conv2d
    const conv2dPattern = /nn\.Conv2d\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(?:kernel_size\s*=\s*)?(\d+|\(\d+,\s*\d+\))/g;
    let match;
    let layerIndex = 0;

    while ((match = conv2dPattern.exec(content)) !== null) {
      const inChannels = parseInt(match[1]);
      const outChannels = parseInt(match[2]);
      const kernelSize = match[3].includes('(')
        ? parseInt(match[3].match(/\d+/)![0])
        : parseInt(match[3]);

      const params = outChannels * (inChannels * kernelSize * kernelSize + 1);
      const inputShape = this.params.input_shape || [1, inChannels, 224, 224];
      const outputH = inputShape[2]; // Simplified
      const outputW = inputShape[3];
      const flops = 2 * inChannels * kernelSize * kernelSize * outChannels * outputH * outputW;

      layers.push({
        name: `conv2d_${layerIndex++}`,
        type: 'Conv2d',
        inputShape: `[B, ${inChannels}, H, W]`,
        outputShape: `[B, ${outChannels}, H', W']`,
        params,
        trainableParams: params,
        flops,
        memoryMB: (params * 4) / (1024 * 1024),
      });
    }

    // Parse nn.Linear
    const linearPattern = /nn\.Linear\s*\(\s*(\d+)\s*,\s*(\d+)/g;
    while ((match = linearPattern.exec(content)) !== null) {
      const inFeatures = parseInt(match[1]);
      const outFeatures = parseInt(match[2]);
      const params = inFeatures * outFeatures + outFeatures;
      const flops = 2 * inFeatures * outFeatures;

      layers.push({
        name: `linear_${layerIndex++}`,
        type: 'Linear',
        inputShape: `[B, ${inFeatures}]`,
        outputShape: `[B, ${outFeatures}]`,
        params,
        trainableParams: params,
        flops,
        memoryMB: (params * 4) / (1024 * 1024),
      });
    }

    // Parse nn.BatchNorm2d
    const bnPattern = /nn\.BatchNorm2d\s*\(\s*(\d+)/g;
    while ((match = bnPattern.exec(content)) !== null) {
      const numFeatures = parseInt(match[1]);
      const params = numFeatures * 4; // gamma, beta, running_mean, running_var

      layers.push({
        name: `batchnorm_${layerIndex++}`,
        type: 'BatchNorm2d',
        inputShape: `[B, ${numFeatures}, H, W]`,
        outputShape: `[B, ${numFeatures}, H, W]`,
        params,
        trainableParams: numFeatures * 2,
        flops: numFeatures * 2,
        memoryMB: (params * 4) / (1024 * 1024),
      });
    }

    // Parse nn.LayerNorm
    const lnPattern = /nn\.LayerNorm\s*\(\s*(\d+|\[[\d,\s]+\])/g;
    while ((match = lnPattern.exec(content)) !== null) {
      const sizeStr = match[1];
      const size = sizeStr.includes('[')
        ? sizeStr.match(/\d+/g)!.reduce((a, b) => parseInt(a.toString()) * parseInt(b), 1) as number
        : parseInt(sizeStr);
      const params = size * 2;

      layers.push({
        name: `layernorm_${layerIndex++}`,
        type: 'LayerNorm',
        inputShape: `[B, *, ${size}]`,
        outputShape: `[B, *, ${size}]`,
        params,
        trainableParams: params,
        flops: size * 5,
        memoryMB: (params * 4) / (1024 * 1024),
      });
    }

    // Parse nn.MultiheadAttention
    const mhaPattern = /nn\.MultiheadAttention\s*\(\s*(?:embed_dim\s*=\s*)?(\d+)\s*,\s*(?:num_heads\s*=\s*)?(\d+)/g;
    while ((match = mhaPattern.exec(content)) !== null) {
      const embedDim = parseInt(match[1]);
      const numHeads = parseInt(match[2]);
      const params = 4 * embedDim * embedDim + 4 * embedDim; // Q, K, V, O projections
      const seqLen = 512; // Default assumption
      const flops = 4 * seqLen * embedDim * embedDim + 2 * seqLen * seqLen * embedDim;

      layers.push({
        name: `multihead_attention_${layerIndex++}`,
        type: 'MultiheadAttention',
        inputShape: `[B, S, ${embedDim}]`,
        outputShape: `[B, S, ${embedDim}]`,
        params,
        trainableParams: params,
        flops,
        memoryMB: (params * 4) / (1024 * 1024),
      });
    }

    // Parse nn.LSTM
    const lstmPattern = /nn\.LSTM\s*\(\s*(\d+)\s*,\s*(\d+)/g;
    while ((match = lstmPattern.exec(content)) !== null) {
      const inputSize = parseInt(match[1]);
      const hiddenSize = parseInt(match[2]);
      const params = 4 * hiddenSize * (inputSize + hiddenSize + 1) + 4 * hiddenSize;

      layers.push({
        name: `lstm_${layerIndex++}`,
        type: 'LSTM',
        inputShape: `[B, S, ${inputSize}]`,
        outputShape: `[B, S, ${hiddenSize}]`,
        params,
        trainableParams: params,
        flops: params * 2,
        memoryMB: (params * 4) / (1024 * 1024),
      });
    }

    // Parse nn.Embedding
    const embPattern = /nn\.Embedding\s*\(\s*(\d+)\s*,\s*(\d+)/g;
    while ((match = embPattern.exec(content)) !== null) {
      const numEmbeddings = parseInt(match[1]);
      const embeddingDim = parseInt(match[2]);
      const params = numEmbeddings * embeddingDim;

      layers.push({
        name: `embedding_${layerIndex++}`,
        type: 'Embedding',
        inputShape: `[B, S]`,
        outputShape: `[B, S, ${embeddingDim}]`,
        params,
        trainableParams: params,
        flops: 0, // Lookup only
        memoryMB: (params * 4) / (1024 * 1024),
      });
    }
  }

  private parseTensorFlowLayers(content: string, layers: LayerInfo[]): void {
    let layerIndex = 0;
    let match;

    // Parse tf.keras.layers.Conv2D
    const conv2dPattern = /(?:layers\.)?Conv2D\s*\(\s*(\d+)\s*,\s*(?:\()?(\d+)/g;
    while ((match = conv2dPattern.exec(content)) !== null) {
      const filters = parseInt(match[1]);
      const kernelSize = parseInt(match[2]);
      const inChannels = 3; // Default assumption
      const params = filters * (inChannels * kernelSize * kernelSize + 1);

      layers.push({
        name: `conv2d_${layerIndex++}`,
        type: 'Conv2D',
        inputShape: `[B, H, W, C]`,
        outputShape: `[B, H', W', ${filters}]`,
        params,
        trainableParams: params,
        flops: params * 100, // Simplified
        memoryMB: (params * 4) / (1024 * 1024),
      });
    }

    // Parse tf.keras.layers.Dense
    const densePattern = /(?:layers\.)?Dense\s*\(\s*(\d+)/g;
    while ((match = densePattern.exec(content)) !== null) {
      const units = parseInt(match[1]);
      const inFeatures = 512; // Default assumption
      const params = inFeatures * units + units;

      layers.push({
        name: `dense_${layerIndex++}`,
        type: 'Dense',
        inputShape: `[B, ${inFeatures}]`,
        outputShape: `[B, ${units}]`,
        params,
        trainableParams: params,
        flops: 2 * params,
        memoryMB: (params * 4) / (1024 * 1024),
      });
    }

    // Parse BatchNormalization
    const bnPattern = /(?:layers\.)?BatchNormalization\s*\(/g;
    while ((match = bnPattern.exec(content)) !== null) {
      const features = 64; // Default assumption
      const params = features * 4;

      layers.push({
        name: `batchnorm_${layerIndex++}`,
        type: 'BatchNormalization',
        inputShape: `[B, ...]`,
        outputShape: `[B, ...]`,
        params,
        trainableParams: features * 2,
        flops: features * 2,
        memoryMB: (params * 4) / (1024 * 1024),
      });
    }

    // Parse MultiHeadAttention
    const mhaPattern = /(?:layers\.)?MultiHeadAttention\s*\(\s*(?:num_heads\s*=\s*)?(\d+)\s*,\s*(?:key_dim\s*=\s*)?(\d+)/g;
    while ((match = mhaPattern.exec(content)) !== null) {
      const numHeads = parseInt(match[1]);
      const keyDim = parseInt(match[2]);
      const embedDim = numHeads * keyDim;
      const params = 4 * embedDim * embedDim;

      layers.push({
        name: `multihead_attention_${layerIndex++}`,
        type: 'MultiHeadAttention',
        inputShape: `[B, S, ${embedDim}]`,
        outputShape: `[B, S, ${embedDim}]`,
        params,
        trainableParams: params,
        flops: params * 4,
        memoryMB: (params * 4) / (1024 * 1024),
      });
    }
  }

  private parseJAXLayers(content: string, layers: LayerInfo[]): void {
    let layerIndex = 0;
    let match;

    // Parse flax.linen.Conv
    const convPattern = /(?:nn\.)?Conv\s*\(\s*features\s*=\s*(\d+)\s*,\s*kernel_size\s*=\s*\((\d+)/g;
    while ((match = convPattern.exec(content)) !== null) {
      const features = parseInt(match[1]);
      const kernelSize = parseInt(match[2]);
      const inFeatures = 3;
      const params = features * (inFeatures * kernelSize * kernelSize + 1);

      layers.push({
        name: `conv_${layerIndex++}`,
        type: 'Conv',
        inputShape: `[B, H, W, C]`,
        outputShape: `[B, H', W', ${features}]`,
        params,
        trainableParams: params,
        flops: params * 100,
        memoryMB: (params * 4) / (1024 * 1024),
      });
    }

    // Parse flax.linen.Dense
    const densePattern = /(?:nn\.)?Dense\s*\(\s*features\s*=\s*(\d+)/g;
    while ((match = densePattern.exec(content)) !== null) {
      const features = parseInt(match[1]);
      const inFeatures = 512;
      const params = inFeatures * features + features;

      layers.push({
        name: `dense_${layerIndex++}`,
        type: 'Dense',
        inputShape: `[B, ${inFeatures}]`,
        outputShape: `[B, ${features}]`,
        params,
        trainableParams: params,
        flops: 2 * params,
        memoryMB: (params * 4) / (1024 * 1024),
      });
    }

    // Parse flax.linen.LayerNorm
    const lnPattern = /(?:nn\.)?LayerNorm\s*\(/g;
    while ((match = lnPattern.exec(content)) !== null) {
      const size = 768; // Default assumption
      const params = size * 2;

      layers.push({
        name: `layernorm_${layerIndex++}`,
        type: 'LayerNorm',
        inputShape: `[B, *, ${size}]`,
        outputShape: `[B, *, ${size}]`,
        params,
        trainableParams: params,
        flops: size * 5,
        memoryMB: (params * 4) / (1024 * 1024),
      });
    }
  }

  private detectPatterns(
    content: string,
    layers: LayerInfo[],
    patterns: ArchitecturePattern[],
  ): void {
    // Detect skip connections / residual connections
    if (content.includes('+ x') || content.includes('+= ') || content.includes('residual')) {
      patterns.push({
        name: 'Skip Connections',
        description: 'Residual connections that bypass one or more layers, helping with gradient flow',
        locations: ['Detected via addition operations with identity mappings'],
        confidence: 0.85,
      });
    }

    // Detect attention mechanisms
    if (content.includes('attention') || content.includes('Attention') ||
        content.includes('MultiheadAttention') || content.includes('softmax')) {
      patterns.push({
        name: 'Attention Mechanism',
        description: 'Self-attention or cross-attention layers for capturing long-range dependencies',
        locations: ['Attention-related layers detected'],
        confidence: 0.95,
      });
    }

    // Detect normalization patterns
    const hasLayerNorm = content.includes('LayerNorm');
    const hasBatchNorm = content.includes('BatchNorm');
    if (hasLayerNorm && hasBatchNorm) {
      patterns.push({
        name: 'Mixed Normalization',
        description: 'Uses both LayerNorm and BatchNorm for different parts of the network',
        locations: ['Throughout the model'],
        confidence: 0.9,
      });
    } else if (hasLayerNorm) {
      patterns.push({
        name: 'Layer Normalization',
        description: 'Uses LayerNorm, common in transformers and sequence models',
        locations: ['Throughout the model'],
        confidence: 0.95,
      });
    } else if (hasBatchNorm) {
      patterns.push({
        name: 'Batch Normalization',
        description: 'Uses BatchNorm, common in CNNs for training stability',
        locations: ['Throughout the model'],
        confidence: 0.95,
      });
    }

    // Detect dropout
    if (content.includes('Dropout') || content.includes('dropout')) {
      patterns.push({
        name: 'Dropout Regularization',
        description: 'Uses dropout for regularization during training',
        locations: ['Multiple layers'],
        confidence: 0.9,
      });
    }

    // Detect encoder-decoder architecture
    if ((content.includes('encoder') || content.includes('Encoder')) &&
        (content.includes('decoder') || content.includes('Decoder'))) {
      patterns.push({
        name: 'Encoder-Decoder Architecture',
        description: 'Sequence-to-sequence architecture with separate encoder and decoder',
        locations: ['Model structure'],
        confidence: 0.9,
      });
    }

    // Detect bottleneck blocks (common in ResNet)
    const conv1x1Count = (content.match(/kernel_size\s*=\s*1|Conv2d\s*\([^)]*,\s*1\s*\)/g) || []).length;
    const conv3x3Count = (content.match(/kernel_size\s*=\s*3|Conv2d\s*\([^)]*,\s*3\s*\)/g) || []).length;
    if (conv1x1Count >= 2 && conv3x3Count >= 1) {
      patterns.push({
        name: 'Bottleneck Blocks',
        description: 'Uses 1x1 convolutions for dimension reduction/expansion (ResNet-style)',
        locations: ['Convolutional blocks'],
        confidence: 0.8,
      });
    }

    // Detect depthwise separable convolutions
    if (content.includes('groups=') || content.includes('depthwise') || content.includes('SeparableConv')) {
      patterns.push({
        name: 'Depthwise Separable Convolutions',
        description: 'Efficient convolutions that separate spatial and channel operations',
        locations: ['Convolutional layers'],
        confidence: 0.85,
      });
    }

    // Detect positional encoding
    if (content.includes('positional') || content.includes('PositionalEncoding') || content.includes('pos_embed')) {
      patterns.push({
        name: 'Positional Encoding',
        description: 'Adds position information to embeddings for sequence modeling',
        locations: ['Input processing'],
        confidence: 0.9,
      });
    }
  }

  private generateSummary(
    layers: LayerInfo[],
    totalParams: number,
    totalFlops: number,
    patterns: ArchitecturePattern[],
  ): string {
    const layerTypes = layers.reduce((acc, l) => {
      acc[l.type] = (acc[l.type] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    let summary = `Model Summary:\n`;
    summary += `- Total Layers: ${layers.length}\n`;
    summary += `- Total Parameters: ${totalParams.toLocaleString()} (${(totalParams / 1e6).toFixed(2)}M)\n`;
    summary += `- Total FLOPs: ${this.formatFlops(totalFlops)}\n`;
    summary += `- Layer Distribution:\n`;

    for (const [type, count] of Object.entries(layerTypes)) {
      summary += `  - ${type}: ${count}\n`;
    }

    if (patterns.length > 0) {
      summary += `- Detected Patterns:\n`;
      for (const pattern of patterns) {
        summary += `  - ${pattern.name} (${(pattern.confidence * 100).toFixed(0)}% confidence)\n`;
      }
    }

    return summary;
  }

  private generateVisualization(
    analysis: AnalysisResult,
    format: 'ascii' | 'mermaid',
  ): string {
    if (format === 'mermaid') {
      return this.generateMermaidVisualization(analysis);
    }
    return this.generateAsciiVisualization(analysis);
  }

  private generateAsciiVisualization(analysis: AnalysisResult): string {
    let viz = '\n' + '='.repeat(80) + '\n';
    viz += `ARCHITECTURE: ${analysis.modelName}\n`;
    viz += '='.repeat(80) + '\n\n';

    viz += 'Input\n';
    viz += '  |\n';
    viz += '  v\n';

    for (let i = 0; i < analysis.layers.length; i++) {
      const layer = analysis.layers[i];
      const box = '+' + '-'.repeat(40) + '+';
      const content = `| ${layer.type.padEnd(20)} ${this.formatParams(layer.params).padStart(16)} |`;

      viz += box + '\n';
      viz += content + '\n';
      viz += box + '\n';

      if (i < analysis.layers.length - 1) {
        viz += '  |\n';
        viz += '  v\n';
      }
    }

    viz += '\n  |\n';
    viz += '  v\n';
    viz += 'Output\n';

    return viz;
  }

  private generateMermaidVisualization(analysis: AnalysisResult): string {
    let mermaid = '```mermaid\ngraph TD\n';
    mermaid += '    Input([Input])\n';

    const nodeIds: string[] = [];
    for (let i = 0; i < analysis.layers.length; i++) {
      const layer = analysis.layers[i];
      const nodeId = `L${i}`;
      nodeIds.push(nodeId);
      mermaid += `    ${nodeId}["${layer.type}<br/>${this.formatParams(layer.params)}"]\n`;
    }

    mermaid += '    Output([Output])\n\n';

    // Add connections
    mermaid += `    Input --> ${nodeIds[0]}\n`;
    for (let i = 0; i < nodeIds.length - 1; i++) {
      mermaid += `    ${nodeIds[i]} --> ${nodeIds[i + 1]}\n`;
    }
    mermaid += `    ${nodeIds[nodeIds.length - 1]} --> Output\n`;

    mermaid += '```';
    return mermaid;
  }

  private formatParams(params: number): string {
    if (params >= 1e9) return `${(params / 1e9).toFixed(2)}B`;
    if (params >= 1e6) return `${(params / 1e6).toFixed(2)}M`;
    if (params >= 1e3) return `${(params / 1e3).toFixed(2)}K`;
    return params.toString();
  }

  private formatFlops(flops: number): string {
    if (flops >= 1e15) return `${(flops / 1e15).toFixed(2)} PFLOPs`;
    if (flops >= 1e12) return `${(flops / 1e12).toFixed(2)} TFLOPs`;
    if (flops >= 1e9) return `${(flops / 1e9).toFixed(2)} GFLOPs`;
    if (flops >= 1e6) return `${(flops / 1e6).toFixed(2)} MFLOPs`;
    if (flops >= 1e3) return `${(flops / 1e3).toFixed(2)} KFLOPs`;
    return `${flops} FLOPs`;
  }

  private formatOutput(analysis: AnalysisResult): string {
    let output = `# Neural Network Analysis: ${analysis.modelName}\n\n`;
    output += `**Framework**: ${analysis.framework}\n\n`;

    output += `## Summary\n`;
    output += `- **Total Parameters**: ${analysis.totalParams.toLocaleString()} (${this.formatParams(analysis.totalParams)})\n`;
    output += `- **Trainable Parameters**: ${analysis.trainableParams.toLocaleString()} (${this.formatParams(analysis.trainableParams)})\n`;
    output += `- **Total FLOPs**: ${this.formatFlops(analysis.totalFlops)}\n`;
    output += `- **Memory Footprint**: ${analysis.totalMemoryMB.toFixed(2)} MB\n\n`;

    output += `## Layer Details\n\n`;
    output += `| Layer | Type | Input Shape | Output Shape | Params | FLOPs |\n`;
    output += `|-------|------|-------------|--------------|--------|-------|\n`;

    for (const layer of analysis.layers) {
      output += `| ${layer.name} | ${layer.type} | ${layer.inputShape} | ${layer.outputShape} | ${this.formatParams(layer.params)} | ${this.formatFlops(layer.flops)} |\n`;
    }

    if (analysis.patterns.length > 0) {
      output += `\n## Architecture Patterns\n\n`;
      for (const pattern of analysis.patterns) {
        output += `### ${pattern.name}\n`;
        output += `- **Description**: ${pattern.description}\n`;
        output += `- **Confidence**: ${(pattern.confidence * 100).toFixed(0)}%\n`;
        output += `- **Locations**: ${pattern.locations.join(', ')}\n\n`;
      }
    }

    if (analysis.visualization) {
      output += `\n## Architecture Visualization\n\n`;
      output += analysis.visualization;
    }

    return output;
  }
}

/**
 * Tool for analyzing neural network architectures
 */
export class NeuralNetworkAnalyzerTool extends BaseDeclarativeTool<
  NeuralNetworkAnalyzerParams,
  ToolResult
> {
  static readonly Name = 'analyze_neural_network';

  constructor(
    private readonly config: Config,
    messageBus?: MessageBus,
  ) {
    super(
      NeuralNetworkAnalyzerTool.Name,
      'NeuralNetworkAnalyzer',
      `Analyzes neural network architectures from model definition files. Supports PyTorch, TensorFlow, and JAX frameworks.

Features:
- Parse model definitions and extract layer information
- Calculate parameter counts per layer and total
- Compute FLOPs/MACs for computational cost estimation
- Analyze memory footprint for deployment planning
- Detect architecture patterns (skip connections, attention, normalization, etc.)
- Generate architecture visualizations in ASCII or Mermaid format

Use this tool to understand model complexity, compare architectures, and identify potential optimizations.`,
      Kind.Read,
      {
        properties: {
          model_path: {
            description: 'Path to the model definition file (e.g., model.py, network.py)',
            type: 'string',
          },
          framework: {
            description: 'Deep learning framework (pytorch, tensorflow, jax, or auto for automatic detection)',
            type: 'string',
            enum: ['pytorch', 'tensorflow', 'jax', 'auto'],
          },
          input_shape: {
            description: 'Input tensor shape for FLOPs calculation (e.g., [1, 3, 224, 224] for image classification)',
            type: 'array',
            items: { type: 'number' },
          },
          generate_visualization: {
            description: 'Whether to generate architecture visualization',
            type: 'boolean',
          },
          visualization_format: {
            description: 'Format for visualization (ascii or mermaid)',
            type: 'string',
            enum: ['ascii', 'mermaid'],
          },
          analyze_memory: {
            description: 'Whether to analyze memory footprint',
            type: 'boolean',
          },
          detect_patterns: {
            description: 'Whether to detect architecture patterns like skip connections, attention, etc.',
            type: 'boolean',
          },
        },
        required: ['model_path'],
        type: 'object',
      },
      true,
      false,
      messageBus,
    );
  }

  protected override validateToolParamValues(
    params: NeuralNetworkAnalyzerParams,
  ): string | null {
    if (!params.model_path || params.model_path.trim() === '') {
      return "The 'model_path' parameter must be non-empty.";
    }

    if (params.input_shape && !Array.isArray(params.input_shape)) {
      return "The 'input_shape' parameter must be an array of numbers.";
    }

    if (params.visualization_format &&
        !['ascii', 'mermaid'].includes(params.visualization_format)) {
      return "The 'visualization_format' must be 'ascii' or 'mermaid'.";
    }

    return null;
  }

  protected createInvocation(
    params: NeuralNetworkAnalyzerParams,
    messageBus?: MessageBus,
    _toolName?: string,
    _toolDisplayName?: string,
  ): ToolInvocation<NeuralNetworkAnalyzerParams, ToolResult> {
    return new NeuralNetworkAnalyzerInvocation(
      this.config,
      params,
      messageBus,
      _toolName,
      _toolDisplayName,
    );
  }
}
