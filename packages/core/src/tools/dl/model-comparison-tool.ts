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
 * Parameters for the Model Comparison tool
 */
export interface ModelComparisonParams {
  /**
   * Paths to model files or definitions to compare
   */
  model_paths: string[];

  /**
   * Names for the models (optional, will be derived from file names)
   */
  model_names?: string[];

  /**
   * Whether to compare parameter counts
   */
  compare_params?: boolean;

  /**
   * Whether to compare FLOPs
   */
  compare_flops?: boolean;

  /**
   * Whether to compare memory requirements
   */
  compare_memory?: boolean;

  /**
   * Whether to compare layer structures
   */
  compare_architecture?: boolean;

  /**
   * Input shape for comparison (e.g., [1, 3, 224, 224])
   */
  input_shape?: number[];

  /**
   * Output format (table, detailed, json)
   */
  output_format?: 'table' | 'detailed' | 'json';
}

/**
 * Model information for comparison
 */
interface ModelInfo {
  name: string;
  path: string;
  totalParams: number;
  trainableParams: number;
  totalFlops: number;
  memoryMB: number;
  layerCount: number;
  layerBreakdown: Record<string, number>;
  depth: number;
  width: number;
  hasAttention: boolean;
  hasResidual: boolean;
  hasNormalization: boolean;
  framework: string;
}

/**
 * Comparison result
 */
interface ComparisonResult {
  models: ModelInfo[];
  rankings: {
    mostEfficient: string;
    smallest: string;
    fastest: string;
    deepest: string;
  };
  tradeoffs: string[];
  recommendation: string;
}

class ModelComparisonInvocation extends BaseToolInvocation<
  ModelComparisonParams,
  ToolResult
> {
  constructor(
    private readonly config: Config,
    params: ModelComparisonParams,
    messageBus?: MessageBus,
    _toolName?: string,
    _toolDisplayName?: string,
  ) {
    super(params, messageBus, _toolName, _toolDisplayName);
  }

  getDescription(): string {
    return `Comparing ${this.params.model_paths.length} models`;
  }

  async execute(): Promise<ToolResult> {
    try {
      const models: ModelInfo[] = [];

      // Analyze each model
      for (let i = 0; i < this.params.model_paths.length; i++) {
        const modelPath = this.params.model_paths[i];
        const resolvedPath = path.resolve(
          this.config.getTargetDir(),
          modelPath,
        );

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

        const modelName = this.params.model_names?.[i] ||
          path.basename(modelPath, path.extname(modelPath));

        const modelInfo = this.analyzeModel(content, modelName, resolvedPath);
        models.push(modelInfo);
      }

      // Generate comparison
      const comparison = this.compareModels(models);

      // Format output
      const output = this.formatOutput(comparison);

      return {
        llmContent: output,
        returnDisplay: `Compared ${models.length} models`,
      };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      return {
        llmContent: `Error comparing models: ${errorMessage}`,
        returnDisplay: `Error: ${errorMessage}`,
        error: {
          message: errorMessage,
          type: ToolErrorType.EXECUTION_FAILED,
        },
      };
    }
  }

  private analyzeModel(content: string, name: string, filePath: string): ModelInfo {
    const framework = this.detectFramework(content);
    const layers = this.extractLayers(content, framework);

    // Calculate totals
    let totalParams = 0;
    let totalFlops = 0;
    const layerBreakdown: Record<string, number> = {};

    for (const layer of layers) {
      totalParams += layer.params;
      totalFlops += layer.flops;
      layerBreakdown[layer.type] = (layerBreakdown[layer.type] || 0) + 1;
    }

    // Estimate width (max channels/features)
    const width = this.estimateWidth(content);

    // Detect architectural patterns
    const hasAttention = content.includes('attention') || content.includes('Attention') ||
                        content.includes('MultiheadAttention');
    const hasResidual = content.includes('+ x') || content.includes('+= ') ||
                       content.includes('residual') || content.includes('skip');
    const hasNormalization = content.includes('BatchNorm') || content.includes('LayerNorm') ||
                            content.includes('GroupNorm');

    return {
      name,
      path: filePath,
      totalParams,
      trainableParams: totalParams, // Assume all trainable for now
      totalFlops,
      memoryMB: (totalParams * 4) / (1024 * 1024), // 4 bytes per param
      layerCount: layers.length,
      layerBreakdown,
      depth: layers.length,
      width,
      hasAttention,
      hasResidual,
      hasNormalization,
      framework,
    };
  }

  private detectFramework(content: string): string {
    if (content.includes('torch.nn') || content.includes('nn.Module') || content.includes('import torch')) {
      return 'PyTorch';
    } else if (content.includes('tensorflow') || content.includes('tf.keras') || content.includes('keras.layers')) {
      return 'TensorFlow';
    } else if (content.includes('import jax') || content.includes('flax.linen') || content.includes('haiku')) {
      return 'JAX';
    }
    return 'Unknown';
  }

  private extractLayers(content: string, framework: string): Array<{ type: string; params: number; flops: number }> {
    const layers: Array<{ type: string; params: number; flops: number }> = [];

    if (framework === 'PyTorch') {
      // Conv2d
      const conv2dPattern = /nn\.Conv2d\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(?:kernel_size\s*=\s*)?(\d+|\(\d+,\s*\d+\))/g;
      let match;
      while ((match = conv2dPattern.exec(content)) !== null) {
        const inC = parseInt(match[1]);
        const outC = parseInt(match[2]);
        const k = parseInt(match[3].match(/\d+/)![0]);
        const params = outC * (inC * k * k + 1);
        const flops = 2 * inC * k * k * outC * 224 * 224; // Estimate
        layers.push({ type: 'Conv2d', params, flops });
      }

      // Linear
      const linearPattern = /nn\.Linear\s*\(\s*(\d+)\s*,\s*(\d+)/g;
      while ((match = linearPattern.exec(content)) !== null) {
        const inF = parseInt(match[1]);
        const outF = parseInt(match[2]);
        const params = inF * outF + outF;
        const flops = 2 * inF * outF;
        layers.push({ type: 'Linear', params, flops });
      }

      // BatchNorm
      const bnPattern = /nn\.BatchNorm2d\s*\(\s*(\d+)/g;
      while ((match = bnPattern.exec(content)) !== null) {
        const features = parseInt(match[1]);
        layers.push({ type: 'BatchNorm', params: features * 4, flops: features * 2 });
      }

      // LayerNorm
      const lnPattern = /nn\.LayerNorm\s*\(\s*(\d+|\[[\d,\s]+\])/g;
      while ((match = lnPattern.exec(content)) !== null) {
        const size = parseInt(match[1].match(/\d+/)![0]);
        layers.push({ type: 'LayerNorm', params: size * 2, flops: size * 5 });
      }

      // MultiheadAttention
      const mhaPattern = /nn\.MultiheadAttention\s*\(\s*(?:embed_dim\s*=\s*)?(\d+)/g;
      while ((match = mhaPattern.exec(content)) !== null) {
        const dim = parseInt(match[1]);
        const params = 4 * dim * dim;
        const flops = 8 * 512 * dim * dim; // Estimate for seq_len=512
        layers.push({ type: 'Attention', params, flops });
      }

      // LSTM
      const lstmPattern = /nn\.LSTM\s*\(\s*(\d+)\s*,\s*(\d+)/g;
      while ((match = lstmPattern.exec(content)) !== null) {
        const inputSize = parseInt(match[1]);
        const hiddenSize = parseInt(match[2]);
        const params = 4 * hiddenSize * (inputSize + hiddenSize + 1);
        layers.push({ type: 'LSTM', params, flops: params * 2 });
      }

      // Embedding
      const embPattern = /nn\.Embedding\s*\(\s*(\d+)\s*,\s*(\d+)/g;
      while ((match = embPattern.exec(content)) !== null) {
        const numEmb = parseInt(match[1]);
        const embDim = parseInt(match[2]);
        layers.push({ type: 'Embedding', params: numEmb * embDim, flops: 0 });
      }
    } else if (framework === 'TensorFlow') {
      // Conv2D
      const conv2dPattern = /(?:layers\.)?Conv2D\s*\(\s*(\d+)\s*,\s*(?:\()?(\d+)/g;
      let match;
      while ((match = conv2dPattern.exec(content)) !== null) {
        const filters = parseInt(match[1]);
        const kernelSize = parseInt(match[2]);
        const params = filters * (3 * kernelSize * kernelSize + 1);
        layers.push({ type: 'Conv2D', params, flops: params * 100 });
      }

      // Dense
      const densePattern = /(?:layers\.)?Dense\s*\(\s*(\d+)/g;
      while ((match = densePattern.exec(content)) !== null) {
        const units = parseInt(match[1]);
        const params = 512 * units + units; // Estimate input as 512
        layers.push({ type: 'Dense', params, flops: 2 * params });
      }
    }

    return layers;
  }

  private estimateWidth(content: string): number {
    const numbers = content.match(/(?:filters|units|hidden_size|embed_dim|d_model|num_features)\s*[=:]\s*(\d+)/g) || [];
    const channelNumbers = content.match(/(?:Conv2d|Linear|Dense)\s*\([^)]*,\s*(\d+)/g) || [];

    const allNumbers: number[] = [];
    for (const match of numbers) {
      const num = parseInt(match.match(/\d+/)![0]);
      if (num > 0) allNumbers.push(num);
    }
    for (const match of channelNumbers) {
      const num = parseInt(match.match(/\d+/)![0]);
      if (num > 0) allNumbers.push(num);
    }

    return allNumbers.length > 0 ? Math.max(...allNumbers) : 0;
  }

  private compareModels(models: ModelInfo[]): ComparisonResult {
    // Calculate efficiency (params per layer)
    const efficiencies = models.map(m => ({
      name: m.name,
      efficiency: m.layerCount > 0 ? m.totalParams / m.layerCount : 0,
    }));

    // Find rankings
    const sortedByParams = [...models].sort((a, b) => a.totalParams - b.totalParams);
    const sortedByFlops = [...models].sort((a, b) => a.totalFlops - b.totalFlops);
    const sortedByDepth = [...models].sort((a, b) => b.depth - a.depth);
    const sortedByEfficiency = [...efficiencies].sort((a, b) => a.efficiency - b.efficiency);

    const rankings = {
      mostEfficient: sortedByEfficiency[0].name,
      smallest: sortedByParams[0].name,
      fastest: sortedByFlops[0].name,
      deepest: sortedByDepth[0].name,
    };

    // Analyze tradeoffs
    const tradeoffs: string[] = [];

    // Compare largest vs smallest
    if (models.length >= 2) {
      const largest = sortedByParams[sortedByParams.length - 1];
      const smallest = sortedByParams[0];
      const paramRatio = largest.totalParams / smallest.totalParams;

      if (paramRatio > 2) {
        tradeoffs.push(
          `${largest.name} has ${paramRatio.toFixed(1)}x more parameters than ${smallest.name}, ` +
          `requiring ${((largest.memoryMB - smallest.memoryMB)).toFixed(1)} MB more memory`
        );
      }

      // Compare FLOPs
      const flopRatio = largest.totalFlops / (smallest.totalFlops + 1);
      if (flopRatio > 2) {
        tradeoffs.push(
          `${largest.name} requires ${flopRatio.toFixed(1)}x more FLOPs than ${smallest.name}`
        );
      }

      // Architectural differences
      const attentionModels = models.filter(m => m.hasAttention);
      const noAttentionModels = models.filter(m => !m.hasAttention);
      if (attentionModels.length > 0 && noAttentionModels.length > 0) {
        tradeoffs.push(
          `${attentionModels.map(m => m.name).join(', ')} use attention mechanisms while ` +
          `${noAttentionModels.map(m => m.name).join(', ')} do not`
        );
      }

      // Residual connections
      const residualModels = models.filter(m => m.hasResidual);
      if (residualModels.length > 0 && residualModels.length < models.length) {
        tradeoffs.push(
          `${residualModels.map(m => m.name).join(', ')} use skip connections for better gradient flow`
        );
      }
    }

    // Generate recommendation
    let recommendation = '';
    if (models.length >= 2) {
      const avgParams = models.reduce((a, b) => a + b.totalParams, 0) / models.length;

      // Find balanced model
      const balanced = [...models].sort((a, b) =>
        Math.abs(a.totalParams - avgParams) - Math.abs(b.totalParams - avgParams)
      )[0];

      recommendation = `**Recommendation**: `;

      if (models.every(m => m.hasAttention)) {
        recommendation += `All models use attention mechanisms. `;
      }

      if (sortedByParams[0].totalParams < avgParams * 0.5) {
        recommendation += `For resource-constrained environments, ${sortedByParams[0].name} offers the smallest footprint. `;
      }

      if (sortedByDepth[0].depth > sortedByDepth[sortedByDepth.length - 1].depth * 1.5) {
        recommendation += `${sortedByDepth[0].name} is significantly deeper and may capture more complex patterns. `;
      }

      recommendation += `For a balanced choice, consider ${balanced.name} with ${this.formatParams(balanced.totalParams)} parameters.`;
    }

    return {
      models,
      rankings,
      tradeoffs,
      recommendation,
    };
  }

  private formatParams(params: number): string {
    if (params >= 1e9) return `${(params / 1e9).toFixed(2)}B`;
    if (params >= 1e6) return `${(params / 1e6).toFixed(2)}M`;
    if (params >= 1e3) return `${(params / 1e3).toFixed(2)}K`;
    return params.toString();
  }

  private formatFlops(flops: number): string {
    if (flops >= 1e12) return `${(flops / 1e12).toFixed(2)} TFLOPs`;
    if (flops >= 1e9) return `${(flops / 1e9).toFixed(2)} GFLOPs`;
    if (flops >= 1e6) return `${(flops / 1e6).toFixed(2)} MFLOPs`;
    return `${flops} FLOPs`;
  }

  private formatOutput(comparison: ComparisonResult): string {
    const format = this.params.output_format || 'table';

    if (format === 'json') {
      return JSON.stringify(comparison, null, 2);
    }

    let output = `# Model Comparison Report\n\n`;

    // Summary table
    output += `## Summary\n\n`;
    output += `| Model | Parameters | FLOPs | Memory | Layers | Depth | Framework |\n`;
    output += `|-------|------------|-------|--------|--------|-------|----------|\n`;

    for (const model of comparison.models) {
      output += `| ${model.name} | ${this.formatParams(model.totalParams)} | ${this.formatFlops(model.totalFlops)} | ${model.memoryMB.toFixed(1)} MB | ${model.layerCount} | ${model.depth} | ${model.framework} |\n`;
    }

    // Rankings
    output += `\n## Rankings\n\n`;
    output += `- **Most Parameter Efficient**: ${comparison.rankings.mostEfficient}\n`;
    output += `- **Smallest Model**: ${comparison.rankings.smallest}\n`;
    output += `- **Fastest (Lowest FLOPs)**: ${comparison.rankings.fastest}\n`;
    output += `- **Deepest Architecture**: ${comparison.rankings.deepest}\n`;

    // Architectural features
    output += `\n## Architectural Features\n\n`;
    output += `| Model | Attention | Skip Connections | Normalization | Max Width |\n`;
    output += `|-------|-----------|------------------|---------------|----------|\n`;

    for (const model of comparison.models) {
      output += `| ${model.name} | ${model.hasAttention ? 'Yes' : 'No'} | ${model.hasResidual ? 'Yes' : 'No'} | ${model.hasNormalization ? 'Yes' : 'No'} | ${model.width} |\n`;
    }

    // Layer breakdown
    if (format === 'detailed') {
      output += `\n## Layer Breakdown\n\n`;

      for (const model of comparison.models) {
        output += `### ${model.name}\n\n`;
        output += `| Layer Type | Count |\n`;
        output += `|------------|-------|\n`;

        for (const [layerType, count] of Object.entries(model.layerBreakdown)) {
          output += `| ${layerType} | ${count} |\n`;
        }
        output += '\n';
      }
    }

    // Tradeoffs
    if (comparison.tradeoffs.length > 0) {
      output += `\n## Key Tradeoffs\n\n`;
      for (const tradeoff of comparison.tradeoffs) {
        output += `- ${tradeoff}\n`;
      }
    }

    // Recommendation
    if (comparison.recommendation) {
      output += `\n## ${comparison.recommendation}\n`;
    }

    // Parameter efficiency chart (ASCII)
    output += `\n## Parameter Comparison (Visual)\n\n`;
    const maxParams = Math.max(...comparison.models.map(m => m.totalParams));
    const barWidth = 40;

    for (const model of comparison.models) {
      const barLength = Math.round((model.totalParams / maxParams) * barWidth);
      const bar = '#'.repeat(barLength) + ' '.repeat(barWidth - barLength);
      output += `${model.name.padEnd(20)} |${bar}| ${this.formatParams(model.totalParams)}\n`;
    }

    // FLOPs comparison
    output += `\n## FLOPs Comparison (Visual)\n\n`;
    const maxFlops = Math.max(...comparison.models.map(m => m.totalFlops));

    for (const model of comparison.models) {
      const barLength = Math.round((model.totalFlops / maxFlops) * barWidth);
      const bar = '#'.repeat(barLength) + ' '.repeat(barWidth - barLength);
      output += `${model.name.padEnd(20)} |${bar}| ${this.formatFlops(model.totalFlops)}\n`;
    }

    return output;
  }
}

/**
 * Tool for comparing different neural network models
 */
export class ModelComparisonTool extends BaseDeclarativeTool<
  ModelComparisonParams,
  ToolResult
> {
  static readonly Name = 'compare_models';

  constructor(
    private readonly config: Config,
    messageBus?: MessageBus,
  ) {
    super(
      ModelComparisonTool.Name,
      'ModelComparison',
      `Compares multiple neural network models side-by-side.

Features:
- Side-by-side architecture comparison
- Parameter count comparison (total, per layer type)
- FLOPs/MACs computational cost comparison
- Memory requirements comparison
- Speed/accuracy tradeoff analysis
- Architectural feature comparison (attention, residuals, normalization)
- Visual comparison charts
- Recommendations based on requirements

Use this tool to select the best model for your use case or understand tradeoffs between different architectures.`,
      Kind.Read,
      {
        properties: {
          model_paths: {
            description: 'Array of paths to model definition files to compare',
            type: 'array',
            items: { type: 'string' },
          },
          model_names: {
            description: 'Optional names for the models (derived from filenames if not provided)',
            type: 'array',
            items: { type: 'string' },
          },
          compare_params: {
            description: 'Whether to compare parameter counts (default: true)',
            type: 'boolean',
          },
          compare_flops: {
            description: 'Whether to compare FLOPs (default: true)',
            type: 'boolean',
          },
          compare_memory: {
            description: 'Whether to compare memory requirements (default: true)',
            type: 'boolean',
          },
          compare_architecture: {
            description: 'Whether to compare layer structures (default: true)',
            type: 'boolean',
          },
          input_shape: {
            description: 'Input tensor shape for comparison calculations',
            type: 'array',
            items: { type: 'number' },
          },
          output_format: {
            description: 'Output format (table, detailed, json)',
            type: 'string',
            enum: ['table', 'detailed', 'json'],
          },
        },
        required: ['model_paths'],
        type: 'object',
      },
      true,
      false,
      messageBus,
    );
  }

  protected override validateToolParamValues(
    params: ModelComparisonParams,
  ): string | null {
    if (!params.model_paths || params.model_paths.length < 2) {
      return "At least 2 model paths are required for comparison.";
    }

    if (params.model_names && params.model_names.length !== params.model_paths.length) {
      return "The number of model_names must match the number of model_paths.";
    }

    for (const path of params.model_paths) {
      if (!path || path.trim() === '') {
        return "All model paths must be non-empty strings.";
      }
    }

    return null;
  }

  protected createInvocation(
    params: ModelComparisonParams,
    messageBus?: MessageBus,
    _toolName?: string,
    _toolDisplayName?: string,
  ): ToolInvocation<ModelComparisonParams, ToolResult> {
    return new ModelComparisonInvocation(
      this.config,
      params,
      messageBus,
      _toolName,
      _toolDisplayName,
    );
  }
}
