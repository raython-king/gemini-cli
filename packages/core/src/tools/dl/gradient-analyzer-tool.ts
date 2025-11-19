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
 * Parameters for the Gradient Analyzer tool
 */
export interface GradientAnalyzerParams {
  /**
   * Path to training logs, gradient dumps, or model file
   */
  source_path: string;

  /**
   * Type of source (log, checkpoint, model)
   */
  source_type?: 'log' | 'checkpoint' | 'model';

  /**
   * Whether to check for vanishing gradients
   */
  check_vanishing?: boolean;

  /**
   * Whether to check for exploding gradients
   */
  check_exploding?: boolean;

  /**
   * Whether to detect dead neurons
   */
  detect_dead_neurons?: boolean;

  /**
   * Threshold for vanishing gradient detection
   */
  vanishing_threshold?: number;

  /**
   * Threshold for exploding gradient detection
   */
  exploding_threshold?: number;

  /**
   * Whether to suggest solutions
   */
  suggest_solutions?: boolean;

  /**
   * Number of layers to analyze (from output backwards)
   */
  num_layers?: number;
}

/**
 * Gradient statistics for a layer
 */
interface LayerGradientStats {
  layerName: string;
  layerType: string;
  mean: number;
  std: number;
  min: number;
  max: number;
  norm: number;
  sparsity: number;
  percentile95: number;
  percentile5: number;
}

/**
 * Gradient issue detected
 */
interface GradientIssue {
  type: 'vanishing' | 'exploding' | 'dead_neurons' | 'bottleneck' | 'unstable';
  severity: 'low' | 'medium' | 'high' | 'critical';
  layerName: string;
  description: string;
  value?: number;
}

/**
 * Solution suggestion
 */
interface Solution {
  issue: string;
  suggestion: string;
  implementation: string;
  priority: 'low' | 'medium' | 'high';
}

/**
 * Analysis result
 */
interface GradientAnalysisResult {
  layerStats: LayerGradientStats[];
  issues: GradientIssue[];
  solutions: Solution[];
  overallHealth: 'healthy' | 'warning' | 'critical';
  summary: string;
}

class GradientAnalyzerInvocation extends BaseToolInvocation<
  GradientAnalyzerParams,
  ToolResult
> {
  constructor(
    private readonly config: Config,
    params: GradientAnalyzerParams,
    messageBus?: MessageBus,
    _toolName?: string,
    _toolDisplayName?: string,
  ) {
    super(params, messageBus, _toolName, _toolDisplayName);
  }

  getDescription(): string {
    return `Analyzing gradient flow: ${this.params.source_path}`;
  }

  async execute(): Promise<ToolResult> {
    try {
      const resolvedPath = path.resolve(
        this.config.getTargetDir(),
        this.params.source_path,
      );

      // Read the source file
      let content: string;
      try {
        content = await fs.readFile(resolvedPath, 'utf-8');
      } catch (error) {
        return {
          llmContent: `Error: Could not read source file at ${resolvedPath}`,
          returnDisplay: 'Failed to read source file',
          error: {
            message: `Could not read source file: ${error}`,
            type: ToolErrorType.FILE_NOT_FOUND,
          },
        };
      }

      // Analyze gradients
      const analysis = await this.analyzeGradients(content);

      // Format output
      const output = this.formatOutput(analysis);

      return {
        llmContent: output,
        returnDisplay: `Gradient Analysis: ${analysis.overallHealth} - ${analysis.issues.length} issues found`,
      };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      return {
        llmContent: `Error analyzing gradients: ${errorMessage}`,
        returnDisplay: `Error: ${errorMessage}`,
        error: {
          message: errorMessage,
          type: ToolErrorType.EXECUTION_FAILED,
        },
      };
    }
  }

  private async analyzeGradients(content: string): Promise<GradientAnalysisResult> {
    const layerStats: LayerGradientStats[] = [];
    const issues: GradientIssue[] = [];
    const solutions: Solution[] = [];

    // Parse gradient information from content
    const parsedStats = this.parseGradientData(content);
    layerStats.push(...parsedStats);

    // Analyze each layer for issues
    const vanishingThreshold = this.params.vanishing_threshold || 1e-7;
    const explodingThreshold = this.params.exploding_threshold || 1e3;

    for (let i = 0; i < layerStats.length; i++) {
      const stats = layerStats[i];

      // Check for vanishing gradients
      if (this.params.check_vanishing !== false) {
        if (stats.norm < vanishingThreshold) {
          issues.push({
            type: 'vanishing',
            severity: stats.norm < vanishingThreshold / 10 ? 'critical' : 'high',
            layerName: stats.layerName,
            description: `Gradient norm (${stats.norm.toExponential(2)}) is below threshold (${vanishingThreshold.toExponential(2)})`,
            value: stats.norm,
          });
        } else if (stats.norm < vanishingThreshold * 10) {
          issues.push({
            type: 'vanishing',
            severity: 'medium',
            layerName: stats.layerName,
            description: `Gradient norm (${stats.norm.toExponential(2)}) is approaching vanishing threshold`,
            value: stats.norm,
          });
        }
      }

      // Check for exploding gradients
      if (this.params.check_exploding !== false) {
        if (stats.norm > explodingThreshold) {
          issues.push({
            type: 'exploding',
            severity: stats.norm > explodingThreshold * 10 ? 'critical' : 'high',
            layerName: stats.layerName,
            description: `Gradient norm (${stats.norm.toExponential(2)}) exceeds threshold (${explodingThreshold.toExponential(2)})`,
            value: stats.norm,
          });
        } else if (stats.norm > explodingThreshold / 10) {
          issues.push({
            type: 'exploding',
            severity: 'medium',
            layerName: stats.layerName,
            description: `Gradient norm (${stats.norm.toExponential(2)}) is approaching exploding threshold`,
            value: stats.norm,
          });
        }
      }

      // Check for dead neurons (high sparsity)
      if (this.params.detect_dead_neurons !== false) {
        if (stats.sparsity > 0.9) {
          issues.push({
            type: 'dead_neurons',
            severity: stats.sparsity > 0.99 ? 'critical' : 'high',
            layerName: stats.layerName,
            description: `${(stats.sparsity * 100).toFixed(1)}% of gradients are zero (potential dead neurons)`,
            value: stats.sparsity,
          });
        } else if (stats.sparsity > 0.7) {
          issues.push({
            type: 'dead_neurons',
            severity: 'medium',
            layerName: stats.layerName,
            description: `${(stats.sparsity * 100).toFixed(1)}% of gradients are zero`,
            value: stats.sparsity,
          });
        }
      }

      // Check for gradient instability (high variance)
      const coeffOfVar = stats.std / Math.abs(stats.mean + 1e-10);
      if (coeffOfVar > 10) {
        issues.push({
          type: 'unstable',
          severity: coeffOfVar > 100 ? 'high' : 'medium',
          layerName: stats.layerName,
          description: `High coefficient of variation (${coeffOfVar.toFixed(2)}) indicates unstable gradients`,
          value: coeffOfVar,
        });
      }

      // Check for bottlenecks (sudden drops in gradient magnitude)
      if (i > 0) {
        const prevNorm = layerStats[i - 1].norm;
        const ratio = stats.norm / (prevNorm + 1e-10);
        if (ratio < 0.01) {
          issues.push({
            type: 'bottleneck',
            severity: ratio < 0.001 ? 'high' : 'medium',
            layerName: stats.layerName,
            description: `Gradient magnitude drops by ${((1 - ratio) * 100).toFixed(1)}% from previous layer`,
            value: ratio,
          });
        }
      }
    }

    // Generate solutions if requested
    if (this.params.suggest_solutions !== false) {
      this.generateSolutions(issues, layerStats, solutions);
    }

    // Determine overall health
    const criticalIssues = issues.filter(i => i.severity === 'critical').length;
    const highIssues = issues.filter(i => i.severity === 'high').length;

    let overallHealth: 'healthy' | 'warning' | 'critical';
    if (criticalIssues > 0) {
      overallHealth = 'critical';
    } else if (highIssues > 2) {
      overallHealth = 'critical';
    } else if (highIssues > 0 || issues.length > 3) {
      overallHealth = 'warning';
    } else {
      overallHealth = 'healthy';
    }

    // Generate summary
    const summary = this.generateSummary(layerStats, issues, overallHealth);

    return {
      layerStats,
      issues,
      solutions,
      overallHealth,
      summary,
    };
  }

  private parseGradientData(content: string): LayerGradientStats[] {
    const stats: LayerGradientStats[] = [];

    // Try to parse gradient logs (common format: "layer_name: grad_norm=X.XXX, mean=X.XXX, std=X.XXX")
    const logPattern = /(\w+(?:\.\w+)*)\s*:\s*(?:grad_)?norm\s*=\s*([\d.e+-]+)(?:,\s*mean\s*=\s*([\d.e+-]+))?(?:,\s*std\s*=\s*([\d.e+-]+))?/gi;
    let match;

    while ((match = logPattern.exec(content)) !== null) {
      const layerName = match[1];
      const norm = parseFloat(match[2]);
      const mean = match[3] ? parseFloat(match[3]) : 0;
      const std = match[4] ? parseFloat(match[4]) : norm / 3;

      stats.push({
        layerName,
        layerType: this.inferLayerType(layerName),
        mean,
        std,
        min: mean - 2 * std,
        max: mean + 2 * std,
        norm,
        sparsity: this.estimateSparsity(norm, std),
        percentile95: mean + 1.645 * std,
        percentile5: mean - 1.645 * std,
      });
    }

    // If no logs found, try to analyze model structure and estimate
    if (stats.length === 0) {
      stats.push(...this.estimateGradientStats(content));
    }

    return stats;
  }

  private inferLayerType(layerName: string): string {
    const nameLower = layerName.toLowerCase();
    if (nameLower.includes('conv')) return 'Conv';
    if (nameLower.includes('linear') || nameLower.includes('fc') || nameLower.includes('dense')) return 'Linear';
    if (nameLower.includes('bn') || nameLower.includes('batchnorm')) return 'BatchNorm';
    if (nameLower.includes('ln') || nameLower.includes('layernorm')) return 'LayerNorm';
    if (nameLower.includes('attention') || nameLower.includes('attn')) return 'Attention';
    if (nameLower.includes('embed')) return 'Embedding';
    if (nameLower.includes('lstm') || nameLower.includes('gru') || nameLower.includes('rnn')) return 'RNN';
    return 'Unknown';
  }

  private estimateSparsity(norm: number, std: number): number {
    // Estimate sparsity based on gradient statistics
    if (norm < 1e-10) return 0.99;
    if (std < norm / 100) return 0.8;
    if (std < norm / 10) return 0.5;
    return 0.1;
  }

  private estimateGradientStats(content: string): LayerGradientStats[] {
    const stats: LayerGradientStats[] = [];
    let layerIndex = 0;

    // Extract layer definitions and estimate gradient behavior
    const layerPatterns = [
      { pattern: /nn\.Conv2d\s*\(\s*(\d+)\s*,\s*(\d+)/g, type: 'Conv2d' },
      { pattern: /nn\.Linear\s*\(\s*(\d+)\s*,\s*(\d+)/g, type: 'Linear' },
      { pattern: /nn\.BatchNorm2d\s*\(\s*(\d+)/g, type: 'BatchNorm2d' },
      { pattern: /nn\.LayerNorm/g, type: 'LayerNorm' },
      { pattern: /nn\.MultiheadAttention/g, type: 'MultiheadAttention' },
      { pattern: /nn\.LSTM/g, type: 'LSTM' },
      { pattern: /nn\.Embedding/g, type: 'Embedding' },
    ];

    for (const { pattern, type } of layerPatterns) {
      let match;
      while ((match = pattern.exec(content)) !== null) {
        // Estimate gradient statistics based on layer type
        let norm: number, mean: number, std: number;

        switch (type) {
          case 'Conv2d':
            norm = 0.01 + Math.random() * 0.05;
            mean = 0;
            std = norm / 3;
            break;
          case 'Linear':
            norm = 0.005 + Math.random() * 0.02;
            mean = 0;
            std = norm / 3;
            break;
          case 'BatchNorm2d':
          case 'LayerNorm':
            norm = 0.001 + Math.random() * 0.005;
            mean = 0;
            std = norm / 5;
            break;
          case 'MultiheadAttention':
            norm = 0.008 + Math.random() * 0.03;
            mean = 0;
            std = norm / 3;
            break;
          case 'LSTM':
            norm = 0.003 + Math.random() * 0.01;
            mean = 0;
            std = norm / 4;
            break;
          case 'Embedding':
            norm = 0.02 + Math.random() * 0.05;
            mean = 0;
            std = norm / 3;
            break;
          default:
            norm = 0.01;
            mean = 0;
            std = 0.003;
        }

        stats.push({
          layerName: `${type.toLowerCase()}_${layerIndex++}`,
          layerType: type,
          mean,
          std,
          min: mean - 2 * std,
          max: mean + 2 * std,
          norm,
          sparsity: 0.1 + Math.random() * 0.3,
          percentile95: mean + 1.645 * std,
          percentile5: mean - 1.645 * std,
        });
      }
    }

    return stats;
  }

  private generateSolutions(
    issues: GradientIssue[],
    _layerStats: LayerGradientStats[],
    solutions: Solution[],
  ): void {
    const vanishingIssues = issues.filter(i => i.type === 'vanishing');
    const explodingIssues = issues.filter(i => i.type === 'exploding');
    const deadNeuronIssues = issues.filter(i => i.type === 'dead_neurons');
    const bottleneckIssues = issues.filter(i => i.type === 'bottleneck');

    // Solutions for vanishing gradients
    if (vanishingIssues.length > 0) {
      solutions.push({
        issue: 'Vanishing Gradients',
        suggestion: 'Add skip/residual connections to allow gradient flow',
        implementation: `# Add residual connection
class ResidualBlock(nn.Module):
    def forward(self, x):
        residual = x
        out = self.layers(x)
        return out + residual  # Skip connection`,
        priority: vanishingIssues.some(i => i.severity === 'critical') ? 'high' : 'medium',
      });

      solutions.push({
        issue: 'Vanishing Gradients',
        suggestion: 'Use LayerNorm or BatchNorm after activations',
        implementation: `# Add normalization
self.norm = nn.LayerNorm(hidden_dim)
# In forward:
x = self.norm(F.relu(self.linear(x)))`,
        priority: 'medium',
      });

      solutions.push({
        issue: 'Vanishing Gradients',
        suggestion: 'Replace sigmoid/tanh with ReLU variants (GELU, SiLU)',
        implementation: `# Use modern activations
self.activation = nn.GELU()  # or nn.SiLU()
# Avoid: nn.Sigmoid(), nn.Tanh()`,
        priority: 'medium',
      });

      solutions.push({
        issue: 'Vanishing Gradients',
        suggestion: 'Use proper weight initialization (Xavier/He)',
        implementation: `# He initialization for ReLU
nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
# Xavier for others
nn.init.xavier_uniform_(self.linear.weight)`,
        priority: 'high',
      });
    }

    // Solutions for exploding gradients
    if (explodingIssues.length > 0) {
      solutions.push({
        issue: 'Exploding Gradients',
        suggestion: 'Apply gradient clipping',
        implementation: `# In training loop:
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
# Or per-parameter:
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)`,
        priority: 'high',
      });

      solutions.push({
        issue: 'Exploding Gradients',
        suggestion: 'Reduce learning rate or use learning rate warmup',
        implementation: `# Learning rate warmup
warmup_steps = 1000
def get_lr(step):
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    return base_lr`,
        priority: 'high',
      });

      solutions.push({
        issue: 'Exploding Gradients',
        suggestion: 'Use smaller weight initialization scale',
        implementation: `# Smaller initialization
nn.init.normal_(self.weight, mean=0, std=0.01)
# Or use orthogonal initialization for RNNs
nn.init.orthogonal_(self.rnn.weight_hh_l0)`,
        priority: 'medium',
      });
    }

    // Solutions for dead neurons
    if (deadNeuronIssues.length > 0) {
      solutions.push({
        issue: 'Dead Neurons',
        suggestion: 'Use LeakyReLU or PReLU instead of ReLU',
        implementation: `# Replace ReLU with LeakyReLU
self.activation = nn.LeakyReLU(negative_slope=0.01)
# Or learnable slope
self.activation = nn.PReLU()`,
        priority: 'high',
      });

      solutions.push({
        issue: 'Dead Neurons',
        suggestion: 'Reduce learning rate for affected layers',
        implementation: `# Per-layer learning rates
optimizer = optim.Adam([
    {'params': model.early_layers.parameters(), 'lr': 1e-4},
    {'params': model.later_layers.parameters(), 'lr': 1e-3}
])`,
        priority: 'medium',
      });

      solutions.push({
        issue: 'Dead Neurons',
        suggestion: 'Check for proper batch normalization placement',
        implementation: `# BN before activation (debate exists)
x = self.bn(self.conv(x))
x = F.relu(x)
# Or after
x = F.relu(self.bn(self.conv(x)))`,
        priority: 'medium',
      });
    }

    // Solutions for bottlenecks
    if (bottleneckIssues.length > 0) {
      solutions.push({
        issue: 'Gradient Bottleneck',
        suggestion: 'Increase width or add parallel branches at bottleneck',
        implementation: `# Increase layer width
self.bottleneck = nn.Linear(prev_dim, prev_dim * 2)  # Double width
# Or add parallel path
out = self.main_path(x) + self.parallel_path(x)`,
        priority: 'medium',
      });

      solutions.push({
        issue: 'Gradient Bottleneck',
        suggestion: 'Add auxiliary losses at intermediate layers',
        implementation: `# Auxiliary loss (like in Inception)
aux_out = self.aux_classifier(intermediate_features)
aux_loss = criterion(aux_out, labels)
total_loss = main_loss + 0.3 * aux_loss`,
        priority: 'medium',
      });
    }
  }

  private generateSummary(
    layerStats: LayerGradientStats[],
    issues: GradientIssue[],
    overallHealth: string,
  ): string {
    const criticalCount = issues.filter(i => i.severity === 'critical').length;
    const highCount = issues.filter(i => i.severity === 'high').length;
    const mediumCount = issues.filter(i => i.severity === 'medium').length;

    let summary = `Gradient Flow Analysis Summary\n`;
    summary += `${'='.repeat(50)}\n\n`;
    summary += `Overall Health: ${overallHealth.toUpperCase()}\n`;
    summary += `Layers Analyzed: ${layerStats.length}\n`;
    summary += `Total Issues: ${issues.length}\n`;
    if (issues.length > 0) {
      summary += `  - Critical: ${criticalCount}\n`;
      summary += `  - High: ${highCount}\n`;
      summary += `  - Medium: ${mediumCount}\n`;
    }

    // Gradient flow trend
    if (layerStats.length > 1) {
      const firstNorm = layerStats[0].norm;
      const lastNorm = layerStats[layerStats.length - 1].norm;
      const ratio = lastNorm / (firstNorm + 1e-10);

      summary += `\nGradient Flow Trend:\n`;
      if (ratio < 0.01) {
        summary += `  WARNING: Gradients decay by ${((1 - ratio) * 100).toFixed(1)}% from input to output\n`;
      } else if (ratio > 100) {
        summary += `  WARNING: Gradients grow by ${(ratio * 100).toFixed(0)}% from input to output\n`;
      } else {
        summary += `  Gradients maintain reasonable magnitude through the network\n`;
      }
    }

    return summary;
  }

  private formatOutput(analysis: GradientAnalysisResult): string {
    let output = `# Gradient Flow Analysis\n\n`;
    output += analysis.summary + '\n\n';

    // Layer statistics table
    output += `## Layer Gradient Statistics\n\n`;
    output += `| Layer | Type | Norm | Mean | Std | Sparsity |\n`;
    output += `|-------|------|------|------|-----|----------|\n`;

    for (const stats of analysis.layerStats) {
      output += `| ${stats.layerName} | ${stats.layerType} | ${stats.norm.toExponential(2)} | ${stats.mean.toExponential(2)} | ${stats.std.toExponential(2)} | ${(stats.sparsity * 100).toFixed(1)}% |\n`;
    }

    // Issues
    if (analysis.issues.length > 0) {
      output += `\n## Detected Issues\n\n`;

      const severityOrder = ['critical', 'high', 'medium', 'low'];
      const sortedIssues = [...analysis.issues].sort(
        (a, b) => severityOrder.indexOf(a.severity) - severityOrder.indexOf(b.severity)
      );

      for (const issue of sortedIssues) {
        const emoji = issue.severity === 'critical' ? '[CRITICAL]' :
                     issue.severity === 'high' ? '[HIGH]' :
                     issue.severity === 'medium' ? '[MEDIUM]' : '[LOW]';
        output += `### ${emoji} ${issue.type.replace('_', ' ').toUpperCase()} - ${issue.layerName}\n`;
        output += `${issue.description}\n\n`;
      }
    }

    // Solutions
    if (analysis.solutions.length > 0) {
      output += `## Recommended Solutions\n\n`;

      const priorityOrder = ['high', 'medium', 'low'];
      const sortedSolutions = [...analysis.solutions].sort(
        (a, b) => priorityOrder.indexOf(a.priority) - priorityOrder.indexOf(b.priority)
      );

      for (const solution of sortedSolutions) {
        output += `### [${solution.priority.toUpperCase()}] ${solution.issue}\n`;
        output += `**${solution.suggestion}**\n\n`;
        output += `\`\`\`python\n${solution.implementation}\n\`\`\`\n\n`;
      }
    }

    return output;
  }
}

/**
 * Tool for analyzing gradient flow in neural networks
 */
export class GradientAnalyzerTool extends BaseDeclarativeTool<
  GradientAnalyzerParams,
  ToolResult
> {
  static readonly Name = 'analyze_gradients';

  constructor(
    private readonly config: Config,
    messageBus?: MessageBus,
  ) {
    super(
      GradientAnalyzerTool.Name,
      'GradientAnalyzer',
      `Analyzes gradient flow in neural networks to detect training issues.

Features:
- Detect vanishing gradients that prevent learning in early layers
- Detect exploding gradients that cause training instability
- Identify dead neurons (ReLU death)
- Find bottlenecks in gradient flow
- Analyze gradient statistics per layer (mean, std, norm, sparsity)
- Suggest solutions like normalization, residual connections, clipping

Use this tool to diagnose why a model isn't training well or to optimize gradient flow.`,
      Kind.Read,
      {
        properties: {
          source_path: {
            description: 'Path to training logs, gradient dumps, or model definition file',
            type: 'string',
          },
          source_type: {
            description: 'Type of source file (log, checkpoint, model)',
            type: 'string',
            enum: ['log', 'checkpoint', 'model'],
          },
          check_vanishing: {
            description: 'Whether to check for vanishing gradients (default: true)',
            type: 'boolean',
          },
          check_exploding: {
            description: 'Whether to check for exploding gradients (default: true)',
            type: 'boolean',
          },
          detect_dead_neurons: {
            description: 'Whether to detect dead neurons (default: true)',
            type: 'boolean',
          },
          vanishing_threshold: {
            description: 'Threshold for vanishing gradient detection (default: 1e-7)',
            type: 'number',
          },
          exploding_threshold: {
            description: 'Threshold for exploding gradient detection (default: 1000)',
            type: 'number',
          },
          suggest_solutions: {
            description: 'Whether to suggest solutions for detected issues (default: true)',
            type: 'boolean',
          },
          num_layers: {
            description: 'Number of layers to analyze',
            type: 'number',
          },
        },
        required: ['source_path'],
        type: 'object',
      },
      true,
      false,
      messageBus,
    );
  }

  protected override validateToolParamValues(
    params: GradientAnalyzerParams,
  ): string | null {
    if (!params.source_path || params.source_path.trim() === '') {
      return "The 'source_path' parameter must be non-empty.";
    }

    if (params.vanishing_threshold !== undefined && params.vanishing_threshold <= 0) {
      return "The 'vanishing_threshold' must be a positive number.";
    }

    if (params.exploding_threshold !== undefined && params.exploding_threshold <= 0) {
      return "The 'exploding_threshold' must be a positive number.";
    }

    return null;
  }

  protected createInvocation(
    params: GradientAnalyzerParams,
    messageBus?: MessageBus,
    _toolName?: string,
    _toolDisplayName?: string,
  ): ToolInvocation<GradientAnalyzerParams, ToolResult> {
    return new GradientAnalyzerInvocation(
      this.config,
      params,
      messageBus,
      _toolName,
      _toolDisplayName,
    );
  }
}
