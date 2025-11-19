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
 * Parameters for the Training Diagnostics tool
 */
export interface TrainingDiagnosticsParams {
  /**
   * Path to training logs or TensorBoard logs
   */
  log_path: string;

  /**
   * Metrics to analyze (loss, accuracy, lr, etc.)
   */
  metrics?: string[];

  /**
   * Whether to check for overfitting
   */
  check_overfitting?: boolean;

  /**
   * Whether to check for underfitting
   */
  check_underfitting?: boolean;

  /**
   * Whether to analyze learning rate schedule
   */
  analyze_lr?: boolean;

  /**
   * Whether to check for data leakage signs
   */
  check_data_leakage?: boolean;

  /**
   * Whether to analyze GPU utilization
   */
  analyze_gpu?: boolean;

  /**
   * Window size for moving average smoothing
   */
  smoothing_window?: number;

  /**
   * Whether to suggest optimization strategies
   */
  suggest_optimizations?: boolean;
}

/**
 * Metric data point
 */
interface MetricPoint {
  step: number;
  epoch?: number;
  value: number;
  timestamp?: number;
}

/**
 * Metric statistics
 */
interface MetricStats {
  name: string;
  dataPoints: MetricPoint[];
  min: number;
  max: number;
  mean: number;
  std: number;
  final: number;
  trend: 'improving' | 'degrading' | 'stable' | 'oscillating';
  convergenceRate: number;
}

/**
 * Training issue
 */
interface TrainingIssue {
  type: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  description: string;
  evidence: string;
  step?: number;
}

/**
 * Optimization suggestion
 */
interface OptimizationSuggestion {
  category: string;
  suggestion: string;
  implementation: string;
  expectedImpact: string;
  priority: 'low' | 'medium' | 'high';
}

/**
 * GPU utilization data
 */
interface GPUUtilization {
  avgUtilization: number;
  avgMemoryUsage: number;
  peakMemoryUsage: number;
  bottleneck?: string;
}

/**
 * Diagnostics result
 */
interface DiagnosticsResult {
  metrics: MetricStats[];
  issues: TrainingIssue[];
  suggestions: OptimizationSuggestion[];
  gpuUtilization?: GPUUtilization;
  overallStatus: 'healthy' | 'warning' | 'critical';
  summary: string;
}

class TrainingDiagnosticsInvocation extends BaseToolInvocation<
  TrainingDiagnosticsParams,
  ToolResult
> {
  constructor(
    private readonly config: Config,
    params: TrainingDiagnosticsParams,
    messageBus?: MessageBus,
    _toolName?: string,
    _toolDisplayName?: string,
  ) {
    super(params, messageBus, _toolName, _toolDisplayName);
  }

  getDescription(): string {
    return `Analyzing training logs: ${this.params.log_path}`;
  }

  async execute(): Promise<ToolResult> {
    try {
      const resolvedPath = path.resolve(
        this.config.getTargetDir(),
        this.params.log_path,
      );

      // Read the log file
      let content: string;
      try {
        content = await fs.readFile(resolvedPath, 'utf-8');
      } catch (error) {
        return {
          llmContent: `Error: Could not read log file at ${resolvedPath}`,
          returnDisplay: 'Failed to read log file',
          error: {
            message: `Could not read log file: ${error}`,
            type: ToolErrorType.FILE_NOT_FOUND,
          },
        };
      }

      // Perform diagnostics
      const diagnostics = await this.runDiagnostics(content);

      // Format output
      const output = this.formatOutput(diagnostics);

      return {
        llmContent: output,
        returnDisplay: `Training Diagnostics: ${diagnostics.overallStatus} - ${diagnostics.issues.length} issues found`,
      };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      return {
        llmContent: `Error in training diagnostics: ${errorMessage}`,
        returnDisplay: `Error: ${errorMessage}`,
        error: {
          message: errorMessage,
          type: ToolErrorType.EXECUTION_FAILED,
        },
      };
    }
  }

  private async runDiagnostics(content: string): Promise<DiagnosticsResult> {
    const metrics: MetricStats[] = [];
    const issues: TrainingIssue[] = [];
    const suggestions: OptimizationSuggestion[] = [];

    // Parse metrics from logs
    const parsedMetrics = this.parseMetrics(content);
    metrics.push(...parsedMetrics);

    // Check for overfitting
    if (this.params.check_overfitting !== false) {
      this.checkOverfitting(metrics, issues);
    }

    // Check for underfitting
    if (this.params.check_underfitting !== false) {
      this.checkUnderfitting(metrics, issues);
    }

    // Analyze learning rate
    if (this.params.analyze_lr !== false) {
      this.analyzeLearningRate(content, metrics, issues);
    }

    // Check for data leakage
    if (this.params.check_data_leakage !== false) {
      this.checkDataLeakage(metrics, issues);
    }

    // Analyze GPU utilization
    let gpuUtilization: GPUUtilization | undefined;
    if (this.params.analyze_gpu !== false) {
      gpuUtilization = this.analyzeGPUUtilization(content, issues);
    }

    // Check for other common issues
    this.checkCommonIssues(metrics, issues);

    // Generate optimization suggestions
    if (this.params.suggest_optimizations !== false) {
      this.generateSuggestions(metrics, issues, suggestions);
    }

    // Determine overall status
    const criticalIssues = issues.filter(i => i.severity === 'critical').length;
    const highIssues = issues.filter(i => i.severity === 'high').length;

    let overallStatus: 'healthy' | 'warning' | 'critical';
    if (criticalIssues > 0) {
      overallStatus = 'critical';
    } else if (highIssues > 0 || issues.length > 4) {
      overallStatus = 'warning';
    } else {
      overallStatus = 'healthy';
    }

    // Generate summary
    const summary = this.generateSummary(metrics, issues, overallStatus, gpuUtilization);

    return {
      metrics,
      issues,
      suggestions,
      gpuUtilization,
      overallStatus,
      summary,
    };
  }

  private parseMetrics(content: string): MetricStats[] {
    const metrics: MetricStats[] = [];
    const metricsToFind = this.params.metrics || ['loss', 'train_loss', 'val_loss', 'accuracy', 'train_acc', 'val_acc', 'lr', 'learning_rate'];

    for (const metricName of metricsToFind) {
      const dataPoints: MetricPoint[] = [];

      // Common log formats:
      // "epoch 1: loss=0.5, acc=0.8"
      // "step 100 - loss: 0.45"
      // {"step": 100, "loss": 0.45}
      // "[2024-01-01 10:00:00] loss: 0.45"

      // Pattern 1: key=value or key: value
      const pattern1 = new RegExp(
        `(?:epoch|step|iter)\\s*(\\d+).*?${metricName}\\s*[=:]\\s*([\\d.e+-]+)`,
        'gi'
      );

      // Pattern 2: JSON format
      const pattern2 = new RegExp(
        `"(?:step|epoch)"\\s*:\\s*(\\d+)[^}]*"${metricName}"\\s*:\\s*([\\d.e+-]+)`,
        'gi'
      );

      let match;
      while ((match = pattern1.exec(content)) !== null) {
        dataPoints.push({
          step: parseInt(match[1]),
          value: parseFloat(match[2]),
        });
      }

      while ((match = pattern2.exec(content)) !== null) {
        dataPoints.push({
          step: parseInt(match[1]),
          value: parseFloat(match[2]),
        });
      }

      if (dataPoints.length > 0) {
        // Sort by step
        dataPoints.sort((a, b) => a.step - b.step);

        // Calculate statistics
        const values = dataPoints.map(d => d.value);
        const min = Math.min(...values);
        const max = Math.max(...values);
        const mean = values.reduce((a, b) => a + b, 0) / values.length;
        const variance = values.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / values.length;
        const std = Math.sqrt(variance);
        const final = values[values.length - 1];

        // Determine trend
        const trend = this.determineTrend(dataPoints);

        // Calculate convergence rate
        const convergenceRate = this.calculateConvergenceRate(dataPoints);

        metrics.push({
          name: metricName,
          dataPoints,
          min,
          max,
          mean,
          std,
          final,
          trend,
          convergenceRate,
        });
      }
    }

    return metrics;
  }

  private determineTrend(dataPoints: MetricPoint[]): 'improving' | 'degrading' | 'stable' | 'oscillating' {
    if (dataPoints.length < 3) return 'stable';

    const windowSize = Math.min(10, Math.floor(dataPoints.length / 3));
    const earlyAvg = dataPoints.slice(0, windowSize).reduce((a, b) => a + b.value, 0) / windowSize;
    const lateAvg = dataPoints.slice(-windowSize).reduce((a, b) => a + b.value, 0) / windowSize;

    // Check for oscillation
    let directionChanges = 0;
    let prevDirection = 0;
    for (let i = 1; i < dataPoints.length; i++) {
      const direction = dataPoints[i].value > dataPoints[i - 1].value ? 1 : -1;
      if (prevDirection !== 0 && direction !== prevDirection) {
        directionChanges++;
      }
      prevDirection = direction;
    }

    const oscillationRatio = directionChanges / (dataPoints.length - 1);
    if (oscillationRatio > 0.4) return 'oscillating';

    // Determine if improving or degrading (assuming lower is better for loss)
    const isLossMetric = dataPoints[0] && /loss/i.test('loss');
    const improvement = isLossMetric ? earlyAvg - lateAvg : lateAvg - earlyAvg;
    const threshold = Math.abs(earlyAvg) * 0.05;

    if (Math.abs(improvement) < threshold) return 'stable';
    return improvement > 0 ? 'improving' : 'degrading';
  }

  private calculateConvergenceRate(dataPoints: MetricPoint[]): number {
    if (dataPoints.length < 10) return 0;

    // Calculate average rate of change in last portion
    const lastPortion = dataPoints.slice(-Math.floor(dataPoints.length / 3));
    let totalChange = 0;

    for (let i = 1; i < lastPortion.length; i++) {
      totalChange += Math.abs(lastPortion[i].value - lastPortion[i - 1].value);
    }

    return totalChange / (lastPortion.length - 1);
  }

  private checkOverfitting(metrics: MetricStats[], issues: TrainingIssue[]): void {
    const trainLoss = metrics.find(m => m.name.includes('train') && m.name.includes('loss'));
    const valLoss = metrics.find(m => m.name.includes('val') && m.name.includes('loss'));
    const trainAcc = metrics.find(m => m.name.includes('train') && m.name.includes('acc'));
    const valAcc = metrics.find(m => m.name.includes('val') && m.name.includes('acc'));

    if (trainLoss && valLoss) {
      // Check if training loss decreases but validation loss increases
      const trainTrend = trainLoss.trend;
      const valTrend = valLoss.trend;

      if (trainTrend === 'improving' && valTrend === 'degrading') {
        const gap = valLoss.final - trainLoss.final;
        const severity = gap > trainLoss.final ? 'critical' :
                        gap > trainLoss.final * 0.5 ? 'high' : 'medium';

        issues.push({
          type: 'overfitting',
          severity,
          description: 'Model is overfitting: training loss decreases while validation loss increases',
          evidence: `Train loss trend: ${trainTrend}, Val loss trend: ${valTrend}, Gap: ${gap.toFixed(4)}`,
        });
      }

      // Check generalization gap
      const gapRatio = (valLoss.final - trainLoss.final) / (trainLoss.final + 1e-10);
      if (gapRatio > 0.5) {
        issues.push({
          type: 'large_generalization_gap',
          severity: gapRatio > 1 ? 'high' : 'medium',
          description: `Large gap between training and validation loss (${(gapRatio * 100).toFixed(1)}%)`,
          evidence: `Train loss: ${trainLoss.final.toFixed(4)}, Val loss: ${valLoss.final.toFixed(4)}`,
        });
      }
    }

    if (trainAcc && valAcc) {
      const accGap = trainAcc.final - valAcc.final;
      if (accGap > 0.1) {
        issues.push({
          type: 'accuracy_gap',
          severity: accGap > 0.2 ? 'high' : 'medium',
          description: `Training accuracy significantly higher than validation (${(accGap * 100).toFixed(1)}% gap)`,
          evidence: `Train acc: ${(trainAcc.final * 100).toFixed(1)}%, Val acc: ${(valAcc.final * 100).toFixed(1)}%`,
        });
      }
    }
  }

  private checkUnderfitting(metrics: MetricStats[], issues: TrainingIssue[]): void {
    const trainLoss = metrics.find(m => m.name.includes('train') && m.name.includes('loss'));
    const valLoss = metrics.find(m => m.name.includes('val') && m.name.includes('loss'));
    const trainAcc = metrics.find(m => m.name.includes('train') && m.name.includes('acc'));

    // Check if both train and val loss are high and stable
    if (trainLoss && valLoss) {
      if (trainLoss.trend === 'stable' && valLoss.trend === 'stable') {
        if (trainLoss.convergenceRate < 0.001 && valLoss.convergenceRate < 0.001) {
          issues.push({
            type: 'premature_convergence',
            severity: 'high',
            description: 'Training appears to have converged prematurely with both losses remaining high',
            evidence: `Train loss: ${trainLoss.final.toFixed(4)}, Val loss: ${valLoss.final.toFixed(4)}, both stable`,
          });
        }
      }
    }

    // Check for low training accuracy
    if (trainAcc && trainAcc.final < 0.6) {
      issues.push({
        type: 'low_training_accuracy',
        severity: trainAcc.final < 0.4 ? 'high' : 'medium',
        description: `Training accuracy is low (${(trainAcc.final * 100).toFixed(1)}%), model may be underfitting`,
        evidence: `Final training accuracy: ${(trainAcc.final * 100).toFixed(1)}%`,
      });
    }

    // Check if loss is not decreasing
    if (trainLoss && trainLoss.trend !== 'improving') {
      if (trainLoss.dataPoints.length > 10) {
        const earlyLoss = trainLoss.dataPoints.slice(0, 5).reduce((a, b) => a + b.value, 0) / 5;
        const lateLoss = trainLoss.dataPoints.slice(-5).reduce((a, b) => a + b.value, 0) / 5;

        if (Math.abs(earlyLoss - lateLoss) / earlyLoss < 0.1) {
          issues.push({
            type: 'no_learning',
            severity: 'critical',
            description: 'Training loss is not decreasing - model may not be learning',
            evidence: `Early loss: ${earlyLoss.toFixed(4)}, Late loss: ${lateLoss.toFixed(4)}`,
          });
        }
      }
    }
  }

  private analyzeLearningRate(content: string, metrics: MetricStats[], issues: TrainingIssue[]): void {
    const lrMetric = metrics.find(m => m.name === 'lr' || m.name === 'learning_rate');

    if (lrMetric) {
      // Check if LR is too high (loss oscillating)
      const trainLoss = metrics.find(m => m.name.includes('train') && m.name.includes('loss'));
      if (trainLoss && trainLoss.trend === 'oscillating' && lrMetric.final > 1e-3) {
        issues.push({
          type: 'high_learning_rate',
          severity: 'high',
          description: 'Learning rate may be too high causing loss oscillation',
          evidence: `LR: ${lrMetric.final}, Loss trend: oscillating`,
        });
      }

      // Check if LR is too low (very slow convergence)
      if (trainLoss && trainLoss.convergenceRate < 0.0001 && lrMetric.final < 1e-5) {
        issues.push({
          type: 'low_learning_rate',
          severity: 'medium',
          description: 'Learning rate may be too low causing very slow convergence',
          evidence: `LR: ${lrMetric.final.toExponential(2)}, Convergence rate: ${trainLoss.convergenceRate.toExponential(2)}`,
        });
      }

      // Check for learning rate decay
      if (lrMetric.dataPoints.length > 1) {
        const initialLR = lrMetric.dataPoints[0].value;
        const finalLR = lrMetric.final;
        const decayRatio = finalLR / initialLR;

        if (decayRatio < 0.001) {
          issues.push({
            type: 'excessive_lr_decay',
            severity: 'medium',
            description: `Learning rate decayed by ${((1 - decayRatio) * 100).toFixed(1)}%, may be too aggressive`,
            evidence: `Initial LR: ${initialLR.toExponential(2)}, Final LR: ${finalLR.toExponential(2)}`,
          });
        }
      }
    }

    // Check for NaN or Inf in loss (often caused by LR issues)
    if (content.includes('nan') || content.includes('NaN') || content.includes('inf') || content.includes('Inf')) {
      issues.push({
        type: 'numerical_instability',
        severity: 'critical',
        description: 'NaN or Inf values detected in training - likely due to learning rate or gradient issues',
        evidence: 'Found nan/inf values in logs',
      });
    }
  }

  private checkDataLeakage(metrics: MetricStats[], issues: TrainingIssue[]): void {
    const valAcc = metrics.find(m => m.name.includes('val') && m.name.includes('acc'));
    const trainAcc = metrics.find(m => m.name.includes('train') && m.name.includes('acc'));

    // Suspiciously high validation accuracy from the start
    if (valAcc && valAcc.dataPoints.length > 0) {
      const initialValAcc = valAcc.dataPoints[0].value;
      if (initialValAcc > 0.9) {
        issues.push({
          type: 'possible_data_leakage',
          severity: 'critical',
          description: 'Suspiciously high initial validation accuracy - possible data leakage',
          evidence: `Initial validation accuracy: ${(initialValAcc * 100).toFixed(1)}%`,
        });
      }
    }

    // Validation accuracy higher than training accuracy (unusual)
    if (valAcc && trainAcc) {
      if (valAcc.final > trainAcc.final + 0.05) {
        issues.push({
          type: 'val_better_than_train',
          severity: 'medium',
          description: 'Validation accuracy is higher than training accuracy - check for data issues',
          evidence: `Train acc: ${(trainAcc.final * 100).toFixed(1)}%, Val acc: ${(valAcc.final * 100).toFixed(1)}%`,
        });
      }
    }

    // Perfect or near-perfect training accuracy
    if (trainAcc && trainAcc.final > 0.99) {
      issues.push({
        type: 'near_perfect_training',
        severity: 'medium',
        description: 'Near-perfect training accuracy may indicate overfitting or data issues',
        evidence: `Training accuracy: ${(trainAcc.final * 100).toFixed(2)}%`,
      });
    }
  }

  private analyzeGPUUtilization(content: string, issues: TrainingIssue[]): GPUUtilization | undefined {
    // Try to extract GPU utilization info from logs
    const gpuUtilPattern = /gpu[_\s]?(?:utilization|util)[:\s]*(\d+(?:\.\d+)?)\s*%/gi;
    const gpuMemPattern = /gpu[_\s]?(?:memory|mem)[:\s]*(\d+(?:\.\d+)?)\s*(?:%|MB|GB)/gi;

    const utilValues: number[] = [];
    const memValues: number[] = [];

    let match;
    while ((match = gpuUtilPattern.exec(content)) !== null) {
      utilValues.push(parseFloat(match[1]));
    }

    while ((match = gpuMemPattern.exec(content)) !== null) {
      memValues.push(parseFloat(match[1]));
    }

    if (utilValues.length === 0 && memValues.length === 0) {
      return undefined;
    }

    const avgUtilization = utilValues.length > 0
      ? utilValues.reduce((a, b) => a + b, 0) / utilValues.length
      : 0;

    const avgMemoryUsage = memValues.length > 0
      ? memValues.reduce((a, b) => a + b, 0) / memValues.length
      : 0;

    const peakMemoryUsage = memValues.length > 0
      ? Math.max(...memValues)
      : 0;

    // Check for low GPU utilization
    if (avgUtilization > 0 && avgUtilization < 50) {
      issues.push({
        type: 'low_gpu_utilization',
        severity: avgUtilization < 30 ? 'high' : 'medium',
        description: `Low GPU utilization (${avgUtilization.toFixed(1)}%) - possible CPU bottleneck`,
        evidence: `Average GPU utilization: ${avgUtilization.toFixed(1)}%`,
      });
    }

    // Determine bottleneck
    let bottleneck: string | undefined;
    if (avgUtilization < 50) {
      bottleneck = 'Data loading or CPU preprocessing';
    } else if (peakMemoryUsage > 90) {
      bottleneck = 'GPU memory - consider gradient checkpointing or smaller batch';
    }

    return {
      avgUtilization,
      avgMemoryUsage,
      peakMemoryUsage,
      bottleneck,
    };
  }

  private checkCommonIssues(metrics: MetricStats[], issues: TrainingIssue[]): void {
    // Check for loss spikes
    for (const metric of metrics) {
      if (metric.name.includes('loss')) {
        for (let i = 1; i < metric.dataPoints.length; i++) {
          const prev = metric.dataPoints[i - 1].value;
          const curr = metric.dataPoints[i].value;
          const spike = (curr - prev) / (prev + 1e-10);

          if (spike > 2) {
            issues.push({
              type: 'loss_spike',
              severity: spike > 5 ? 'high' : 'medium',
              description: `Loss spike detected at step ${metric.dataPoints[i].step} (${(spike * 100).toFixed(0)}% increase)`,
              evidence: `Previous: ${prev.toFixed(4)}, Current: ${curr.toFixed(4)}`,
              step: metric.dataPoints[i].step,
            });
            break; // Only report first spike
          }
        }
      }
    }

    // Check for training instability (high variance in recent values)
    for (const metric of metrics) {
      if (metric.name.includes('loss') && metric.dataPoints.length > 10) {
        const recentValues = metric.dataPoints.slice(-10).map(d => d.value);
        const recentMean = recentValues.reduce((a, b) => a + b, 0) / recentValues.length;
        const recentVariance = recentValues.reduce((a, b) => a + Math.pow(b - recentMean, 2), 0) / recentValues.length;
        const cv = Math.sqrt(recentVariance) / recentMean;

        if (cv > 0.3) {
          issues.push({
            type: 'training_instability',
            severity: cv > 0.5 ? 'high' : 'medium',
            description: `High variance in recent ${metric.name} values (CV: ${(cv * 100).toFixed(1)}%)`,
            evidence: `Recent mean: ${recentMean.toFixed(4)}, CV: ${(cv * 100).toFixed(1)}%`,
          });
        }
      }
    }
  }

  private generateSuggestions(
    metrics: MetricStats[],
    issues: TrainingIssue[],
    suggestions: OptimizationSuggestion[],
  ): void {
    // Suggestions based on issues
    const issueTypes = new Set(issues.map(i => i.type));

    if (issueTypes.has('overfitting') || issueTypes.has('large_generalization_gap')) {
      suggestions.push({
        category: 'Regularization',
        suggestion: 'Add dropout layers to reduce overfitting',
        implementation: `# Add dropout after dense layers
self.dropout = nn.Dropout(p=0.3)
x = self.dropout(F.relu(self.fc(x)))`,
        expectedImpact: 'Reduce generalization gap by 10-30%',
        priority: 'high',
      });

      suggestions.push({
        category: 'Regularization',
        suggestion: 'Apply weight decay (L2 regularization)',
        implementation: `optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)`,
        expectedImpact: 'Reduce overfitting, more generalizable weights',
        priority: 'high',
      });

      suggestions.push({
        category: 'Data Augmentation',
        suggestion: 'Add data augmentation to increase effective dataset size',
        implementation: `transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
])`,
        expectedImpact: 'Reduce overfitting, improve generalization',
        priority: 'high',
      });
    }

    if (issueTypes.has('underfitting') || issueTypes.has('no_learning') || issueTypes.has('low_training_accuracy')) {
      suggestions.push({
        category: 'Model Capacity',
        suggestion: 'Increase model capacity (more layers/units)',
        implementation: `# Example: Increase hidden dim
self.fc1 = nn.Linear(input_dim, hidden_dim * 2)  # Double the hidden size
# Or add more layers`,
        expectedImpact: 'Better fit to training data',
        priority: 'high',
      });

      suggestions.push({
        category: 'Learning Rate',
        suggestion: 'Increase learning rate or use learning rate finder',
        implementation: `# Use learning rate finder
from torch_lr_finder import LRFinder
lr_finder = LRFinder(model, optimizer, criterion)
lr_finder.range_test(train_loader, end_lr=1, num_iter=100)
lr_finder.plot()  # Find the steepest part`,
        expectedImpact: 'Faster convergence, better training',
        priority: 'high',
      });
    }

    if (issueTypes.has('high_learning_rate') || issueTypes.has('numerical_instability') || issueTypes.has('loss_spike')) {
      suggestions.push({
        category: 'Learning Rate',
        suggestion: 'Reduce learning rate or add warmup',
        implementation: `# Learning rate warmup
def get_lr(step, warmup_steps=1000, base_lr=1e-4):
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    return base_lr`,
        expectedImpact: 'More stable training, avoid divergence',
        priority: 'high',
      });

      suggestions.push({
        category: 'Gradient Clipping',
        suggestion: 'Apply gradient clipping to prevent explosions',
        implementation: `# In training loop
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`,
        expectedImpact: 'Prevent training instability',
        priority: 'high',
      });
    }

    if (issueTypes.has('low_gpu_utilization')) {
      suggestions.push({
        category: 'Data Loading',
        suggestion: 'Optimize data loading with more workers and prefetching',
        implementation: `train_loader = DataLoader(
    dataset,
    batch_size=64,
    num_workers=4,  # Increase workers
    pin_memory=True,  # Faster CPU to GPU transfer
    prefetch_factor=2,
)`,
        expectedImpact: 'Increase GPU utilization by 20-50%',
        priority: 'high',
      });

      suggestions.push({
        category: 'Batch Size',
        suggestion: 'Increase batch size to better utilize GPU',
        implementation: `# Increase batch size (if memory allows)
batch_size = 128  # or use gradient accumulation for larger effective batch`,
        expectedImpact: 'Better GPU utilization, faster training',
        priority: 'medium',
      });
    }

    // General optimization suggestions
    if (suggestions.length < 3) {
      suggestions.push({
        category: 'Mixed Precision',
        suggestion: 'Use mixed precision training for faster computation',
        implementation: `from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    output = model(input)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()`,
        expectedImpact: '1.5-3x training speedup with minimal accuracy loss',
        priority: 'medium',
      });
    }
  }

  private generateSummary(
    metrics: MetricStats[],
    issues: TrainingIssue[],
    overallStatus: string,
    gpuUtilization?: GPUUtilization,
  ): string {
    let summary = `Training Diagnostics Summary\n`;
    summary += `${'='.repeat(50)}\n\n`;
    summary += `Overall Status: ${overallStatus.toUpperCase()}\n\n`;

    // Metrics summary
    summary += `Metrics Analyzed: ${metrics.length}\n`;
    for (const metric of metrics) {
      summary += `  - ${metric.name}: ${metric.final.toFixed(4)} (trend: ${metric.trend})\n`;
    }

    // Issues summary
    summary += `\nIssues Found: ${issues.length}\n`;
    const criticalCount = issues.filter(i => i.severity === 'critical').length;
    const highCount = issues.filter(i => i.severity === 'high').length;
    if (criticalCount > 0) summary += `  - Critical: ${criticalCount}\n`;
    if (highCount > 0) summary += `  - High: ${highCount}\n`;

    // GPU summary
    if (gpuUtilization) {
      summary += `\nGPU Utilization: ${gpuUtilization.avgUtilization.toFixed(1)}%\n`;
      summary += `GPU Memory: ${gpuUtilization.avgMemoryUsage.toFixed(1)}% (peak: ${gpuUtilization.peakMemoryUsage.toFixed(1)}%)\n`;
      if (gpuUtilization.bottleneck) {
        summary += `Bottleneck: ${gpuUtilization.bottleneck}\n`;
      }
    }

    return summary;
  }

  private formatOutput(diagnostics: DiagnosticsResult): string {
    let output = `# Training Diagnostics Report\n\n`;
    output += diagnostics.summary + '\n\n';

    // Metrics details
    output += `## Metric Details\n\n`;
    output += `| Metric | Final | Min | Max | Mean | Std | Trend |\n`;
    output += `|--------|-------|-----|-----|------|-----|-------|\n`;

    for (const metric of diagnostics.metrics) {
      output += `| ${metric.name} | ${metric.final.toFixed(4)} | ${metric.min.toFixed(4)} | ${metric.max.toFixed(4)} | ${metric.mean.toFixed(4)} | ${metric.std.toFixed(4)} | ${metric.trend} |\n`;
    }

    // Issues
    if (diagnostics.issues.length > 0) {
      output += `\n## Detected Issues\n\n`;

      const severityOrder = ['critical', 'high', 'medium', 'low'];
      const sortedIssues = [...diagnostics.issues].sort(
        (a, b) => severityOrder.indexOf(a.severity) - severityOrder.indexOf(b.severity)
      );

      for (const issue of sortedIssues) {
        const badge = issue.severity === 'critical' ? '[CRITICAL]' :
                     issue.severity === 'high' ? '[HIGH]' :
                     issue.severity === 'medium' ? '[MEDIUM]' : '[LOW]';
        output += `### ${badge} ${issue.type.replace(/_/g, ' ').toUpperCase()}\n`;
        output += `${issue.description}\n`;
        output += `*Evidence: ${issue.evidence}*\n\n`;
      }
    }

    // Suggestions
    if (diagnostics.suggestions.length > 0) {
      output += `## Optimization Suggestions\n\n`;

      const priorityOrder = ['high', 'medium', 'low'];
      const sortedSuggestions = [...diagnostics.suggestions].sort(
        (a, b) => priorityOrder.indexOf(a.priority) - priorityOrder.indexOf(b.priority)
      );

      for (const suggestion of sortedSuggestions) {
        output += `### [${suggestion.priority.toUpperCase()}] ${suggestion.category}\n`;
        output += `**${suggestion.suggestion}**\n\n`;
        output += `*Expected Impact: ${suggestion.expectedImpact}*\n\n`;
        output += `\`\`\`python\n${suggestion.implementation}\n\`\`\`\n\n`;
      }
    }

    return output;
  }
}

/**
 * Tool for training diagnostics
 */
export class TrainingDiagnosticsTool extends BaseDeclarativeTool<
  TrainingDiagnosticsParams,
  ToolResult
> {
  static readonly Name = 'training_diagnostics';

  constructor(
    private readonly config: Config,
    messageBus?: MessageBus,
  ) {
    super(
      TrainingDiagnosticsTool.Name,
      'TrainingDiagnostics',
      `Analyzes training logs to diagnose issues and optimize deep learning training.

Features:
- Analyze training and validation loss/accuracy curves
- Detect overfitting and underfitting patterns
- Identify learning rate issues (too high, too low, bad schedule)
- Check for signs of data leakage
- Monitor GPU utilization and identify bottlenecks
- Detect training instabilities (loss spikes, NaN values)
- Suggest optimization strategies based on detected issues

Use this tool to debug training problems and improve model performance.`,
      Kind.Read,
      {
        properties: {
          log_path: {
            description: 'Path to training logs (text logs, CSV, or TensorBoard logs)',
            type: 'string',
          },
          metrics: {
            description: 'List of metrics to analyze (default: loss, accuracy, lr)',
            type: 'array',
            items: { type: 'string' },
          },
          check_overfitting: {
            description: 'Whether to check for overfitting (default: true)',
            type: 'boolean',
          },
          check_underfitting: {
            description: 'Whether to check for underfitting (default: true)',
            type: 'boolean',
          },
          analyze_lr: {
            description: 'Whether to analyze learning rate schedule (default: true)',
            type: 'boolean',
          },
          check_data_leakage: {
            description: 'Whether to check for data leakage signs (default: true)',
            type: 'boolean',
          },
          analyze_gpu: {
            description: 'Whether to analyze GPU utilization (default: true)',
            type: 'boolean',
          },
          smoothing_window: {
            description: 'Window size for moving average smoothing',
            type: 'number',
          },
          suggest_optimizations: {
            description: 'Whether to suggest optimization strategies (default: true)',
            type: 'boolean',
          },
        },
        required: ['log_path'],
        type: 'object',
      },
      true,
      false,
      messageBus,
    );
  }

  protected override validateToolParamValues(
    params: TrainingDiagnosticsParams,
  ): string | null {
    if (!params.log_path || params.log_path.trim() === '') {
      return "The 'log_path' parameter must be non-empty.";
    }

    if (params.smoothing_window !== undefined && params.smoothing_window < 1) {
      return "The 'smoothing_window' must be at least 1.";
    }

    return null;
  }

  protected createInvocation(
    params: TrainingDiagnosticsParams,
    messageBus?: MessageBus,
    _toolName?: string,
    _toolDisplayName?: string,
  ): ToolInvocation<TrainingDiagnosticsParams, ToolResult> {
    return new TrainingDiagnosticsInvocation(
      this.config,
      params,
      messageBus,
      _toolName,
      _toolDisplayName,
    );
  }
}
