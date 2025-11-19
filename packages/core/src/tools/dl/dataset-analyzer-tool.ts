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
 * Parameters for the Dataset Analyzer tool
 */
export interface DatasetAnalyzerParams {
  /**
   * Path to dataset directory, CSV file, or manifest file
   */
  dataset_path: string;

  /**
   * Type of dataset (image, text, tabular, audio)
   */
  dataset_type?: 'image' | 'text' | 'tabular' | 'audio' | 'auto';

  /**
   * Whether to analyze class distribution
   */
  analyze_distribution?: boolean;

  /**
   * Whether to check for data quality issues
   */
  check_quality?: boolean;

  /**
   * Whether to detect outliers
   */
  detect_outliers?: boolean;

  /**
   * Whether to suggest data augmentations
   */
  suggest_augmentation?: boolean;

  /**
   * Whether to generate sample visualizations (descriptions)
   */
  visualize_samples?: boolean;

  /**
   * Number of samples to analyze (for large datasets)
   */
  sample_size?: number;

  /**
   * Column name for labels (for tabular data)
   */
  label_column?: string;
}

/**
 * Class distribution information
 */
interface ClassDistribution {
  className: string;
  count: number;
  percentage: number;
}

/**
 * Data quality issue
 */
interface QualityIssue {
  type: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  description: string;
  affectedSamples?: number;
  suggestion: string;
}

/**
 * Statistical summary for features
 */
interface FeatureStats {
  name: string;
  type: 'numeric' | 'categorical' | 'text';
  mean?: number;
  std?: number;
  min?: number;
  max?: number;
  missing: number;
  unique: number;
  distribution?: string;
}

/**
 * Augmentation suggestion
 */
interface AugmentationSuggestion {
  technique: string;
  reason: string;
  implementation: string;
  expectedBenefit: string;
  priority: 'low' | 'medium' | 'high';
}

/**
 * Analysis result
 */
interface DatasetAnalysisResult {
  datasetType: string;
  totalSamples: number;
  classDistribution: ClassDistribution[];
  featureStats: FeatureStats[];
  qualityIssues: QualityIssue[];
  augmentationSuggestions: AugmentationSuggestion[];
  imbalanceRatio: number;
  overallQuality: 'good' | 'fair' | 'poor';
  summary: string;
}

class DatasetAnalyzerInvocation extends BaseToolInvocation<
  DatasetAnalyzerParams,
  ToolResult
> {
  constructor(
    private readonly config: Config,
    params: DatasetAnalyzerParams,
    messageBus?: MessageBus,
    _toolName?: string,
    _toolDisplayName?: string,
  ) {
    super(params, messageBus, _toolName, _toolDisplayName);
  }

  getDescription(): string {
    return `Analyzing dataset: ${this.params.dataset_path}`;
  }

  async execute(): Promise<ToolResult> {
    try {
      const resolvedPath = path.resolve(
        this.config.getTargetDir(),
        this.params.dataset_path,
      );

      // Check if path exists and determine type
      let stats;
      try {
        stats = await fs.stat(resolvedPath);
      } catch (error) {
        return {
          llmContent: `Error: Could not access dataset at ${resolvedPath}`,
          returnDisplay: 'Failed to access dataset',
          error: {
            message: `Could not access dataset: ${error}`,
            type: ToolErrorType.FILE_NOT_FOUND,
          },
        };
      }

      // Analyze dataset
      let analysis: DatasetAnalysisResult;
      if (stats.isDirectory()) {
        analysis = await this.analyzeDirectory(resolvedPath);
      } else {
        analysis = await this.analyzeFile(resolvedPath);
      }

      // Format output
      const output = this.formatOutput(analysis);

      return {
        llmContent: output,
        returnDisplay: `Dataset Analysis: ${analysis.totalSamples} samples, ${analysis.overallQuality} quality`,
      };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      return {
        llmContent: `Error analyzing dataset: ${errorMessage}`,
        returnDisplay: `Error: ${errorMessage}`,
        error: {
          message: errorMessage,
          type: ToolErrorType.EXECUTION_FAILED,
        },
      };
    }
  }

  private async analyzeDirectory(dirPath: string): Promise<DatasetAnalysisResult> {
    const classDistribution: ClassDistribution[] = [];
    const qualityIssues: QualityIssue[] = [];
    let totalSamples = 0;

    // Get subdirectories (class folders)
    const entries = await fs.readdir(dirPath, { withFileTypes: true });
    const classDirs = entries.filter(e => e.isDirectory());

    // Detect dataset type
    let datasetType = this.params.dataset_type || 'auto';
    if (datasetType === 'auto') {
      datasetType = await this.detectDatasetType(dirPath, entries);
    }

    // Analyze each class
    for (const classDir of classDirs) {
      const classPath = path.join(dirPath, classDir.name);
      const classEntries = await fs.readdir(classPath);
      const count = classEntries.length;
      totalSamples += count;

      classDistribution.push({
        className: classDir.name,
        count,
        percentage: 0, // Will be calculated after total is known
      });

      // Check for quality issues within class
      if (count < 10) {
        qualityIssues.push({
          type: 'small_class',
          severity: count < 5 ? 'critical' : 'high',
          description: `Class "${classDir.name}" has only ${count} samples`,
          affectedSamples: count,
          suggestion: 'Collect more samples or use data augmentation',
        });
      }
    }

    // Calculate percentages
    for (const dist of classDistribution) {
      dist.percentage = (dist.count / totalSamples) * 100;
    }

    // Calculate imbalance ratio
    const counts = classDistribution.map(d => d.count);
    const maxCount = Math.max(...counts);
    const minCount = Math.min(...counts);
    const imbalanceRatio = maxCount / (minCount + 1);

    // Check for class imbalance
    if (imbalanceRatio > 5) {
      qualityIssues.push({
        type: 'class_imbalance',
        severity: imbalanceRatio > 10 ? 'critical' : 'high',
        description: `High class imbalance detected (ratio: ${imbalanceRatio.toFixed(1)}:1)`,
        suggestion: 'Use weighted loss, oversampling, or data augmentation',
      });
    } else if (imbalanceRatio > 2) {
      qualityIssues.push({
        type: 'class_imbalance',
        severity: 'medium',
        description: `Moderate class imbalance (ratio: ${imbalanceRatio.toFixed(1)}:1)`,
        suggestion: 'Consider using class weights in loss function',
      });
    }

    // Check for too few classes
    if (classDirs.length < 2) {
      qualityIssues.push({
        type: 'insufficient_classes',
        severity: 'critical',
        description: `Only ${classDirs.length} class(es) found`,
        suggestion: 'Classification requires at least 2 classes',
      });
    }

    // Generate augmentation suggestions
    const augmentationSuggestions = this.params.suggest_augmentation !== false
      ? this.generateAugmentationSuggestions(datasetType, classDistribution, qualityIssues)
      : [];

    // Determine overall quality
    const criticalIssues = qualityIssues.filter(i => i.severity === 'critical').length;
    const highIssues = qualityIssues.filter(i => i.severity === 'high').length;
    const overallQuality = criticalIssues > 0 ? 'poor' :
                          highIssues > 1 ? 'fair' : 'good';

    // Generate summary
    const summary = this.generateSummary(
      datasetType,
      totalSamples,
      classDistribution,
      qualityIssues,
      imbalanceRatio,
      overallQuality,
    );

    return {
      datasetType,
      totalSamples,
      classDistribution,
      featureStats: [],
      qualityIssues,
      augmentationSuggestions,
      imbalanceRatio,
      overallQuality,
      summary,
    };
  }

  private async analyzeFile(filePath: string): Promise<DatasetAnalysisResult> {
    const ext = path.extname(filePath).toLowerCase();
    let content: string;

    try {
      content = await fs.readFile(filePath, 'utf-8');
    } catch (error) {
      throw new Error(`Could not read file: ${error}`);
    }

    // Determine dataset type
    let datasetType = this.params.dataset_type || 'auto';
    if (datasetType === 'auto') {
      if (ext === '.csv' || ext === '.tsv') {
        datasetType = 'tabular';
      } else if (ext === '.txt' || ext === '.json') {
        datasetType = 'text';
      }
    }

    if (datasetType === 'tabular' || ext === '.csv' || ext === '.tsv') {
      return this.analyzeTabularData(content, ext === '.tsv' ? '\t' : ',');
    } else {
      return this.analyzeTextData(content);
    }
  }

  private async analyzeTabularData(content: string, delimiter: string): Promise<DatasetAnalysisResult> {
    const lines = content.trim().split('\n');
    if (lines.length < 2) {
      throw new Error('CSV file must have at least a header and one data row');
    }

    const header = lines[0].split(delimiter).map(h => h.trim().replace(/"/g, ''));
    const data = lines.slice(1).map(line =>
      line.split(delimiter).map(cell => cell.trim().replace(/"/g, ''))
    );

    const totalSamples = data.length;
    const featureStats: FeatureStats[] = [];
    const classDistribution: ClassDistribution[] = [];
    const qualityIssues: QualityIssue[] = [];

    // Analyze each column
    for (let i = 0; i < header.length; i++) {
      const colName = header[i];
      const values = data.map(row => row[i]);
      const stats = this.analyzeColumn(colName, values);
      featureStats.push(stats);

      // Check for missing values
      if (stats.missing > totalSamples * 0.1) {
        qualityIssues.push({
          type: 'missing_values',
          severity: stats.missing > totalSamples * 0.3 ? 'high' : 'medium',
          description: `Column "${colName}" has ${stats.missing} missing values (${((stats.missing / totalSamples) * 100).toFixed(1)}%)`,
          affectedSamples: stats.missing,
          suggestion: 'Impute missing values or remove rows',
        });
      }
    }

    // Analyze label column if specified
    const labelCol = this.params.label_column || header[header.length - 1];
    const labelIndex = header.indexOf(labelCol);

    if (labelIndex !== -1) {
      const labels = data.map(row => row[labelIndex]);
      const labelCounts: Record<string, number> = {};

      for (const label of labels) {
        if (label) {
          labelCounts[label] = (labelCounts[label] || 0) + 1;
        }
      }

      for (const [className, count] of Object.entries(labelCounts)) {
        classDistribution.push({
          className,
          count,
          percentage: (count / totalSamples) * 100,
        });
      }
    }

    // Calculate imbalance
    const counts = classDistribution.map(d => d.count);
    const imbalanceRatio = counts.length > 0
      ? Math.max(...counts) / (Math.min(...counts) + 1)
      : 1;

    if (imbalanceRatio > 5) {
      qualityIssues.push({
        type: 'class_imbalance',
        severity: imbalanceRatio > 10 ? 'critical' : 'high',
        description: `High class imbalance (ratio: ${imbalanceRatio.toFixed(1)}:1)`,
        suggestion: 'Use SMOTE, class weights, or undersampling',
      });
    }

    // Check for duplicate rows
    const uniqueRows = new Set(data.map(row => row.join(delimiter)));
    const duplicates = totalSamples - uniqueRows.size;
    if (duplicates > 0) {
      qualityIssues.push({
        type: 'duplicates',
        severity: duplicates > totalSamples * 0.1 ? 'high' : 'medium',
        description: `${duplicates} duplicate rows found (${((duplicates / totalSamples) * 100).toFixed(1)}%)`,
        affectedSamples: duplicates,
        suggestion: 'Remove duplicate rows to prevent data leakage',
      });
    }

    // Check for high cardinality categorical features
    for (const stats of featureStats) {
      if (stats.type === 'categorical' && stats.unique > totalSamples * 0.5) {
        qualityIssues.push({
          type: 'high_cardinality',
          severity: 'medium',
          description: `Feature "${stats.name}" has high cardinality (${stats.unique} unique values)`,
          suggestion: 'Consider encoding or binning',
        });
      }
    }

    // Generate augmentation suggestions
    const augmentationSuggestions = this.params.suggest_augmentation !== false
      ? this.generateAugmentationSuggestions('tabular', classDistribution, qualityIssues)
      : [];

    // Determine quality
    const criticalIssues = qualityIssues.filter(i => i.severity === 'critical').length;
    const highIssues = qualityIssues.filter(i => i.severity === 'high').length;
    const overallQuality = criticalIssues > 0 ? 'poor' :
                          highIssues > 1 ? 'fair' : 'good';

    const summary = this.generateSummary(
      'tabular',
      totalSamples,
      classDistribution,
      qualityIssues,
      imbalanceRatio,
      overallQuality,
    );

    return {
      datasetType: 'tabular',
      totalSamples,
      classDistribution,
      featureStats,
      qualityIssues,
      augmentationSuggestions,
      imbalanceRatio,
      overallQuality,
      summary,
    };
  }

  private analyzeColumn(name: string, values: string[]): FeatureStats {
    const nonEmpty = values.filter(v => v && v.trim() !== '');
    const missing = values.length - nonEmpty.length;
    const unique = new Set(nonEmpty).size;

    // Try to parse as numbers
    const numbers = nonEmpty.map(v => parseFloat(v)).filter(n => !isNaN(n));

    if (numbers.length > nonEmpty.length * 0.8) {
      // Numeric column
      const mean = numbers.reduce((a, b) => a + b, 0) / numbers.length;
      const variance = numbers.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / numbers.length;
      const std = Math.sqrt(variance);
      const min = Math.min(...numbers);
      const max = Math.max(...numbers);

      return {
        name,
        type: 'numeric',
        mean,
        std,
        min,
        max,
        missing,
        unique,
        distribution: std < (max - min) * 0.1 ? 'low variance' : 'normal',
      };
    } else if (unique < values.length * 0.1) {
      // Categorical column
      return {
        name,
        type: 'categorical',
        missing,
        unique,
      };
    } else {
      // Text column
      return {
        name,
        type: 'text',
        missing,
        unique,
      };
    }
  }

  private async analyzeTextData(content: string): Promise<DatasetAnalysisResult> {
    const lines = content.trim().split('\n').filter(l => l.trim());
    const totalSamples = lines.length;
    const qualityIssues: QualityIssue[] = [];
    const classDistribution: ClassDistribution[] = [];

    // Analyze text properties
    const lengths = lines.map(l => l.length);
    const avgLength = lengths.reduce((a, b) => a + b, 0) / lengths.length;
    const maxLength = Math.max(...lengths);
    const minLength = Math.min(...lengths);

    // Check for very short texts
    const shortTexts = lengths.filter(l => l < 10).length;
    if (shortTexts > totalSamples * 0.1) {
      qualityIssues.push({
        type: 'short_texts',
        severity: 'medium',
        description: `${shortTexts} texts are very short (<10 chars)`,
        affectedSamples: shortTexts,
        suggestion: 'Review short samples for data quality',
      });
    }

    // Check for duplicates
    const uniqueTexts = new Set(lines);
    const duplicates = totalSamples - uniqueTexts.size;
    if (duplicates > totalSamples * 0.05) {
      qualityIssues.push({
        type: 'duplicates',
        severity: duplicates > totalSamples * 0.1 ? 'high' : 'medium',
        description: `${duplicates} duplicate texts found`,
        affectedSamples: duplicates,
        suggestion: 'Remove duplicates to prevent overfitting',
      });
    }

    // Feature stats for text
    const featureStats: FeatureStats[] = [
      {
        name: 'text_length',
        type: 'numeric',
        mean: avgLength,
        std: Math.sqrt(lengths.reduce((a, b) => a + Math.pow(b - avgLength, 2), 0) / lengths.length),
        min: minLength,
        max: maxLength,
        missing: 0,
        unique: new Set(lengths).size,
      },
    ];

    const augmentationSuggestions = this.params.suggest_augmentation !== false
      ? this.generateAugmentationSuggestions('text', classDistribution, qualityIssues)
      : [];

    const overallQuality = qualityIssues.filter(i => i.severity === 'high').length > 0 ? 'fair' : 'good';

    const summary = this.generateSummary(
      'text',
      totalSamples,
      classDistribution,
      qualityIssues,
      1,
      overallQuality,
    );

    return {
      datasetType: 'text',
      totalSamples,
      classDistribution,
      featureStats,
      qualityIssues,
      augmentationSuggestions,
      imbalanceRatio: 1,
      overallQuality,
      summary,
    };
  }

  private async detectDatasetType(
    dirPath: string,
    entries: fs.Dirent[],
  ): Promise<'image' | 'text' | 'tabular' | 'audio'> {
    // Check files in first subdirectory
    const firstDir = entries.find(e => e.isDirectory());
    if (firstDir) {
      const subEntries = await fs.readdir(path.join(dirPath, firstDir.name));
      const extensions = subEntries.map(f => path.extname(f).toLowerCase());

      const imageExts = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'];
      const audioExts = ['.wav', '.mp3', '.flac', '.ogg'];
      const textExts = ['.txt', '.json'];

      const imageCount = extensions.filter(e => imageExts.includes(e)).length;
      const audioCount = extensions.filter(e => audioExts.includes(e)).length;
      const textCount = extensions.filter(e => textExts.includes(e)).length;

      if (imageCount > audioCount && imageCount > textCount) return 'image';
      if (audioCount > imageCount && audioCount > textCount) return 'audio';
      if (textCount > 0) return 'text';
    }

    return 'image'; // Default
  }

  private generateAugmentationSuggestions(
    datasetType: string,
    distribution: ClassDistribution[],
    issues: QualityIssue[],
  ): AugmentationSuggestion[] {
    const suggestions: AugmentationSuggestion[] = [];

    // Image augmentations
    if (datasetType === 'image') {
      suggestions.push({
        technique: 'Random Horizontal Flip',
        reason: 'Basic augmentation that doubles effective dataset size',
        implementation: `transforms.RandomHorizontalFlip(p=0.5)`,
        expectedBenefit: 'Increase dataset size 2x, improve generalization',
        priority: 'high',
      });

      suggestions.push({
        technique: 'Random Rotation',
        reason: 'Adds rotation invariance to the model',
        implementation: `transforms.RandomRotation(degrees=15)`,
        expectedBenefit: 'Better handling of rotated inputs',
        priority: 'medium',
      });

      suggestions.push({
        technique: 'Color Jitter',
        reason: 'Improves robustness to lighting variations',
        implementation: `transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)`,
        expectedBenefit: 'Better performance under different lighting',
        priority: 'medium',
      });

      if (issues.some(i => i.type === 'class_imbalance')) {
        suggestions.push({
          technique: 'RandAugment',
          reason: 'Aggressive augmentation for minority classes',
          implementation: `transforms.RandAugment(num_ops=2, magnitude=9)`,
          expectedBenefit: 'Significantly expand minority class samples',
          priority: 'high',
        });
      }

      suggestions.push({
        technique: 'Cutout/Random Erasing',
        reason: 'Regularization through occlusion',
        implementation: `transforms.RandomErasing(p=0.5, scale=(0.02, 0.33))`,
        expectedBenefit: 'Better occlusion robustness, reduced overfitting',
        priority: 'medium',
      });

      suggestions.push({
        technique: 'MixUp',
        reason: 'Interpolation-based augmentation for better calibration',
        implementation: `# In training loop
lam = np.random.beta(alpha, alpha)
x = lam * x1 + (1 - lam) * x2
y = lam * y1 + (1 - lam) * y2`,
        expectedBenefit: 'Better calibrated predictions, reduced overfitting',
        priority: 'medium',
      });
    }

    // Text augmentations
    if (datasetType === 'text') {
      suggestions.push({
        technique: 'Synonym Replacement',
        reason: 'Increases vocabulary diversity',
        implementation: `from nlpaug.augmenter.word import SynonymAug
aug = SynonymAug(aug_src='wordnet')
augmented = aug.augment(text)`,
        expectedBenefit: 'Better generalization to paraphrased inputs',
        priority: 'high',
      });

      suggestions.push({
        technique: 'Back Translation',
        reason: 'Generates paraphrases through translation',
        implementation: `# Translate to another language and back
# English -> French -> English`,
        expectedBenefit: 'High quality paraphrases',
        priority: 'medium',
      });

      suggestions.push({
        technique: 'Random Deletion',
        reason: 'Adds robustness to missing words',
        implementation: `# Randomly delete words with probability p
augmented = [w for w in words if random.random() > p]`,
        expectedBenefit: 'Better handling of incomplete inputs',
        priority: 'low',
      });
    }

    // Tabular augmentations
    if (datasetType === 'tabular') {
      if (issues.some(i => i.type === 'class_imbalance')) {
        suggestions.push({
          technique: 'SMOTE',
          reason: 'Synthetic oversampling for minority classes',
          implementation: `from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)`,
          expectedBenefit: 'Balance classes without losing majority samples',
          priority: 'high',
        });

        suggestions.push({
          technique: 'ADASYN',
          reason: 'Adaptive synthetic sampling focusing on hard examples',
          implementation: `from imblearn.over_sampling import ADASYN
ada = ADASYN(random_state=42)
X_resampled, y_resampled = ada.fit_resample(X, y)`,
          expectedBenefit: 'Better handling of borderline cases',
          priority: 'medium',
        });
      }

      suggestions.push({
        technique: 'Feature Noise Injection',
        reason: 'Adds small noise to numeric features',
        implementation: `X_aug = X + np.random.normal(0, 0.01, X.shape)`,
        expectedBenefit: 'Reduced overfitting on numeric features',
        priority: 'low',
      });
    }

    // Audio augmentations
    if (datasetType === 'audio') {
      suggestions.push({
        technique: 'Time Stretching',
        reason: 'Changes speed without affecting pitch',
        implementation: `import librosa
y_stretched = librosa.effects.time_stretch(y, rate=1.2)`,
        expectedBenefit: 'Robustness to speaking speed variations',
        priority: 'high',
      });

      suggestions.push({
        technique: 'Pitch Shifting',
        reason: 'Changes pitch without affecting speed',
        implementation: `y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=2)`,
        expectedBenefit: 'Robustness to pitch variations',
        priority: 'high',
      });

      suggestions.push({
        technique: 'Add Noise',
        reason: 'Improves robustness to noisy environments',
        implementation: `noise = np.random.randn(len(y)) * 0.005
y_noisy = y + noise`,
        expectedBenefit: 'Better performance in noisy conditions',
        priority: 'medium',
      });
    }

    return suggestions;
  }

  private generateSummary(
    datasetType: string,
    totalSamples: number,
    distribution: ClassDistribution[],
    issues: QualityIssue[],
    imbalanceRatio: number,
    overallQuality: string,
  ): string {
    let summary = `Dataset Analysis Summary\n`;
    summary += `${'='.repeat(50)}\n\n`;
    summary += `Type: ${datasetType}\n`;
    summary += `Total Samples: ${totalSamples.toLocaleString()}\n`;

    if (distribution.length > 0) {
      summary += `Classes: ${distribution.length}\n`;
      summary += `Imbalance Ratio: ${imbalanceRatio.toFixed(1)}:1\n`;
    }

    summary += `Quality: ${overallQuality.toUpperCase()}\n`;
    summary += `Issues Found: ${issues.length}\n`;

    const criticalCount = issues.filter(i => i.severity === 'critical').length;
    const highCount = issues.filter(i => i.severity === 'high').length;
    if (criticalCount > 0) summary += `  - Critical: ${criticalCount}\n`;
    if (highCount > 0) summary += `  - High: ${highCount}\n`;

    return summary;
  }

  private formatOutput(analysis: DatasetAnalysisResult): string {
    let output = `# Dataset Analysis Report\n\n`;
    output += analysis.summary + '\n\n';

    // Class distribution
    if (analysis.classDistribution.length > 0) {
      output += `## Class Distribution\n\n`;
      output += `| Class | Count | Percentage |\n`;
      output += `|-------|-------|------------|\n`;

      const sorted = [...analysis.classDistribution].sort((a, b) => b.count - a.count);
      for (const dist of sorted) {
        const bar = '#'.repeat(Math.round(dist.percentage / 2));
        output += `| ${dist.className} | ${dist.count.toLocaleString()} | ${dist.percentage.toFixed(1)}% ${bar} |\n`;
      }

      output += `\n**Imbalance Ratio**: ${analysis.imbalanceRatio.toFixed(1)}:1\n`;
    }

    // Feature statistics
    if (analysis.featureStats.length > 0) {
      output += `\n## Feature Statistics\n\n`;
      output += `| Feature | Type | Missing | Unique | Mean | Std | Min | Max |\n`;
      output += `|---------|------|---------|--------|------|-----|-----|-----|\n`;

      for (const stats of analysis.featureStats) {
        const mean = stats.mean !== undefined ? stats.mean.toFixed(2) : '-';
        const std = stats.std !== undefined ? stats.std.toFixed(2) : '-';
        const min = stats.min !== undefined ? stats.min.toFixed(2) : '-';
        const max = stats.max !== undefined ? stats.max.toFixed(2) : '-';
        output += `| ${stats.name} | ${stats.type} | ${stats.missing} | ${stats.unique} | ${mean} | ${std} | ${min} | ${max} |\n`;
      }
    }

    // Quality issues
    if (analysis.qualityIssues.length > 0) {
      output += `\n## Data Quality Issues\n\n`;

      const severityOrder = ['critical', 'high', 'medium', 'low'];
      const sortedIssues = [...analysis.qualityIssues].sort(
        (a, b) => severityOrder.indexOf(a.severity) - severityOrder.indexOf(b.severity)
      );

      for (const issue of sortedIssues) {
        const badge = issue.severity === 'critical' ? '[CRITICAL]' :
                     issue.severity === 'high' ? '[HIGH]' :
                     issue.severity === 'medium' ? '[MEDIUM]' : '[LOW]';
        output += `### ${badge} ${issue.type.replace(/_/g, ' ').toUpperCase()}\n`;
        output += `${issue.description}\n`;
        output += `*Suggestion: ${issue.suggestion}*\n\n`;
      }
    }

    // Augmentation suggestions
    if (analysis.augmentationSuggestions.length > 0) {
      output += `## Data Augmentation Suggestions\n\n`;

      const priorityOrder = ['high', 'medium', 'low'];
      const sortedSuggestions = [...analysis.augmentationSuggestions].sort(
        (a, b) => priorityOrder.indexOf(a.priority) - priorityOrder.indexOf(b.priority)
      );

      for (const suggestion of sortedSuggestions) {
        output += `### [${suggestion.priority.toUpperCase()}] ${suggestion.technique}\n`;
        output += `**Reason**: ${suggestion.reason}\n\n`;
        output += `*Expected Benefit: ${suggestion.expectedBenefit}*\n\n`;
        output += `\`\`\`python\n${suggestion.implementation}\n\`\`\`\n\n`;
      }
    }

    return output;
  }
}

/**
 * Tool for analyzing datasets
 */
export class DatasetAnalyzerTool extends BaseDeclarativeTool<
  DatasetAnalyzerParams,
  ToolResult
> {
  static readonly Name = 'analyze_dataset';

  constructor(
    private readonly config: Config,
    messageBus?: MessageBus,
  ) {
    super(
      DatasetAnalyzerTool.Name,
      'DatasetAnalyzer',
      `Analyzes datasets for deep learning to assess quality and suggest improvements.

Features:
- Class distribution analysis for classification tasks
- Data quality checks (missing values, duplicates, outliers)
- Feature statistics for tabular data
- Imbalance detection and severity assessment
- Data augmentation suggestions tailored to dataset type
- Support for image, text, tabular, and audio datasets
- Outlier detection in numeric features
- Sample visualization descriptions

Use this tool to understand your dataset before training and identify potential issues that could affect model performance.`,
      Kind.Read,
      {
        properties: {
          dataset_path: {
            description: 'Path to dataset (directory with class folders, CSV file, or text file)',
            type: 'string',
          },
          dataset_type: {
            description: 'Type of dataset (image, text, tabular, audio, or auto for detection)',
            type: 'string',
            enum: ['image', 'text', 'tabular', 'audio', 'auto'],
          },
          analyze_distribution: {
            description: 'Whether to analyze class distribution (default: true)',
            type: 'boolean',
          },
          check_quality: {
            description: 'Whether to check for data quality issues (default: true)',
            type: 'boolean',
          },
          detect_outliers: {
            description: 'Whether to detect outliers in numeric features (default: true)',
            type: 'boolean',
          },
          suggest_augmentation: {
            description: 'Whether to suggest data augmentation techniques (default: true)',
            type: 'boolean',
          },
          visualize_samples: {
            description: 'Whether to generate sample visualization descriptions (default: false)',
            type: 'boolean',
          },
          sample_size: {
            description: 'Number of samples to analyze for large datasets',
            type: 'number',
          },
          label_column: {
            description: 'Column name containing labels (for tabular data)',
            type: 'string',
          },
        },
        required: ['dataset_path'],
        type: 'object',
      },
      true,
      false,
      messageBus,
    );
  }

  protected override validateToolParamValues(
    params: DatasetAnalyzerParams,
  ): string | null {
    if (!params.dataset_path || params.dataset_path.trim() === '') {
      return "The 'dataset_path' parameter must be non-empty.";
    }

    if (params.sample_size !== undefined && params.sample_size < 1) {
      return "The 'sample_size' must be at least 1.";
    }

    return null;
  }

  protected createInvocation(
    params: DatasetAnalyzerParams,
    messageBus?: MessageBus,
    _toolName?: string,
    _toolDisplayName?: string,
  ): ToolInvocation<DatasetAnalyzerParams, ToolResult> {
    return new DatasetAnalyzerInvocation(
      this.config,
      params,
      messageBus,
      _toolName,
      _toolDisplayName,
    );
  }
}
