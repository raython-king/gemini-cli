/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Hardware topology analysis and training optimization module
 *
 * This module provides comprehensive hardware detection, analysis, and training strategy optimization
 * for distributed deep learning workloads.
 */

export * from './types.js';
export { HardwareDetector } from './detector.js';
export { CommunicationBenchmarker } from './benchmark.js';
export { HardwareAnalyzer } from './analyzer.js';
export { TrainingStrategyOptimizer } from './strategy-optimizer.js';
export type { ModelConfig, TrainingConfig } from './strategy-optimizer.js';
