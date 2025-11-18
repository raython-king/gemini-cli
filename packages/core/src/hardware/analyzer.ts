/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Hardware analysis and recommendation engine
 */

import { HardwareDetector } from './detector.js';
import { CommunicationBenchmarker } from './benchmark.js';
import type {
  HardwareAnalysisResult,
  HardwareTopology,
  CommunicationBenchmark,
  InterconnectInfo,
  GPUInfo,
} from './types.js';

export class HardwareAnalyzer {
  private detector: HardwareDetector;
  private benchmarker: CommunicationBenchmarker;

  constructor() {
    this.detector = new HardwareDetector();
    this.benchmarker = new CommunicationBenchmarker();
  }

  /**
   * Perform comprehensive hardware analysis
   */
  async analyze(): Promise<HardwareAnalysisResult> {
    // Detect hardware topology
    const topology = await this.detector.detectTopology();

    // Run communication benchmarks
    const benchmarks = await this.benchmarker.runBenchmarks(topology.gpus);

    // Analyze and generate recommendations
    const recommendations = this.generateRecommendations(topology, benchmarks);
    const bottlenecks = this.identifyBottlenecks(topology, benchmarks);
    const score = this.calculateScore(topology);

    return {
      topology,
      benchmarks,
      recommendations,
      bottlenecks,
      score,
    };
  }

  /**
   * Generate recommendations based on hardware analysis
   */
  private generateRecommendations(
    topology: HardwareTopology,
    benchmarks: CommunicationBenchmark[],
  ): string[] {
    const recommendations: string[] = [];

    // GPU recommendations
    if (topology.gpus.length === 0) {
      recommendations.push(
        'No GPUs detected. Training will run on CPU only, which is significantly slower for deep learning workloads.',
      );
    } else if (topology.gpus.length === 1) {
      recommendations.push(
        'Single GPU detected. Consider data parallelism with multiple GPUs for faster training on large datasets.',
      );
    } else if (topology.gpus.length >= 2 && topology.gpus.length <= 4) {
      const hasNVLink = topology.interconnects.some(
        (ic: InterconnectInfo) => ic.type === 'NVLink',
      );
      if (hasNVLink) {
        recommendations.push(
          `${topology.gpus.length} GPUs with NVLink detected. Excellent for data parallelism and small model parallelism. Consider using NCCL backend for optimal communication.`,
        );
      } else {
        recommendations.push(
          `${topology.gpus.length} GPUs with PCIe interconnect. Good for data parallelism. PCIe bandwidth may become a bottleneck for frequent synchronization.`,
        );
      }
    } else if (topology.gpus.length >= 8) {
      recommendations.push(
        `${topology.gpus.length} GPUs detected. Excellent for hybrid parallelism strategies. Consider combining data parallelism with model/pipeline parallelism for very large models.`,
      );
    }

    // Memory recommendations
    const avgGPUMemory =
      topology.gpus.reduce((sum: number, gpu: GPUInfo) => sum + gpu.memory, 0) /
      topology.gpus.length;

    if (avgGPUMemory > 0 && avgGPUMemory < 8000) {
      recommendations.push(
        `Average GPU memory is ${Math.round(avgGPUMemory / 1024)}GB. Consider using gradient checkpointing and mixed precision training for large models.`,
      );
    } else if (avgGPUMemory >= 40000) {
      recommendations.push(
        `High GPU memory (${Math.round(avgGPUMemory / 1024)}GB) available. You can train larger models or use larger batch sizes.`,
      );
    }

    // CPU recommendations
    if (topology.cpu.cores < topology.gpus.length * 4) {
      recommendations.push(
        `CPU core count (${topology.cpu.cores}) may be insufficient for optimal data loading with ${topology.gpus.length} GPUs. Consider using fewer data loader workers per GPU.`,
      );
    }

    // Communication recommendations
    const avgGPUBandwidth =
      benchmarks
        .filter((b) => b.protocol === 'NCCL')
        .reduce((sum, b) => sum + b.bandwidth, 0) /
      (benchmarks.filter((b) => b.protocol === 'NCCL').length || 1);

    if (avgGPUBandwidth < 20) {
      recommendations.push(
        `GPU communication bandwidth is ${avgGPUBandwidth.toFixed(1)} GB/s. Consider reducing communication frequency through gradient accumulation or larger batch sizes.`,
      );
    } else if (avgGPUBandwidth > 100) {
      recommendations.push(
        `Excellent GPU communication bandwidth (${avgGPUBandwidth.toFixed(1)} GB/s). You can use communication-intensive strategies like pipeline parallelism.`,
      );
    }

    // Network recommendations for multi-node
    const hasNetwork = benchmarks.some(
      (b) => b.protocol === 'TCP' || b.protocol === 'InfiniBand',
    );
    if (hasNetwork) {
      recommendations.push(
        'Multi-node setup detected. Use NCCL with GPUDirect RDMA for optimal cross-node communication if available.',
      );
    }

    return recommendations;
  }

  /**
   * Identify performance bottlenecks
   */
  private identifyBottlenecks(
    topology: HardwareTopology,
    benchmarks: CommunicationBenchmark[],
  ): string[] {
    const bottlenecks: string[] = [];

    // GPU memory bottleneck
    const minGPUMemory = Math.min(
      ...topology.gpus.map((g: GPUInfo) => g.memory),
      Infinity,
    );
    if (minGPUMemory < 8000 && minGPUMemory > 0) {
      bottlenecks.push(
        `Limited GPU memory (${Math.round(minGPUMemory / 1024)}GB) may restrict model size and batch size`,
      );
    }

    // System memory bottleneck
    if (topology.memory.available < topology.gpus.length * 16000) {
      bottlenecks.push(
        `System RAM (${Math.round(topology.memory.total / 1024)}GB) may be insufficient for efficient data loading with ${topology.gpus.length} GPUs`,
      );
    }

    // CPU bottleneck
    const recommendedCores = topology.gpus.length * 4;
    if (topology.cpu.cores < recommendedCores) {
      bottlenecks.push(
        `CPU cores (${topology.cpu.cores}) below recommended (${recommendedCores}) for ${topology.gpus.length} GPUs`,
      );
    }

    // Communication bottleneck
    const pcieBenchmarks = benchmarks.filter(
      (b) => b.protocol === 'NCCL' && b.bandwidth < 30,
    );
    if (pcieBenchmarks.length > 0) {
      bottlenecks.push(
        `GPU communication bandwidth is limited by PCIe. Average: ${pcieBenchmarks[0].bandwidth.toFixed(1)} GB/s`,
      );
    }

    // Network bottleneck
    const networkBandwidth = benchmarks.find(
      (b) => b.protocol === 'TCP',
    )?.bandwidth;
    if (networkBandwidth && networkBandwidth < 1) {
      bottlenecks.push(
        `Network bandwidth (${networkBandwidth.toFixed(1)} GB/s) may bottleneck multi-node training`,
      );
    }

    return bottlenecks;
  }

  /**
   * Calculate overall hardware capability score (0-100)
   */
  private calculateScore(topology: HardwareTopology): number {
    let score = 0;

    // GPU score (40 points max)
    const gpuCount = topology.gpus.length;
    if (gpuCount === 0) {
      score += 0;
    } else if (gpuCount === 1) {
      score += 15;
    } else if (gpuCount <= 4) {
      score += 25;
    } else if (gpuCount <= 8) {
      score += 35;
    } else {
      score += 40;
    }

    // GPU memory score (20 points max)
    const avgGPUMemory =
      topology.gpus.reduce((sum: number, gpu: GPUInfo) => sum + gpu.memory, 0) /
      (topology.gpus.length || 1);
    if (avgGPUMemory >= 80000) {
      score += 20;
    } else if (avgGPUMemory >= 40000) {
      score += 15;
    } else if (avgGPUMemory >= 16000) {
      score += 10;
    } else if (avgGPUMemory >= 8000) {
      score += 5;
    }

    // Interconnect score (20 points max)
    const hasNVLink = topology.interconnects.some(
      (ic: InterconnectInfo) => ic.type === 'NVLink',
    );
    const hasInfiniBand = topology.interconnects.some(
      (ic: InterconnectInfo) => ic.type === 'InfiniBand',
    );
    if (hasNVLink && hasInfiniBand) {
      score += 20;
    } else if (hasNVLink) {
      score += 15;
    } else if (hasInfiniBand) {
      score += 12;
    } else {
      score += 5;
    }

    // CPU score (10 points max)
    const coresPerGPU = topology.cpu.cores / (topology.gpus.length || 1);
    if (coresPerGPU >= 8) {
      score += 10;
    } else if (coresPerGPU >= 4) {
      score += 7;
    } else if (coresPerGPU >= 2) {
      score += 4;
    } else {
      score += 2;
    }

    // Memory score (10 points max)
    const memoryPerGPU = topology.memory.total / (topology.gpus.length || 1);
    if (memoryPerGPU >= 64000) {
      score += 10;
    } else if (memoryPerGPU >= 32000) {
      score += 7;
    } else if (memoryPerGPU >= 16000) {
      score += 4;
    } else {
      score += 2;
    }

    return Math.min(100, Math.round(score));
  }
}
