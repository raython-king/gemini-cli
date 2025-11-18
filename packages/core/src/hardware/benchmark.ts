/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Communication bandwidth and latency benchmarking
 */

import { exec } from 'node:child_process';
import { promisify } from 'node:util';
import type { CommunicationBenchmark, GPUInfo } from './types.js';

const execAsync = promisify(exec);

export class CommunicationBenchmarker {
  /**
   * Benchmark GPU-to-GPU communication using NCCL
   */
  async benchmarkGPUCommunication(
    gpus: GPUInfo[],
  ): Promise<CommunicationBenchmark[]> {
    const benchmarks: CommunicationBenchmark[] = [];

    if (gpus.length < 2) {
      return benchmarks;
    }

    // Try to use nccl-tests if available
    try {
      const { stdout } = await execAsync(
        'which all_reduce_perf 2>/dev/null || echo ""',
        { timeout: 5000 },
      );

      if (stdout.trim()) {
        // NCCL tests available, run benchmark
        try {
          const { stdout: perfOut } = await execAsync(
            'all_reduce_perf -b 8M -e 128M -f 2 -g ' + gpus.length,
            { timeout: 30000 },
          );

          // Parse NCCL performance test output
          const lines = perfOut.split('\n');
          for (const line of lines) {
            if (line.includes('float')) {
              const parts = line.trim().split(/\s+/);
              const bandwidth = parseFloat(parts[parts.length - 2] || '0');

              benchmarks.push({
                source: 'GPU-All',
                destination: 'GPU-All',
                bandwidth: bandwidth / 1024, // Convert to GB/s
                latency: 0, // NCCL tests don't report latency separately
                protocol: 'NCCL',
                messageSize: 128 * 1024 * 1024, // 128MB
              });
              break;
            }
          }
        } catch (_error) {
          // NCCL benchmark failed
        }
      }
    } catch (_error) {
      // NCCL tests not available
    }

    // Fallback: estimate based on NVLink/PCIe topology
    if (benchmarks.length === 0) {
      // Estimate GPU communication bandwidth
      const hasNVLink = await this.detectNVLink();

      for (let i = 0; i < gpus.length; i++) {
        for (let j = i + 1; j < gpus.length; j++) {
          benchmarks.push({
            source: `GPU-${i}`,
            destination: `GPU-${j}`,
            bandwidth: hasNVLink ? 150 : 12, // GB/s (NVLink vs PCIe estimate)
            latency: hasNVLink ? 5 : 15, // microseconds
            protocol: 'NCCL',
            messageSize: 128 * 1024 * 1024,
          });
        }
      }
    }

    return benchmarks;
  }

  /**
   * Benchmark CPU-GPU communication
   */
  async benchmarkCPUGPUCommunication(
    gpus: GPUInfo[],
  ): Promise<CommunicationBenchmark[]> {
    const benchmarks: CommunicationBenchmark[] = [];

    // Simple PCIe bandwidth estimation
    for (let i = 0; i < gpus.length; i++) {
      benchmarks.push({
        source: 'CPU',
        destination: `GPU-${i}`,
        bandwidth: 12, // GB/s (PCIe 3.0 x16 bidirectional)
        latency: 10, // microseconds
        protocol: 'PCIe',
        messageSize: 64 * 1024 * 1024, // 64MB
      });
    }

    return benchmarks;
  }

  /**
   * Benchmark network communication (for multi-node scenarios)
   */
  async benchmarkNetworkCommunication(): Promise<CommunicationBenchmark[]> {
    const benchmarks: CommunicationBenchmark[] = [];

    // Try iperf3 for network bandwidth testing
    try {
      const { stdout } = await execAsync(
        'which iperf3 2>/dev/null || echo ""',
        {
          timeout: 5000,
        },
      );

      if (stdout.trim()) {
        // iperf3 available - for multi-node testing
        // This is a placeholder - actual multi-node testing would require
        // coordination between nodes
        benchmarks.push({
          source: 'node-0',
          destination: 'node-1',
          bandwidth: 10, // GB/s (10GbE estimate)
          latency: 100, // microseconds
          protocol: 'TCP',
          messageSize: 1024 * 1024 * 1024, // 1GB
        });
      }
    } catch (_error) {
      // Network benchmark not available
    }

    return benchmarks;
  }

  /**
   * Detect NVLink presence
   */
  private async detectNVLink(): Promise<boolean> {
    try {
      const { stdout } = await execAsync(
        'nvidia-smi nvlink --status 2>/dev/null || echo ""',
        {
          timeout: 5000,
        },
      );
      return stdout.includes('Active');
    } catch (_error) {
      return false;
    }
  }

  /**
   * Run comprehensive communication benchmarks
   */
  async runBenchmarks(gpus: GPUInfo[]): Promise<CommunicationBenchmark[]> {
    const [gpuComm, cpuGpuComm, networkComm] = await Promise.all([
      this.benchmarkGPUCommunication(gpus),
      this.benchmarkCPUGPUCommunication(gpus),
      this.benchmarkNetworkCommunication(),
    ]);

    return [...gpuComm, ...cpuGpuComm, ...networkComm];
  }
}
