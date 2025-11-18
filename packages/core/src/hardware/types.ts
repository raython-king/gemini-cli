/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Hardware topology and performance analysis types
 */

export interface CPUInfo {
  model: string;
  cores: number;
  threads: number;
  frequency: number; // MHz
  architecture: string;
  vendor: string;
}

export interface GPUInfo {
  id: number;
  name: string;
  memory: number; // MB
  computeCapability?: string;
  cudaCores?: number;
  vendor: string; // NVIDIA, AMD, Intel, etc.
  busId: string;
  temperature?: number;
  powerLimit?: number;
  utilization?: number;
}

export interface MemoryInfo {
  total: number; // MB
  available: number; // MB
  used: number; // MB
  swap: {
    total: number;
    used: number;
    free: number;
  };
}

export interface NetworkInterface {
  name: string;
  speed: number; // Mbps
  type: string; // ethernet, infiniband, etc.
  mtu: number;
  ipAddress?: string;
}

export interface InterconnectInfo {
  type: string; // PCIe, NVLink, InfiniBand, etc.
  bandwidth: number; // GB/s
  latency: number; // microseconds
  topology: string; // mesh, ring, all-to-all, etc.
}

export interface HardwareTopology {
  cpu: CPUInfo;
  gpus: GPUInfo[];
  memory: MemoryInfo;
  networks: NetworkInterface[];
  interconnects: InterconnectInfo[];
  nodeCount: number;
  timestamp: Date;
}

export interface CommunicationBenchmark {
  source: string;
  destination: string;
  bandwidth: number; // GB/s
  latency: number; // microseconds
  protocol: string; // NCCL, MPI, gRPC, etc.
  messageSize: number; // bytes
}

export interface HardwareAnalysisResult {
  topology: HardwareTopology;
  benchmarks: CommunicationBenchmark[];
  recommendations: string[];
  bottlenecks: string[];
  score: number; // 0-100, overall hardware capability score
}

export interface TrainingStrategy {
  parallelismType: 'data' | 'model' | 'pipeline' | 'hybrid';
  dataParallelDegree?: number;
  modelParallelDegree?: number;
  pipelineStages?: number;
  microBatchSize?: number;
  globalBatchSize?: number;
  gradientAccumulationSteps?: number;
  mixedPrecision?: boolean;
  activationCheckpointing?: boolean;
  communicationBackend: 'nccl' | 'gloo' | 'mpi';
  optimizerSharding?: boolean;
  tensorParallel?: boolean;
  sequenceParallel?: boolean;
  estimatedThroughput?: number; // samples/second
  estimatedMemoryUsage?: number; // MB
  reasoning: string[];
}

export interface TrainingExecutionPlan {
  strategy: TrainingStrategy;
  environmentVariables: Record<string, string>;
  launchCommand: string;
  workerDistribution: Array<{
    nodeId: string;
    gpuIds: number[];
    role: string; // master, worker, etc.
  }>;
  monitoringConfig?: {
    metricsInterval: number;
    checkpointInterval: number;
    logLevel: string;
  };
}
