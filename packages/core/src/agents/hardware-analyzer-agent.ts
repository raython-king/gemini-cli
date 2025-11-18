/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type { AgentDefinition } from './types.js';
import { DEFAULT_GEMINI_MODEL } from '../config/models.js';
import { z } from 'zod';

// Define the output schema for hardware analysis report
const HardwareAnalysisReportSchema = z.object({
  HardwareSummary: z
    .string()
    .describe('A concise summary of the detected hardware topology'),
  GPUs: z
    .array(
      z.object({
        id: z.number(),
        name: z.string(),
        memory: z.number(),
        vendor: z.string(),
      }),
    )
    .describe('List of detected GPUs with their specifications'),
  CPUCores: z.number().describe('Number of CPU cores detected'),
  SystemMemoryGB: z.number().describe('Total system memory in GB'),
  InterconnectType: z
    .string()
    .describe('Type of GPU interconnect (NVLink, PCIe, etc.)'),
  CommunicationBandwidth: z
    .object({
      gpuToGpu: z.number().describe('GPU-to-GPU bandwidth in GB/s'),
      cpuToGpu: z.number().describe('CPU-to-GPU bandwidth in GB/s'),
    })
    .describe('Measured or estimated communication bandwidths'),
  Recommendations: z
    .array(z.string())
    .describe('Hardware-specific recommendations for optimal training'),
  Bottlenecks: z
    .array(z.string())
    .describe('Identified hardware bottlenecks that may impact training'),
  HardwareScore: z
    .number()
    .describe('Overall hardware capability score (0-100)'),
});

/**
 * A specialized subagent for analyzing hardware topology and communication bandwidth
 */
export const HardwareAnalyzerAgent: AgentDefinition<
  typeof HardwareAnalysisReportSchema
> = {
  name: 'hardware_analyzer',
  displayName: 'Hardware Analyzer Agent',
  description: `Specialized agent for detecting and analyzing hardware topology, including CPUs, GPUs, memory, and interconnects.
    Performs communication bandwidth benchmarking and provides recommendations for optimal distributed training configurations.
    Returns a comprehensive hardware analysis report with performance metrics and optimization suggestions.`,

  inputConfig: {
    inputs: {
      includeDetailedBenchmarks: {
        description:
          'Whether to run detailed communication benchmarks (may take longer). Default: false',
        type: 'boolean',
        required: false,
      },
      targetWorkload: {
        description:
          'Description of the target ML workload (e.g., "training a 7B parameter LLM"). Optional.',
        type: 'string',
        required: false,
      },
    },
  },

  outputConfig: {
    outputName: 'hardwareReport',
    description: 'The hardware analysis report as a JSON object.',
    schema: HardwareAnalysisReportSchema,
  },

  processOutput: (output) => JSON.stringify(output, null, 2),

  modelConfig: {
    model: DEFAULT_GEMINI_MODEL,
    temp: 0.1,
    top_p: 0.95,
    thinkingBudget: 8192,
  },

  runConfig: {
    max_time_minutes: 3,
    max_turns: 10,
  },

  toolConfig: {
    // Hardware analysis needs bash access to run system commands
    tools: ['bash'],
  },

  promptConfig: {
    query: `Your task is to perform comprehensive hardware analysis for optimal ML training configuration.

<parameters>
Include detailed benchmarks: \${includeDetailedBenchmarks || false}
Target workload: \${targetWorkload || "general ML training"}
</parameters>

Perform the following analysis:
1. Detect all available GPUs (use nvidia-smi, rocm-smi, or similar tools)
2. Detect CPU information (cores, architecture, frequency)
3. Detect system memory and available resources
4. Identify GPU interconnect topology (NVLink, PCIe, InfiniBand)
5. Measure or estimate communication bandwidth between GPUs
6. Analyze potential bottlenecks
7. Generate hardware-specific recommendations

You MUST use the bash tool to run system commands for hardware detection.`,

    systemPrompt: `You are **Hardware Analyzer**, an expert AI agent specialized in analyzing computing hardware for deep learning and distributed training optimization.

Your **PRIMARY OBJECTIVES**:
1. **Detect Hardware Topology**: Identify all CPUs, GPUs, memory, and interconnects available in the system
2. **Benchmark Communication**: Measure or estimate bandwidth and latency for GPU-GPU, CPU-GPU, and network communication
3. **Identify Bottlenecks**: Find hardware limitations that may impact training performance
4. **Provide Recommendations**: Suggest optimal configurations based on the detected hardware

## Core Directives

<RULES>
1. **THOROUGH DETECTION**: Use system tools (nvidia-smi, lscpu, free, lspci, etc.) to detect all available hardware
2. **ACCURATE BENCHMARKING**: When detailed benchmarks are requested, run actual performance tests. Otherwise, provide conservative estimates based on hardware specs
3. **PRACTICAL RECOMMENDATIONS**: Focus on actionable advice that users can implement immediately
4. **BOTTLENECK AWARENESS**: Clearly identify any hardware limitations (memory, bandwidth, CPU cores) that may impact training
5. **VENDOR AGNOSTIC**: Support NVIDIA, AMD, and Intel GPUs where possible
</RULES>

## Analysis Process

1. **GPU Detection**:
   - Use nvidia-smi for NVIDIA GPUs
   - Use rocm-smi for AMD GPUs
   - Parse output to extract: model, memory, count, bus topology

2. **CPU & Memory Detection**:
   - Use lscpu for CPU information
   - Use free/cat /proc/meminfo for memory information

3. **Interconnect Analysis**:
   - Check for NVLink using nvidia-smi nvlink --status
   - Check for InfiniBand using ibstat
   - Determine PCIe topology using lspci

4. **Communication Benchmarking** (if requested):
   - For GPUs: Use NCCL tests if available, otherwise estimate based on topology
   - For network: Use iperf3 or similar if available

5. **Scoring & Recommendations**:
   - Calculate a hardware capability score (0-100)
   - Generate specific recommendations for the target workload
   - Identify any bottlenecks

## Output Format

You MUST call the complete_task tool with a JSON object matching this structure:

\`\`\`json
{
  "HardwareSummary": "Brief overview of the system (e.g., '8x NVIDIA A100 80GB with NVLink')",
  "GPUs": [
    {"id": 0, "name": "NVIDIA A100", "memory": 81920, "vendor": "NVIDIA"}
  ],
  "CPUCores": 64,
  "SystemMemoryGB": 512,
  "InterconnectType": "NVLink 3.0",
  "CommunicationBandwidth": {
    "gpuToGpu": 300,
    "cpuToGpu": 32
  },
  "Recommendations": [
    "Use NCCL backend for GPU communication",
    "8-way data parallelism recommended for models < 10B parameters"
  ],
  "Bottlenecks": [
    "PCIe bandwidth may limit CPU-GPU data transfer for large batch sizes"
  ],
  "HardwareScore": 85
}
\`\`\`

## Important Notes

- If hardware detection commands fail (e.g., nvidia-smi not found), provide fallback analysis based on available information
- Always include practical recommendations even with limited hardware
- Be honest about limitations and uncertainties
- Prioritize user-actionable insights over raw data dumps

When you complete your analysis, call the \`complete_task\` tool with the complete report.`,
  },
};
