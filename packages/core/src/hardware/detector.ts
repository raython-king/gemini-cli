/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Hardware topology detection and analysis
 */

import { exec } from 'node:child_process';
import { promisify } from 'node:util';
import os from 'node:os';
import type {
  CPUInfo,
  GPUInfo,
  MemoryInfo,
  NetworkInterface,
  InterconnectInfo,
  HardwareTopology,
} from './types.js';

const execAsync = promisify(exec);

export class HardwareDetector {
  /**
   * Detect CPU information
   */
  async detectCPU(): Promise<CPUInfo> {
    const cpus = os.cpus();
    const model = cpus[0]?.model || 'Unknown';
    const cores = cpus.length;
    const frequency = cpus[0]?.speed || 0;

    let architecture = os.arch();
    let vendor = 'Unknown';

    // Try to get more detailed CPU info on Linux
    if (os.platform() === 'linux') {
      try {
        const { stdout } = await execAsync('lscpu');
        const lines = stdout.split('\n');

        for (const line of lines) {
          if (line.includes('Vendor ID:')) {
            vendor = line.split(':')[1]?.trim() || vendor;
          }
          if (line.includes('Architecture:')) {
            architecture = line.split(':')[1]?.trim() || architecture;
          }
        }
      } catch (_error) {
        // Fallback to basic detection
      }
    }

    // Detect logical cores (threads)
    const threads = cores;

    return {
      model,
      cores, // Physical cores (approximation)
      threads,
      frequency,
      architecture,
      vendor,
    };
  }

  /**
   * Detect GPU information
   */
  async detectGPUs(): Promise<GPUInfo[]> {
    const gpus: GPUInfo[] = [];

    // Try NVIDIA GPUs first
    try {
      const { stdout } = await execAsync(
        'nvidia-smi --query-gpu=index,name,memory.total,pci.bus_id,temperature.gpu,power.limit,utilization.gpu --format=csv,noheader,nounits',
      );

      const lines = stdout.trim().split('\n');
      for (const line of lines) {
        const [id, name, memory, busId, temp, powerLimit, util] = line
          .split(',')
          .map((s) => s.trim());

        gpus.push({
          id: parseInt(id, 10),
          name,
          memory: parseFloat(memory),
          vendor: 'NVIDIA',
          busId,
          temperature: parseFloat(temp),
          powerLimit: parseFloat(powerLimit),
          utilization: parseFloat(util),
        });
      }

      // Get CUDA compute capability
      try {
        const { stdout: cudaOut } = await execAsync(
          'nvidia-smi --query-gpu=compute_cap --format=csv,noheader',
        );
        const capabilities = cudaOut.trim().split('\n');
        capabilities.forEach((cap, idx) => {
          if (gpus[idx]) {
            gpus[idx].computeCapability = cap.trim();
          }
        });
      } catch (_e) {
        // Compute capability detection failed
      }
    } catch (_error) {
      // NVIDIA GPUs not found or nvidia-smi not available
    }

    // Try AMD GPUs
    try {
      const { stdout } = await execAsync(
        'rocm-smi --showproductname --showmeminfo vram',
      );
      // Parse AMD GPU info (rocm-smi output format)
      // This is a simplified parser
      if (stdout.includes('GPU')) {
        // Add AMD GPU detection logic here
      }
    } catch (_error) {
      // AMD GPUs not found
    }

    // Try Intel GPUs (basic detection)
    try {
      if (os.platform() === 'linux') {
        const { stdout } = await execAsync('lspci | grep -i vga');
        if (stdout.toLowerCase().includes('intel')) {
          // Basic Intel GPU detection
          gpus.push({
            id: gpus.length,
            name: 'Intel Integrated Graphics',
            memory: 0, // Shared memory
            vendor: 'Intel',
            busId: 'unknown',
          });
        }
      }
    } catch (_error) {
      // Intel GPU detection failed
    }

    return gpus;
  }

  /**
   * Detect memory information
   */
  async detectMemory(): Promise<MemoryInfo> {
    const totalMem = os.totalmem() / (1024 * 1024); // Convert to MB
    const freeMem = os.freemem() / (1024 * 1024);
    const usedMem = totalMem - freeMem;

    let swap = {
      total: 0,
      used: 0,
      free: 0,
    };

    // Get swap info on Linux
    if (os.platform() === 'linux') {
      try {
        const { stdout } = await execAsync('free -m');
        const lines = stdout.split('\n');
        const swapLine = lines.find((line) => line.startsWith('Swap:'));

        if (swapLine) {
          const parts = swapLine.split(/\s+/);
          swap = {
            total: parseInt(parts[1] || '0', 10),
            used: parseInt(parts[2] || '0', 10),
            free: parseInt(parts[3] || '0', 10),
          };
        }
      } catch (_error) {
        // Swap detection failed
      }
    }

    return {
      total: totalMem,
      available: freeMem,
      used: usedMem,
      swap,
    };
  }

  /**
   * Detect network interfaces
   */
  async detectNetworkInterfaces(): Promise<NetworkInterface[]> {
    const interfaces: NetworkInterface[] = [];
    const networkInterfaces = os.networkInterfaces();

    for (const [name, addrs] of Object.entries(networkInterfaces)) {
      if (!addrs) continue;

      // Skip loopback
      if (name === 'lo' || name === 'lo0') continue;

      const ipv4 = addrs.find((addr) => addr.family === 'IPv4');

      let speed = 1000; // Default 1 Gbps
      let mtu = 1500;
      let type = 'ethernet';

      // Try to get interface details on Linux
      if (os.platform() === 'linux') {
        try {
          const { stdout } = await execAsync(
            `ethtool ${name} 2>/dev/null || echo ""`,
          );
          const speedMatch = stdout.match(/Speed: (\d+)Mb\/s/);
          if (speedMatch) {
            speed = parseInt(speedMatch[1], 10);
          }

          // Check for InfiniBand
          if (name.startsWith('ib')) {
            type = 'infiniband';
            speed = 100000; // 100 Gbps typical for IB
          }
        } catch (_error) {
          // Use defaults
        }

        try {
          const { stdout: mtuOut } = await execAsync(
            `cat /sys/class/net/${name}/mtu 2>/dev/null || echo "1500"`,
          );
          mtu = parseInt(mtuOut.trim(), 10);
        } catch (_error) {
          // Use default MTU
        }
      }

      interfaces.push({
        name,
        speed,
        type,
        mtu,
        ipAddress: ipv4?.address,
      });
    }

    return interfaces;
  }

  /**
   * Detect interconnect information (PCIe, NVLink, etc.)
   */
  async detectInterconnects(): Promise<InterconnectInfo[]> {
    const interconnects: InterconnectInfo[] = [];

    // Detect NVLink topology (NVIDIA)
    try {
      const { stdout } = await execAsync('nvidia-smi nvlink --status');
      if (stdout.includes('Active')) {
        // Parse NVLink connections
        interconnects.push({
          type: 'NVLink',
          bandwidth: 300, // GB/s (NVLink 3.0 approximate)
          latency: 5, // microseconds
          topology: 'all-to-all',
        });
      }
    } catch (_error) {
      // NVLink not available
    }

    // Detect PCIe topology
    try {
      if (os.platform() === 'linux') {
        const { stdout } = await execAsync('lspci | grep -i pci');
        if (stdout) {
          interconnects.push({
            type: 'PCIe',
            bandwidth: 32, // GB/s (PCIe 4.0 x16 approximate)
            latency: 10, // microseconds
            topology: 'tree',
          });
        }
      }
    } catch (_error) {
      // PCIe detection failed
    }

    // Detect InfiniBand
    try {
      const { stdout } = await execAsync('ibstat 2>/dev/null || echo ""');
      if (stdout.includes('State: Active')) {
        interconnects.push({
          type: 'InfiniBand',
          bandwidth: 100, // GB/s (HDR200)
          latency: 1, // microseconds
          topology: 'mesh',
        });
      }
    } catch (_error) {
      // InfiniBand not available
    }

    // If no specific interconnects found, add generic PCIe
    if (interconnects.length === 0) {
      interconnects.push({
        type: 'PCIe',
        bandwidth: 16, // GB/s (conservative estimate)
        latency: 10,
        topology: 'tree',
      });
    }

    return interconnects;
  }

  /**
   * Detect full hardware topology
   */
  async detectTopology(): Promise<HardwareTopology> {
    const [cpu, gpus, memory, networks, interconnects] = await Promise.all([
      this.detectCPU(),
      this.detectGPUs(),
      this.detectMemory(),
      this.detectNetworkInterfaces(),
      this.detectInterconnects(),
    ]);

    return {
      cpu,
      gpus,
      memory,
      networks,
      interconnects,
      nodeCount: 1, // Single node by default
      timestamp: new Date(),
    };
  }
}
