/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import {
  MultiAgentCoordinator,
  SubtaskStatus,
  type CoordinatorStatus,
} from '../multi-agent-coordinator.js';
import type { Config } from '../../config/config.js';

// Mock config
const mockConfig: Partial<Config> = {
  workingDir: '/test',
  model: 'gemini-pro',
};

describe('MultiAgentCoordinator', () => {
  let coordinator: MultiAgentCoordinator;

  beforeEach(() => {
    coordinator = new MultiAgentCoordinator(mockConfig as Config);
  });

  describe('preview', () => {
    it('should preview task decomposition', () => {
      const result = coordinator.preview(
        'Implement user authentication',
        10,
      );

      expect(result.originalTask).toBe('Implement user authentication');
      expect(result.subtasks.length).toBeGreaterThan(0);
      expect(result.executionPlan.length).toBeGreaterThan(0);
      expect(result.estimatedDuration).toBeDefined();
    });

    it('should respect maxSubtasks limit', () => {
      const result = coordinator.preview('Complex task', 3);

      expect(result.subtasks.length).toBeLessThanOrEqual(3);
    });
  });

  describe('formatDecomposition', () => {
    it('should format decomposition as readable text', () => {
      const decomposition = coordinator.preview('Test task', 5);
      const formatted = coordinator.formatDecomposition(decomposition);

      expect(formatted).toContain('Test task');
      expect(formatted).toContain('Execution Plan');
      expect(formatted).toContain('Stage');
      expect(typeof formatted).toBe('string');
    });

    it('should include all stages in formatted output', () => {
      const decomposition = coordinator.preview(
        'Multi-stage task',
        10,
      );
      const formatted = coordinator.formatDecomposition(decomposition);

      decomposition.executionPlan.forEach((stage) => {
        expect(formatted).toContain(`Stage ${stage.stage}`);
      });
    });
  });

  describe('formatStatus', () => {
    it('should format coordinator status as readable text', () => {
      const mockStatus: CoordinatorStatus = {
        task: 'Test task',
        currentStage: 1,
        totalStages: 3,
        subtasks: [
          {
            subtask: {
              id: 'st1',
              title: 'Test subtask',
              description: 'Test',
              recommendedAgents: ['test_agent'],
              dependencies: [],
              complexity: 'low' as any,
              priority: 1,
              inputs: {},
              canParallelize: false,
            },
            status: SubtaskStatus.COMPLETED,
          },
        ],
        progress: 33,
        isComplete: false,
        success: false,
        startTime: new Date(),
      };

      const formatted = coordinator.formatStatus(mockStatus);

      expect(formatted).toContain('Test task');
      expect(formatted).toContain('Progress: 33%');
      expect(formatted).toContain('Stage 1/3');
      expect(formatted).toContain('Test subtask');
    });

    it('should show completed status when execution is done', () => {
      const mockStatus: CoordinatorStatus = {
        task: 'Test task',
        currentStage: 3,
        totalStages: 3,
        subtasks: [],
        progress: 100,
        isComplete: true,
        success: true,
        startTime: new Date(),
        endTime: new Date(),
      };

      const formatted = coordinator.formatStatus(mockStatus);

      expect(formatted).toContain('Complete');
      expect(formatted).toContain('Success: Yes');
    });

    it('should show failed subtasks', () => {
      const mockStatus: CoordinatorStatus = {
        task: 'Test task',
        currentStage: 1,
        totalStages: 1,
        subtasks: [
          {
            subtask: {
              id: 'st1',
              title: 'Failed subtask',
              description: 'Test',
              recommendedAgents: ['test_agent'],
              dependencies: [],
              complexity: 'low' as any,
              priority: 1,
              inputs: {},
              canParallelize: false,
            },
            status: SubtaskStatus.FAILED,
            error: 'Test error',
          },
        ],
        progress: 0,
        isComplete: true,
        success: false,
        startTime: new Date(),
        endTime: new Date(),
      };

      const formatted = coordinator.formatStatus(mockStatus);

      expect(formatted).toContain('Failed Subtasks');
      expect(formatted).toContain('Failed subtask');
      expect(formatted).toContain('Test error');
    });
  });

  describe('execution flow', () => {
    it('should handle simple tasks without errors', async () => {
      // This is a minimal test - full execution would require mocking agents
      const decomposition = coordinator.preview('Simple task', 1);

      expect(decomposition.subtasks.length).toBe(1);
      expect(decomposition.executionPlan.length).toBeGreaterThan(0);
    });
  });
});
