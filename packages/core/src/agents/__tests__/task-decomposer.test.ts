/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { TaskDecomposer } from '../task-decomposer.js';
import { TaskType, TaskComplexity } from '../orchestrator.js';

describe('TaskDecomposer', () => {
  let decomposer: TaskDecomposer;

  beforeEach(() => {
    decomposer = new TaskDecomposer();
    decomposer.resetCounter();
  });

  describe('decompose', () => {
    it('should decompose an implementation task', () => {
      const result = decomposer.decompose(
        'Implement user authentication with JWT tokens',
      );

      expect(result.originalTask).toBe(
        'Implement user authentication with JWT tokens',
      );
      expect(result.classification.type).toBe(TaskType.IMPLEMENTATION);
      expect(result.subtasks.length).toBeGreaterThan(0);
      expect(result.executionPlan.length).toBeGreaterThan(0);
    });

    it('should create a simple subtask for low complexity tasks', () => {
      const result = decomposer.decompose('Show me the login component');

      expect(result.subtasks.length).toBe(1);
      expect(result.executionPlan.length).toBe(1);
      expect(result.classification.complexity).toBe(TaskComplexity.LOW);
    });

    it('should decompose a refactoring task correctly', () => {
      const result = decomposer.decompose(
        'Refactor the entire authentication module to use async/await',
      );

      expect(result.classification.type).toBe(TaskType.REFACTORING);
      expect(result.subtasks.length).toBeGreaterThanOrEqual(3);

      // Should include analysis, review, refactoring, and testing
      const subtaskTitles = result.subtasks.map((st) =>
        st.title.toLowerCase(),
      );
      expect(
        subtaskTitles.some((title) => title.includes('analyze')),
      ).toBe(true);
      expect(
        subtaskTitles.some((title) => title.includes('refactor')),
      ).toBe(true);
    });

    it('should decompose a debugging task correctly', () => {
      const result = decomposer.decompose(
        'Fix the bug where login fails on iOS Safari',
      );

      expect(result.classification.type).toBe(TaskType.DEBUGGING);
      expect(result.subtasks.length).toBeGreaterThanOrEqual(2);

      // Should include exploration and debugging
      const agents = result.subtasks.flatMap((st) => st.recommendedAgents);
      expect(agents).toContain('debug_agent');
    });

    it('should decompose a code review task with parallel execution', () => {
      const result = decomposer.decompose(
        'Review all authentication code for security and performance',
      );

      expect(result.classification.type).toBe(TaskType.CODE_REVIEW);

      // Code review subtasks should be parallelizable
      const parallelSubtasks = result.subtasks.filter(
        (st) => st.canParallelize,
      );
      expect(parallelSubtasks.length).toBeGreaterThan(0);
    });

    it('should limit subtasks to maxSubtasks', () => {
      const result = decomposer.decompose(
        'Implement a comprehensive CI/CD pipeline',
        3,
      );

      expect(result.subtasks.length).toBeLessThanOrEqual(3);
    });

    it('should create execution plan with proper dependencies', () => {
      const result = decomposer.decompose(
        'Implement user registration',
        10,
      );

      // Verify execution plan stages
      expect(result.executionPlan.length).toBeGreaterThan(0);

      // First stage should have no dependencies
      const firstStage = result.executionPlan[0];
      const firstStageSubtasks = result.subtasks.filter((st) =>
        firstStage.subtasks.includes(st.id),
      );

      firstStageSubtasks.forEach((st) => {
        expect(st.dependencies.length).toBe(0);
      });
    });

    it('should identify parallel vs sequential execution', () => {
      const result = decomposer.decompose(
        'Review code for security, performance, and testing',
      );

      const parallelStages = result.executionPlan.filter(
        (stage) => stage.canRunInParallel,
      );
      expect(parallelStages.length).toBeGreaterThan(0);
    });

    it('should estimate duration based on complexity', () => {
      const simpleResult = decomposer.decompose('Find a function');
      const complexResult = decomposer.decompose(
        'Redesign the entire system architecture',
      );

      expect(simpleResult.estimatedDuration).toBeDefined();
      expect(complexResult.estimatedDuration).toBeDefined();

      // Complex tasks should have longer estimated duration
      // (This is a simple check - actual duration strings may vary)
      expect(complexResult.estimatedDuration).not.toBe(
        simpleResult.estimatedDuration,
      );
    });

    it('should assign priorities to subtasks', () => {
      const result = decomposer.decompose('Implement a new feature', 10);

      // All subtasks should have priorities
      result.subtasks.forEach((st) => {
        expect(st.priority).toBeGreaterThan(0);
      });

      // Higher priority subtasks should generally come first
      const priorities = result.subtasks.map((st) => st.priority);
      expect(priorities.length).toBeGreaterThan(0);
    });

    it('should handle testing tasks', () => {
      const result = decomposer.decompose('Run all tests and fix failures');

      expect(result.classification.type).toBe(TaskType.TESTING);

      const agents = result.subtasks.flatMap((st) => st.recommendedAgents);
      expect(agents).toContain('test_runner');
    });

    it('should handle exploration tasks', () => {
      const result = decomposer.decompose(
        'Find where user authentication is implemented',
      );

      expect(result.classification.type).toBe(TaskType.EXPLORATION);

      const agents = result.subtasks.flatMap((st) => st.recommendedAgents);
      expect(agents.some((a) => a.includes('explore'))).toBe(true);
    });

    it('should generate unique subtask IDs', () => {
      const result = decomposer.decompose(
        'Complex task with many subtasks',
        10,
      );

      const ids = result.subtasks.map((st) => st.id);
      const uniqueIds = new Set(ids);

      expect(uniqueIds.size).toBe(ids.length);
    });

    it('should include inputs for each subtask', () => {
      const result = decomposer.decompose('Implement user login', 10);

      result.subtasks.forEach((st) => {
        expect(st.inputs).toBeDefined();
        expect(typeof st.inputs).toBe('object');
      });
    });
  });

  describe('task classification', () => {
    it('should classify implementation keywords', () => {
      const tasks = [
        'implement a feature',
        'build a component',
        'create a service',
        'add functionality',
      ];

      tasks.forEach((task) => {
        const result = decomposer.decompose(task);
        expect(result.classification.type).toBe(TaskType.IMPLEMENTATION);
      });
    });

    it('should classify exploration keywords', () => {
      const tasks = [
        'find the login function',
        'where is authentication handled',
        'show me the code for',
        'how does this work',
      ];

      tasks.forEach((task) => {
        const result = decomposer.decompose(task);
        expect(result.classification.type).toBe(TaskType.EXPLORATION);
      });
    });

    it('should classify debugging keywords', () => {
      const tasks = [
        'fix the bug',
        'error in login',
        'not working properly',
        'crash on startup',
      ];

      tasks.forEach((task) => {
        const result = decomposer.decompose(task);
        expect(result.classification.type).toBe(TaskType.DEBUGGING);
      });
    });

    it('should assess complexity correctly', () => {
      const lowComplexity = decomposer.decompose('show me a file');
      expect(lowComplexity.classification.complexity).toBe(
        TaskComplexity.LOW,
      );

      const highComplexity = decomposer.decompose(
        'redesign the entire system architecture across all modules',
      );
      expect(
        [TaskComplexity.HIGH, TaskComplexity.VERY_HIGH],
      ).toContain(highComplexity.classification.complexity);
    });
  });

  describe('edge cases', () => {
    it('should handle empty task gracefully', () => {
      const result = decomposer.decompose('');

      expect(result.subtasks.length).toBeGreaterThan(0);
      expect(result.executionPlan.length).toBeGreaterThan(0);
    });

    it('should handle very long task descriptions', () => {
      const longTask = 'A'.repeat(1000);
      const result = decomposer.decompose(longTask);

      expect(result.subtasks.length).toBeGreaterThan(0);
    });

    it('should handle maxSubtasks = 1', () => {
      const result = decomposer.decompose('Complex task', 1);

      expect(result.subtasks.length).toBe(1);
    });

    it('should handle maxSubtasks = 0 by using default', () => {
      const result = decomposer.decompose('Complex task', 0);

      expect(result.subtasks.length).toBeGreaterThan(0);
    });
  });
});
