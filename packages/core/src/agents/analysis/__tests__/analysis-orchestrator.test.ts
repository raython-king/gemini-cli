/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  AnalysisOrchestrator,
  AnalysisWorkflowType,
  type AnalysisTask,
} from '../analysis-orchestrator.js';
import { makeFakeConfig } from '../../../test-utils/config.js';
import type { Config } from '../../../config/config.js';
import { DataFlowAgent } from '../data-flow-agent.js';
import { LiteratureAnalyzerAgent } from '../literature-analyzer-agent.js';
import { SummarizerAgent } from '../summarizer-agent.js';
import { OptimizationStrategy } from '../../context/context-optimizer.js';

// Mock the AgentExecutor to avoid actual agent execution
vi.mock('../../executor.js', () => ({
  AgentExecutor: {
    create: vi.fn(() => ({
      run: vi.fn().mockResolvedValue({
        result: 'Mock result',
        terminate_reason: 'GOAL',
      }),
    })),
  },
}));

// Mock the ParallelAgentExecutor
vi.mock('../../parallel-executor.js', () => ({
  ParallelAgentExecutor: vi.fn().mockImplementation(() => ({
    executeParallel: vi.fn().mockResolvedValue([
      {
        taskId: 'task1',
        agentName: 'test_agent',
        success: true,
        output: {
          result: 'Mock parallel result',
          terminate_reason: 'GOAL',
        },
      },
    ]),
  })),
}));

describe('AnalysisOrchestrator', () => {
  let config: Config;
  let orchestrator: AnalysisOrchestrator;

  beforeEach(() => {
    vi.clearAllMocks();
    config = makeFakeConfig();
    orchestrator = new AnalysisOrchestrator(config, {
      maxContextTokens: 6000,
      optimizationStrategy: OptimizationStrategy.BALANCED,
      maxConcurrency: 3,
    });
  });

  describe('Initialization', () => {
    it('should create orchestrator with default options', () => {
      const defaultOrchestrator = new AnalysisOrchestrator(config);
      expect(defaultOrchestrator).toBeInstanceOf(AnalysisOrchestrator);
    });

    it('should create orchestrator with custom options', () => {
      const customOrchestrator = new AnalysisOrchestrator(config, {
        maxContextTokens: 8000,
        optimizationStrategy: OptimizationStrategy.AGGRESSIVE,
        maxConcurrency: 5,
      });
      expect(customOrchestrator).toBeInstanceOf(AnalysisOrchestrator);
    });

    it('should register default analysis agents', () => {
      const dataFlowDef = (orchestrator as any).agentDefinitions.get(
        'data_flow_agent',
      );
      const litAnalyzerDef = (orchestrator as any).agentDefinitions.get(
        'literature_analyzer_agent',
      );
      const summarizerDef = (orchestrator as any).agentDefinitions.get(
        'summarizer_agent',
      );

      expect(dataFlowDef).toBeDefined();
      expect(dataFlowDef).toEqual(DataFlowAgent);
      expect(litAnalyzerDef).toBeDefined();
      expect(litAnalyzerDef).toEqual(LiteratureAnalyzerAgent);
      expect(summarizerDef).toBeDefined();
      expect(summarizerDef).toEqual(SummarizerAgent);
    });
  });

  describe('Agent Registration', () => {
    it('should register custom agent', () => {
      const customAgent = {
        name: 'custom_agent',
        description: 'Custom test agent',
        inputConfig: { inputs: {} },
        modelConfig: { model: 'test', temp: 0.5, top_p: 0.9 },
        runConfig: { max_time_minutes: 5 },
        promptConfig: { systemPrompt: 'Test prompt' },
      };

      orchestrator.registerAgent(customAgent);

      const registeredAgent = (orchestrator as any).agentDefinitions.get(
        'custom_agent',
      );
      expect(registeredAgent).toEqual(customAgent);
    });

    it('should allow overriding existing agents', () => {
      const newDataFlowAgent = {
        ...DataFlowAgent,
        description: 'Updated description',
      };

      orchestrator.registerAgent(newDataFlowAgent);

      const registeredAgent = (orchestrator as any).agentDefinitions.get(
        'data_flow_agent',
      );
      expect(registeredAgent.description).toBe('Updated description');
    });
  });

  describe('Workflow Types', () => {
    it('should have all workflow types defined', () => {
      expect(AnalysisWorkflowType.DATA_PROCESSING).toBe('data_processing');
      expect(AnalysisWorkflowType.LITERATURE_REVIEW).toBe('literature_review');
      expect(AnalysisWorkflowType.COMPREHENSIVE_ANALYSIS).toBe(
        'comprehensive_analysis',
      );
      expect(AnalysisWorkflowType.MULTI_DOCUMENT_SUMMARY).toBe(
        'multi_document_summary',
      );
    });
  });

  describe('executeAnalysis', () => {
    it('should execute data processing workflow', async () => {
      const task: AnalysisTask = {
        type: AnalysisWorkflowType.DATA_PROCESSING,
        description: 'Process user data',
        sources: ['data/users.json'],
        options: {
          dataValidation: ['email', 'age'],
        },
      };

      const result = await orchestrator.executeAnalysis(task);

      expect(result).toBeDefined();
      expect(result.task).toEqual(task);
      expect(result.steps).toBeInstanceOf(Array);
      expect(result.summary).toBeDefined();
      expect(result.keyFindings).toBeInstanceOf(Array);
      expect(result.recommendations).toBeInstanceOf(Array);
      expect(result.metadata).toBeDefined();
      expect(result.metadata.agentsUsed).toBeInstanceOf(Array);
      expect(result.metadata.totalExecutionTime).toBeGreaterThanOrEqual(0);
    });

    it('should execute literature review workflow', async () => {
      const task: AnalysisTask = {
        type: AnalysisWorkflowType.LITERATURE_REVIEW,
        description: 'Analyze ML papers',
        sources: ['papers/paper1.pdf', 'papers/paper2.pdf'],
        options: {
          depth: 'detailed',
          focus: ['methodology', 'results'],
        },
      };

      const result = await orchestrator.executeAnalysis(task);

      expect(result).toBeDefined();
      expect(result.task.type).toBe(AnalysisWorkflowType.LITERATURE_REVIEW);
      expect(result.steps.length).toBeGreaterThan(0);
    });

    it('should execute comprehensive analysis workflow', async () => {
      const task: AnalysisTask = {
        type: AnalysisWorkflowType.COMPREHENSIVE_ANALYSIS,
        description: 'Full analysis of data and literature',
        sources: ['data/metrics.json', 'docs/paper.pdf'],
      };

      const result = await orchestrator.executeAnalysis(task);

      expect(result).toBeDefined();
      expect(result.task.type).toBe(
        AnalysisWorkflowType.COMPREHENSIVE_ANALYSIS,
      );
    });

    it('should execute multi-document summary workflow', async () => {
      const task: AnalysisTask = {
        type: AnalysisWorkflowType.MULTI_DOCUMENT_SUMMARY,
        description: 'Summarize multiple documents',
        sources: ['doc1.md', 'doc2.md', 'doc3.md'],
        options: {
          depth: 'overview',
          includeActionItems: true,
        },
      };

      const result = await orchestrator.executeAnalysis(task);

      expect(result).toBeDefined();
      expect(result.task.type).toBe(
        AnalysisWorkflowType.MULTI_DOCUMENT_SUMMARY,
      );
    });

    it('should throw error for unknown workflow type', async () => {
      const task: AnalysisTask = {
        type: 'unknown_workflow' as any,
        description: 'Unknown workflow',
        sources: ['test.txt'],
      };

      await expect(orchestrator.executeAnalysis(task)).rejects.toThrow(
        'Unknown workflow type',
      );
    });

    it('should include metadata in result', async () => {
      const task: AnalysisTask = {
        type: AnalysisWorkflowType.DATA_PROCESSING,
        description: 'Test task',
        sources: ['data.json'],
      };

      const result = await orchestrator.executeAnalysis(task);

      expect(result.metadata).toBeDefined();
      expect(result.metadata.totalExecutionTime).toBeGreaterThanOrEqual(0);
      expect(result.metadata.agentsUsed).toBeInstanceOf(Array);
      expect(result.metadata.contextItemsUsed).toBeGreaterThanOrEqual(0);
      expect(result.metadata.memoriesCreated).toBeGreaterThanOrEqual(0);
    });

    it('should handle workflow with multiple sources', async () => {
      const task: AnalysisTask = {
        type: AnalysisWorkflowType.LITERATURE_REVIEW,
        description: 'Analyze multiple papers',
        sources: ['paper1.pdf', 'paper2.pdf', 'paper3.pdf'],
      };

      const result = await orchestrator.executeAnalysis(task);

      expect(result).toBeDefined();
      expect(result.steps.length).toBeGreaterThan(0);
    });
  });

  describe('Context and Memory Management', () => {
    it('should create context items during workflow execution', async () => {
      const task: AnalysisTask = {
        type: AnalysisWorkflowType.DATA_PROCESSING,
        description: 'Process data with context',
        sources: ['data.json'],
      };

      await orchestrator.executeAnalysis(task);

      const stats = orchestrator.getStats();
      expect(stats.context.totalItems).toBeGreaterThan(0);
    });

    it('should consolidate memories after execution', async () => {
      const task: AnalysisTask = {
        type: AnalysisWorkflowType.DATA_PROCESSING,
        description: 'Process data',
        sources: ['data.json'],
      };

      const result = await orchestrator.executeAnalysis(task);

      expect(result.metadata.memoriesCreated).toBeGreaterThanOrEqual(0);
    });
  });

  describe('Analysis History', () => {
    it('should track analysis history', async () => {
      const task: AnalysisTask = {
        type: AnalysisWorkflowType.DATA_PROCESSING,
        description: 'Process data',
        sources: ['data.json'],
      };

      await orchestrator.executeAnalysis(task);

      const history = orchestrator.getAnalysisHistory();
      expect(history).toBeInstanceOf(Array);
    });

    it('should filter history by workflow type', async () => {
      const task: AnalysisTask = {
        type: AnalysisWorkflowType.DATA_PROCESSING,
        description: 'Process data',
        sources: ['data.json'],
      };

      await orchestrator.executeAnalysis(task);

      const history = orchestrator.getAnalysisHistory(
        AnalysisWorkflowType.DATA_PROCESSING,
      );
      expect(history).toBeInstanceOf(Array);
    });
  });

  describe('State Export', () => {
    it('should export analysis state', async () => {
      const task: AnalysisTask = {
        type: AnalysisWorkflowType.DATA_PROCESSING,
        description: 'Process data',
        sources: ['data.json'],
      };

      await orchestrator.executeAnalysis(task);

      const state = orchestrator.exportAnalysisState();

      expect(state).toBeDefined();
      expect(state.workflows).toBeInstanceOf(Array);
      expect(state.insights).toBeInstanceOf(Array);
      expect(state.patterns).toBeInstanceOf(Array);
      expect(state.state).toBeDefined();
    });

    it('should include workflow types in exported state', async () => {
      const task: AnalysisTask = {
        type: AnalysisWorkflowType.LITERATURE_REVIEW,
        description: 'Analyze papers',
        sources: ['paper.pdf'],
      };

      await orchestrator.executeAnalysis(task);

      const state = orchestrator.exportAnalysisState();

      expect(state.workflows).toContain(AnalysisWorkflowType.LITERATURE_REVIEW);
    });
  });

  describe('Task Options', () => {
    it('should handle depth option', async () => {
      const depths: Array<'overview' | 'detailed' | 'comprehensive'> = [
        'overview',
        'detailed',
        'comprehensive',
      ];

      for (const depth of depths) {
        const task: AnalysisTask = {
          type: AnalysisWorkflowType.LITERATURE_REVIEW,
          description: 'Analyze with depth',
          sources: ['paper.pdf'],
          options: { depth },
        };

        const result = await orchestrator.executeAnalysis(task);
        expect(result).toBeDefined();
      }
    });

    it('should handle focus option', async () => {
      const task: AnalysisTask = {
        type: AnalysisWorkflowType.LITERATURE_REVIEW,
        description: 'Focused analysis',
        sources: ['paper.pdf'],
        options: {
          focus: ['methodology', 'results', 'conclusion'],
        },
      };

      const result = await orchestrator.executeAnalysis(task);
      expect(result).toBeDefined();
    });

    it('should handle includeActionItems option', async () => {
      const task: AnalysisTask = {
        type: AnalysisWorkflowType.MULTI_DOCUMENT_SUMMARY,
        description: 'Summary with actions',
        sources: ['doc.md'],
        options: {
          includeActionItems: true,
        },
      };

      const result = await orchestrator.executeAnalysis(task);
      expect(result).toBeDefined();
    });

    it('should handle dataValidation option', async () => {
      const task: AnalysisTask = {
        type: AnalysisWorkflowType.DATA_PROCESSING,
        description: 'Data processing with validation',
        sources: ['data.json'],
        options: {
          dataValidation: ['completeness', 'consistency', 'accuracy'],
        },
      };

      const result = await orchestrator.executeAnalysis(task);
      expect(result).toBeDefined();
    });
  });

  describe('Error Handling', () => {
    it('should handle errors during workflow execution', async () => {
      // Mock the executor to throw an error
      const { AgentExecutor } = await import('../../executor.js');
      vi.mocked(AgentExecutor.create).mockImplementationOnce(() => {
        throw new Error('Mock execution error');
      });

      const task: AnalysisTask = {
        type: AnalysisWorkflowType.DATA_PROCESSING,
        description: 'Failing task',
        sources: ['data.json'],
      };

      await expect(orchestrator.executeAnalysis(task)).rejects.toThrow(
        'Analysis workflow failed',
      );
    });
  });

  describe('Results Synthesis', () => {
    it('should synthesize results from multiple steps', async () => {
      const task: AnalysisTask = {
        type: AnalysisWorkflowType.COMPREHENSIVE_ANALYSIS,
        description: 'Multi-step analysis',
        sources: ['data.json', 'paper.pdf'],
      };

      const result = await orchestrator.executeAnalysis(task);

      expect(result.summary).toBeDefined();
      expect(result.summary).toContain('analysis workflow');
      expect(result.keyFindings).toBeInstanceOf(Array);
      expect(result.recommendations).toBeInstanceOf(Array);
      expect(result.recommendations.length).toBeGreaterThan(0);
    });

    it('should include insights from all steps', async () => {
      const task: AnalysisTask = {
        type: AnalysisWorkflowType.LITERATURE_REVIEW,
        description: 'Multi-document review',
        sources: ['paper1.pdf', 'paper2.pdf'],
      };

      const result = await orchestrator.executeAnalysis(task);

      expect(result.steps.length).toBeGreaterThan(0);
      result.steps.forEach((step) => {
        expect(step.agent).toBeDefined();
        expect(step.output).toBeDefined();
        expect(step.insights).toBeInstanceOf(Array);
      });
    });
  });
});
