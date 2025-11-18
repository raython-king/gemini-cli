/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Analysis Orchestrator - Context-aware orchestration for data flow
 * and literature analysis workflows.
 *
 * This module provides intelligent coordination of DataFlowAgent,
 * LiteratureAnalyzerAgent, and SummarizerAgent using context engineering
 * to maintain continuity and build comprehensive understanding.
 */

import { ContextualOrchestrator } from '../context/contextual-orchestrator.js';
import type { OutputObject } from '../types.js';
import {
  ContextType,
  ContextPriority,
  type ContextItem,
} from '../context/shared-context.js';
import { OptimizationStrategy } from '../context/context-optimizer.js';
import { MemoryType } from '../context/agent-memory.js';

/**
 * Analysis workflow type.
 */
export enum AnalysisWorkflowType {
  DATA_PROCESSING = 'data_processing',
  LITERATURE_REVIEW = 'literature_review',
  COMPREHENSIVE_ANALYSIS = 'comprehensive_analysis',
  MULTI_DOCUMENT_SUMMARY = 'multi_document_summary',
}

/**
 * Analysis task configuration.
 */
export interface AnalysisTask {
  type: AnalysisWorkflowType;
  description: string;
  sources: string[];
  options?: {
    depth?: 'overview' | 'detailed' | 'comprehensive';
    focus?: string[];
    includeActionItems?: boolean;
    dataValidation?: string[];
  };
}

/**
 * Analysis result.
 */
export interface AnalysisResult {
  task: AnalysisTask;
  steps: Array<{
    agent: string;
    output: OutputObject;
    insights: string[];
  }>;
  summary: string;
  keyFindings: string[];
  recommendations: string[];
  metadata: {
    totalExecutionTime: number;
    agentsUsed: string[];
    contextItemsUsed: number;
    memoriesCreated: number;
  };
}

/**
 * AnalysisOrchestrator - Intelligent orchestration for data and literature analysis.
 *
 * This orchestrator extends ContextualOrchestrator with specialized
 * workflows for data processing and literature analysis tasks.
 *
 * @example
 * ```typescript
 * const orchestrator = new AnalysisOrchestrator();
 *
 * // Data processing workflow
 * const dataResult = await orchestrator.executeAnalysis({
 *   type: AnalysisWorkflowType.DATA_PROCESSING,
 *   description: 'Process user activity logs and extract insights',
 *   sources: ['logs/user-activity.json'],
 *   options: {
 *     dataValidation: ['completeness', 'consistency'],
 *   },
 * });
 *
 * // Literature analysis workflow
 * const litResult = await orchestrator.executeAnalysis({
 *   type: AnalysisWorkflowType.LITERATURE_REVIEW,
 *   description: 'Analyze recent papers on machine learning optimization',
 *   sources: ['papers/ml-opt-1.pdf', 'papers/ml-opt-2.pdf'],
 *   options: {
 *     depth: 'comprehensive',
 *     focus: ['methodology', 'results'],
 *   },
 * });
 * ```
 */
export class AnalysisOrchestrator extends ContextualOrchestrator {
  constructor() {
    super({
      maxContextTokens: 6000, // Higher limit for analysis tasks
      optimizationStrategy: OptimizationStrategy.BALANCED,
    });
  }

  /**
   * Execute a complete analysis workflow.
   *
   * This method orchestrates multiple agents to complete a complex
   * analysis task, maintaining context throughout the workflow.
   */
  async executeAnalysis(task: AnalysisTask): Promise<AnalysisResult> {
    const startTime = Date.now();
    const steps: AnalysisResult['steps'] = [];
    const agentsUsed = new Set<string>();

    // Store task in context
    this.getSharedContext().add(
      ContextType.TASK_METADATA,
      {
        type: task.type,
        description: task.description,
        sourceCount: task.sources.length,
      },
      {
        source: 'analysis_orchestrator',
        priority: ContextPriority.HIGH,
        tags: ['task', task.type],
      },
    );

    try {
      // Execute workflow based on type
      switch (task.type) {
        case AnalysisWorkflowType.DATA_PROCESSING:
          await this.executeDataProcessingWorkflow(task, steps, agentsUsed);
          break;

        case AnalysisWorkflowType.LITERATURE_REVIEW:
          await this.executeLiteratureReviewWorkflow(task, steps, agentsUsed);
          break;

        case AnalysisWorkflowType.COMPREHENSIVE_ANALYSIS:
          await this.executeComprehensiveAnalysisWorkflow(
            task,
            steps,
            agentsUsed,
          );
          break;

        case AnalysisWorkflowType.MULTI_DOCUMENT_SUMMARY:
          await this.executeMultiDocumentSummaryWorkflow(
            task,
            steps,
            agentsUsed,
          );
          break;

        default:
          throw new Error(`Unknown workflow type: ${task.type}`);
      }

      // Generate final summary and recommendations
      const { summary, keyFindings, recommendations } =
        this.synthesizeResults(steps);

      // Consolidate important memories
      const memoriesCreated = this.consolidateMemories(0.7);

      const totalExecutionTime = Date.now() - startTime;

      return {
        task,
        steps,
        summary,
        keyFindings,
        recommendations,
        metadata: {
          totalExecutionTime,
          agentsUsed: Array.from(agentsUsed),
          contextItemsUsed: this.getStats().context.totalItems,
          memoriesCreated,
        },
      };
    } catch (error) {
      throw new Error(
        `Analysis workflow failed: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
  }

  /**
   * Execute data processing workflow.
   */
  private async executeDataProcessingWorkflow(
    task: AnalysisTask,
    steps: AnalysisResult['steps'],
    agentsUsed: Set<string>,
  ): Promise<void> {
    // Step 1: Process data with DataFlowAgent
    const dataFlowOutput = await this.simulateAgentExecution(
      'data_flow_agent',
      {
        dataSource: task.sources[0],
        task: task.description,
        validationRules: task.options?.dataValidation,
      },
    );

    const dataInsights = this.extractDataInsights(dataFlowOutput);
    this.recordExecution('data_flow_agent', dataFlowOutput, {
      patterns: dataInsights.patterns,
      issues: dataInsights.issues,
    });

    steps.push({
      agent: 'data_flow_agent',
      output: dataFlowOutput,
      insights: dataInsights.insights,
    });
    agentsUsed.add('data_flow_agent');

    // Step 2: Generate summary with SummarizerAgent
    const summaryOutput = await this.simulateAgentExecution(
      'summarizer_agent',
      {
        sources: [task.sources[0]],
        style: 'technical',
        includeActionItems: task.options?.includeActionItems ?? true,
      },
    );

    this.recordExecution('summarizer_agent', summaryOutput);

    steps.push({
      agent: 'summarizer_agent',
      output: summaryOutput,
      insights: ['Generated comprehensive summary of data processing results'],
    });
    agentsUsed.add('summarizer_agent');
  }

  /**
   * Execute literature review workflow.
   */
  private async executeLiteratureReviewWorkflow(
    task: AnalysisTask,
    steps: AnalysisResult['steps'],
    agentsUsed: Set<string>,
  ): Promise<void> {
    // Step 1: Analyze each document with LiteratureAnalyzerAgent
    for (const source of task.sources) {
      const analysisOutput = await this.simulateAgentExecution(
        'literature_analyzer_agent',
        {
          document: source,
          focus: task.options?.focus?.join(', '),
          depth: task.options?.depth ?? 'detailed',
        },
      );

      const litInsights = this.extractLiteratureInsights(analysisOutput);
      this.recordExecution('literature_analyzer_agent', analysisOutput, {
        patterns: litInsights.concepts,
        files: [source],
      });

      steps.push({
        agent: 'literature_analyzer_agent',
        output: analysisOutput,
        insights: litInsights.insights,
      });
      agentsUsed.add('literature_analyzer_agent');
    }

    // Step 2: Synthesize findings with SummarizerAgent
    const summaryOutput = await this.simulateAgentExecution(
      'summarizer_agent',
      {
        sources: task.sources,
        style: 'academic',
        focus: task.options?.focus,
        includeActionItems: true,
      },
    );

    this.recordExecution('summarizer_agent', summaryOutput);

    steps.push({
      agent: 'summarizer_agent',
      output: summaryOutput,
      insights: ['Synthesized findings from multiple literature sources'],
    });
    agentsUsed.add('summarizer_agent');
  }

  /**
   * Execute comprehensive analysis workflow (data + literature).
   */
  private async executeComprehensiveAnalysisWorkflow(
    task: AnalysisTask,
    steps: AnalysisResult['steps'],
    agentsUsed: Set<string>,
  ): Promise<void> {
    // Combine data processing and literature review
    await this.executeDataProcessingWorkflow(task, steps, agentsUsed);
    await this.executeLiteratureReviewWorkflow(task, steps, agentsUsed);

    // Additional synthesis step
    const finalSummaryOutput = await this.simulateAgentExecution(
      'summarizer_agent',
      {
        sources: task.sources,
        style: 'executive',
        length: 'comprehensive',
        includeActionItems: true,
      },
    );

    this.recordExecution('summarizer_agent', finalSummaryOutput);

    steps.push({
      agent: 'summarizer_agent',
      output: finalSummaryOutput,
      insights: [
        'Comprehensive synthesis of data analysis and literature review',
      ],
    });
  }

  /**
   * Execute multi-document summary workflow.
   */
  private async executeMultiDocumentSummaryWorkflow(
    task: AnalysisTask,
    steps: AnalysisResult['steps'],
    agentsUsed: Set<string>,
  ): Promise<void> {
    // Process all documents with SummarizerAgent
    const summaryOutput = await this.simulateAgentExecution(
      'summarizer_agent',
      {
        sources: task.sources,
        style: task.options?.depth === 'overview' ? 'executive' : 'narrative',
        length: task.options?.depth ?? 'moderate',
        focus: task.options?.focus,
        includeActionItems: task.options?.includeActionItems ?? true,
      },
    );

    this.recordExecution('summarizer_agent', summaryOutput);

    steps.push({
      agent: 'summarizer_agent',
      output: summaryOutput,
      insights: [`Summarized ${task.sources.length} documents`],
    });
    agentsUsed.add('summarizer_agent');
  }

  /**
   * Simulate agent execution (placeholder for actual agent invocation).
   */
  private async simulateAgentExecution(
    agentName: string,
    inputs: Record<string, unknown>,
  ): Promise<OutputObject> {
    // In a real implementation, this would invoke the actual agent
    // For now, return a simulated successful result
    return {
      terminate_reason: 'GOAL',
      result: `${agentName} completed successfully with inputs: ${JSON.stringify(inputs)}`,
    };
  }

  /**
   * Extract insights from data flow agent output.
   */
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  private extractDataInsights(_output: OutputObject): {
    patterns: string[];
    issues: string[];
    insights: string[];
  } {
    // In a real implementation, this would parse the agent's structured output
    return {
      patterns: ['Data pattern 1', 'Data pattern 2'],
      issues: ['Quality issue 1'],
      insights: ['Data processing completed', 'Quality metrics calculated'],
    };
  }

  /**
   * Extract insights from literature analyzer output.
   */
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  private extractLiteratureInsights(_output: OutputObject): {
    concepts: string[];
    insights: string[];
  } {
    return {
      concepts: ['Concept 1', 'Concept 2'],
      insights: ['Literature analysis completed', 'Key themes identified'],
    };
  }

  /**
   * Synthesize results from all workflow steps.
   */
  private synthesizeResults(steps: AnalysisResult['steps']): {
    summary: string;
    keyFindings: string[];
    recommendations: string[];
  } {
    const allInsights = steps.flatMap((step) => step.insights);

    return {
      summary: `Completed analysis workflow with ${steps.length} steps using ${new Set(steps.map((s) => s.agent)).size} agents. ${allInsights.length} insights generated.`,
      keyFindings: allInsights.slice(0, 5),
      recommendations: [
        'Review detailed agent outputs for full analysis',
        'Consider applying findings to related tasks',
        'Update analysis based on new information',
      ],
    };
  }

  /**
   * Get analysis history from memory.
   */
  getAnalysisHistory(workflowType?: AnalysisWorkflowType): Array<{
    type: AnalysisWorkflowType;
    description: string;
    timestamp: Date;
  }> {
    const tags = workflowType ? [workflowType] : ['task'];
    const memories = this.getMemory().getRelevant(tags, 20);

    return memories
      .filter((mem) => mem.type === MemoryType.EPISODIC)
      .map((mem) => ({
        type: AnalysisWorkflowType.DATA_PROCESSING, // Default, should parse from content
        description: mem.summary,
        timestamp: mem.timestamp,
      }));
  }

  /**
   * Export analysis state for persistence.
   */
  exportAnalysisState(): {
    workflows: AnalysisWorkflowType[];
    insights: string[];
    patterns: string[];
    state: ReturnType<typeof this.exportState>;
  } {
    const state = this.exportState();
    const context = this.getSharedContext();

    const workflows = Array.from(
      new Set(
        context
          .get({ types: [ContextType.TASK_METADATA] })
          .map((item: ContextItem) => (item.data as { type?: string })?.type)
          .filter(Boolean),
      ),
    ) as AnalysisWorkflowType[];

    const insights = context
      .get({ types: [ContextType.CODE_INSIGHTS] })
      .map((item: ContextItem) => item.data)
      .flat();

    const patterns =
      Array.from(
        new Set(
          insights
            .map(
              (insight: unknown) => (insight as { pattern?: string })?.pattern,
            )
            .filter(Boolean),
        ),
      ) || [];

    return {
      workflows,
      insights: insights.map((i: unknown) => String(i)),
      patterns: patterns.map((p: unknown) => String(p)),
      state,
    };
  }
}
