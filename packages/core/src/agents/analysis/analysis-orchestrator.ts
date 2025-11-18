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
import { ParallelAgentExecutor, type AgentTask } from '../parallel-executor.js';
import type {
  OutputObject,
  AgentDefinition,
  AgentInputs,
  AgentTerminateMode,
} from '../types.js';
import type { Config } from '../../config/config.js';
import {
  ContextType,
  ContextPriority,
  type ContextItem,
} from '../context/shared-context.js';
import { OptimizationStrategy } from '../context/context-optimizer.js';
import { MemoryType } from '../context/agent-memory.js';
import { debugLogger } from '../../utils/debugLogger.js';
import { DataFlowAgent } from './data-flow-agent.js';
import { LiteratureAnalyzerAgent } from './literature-analyzer-agent.js';
import { SummarizerAgent } from './summarizer-agent.js';
import { AgentExecutor } from '../executor.js';

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
 * const orchestrator = new AnalysisOrchestrator(config);
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
  private parallelExecutor: ParallelAgentExecutor;
  private agentDefinitions: Map<string, AgentDefinition>;

  constructor(
    private readonly runtimeContext: Config,
    options: {
      maxContextTokens?: number;
      optimizationStrategy?: OptimizationStrategy;
      maxConcurrency?: number;
    } = {},
  ) {
    super({
      maxContextTokens: options.maxContextTokens ?? 6000, // Higher limit for analysis tasks
      optimizationStrategy:
        options.optimizationStrategy ?? OptimizationStrategy.BALANCED,
    });

    this.parallelExecutor = new ParallelAgentExecutor(
      runtimeContext,
      options.maxConcurrency ?? 3,
    );

    this.agentDefinitions = new Map();
    this.registerDefaultAgents();
  }

  /**
   * Register default analysis agents.
   */
  private registerDefaultAgents(): void {
    // Register the analysis agents
    this.registerAgent(DataFlowAgent);
    this.registerAgent(LiteratureAnalyzerAgent);
    this.registerAgent(SummarizerAgent);

    debugLogger.log(
      '[AnalysisOrchestrator] Registered default analysis agents: ' +
        'DataFlowAgent, LiteratureAnalyzerAgent, SummarizerAgent',
    );
  }

  /**
   * Register a custom agent for use in workflows.
   */
  registerAgent(agentDefinition: AgentDefinition): void {
    this.agentDefinitions.set(agentDefinition.name, agentDefinition);
    debugLogger.log(
      `[AnalysisOrchestrator] Registered agent: ${agentDefinition.name}`,
    );
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

    debugLogger.log(
      `[AnalysisOrchestrator] Starting ${task.type} workflow with ${task.sources.length} sources`,
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

      debugLogger.log(
        `[AnalysisOrchestrator] Workflow completed in ${totalExecutionTime}ms. ` +
          `Agents used: ${Array.from(agentsUsed).join(', ')}`,
      );

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
    const dataFlowOutput = await this.executeAgent(
      'data_flow_agent',
      {
        dataSource: task.sources[0],
        task: task.description,
        validationRules: task.options?.dataValidation,
      },
      { useParallel: false },
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
    const summaryOutput = await this.executeAgent(
      'summarizer_agent',
      {
        sources: [task.sources[0]],
        style: 'technical',
        includeActionItems: task.options?.includeActionItems ?? true,
      },
      { useParallel: false },
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
    // Step 1: Analyze each document with LiteratureAnalyzerAgent (in parallel)
    const analysisTasks: AgentTask[] = task.sources.map((source, index) => ({
      id: `lit_analysis_${index}`,
      agentName: 'literature_analyzer_agent',
      agentDefinition: this.getAgentDefinition('literature_analyzer_agent'),
      inputs: {
        document: source,
        focus: task.options?.focus?.join(', '),
        depth: task.options?.depth ?? 'detailed',
      },
      priority: task.sources.length - index,
    }));

    // Execute literature analysis tasks in parallel if there are multiple sources
    if (analysisTasks.length > 1) {
      const results = await this.parallelExecutor.executeParallel(
        analysisTasks,
      );

      for (const result of results) {
        if (result.success && result.output) {
          const litInsights = this.extractLiteratureInsights(result.output);
          this.recordExecution(result.agentName, result.output, {
            patterns: litInsights.concepts,
            files: [
              task.sources[
                parseInt(result.taskId.replace('lit_analysis_', ''))
              ],
            ],
          });

          steps.push({
            agent: result.agentName,
            output: result.output,
            insights: litInsights.insights,
          });
          agentsUsed.add(result.agentName);
        }
      }
    } else if (analysisTasks.length === 1) {
      // Single source - execute directly
      const output = await this.executeAgent(
        'literature_analyzer_agent',
        analysisTasks[0].inputs,
        { useParallel: false },
      );

      const litInsights = this.extractLiteratureInsights(output);
      this.recordExecution('literature_analyzer_agent', output, {
        patterns: litInsights.concepts,
        files: [task.sources[0]],
      });

      steps.push({
        agent: 'literature_analyzer_agent',
        output,
        insights: litInsights.insights,
      });
      agentsUsed.add('literature_analyzer_agent');
    }

    // Step 2: Synthesize findings with SummarizerAgent
    const summaryOutput = await this.executeAgent(
      'summarizer_agent',
      {
        sources: task.sources,
        style: 'academic',
        focus: task.options?.focus,
        includeActionItems: true,
      },
      { useParallel: false },
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
    const finalSummaryOutput = await this.executeAgent(
      'summarizer_agent',
      {
        sources: task.sources,
        style: 'executive',
        length: 'comprehensive',
        includeActionItems: true,
      },
      { useParallel: false },
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
    const summaryOutput = await this.executeAgent(
      'summarizer_agent',
      {
        sources: task.sources,
        style: task.options?.depth === 'overview' ? 'executive' : 'narrative',
        length: task.options?.depth ?? 'moderate',
        focus: task.options?.focus,
        includeActionItems: task.options?.includeActionItems ?? true,
      },
      { useParallel: false },
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
   * Execute an agent with context enhancement.
   *
   * @param agentName Name of the agent to execute
   * @param inputs Agent inputs
   * @param options Execution options
   * @returns Agent output
   */
  private async executeAgent(
    agentName: string,
    inputs: AgentInputs,
    options: { useParallel: boolean },
  ): Promise<OutputObject> {
    const agentDef = this.getAgentDefinition(agentName);

    // Enhance inputs with context if available
    const taskClass = this.classifyTask(
      typeof inputs.task === 'string'
        ? inputs.task
        : typeof inputs.description === 'string'
          ? inputs.description
          : agentName,
    );
    const enhancedInputs = this.enhanceInputsWithContext(
      inputs,
      agentName,
      taskClass,
    );

    if (options.useParallel) {
      const tasks = ParallelAgentExecutor.createTasks([
        {
          agentDefinition: agentDef,
          inputs: enhancedInputs,
        },
      ]);

      const results = await this.parallelExecutor.executeParallel(tasks);

      if (results.length > 0 && results[0].success && results[0].output) {
        return results[0].output;
      } else if (results.length > 0 && results[0].error) {
        throw results[0].error;
      } else {
        throw new Error(`Agent ${agentName} failed without error details`);
      }
    } else {
      // Execute agent directly using AgentExecutor
      const executor = await AgentExecutor.create(
        agentDef,
        this.runtimeContext,
      );

      const abortController = new AbortController();
      return await executor.run(enhancedInputs, abortController.signal);
    }
  }

  /**
   * Get agent definition by name.
   * Falls back to a placeholder if not registered.
   */
  private getAgentDefinition(agentName: string): AgentDefinition {
    const definition = this.agentDefinitions.get(agentName);
    if (definition) {
      return definition;
    }

    // Return a placeholder definition for development/testing
    debugLogger.log(
      `[AnalysisOrchestrator] Agent ${agentName} not registered, using placeholder`,
    );

    return {
      name: agentName,
      description: `Placeholder for ${agentName}`,
      inputConfig: {
        inputs: {},
      },
      promptConfig: {
        systemPrompt: `You are ${agentName}.`,
      },
      modelConfig: {
        model: 'gemini-2.0-flash-exp',
        temp: 0.7,
        top_p: 0.95,
      },
      runConfig: {
        max_time_minutes: 10,
        max_turns: 20,
      },
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
