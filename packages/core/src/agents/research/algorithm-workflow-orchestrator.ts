/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { TaskDecomposer, type Subtask, type DecompositionResult } from '../task-decomposer.js';
import { TaskType, TaskComplexity } from '../orchestrator.js';
import type { AgentInputs } from '../types.js';
import { debugLogger } from '../../utils/debugLogger.js';

/**
 * Algorithm development workflow type.
 */
export enum AlgorithmWorkflowType {
  /** Research and implement new algorithm */
  RESEARCH_AND_IMPLEMENT = 'research_and_implement',
  /** Reproduce paper results */
  REPRODUCE_PAPER = 'reproduce_paper',
  /** Optimize existing model */
  OPTIMIZE_MODEL = 'optimize_model',
  /** Benchmark and compare models */
  BENCHMARK_COMPARISON = 'benchmark_comparison',
  /** Full experimental study */
  EXPERIMENTAL_STUDY = 'experimental_study',
  /** Deploy model to production */
  MODEL_DEPLOYMENT = 'model_deployment',
}

/**
 * Algorithm domain.
 */
export enum AlgorithmDomain {
  LLM = 'llm',
  CV = 'cv',
  RL = 'rl',
  GENERAL_ML = 'general_ml',
}

/**
 * Workflow configuration.
 */
export interface AlgorithmWorkflowConfig {
  /** Workflow type */
  workflowType: AlgorithmWorkflowType;
  /** Algorithm domain */
  domain: AlgorithmDomain;
  /** Main objective */
  objective: string;
  /** Research papers (if any) */
  papers?: string[];
  /** Existing code */
  codebase?: string;
  /** Dataset */
  dataset?: string;
  /** Constraints */
  constraints?: {
    computeBudget?: string;
    timeLimit?: string;
    hardware?: string[];
  };
  /** Specific requirements */
  requirements?: {
    includeAblation?: boolean;
    includeBaselines?: boolean;
    targetMetric?: string;
    deploymentTarget?: string;
  };
}

/**
 * Algorithm Workflow Orchestrator - Specialized orchestrator for algorithm development.
 *
 * This orchestrator creates expert-level algorithm development workflows by:
 * 1. Analyzing the research objective and domain
 * 2. Designing a complete research pipeline
 * 3. Coordinating specialized research agents
 * 4. Managing the full lifecycle from research to deployment
 */
export class AlgorithmWorkflowOrchestrator extends TaskDecomposer {
  /**
   * Create a comprehensive algorithm development workflow.
   */
  createAlgorithmWorkflow(config: AlgorithmWorkflowConfig): DecompositionResult {
    debugLogger.log(
      `[AlgorithmWorkflowOrchestrator] Creating ${config.workflowType} workflow for ${config.domain}`,
    );

    const subtasks: Subtask[] = [];
    let taskDescription = config.objective;

    // Generate subtasks based on workflow type and domain
    switch (config.workflowType) {
      case AlgorithmWorkflowType.RESEARCH_AND_IMPLEMENT:
        subtasks.push(...this.createResearchAndImplementWorkflow(config));
        break;

      case AlgorithmWorkflowType.REPRODUCE_PAPER:
        subtasks.push(...this.createReproducePaperWorkflow(config));
        break;

      case AlgorithmWorkflowType.OPTIMIZE_MODEL:
        subtasks.push(...this.createOptimizeModelWorkflow(config));
        break;

      case AlgorithmWorkflowType.BENCHMARK_COMPARISON:
        subtasks.push(...this.createBenchmarkComparisonWorkflow(config));
        break;

      case AlgorithmWorkflowType.EXPERIMENTAL_STUDY:
        subtasks.push(...this.createExperimentalStudyWorkflow(config));
        break;

      case AlgorithmWorkflowType.MODEL_DEPLOYMENT:
        subtasks.push(...this.createModelDeploymentWorkflow(config));
        break;
    }

    // Create execution plan
    const executionPlan = this.createExecutionPlan(subtasks);

    return {
      originalTask: taskDescription,
      classification: {
        type: TaskType.IMPLEMENTATION,
        complexity: TaskComplexity.VERY_HIGH,
        recommendedAgents: this.getRecommendedAgents(config.domain),
        requiresPlan: true,
        canParallelize: true,
      },
      subtasks,
      executionPlan,
      estimatedDuration: this.estimateDuration(subtasks),
    };
  }

  /**
   * Research and implement workflow.
   */
  private createResearchAndImplementWorkflow(
    config: AlgorithmWorkflowConfig,
  ): Subtask[] {
    const subtasks: Subtask[] = [];
    const researchAgent = this.getResearchAgent(config.domain);

    // Phase 1: Literature review and research
    subtasks.push({
      id: this.generateSubtaskId(),
      title: `${config.domain.toUpperCase()} Literature Review`,
      description: `Conduct comprehensive literature review on: ${config.objective}`,
      recommendedAgents: [researchAgent],
      dependencies: [],
      complexity: TaskComplexity.HIGH,
      priority: 10,
      inputs: {
        objective: config.objective,
        papers: config.papers,
        depth: 'comprehensive',
        includeImplementation: true,
      },
      canParallelize: false,
    });

    // Phase 2: Dataset analysis (parallel with phase 3)
    if (config.dataset) {
      subtasks.push({
        id: this.generateSubtaskId(),
        title: 'Dataset Analysis',
        description: 'Analyze dataset characteristics and requirements',
        recommendedAgents: ['data_flow_agent'],
        dependencies: [subtasks[0].id],
        complexity: TaskComplexity.MEDIUM,
        priority: 9,
        inputs: {
          dataSource: config.dataset,
          task: 'Analyze dataset for algorithm development',
        },
        canParallelize: true,
      });
    }

    // Phase 3: Experiment design
    const experimentSubtaskId = this.generateSubtaskId();
    subtasks.push({
      id: experimentSubtaskId,
      title: 'Experiment Design',
      description: 'Design comprehensive experiments to validate approach',
      recommendedAgents: ['experiment_agent'],
      dependencies: [subtasks[0].id],
      complexity: TaskComplexity.HIGH,
      priority: 9,
      inputs: {
        objective: config.objective,
        algorithmType: config.domain,
        experimentType: 'baseline',
        dataset: config.dataset,
        constraints: config.constraints,
      },
      canParallelize: true,
    });

    // Phase 4: Implementation
    subtasks.push({
      id: this.generateSubtaskId(),
      title: 'Algorithm Implementation',
      description: 'Implement the algorithm based on research insights',
      recommendedAgents: ['plan_agent', 'code_reviewer'],
      dependencies: [subtasks[0].id, experimentSubtaskId],
      complexity: TaskComplexity.VERY_HIGH,
      priority: 8,
      inputs: {
        task: `Implement ${config.objective} based on research findings`,
      },
      canParallelize: false,
    });

    // Phase 5: Model evaluation
    const implementationId = subtasks[subtasks.length - 1].id;
    subtasks.push({
      id: this.generateSubtaskId(),
      title: 'Model Evaluation',
      description: 'Comprehensive evaluation of implemented model',
      recommendedAgents: ['model_evaluation_agent'],
      dependencies: [implementationId],
      complexity: TaskComplexity.HIGH,
      priority: 7,
      inputs: {
        modelPath: config.codebase || './model',
        objective: 'Comprehensive performance and efficiency evaluation',
        modelType: config.domain,
        focus: ['accuracy', 'efficiency', 'robustness'],
        analyzeOptimization: true,
      },
      canParallelize: false,
    });

    // Phase 6: Optimization (if needed)
    subtasks.push({
      id: this.generateSubtaskId(),
      title: 'Model Optimization',
      description: 'Apply optimizations based on evaluation results',
      recommendedAgents: ['refactor_agent', 'code_reviewer'],
      dependencies: [subtasks[subtasks.length - 1].id],
      complexity: TaskComplexity.HIGH,
      priority: 6,
      inputs: {
        target: config.codebase || './model',
        strategy: 'performance',
      },
      canParallelize: false,
    });

    return subtasks;
  }

  /**
   * Reproduce paper workflow.
   */
  private createReproducePaperWorkflow(
    config: AlgorithmWorkflowConfig,
  ): Subtask[] {
    const subtasks: Subtask[] = [];
    const researchAgent = this.getResearchAgent(config.domain);

    // Phase 1: Deep paper analysis
    subtasks.push({
      id: this.generateSubtaskId(),
      title: 'Deep Paper Analysis',
      description: 'Analyze paper methodology, architecture, and experiments in detail',
      recommendedAgents: ['literature_analyzer_agent', researchAgent],
      dependencies: [],
      complexity: TaskComplexity.HIGH,
      priority: 10,
      inputs: {
        document: config.papers?.[0] || '',
        focus: 'methodology',
        depth: 'comprehensive',
      },
      canParallelize: false,
    });

    // Phase 2: Implementation planning
    subtasks.push({
      id: this.generateSubtaskId(),
      title: 'Reproduce Implementation Plan',
      description: 'Create detailed plan to reproduce paper results',
      recommendedAgents: ['plan_agent'],
      dependencies: [subtasks[0].id],
      complexity: TaskComplexity.MEDIUM,
      priority: 9,
      inputs: {
        task: 'Create implementation plan to reproduce paper results',
      },
      canParallelize: false,
    });

    // Phase 3: Experiment setup
    subtasks.push({
      id: this.generateSubtaskId(),
      title: 'Experiment Setup',
      description: 'Set up exact experimental conditions from paper',
      recommendedAgents: ['experiment_agent'],
      dependencies: [subtasks[0].id],
      complexity: TaskComplexity.MEDIUM,
      priority: 9,
      inputs: {
        objective: 'Reproduce exact experiments from paper',
        algorithmType: config.domain,
        experimentType: 'reproducibility',
        dataset: config.dataset,
      },
      canParallelize: true,
    });

    // Phase 4: Implementation
    subtasks.push({
      id: this.generateSubtaskId(),
      title: 'Implement Paper Method',
      description: 'Implement the method exactly as described in paper',
      recommendedAgents: ['plan_agent'],
      dependencies: [subtasks[1].id, subtasks[2].id],
      complexity: TaskComplexity.VERY_HIGH,
      priority: 8,
      inputs: {
        task: 'Implement paper method',
      },
      canParallelize: false,
    });

    // Phase 5: Validation
    subtasks.push({
      id: this.generateSubtaskId(),
      title: 'Result Validation',
      description: 'Validate that results match paper claims',
      recommendedAgents: ['model_evaluation_agent'],
      dependencies: [subtasks[subtasks.length - 1].id],
      complexity: TaskComplexity.HIGH,
      priority: 7,
      inputs: {
        modelPath: './reproduced_model',
        objective: 'Validate reproduction against paper results',
        modelType: config.domain,
      },
      canParallelize: false,
    });

    return subtasks;
  }

  /**
   * Optimize model workflow.
   */
  private createOptimizeModelWorkflow(
    config: AlgorithmWorkflowConfig,
  ): Subtask[] {
    const subtasks: Subtask[] = [];

    // Phase 1: Current model analysis
    subtasks.push({
      id: this.generateSubtaskId(),
      title: 'Baseline Model Evaluation',
      description: 'Comprehensive analysis of current model',
      recommendedAgents: ['model_evaluation_agent'],
      dependencies: [],
      complexity: TaskComplexity.MEDIUM,
      priority: 10,
      inputs: {
        modelPath: config.codebase || './model',
        objective: 'Identify optimization opportunities',
        modelType: config.domain,
        focus: ['efficiency', 'accuracy'],
        analyzeOptimization: true,
      },
      canParallelize: false,
    });

    // Phase 2: Optimization strategy research
    subtasks.push({
      id: this.generateSubtaskId(),
      title: 'Optimization Strategy Research',
      description: 'Research applicable optimization techniques',
      recommendedAgents: [this.getResearchAgent(config.domain)],
      dependencies: [subtasks[0].id],
      complexity: TaskComplexity.MEDIUM,
      priority: 9,
      inputs: {
        objective: `Optimization techniques for ${config.domain} models`,
        topic: 'model_optimization',
        depth: 'thorough',
      },
      canParallelize: false,
    });

    // Phase 3: Optimization experiments
    subtasks.push({
      id: this.generateSubtaskId(),
      title: 'Optimization Experiments',
      description: 'Design and run optimization experiments',
      recommendedAgents: ['experiment_agent'],
      dependencies: [subtasks[1].id],
      complexity: TaskComplexity.HIGH,
      priority: 8,
      inputs: {
        objective: 'Test various optimization strategies',
        algorithmType: config.domain,
        experimentType: 'ablation',
        codebase: config.codebase,
      },
      canParallelize: false,
    });

    // Phase 4: Apply optimizations
    subtasks.push({
      id: this.generateSubtaskId(),
      title: 'Apply Optimizations',
      description: 'Implement best optimization strategies',
      recommendedAgents: ['refactor_agent'],
      dependencies: [subtasks[2].id],
      complexity: TaskComplexity.HIGH,
      priority: 7,
      inputs: {
        target: config.codebase || './model',
        strategy: 'performance',
      },
      canParallelize: false,
    });

    // Phase 5: Final evaluation
    subtasks.push({
      id: this.generateSubtaskId(),
      title: 'Optimized Model Evaluation',
      description: 'Evaluate optimized model performance',
      recommendedAgents: ['model_evaluation_agent'],
      dependencies: [subtasks[subtasks.length - 1].id],
      complexity: TaskComplexity.MEDIUM,
      priority: 6,
      inputs: {
        modelPath: './optimized_model',
        objective: 'Compare optimized vs baseline',
        modelType: config.domain,
        baselines: [config.codebase || './model'],
      },
      canParallelize: false,
    });

    return subtasks;
  }

  /**
   * Benchmark comparison workflow.
   */
  private createBenchmarkComparisonWorkflow(
    config: AlgorithmWorkflowConfig,
  ): Subtask[] {
    const subtasks: Subtask[] = [];

    // Phase 1: Identify baselines and benchmarks
    subtasks.push({
      id: this.generateSubtaskId(),
      title: 'Identify Baselines and Benchmarks',
      description: 'Research relevant baselines and standard benchmarks',
      recommendedAgents: [this.getResearchAgent(config.domain)],
      dependencies: [],
      complexity: TaskComplexity.MEDIUM,
      priority: 10,
      inputs: {
        objective: `Identify benchmarks and baselines for: ${config.objective}`,
        depth: 'thorough',
      },
      canParallelize: false,
    });

    // Phase 2: Design comparison experiments
    subtasks.push({
      id: this.generateSubtaskId(),
      title: 'Design Comparison Experiments',
      description: 'Design fair comparison experiments',
      recommendedAgents: ['experiment_agent'],
      dependencies: [subtasks[0].id],
      complexity: TaskComplexity.HIGH,
      priority: 9,
      inputs: {
        objective: 'Compare models fairly across benchmarks',
        algorithmType: config.domain,
        experimentType: 'comparison',
      },
      canParallelize: false,
    });

    // Phase 3: Evaluate all models (can be parallelized)
    subtasks.push({
      id: this.generateSubtaskId(),
      title: 'Run Benchmark Evaluations',
      description: 'Evaluate all models on standard benchmarks',
      recommendedAgents: ['model_evaluation_agent'],
      dependencies: [subtasks[1].id],
      complexity: TaskComplexity.HIGH,
      priority: 8,
      inputs: {
        modelPath: config.codebase || './models',
        objective: 'Benchmark on standard datasets',
        modelType: config.domain,
      },
      canParallelize: true,
    });

    // Phase 4: Analysis and visualization
    subtasks.push({
      id: this.generateSubtaskId(),
      title: 'Comparative Analysis',
      description: 'Analyze results and create comparison report',
      recommendedAgents: ['summarizer_agent', 'data_flow_agent'],
      dependencies: [subtasks[2].id],
      complexity: TaskComplexity.MEDIUM,
      priority: 7,
      inputs: {
        sources: ['./benchmark_results'],
        style: 'technical',
        length: 'comprehensive',
      },
      canParallelize: false,
    });

    return subtasks;
  }

  /**
   * Experimental study workflow.
   */
  private createExperimentalStudyWorkflow(
    config: AlgorithmWorkflowConfig,
  ): Subtask[] {
    const subtasks: Subtask[] = [];

    // Comprehensive research study with multiple experiments
    subtasks.push(...this.createResearchAndImplementWorkflow(config));

    // Add ablation studies
    const lastId = subtasks[subtasks.length - 1].id;
    subtasks.push({
      id: this.generateSubtaskId(),
      title: 'Ablation Studies',
      description: 'Conduct ablation studies to understand component contributions',
      recommendedAgents: ['experiment_agent'],
      dependencies: [lastId],
      complexity: TaskComplexity.HIGH,
      priority: 5,
      inputs: {
        objective: 'Ablation study for all model components',
        algorithmType: config.domain,
        experimentType: 'ablation',
        codebase: config.codebase,
      },
      canParallelize: false,
    });

    return subtasks;
  }

  /**
   * Model deployment workflow.
   */
  private createModelDeploymentWorkflow(
    config: AlgorithmWorkflowConfig,
  ): Subtask[] {
    const subtasks: Subtask[] = [];

    // Phase 1: Deployment readiness assessment
    subtasks.push({
      id: this.generateSubtaskId(),
      title: 'Deployment Readiness Assessment',
      description: 'Assess model readiness for production deployment',
      recommendedAgents: ['model_evaluation_agent'],
      dependencies: [],
      complexity: TaskComplexity.MEDIUM,
      priority: 10,
      inputs: {
        modelPath: config.codebase || './model',
        objective: 'Assess deployment readiness',
        modelType: config.domain,
        focus: ['efficiency', 'robustness', 'all'],
      },
      canParallelize: false,
    });

    // Phase 2: Optimization for deployment
    subtasks.push({
      id: this.generateSubtaskId(),
      title: 'Deployment Optimization',
      description: 'Optimize model for production environment',
      recommendedAgents: ['refactor_agent'],
      dependencies: [subtasks[0].id],
      complexity: TaskComplexity.HIGH,
      priority: 9,
      inputs: {
        target: config.codebase || './model',
        strategy: 'performance',
      },
      canParallelize: false,
    });

    // Phase 3: Testing and validation
    subtasks.push({
      id: this.generateSubtaskId(),
      title: 'Production Testing',
      description: 'Comprehensive testing for production',
      recommendedAgents: ['test_runner'],
      dependencies: [subtasks[1].id],
      complexity: TaskComplexity.MEDIUM,
      priority: 8,
      inputs: {
        testCommand: 'npm test',
        targetFiles: 'tests/production/**',
      },
      canParallelize: false,
    });

    return subtasks;
  }

  /**
   * Get research agent based on domain.
   */
  private getResearchAgent(domain: AlgorithmDomain): string {
    switch (domain) {
      case AlgorithmDomain.LLM:
        return 'llm_research_agent';
      case AlgorithmDomain.CV:
        return 'cv_research_agent';
      default:
        return 'explore_agent';
    }
  }

  /**
   * Get recommended agents for domain.
   */
  private getRecommendedAgents(domain: AlgorithmDomain): string[] {
    const commonAgents = [
      'experiment_agent',
      'model_evaluation_agent',
      'data_flow_agent',
    ];

    switch (domain) {
      case AlgorithmDomain.LLM:
        return ['llm_research_agent', ...commonAgents];
      case AlgorithmDomain.CV:
        return ['cv_research_agent', ...commonAgents];
      default:
        return ['explore_agent', ...commonAgents];
    }
  }
}
