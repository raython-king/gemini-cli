/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Prompts Module Index
 *
 * This module exports all system prompts and persona configurations
 * for the gemini-cli application.
 */

// Core prompt infrastructure
export { PromptRegistry } from './prompt-registry.js';
export { getMCPServerPrompts } from './mcp-prompts.js';

// Deep Learning Expert System
export {
  DEEP_LEARNING_EXPERT_SYSTEM_PROMPT,
  DEEP_LEARNING_EXPERT_PERSONA,
  DOMAIN_EXPERTISE_AREAS,
  CODE_STANDARDS,
  RESEARCH_PAPER_GUIDELINES,
  ARCHITECTURE_DESIGN_PRINCIPLES,
  TRAINING_BEST_PRACTICES,
  PERFORMANCE_OPTIMIZATION,
  HARDWARE_RECOMMENDATIONS,
  SAFETY_AND_ETHICS,
  MATHEMATICAL_NOTATION_STYLE,
  COMMON_ANTI_PATTERNS,
  GRADIENT_FLOW_ANALYSIS,
  HYPERPARAMETER_TUNING,
  REPRODUCIBILITY_PRACTICES,
} from './deep-learning-expert-prompt.js';

// Deep Learning Specialized Agents
export {
  DL_SPECIALIZED_AGENTS,
  ARCHITECTURE_DESIGN_AGENT_PROMPT,
  TRAINING_OPTIMIZATION_AGENT_PROMPT,
  DEBUGGING_PROFILING_AGENT_PROMPT,
  LITERATURE_REVIEW_AGENT_PROMPT,
  EXPERIMENT_DESIGN_AGENT_PROMPT,
  MODEL_EVALUATION_AGENT_PROMPT,
} from './dl-specialized-agents-prompts.js';

// Type definitions for prompts
export interface SystemPromptConfig {
  /** The main system prompt content */
  systemPrompt: string;
  /** Optional persona name */
  personaName?: string;
  /** Optional list of expertise domains */
  expertiseDomains?: string[];
  /** Optional temperature setting recommendation */
  recommendedTemperature?: number;
  /** Optional max tokens recommendation */
  recommendedMaxTokens?: number;
}

export interface AgentPromptConfig {
  /** Agent identifier */
  id: string;
  /** Agent display name */
  name: string;
  /** Agent's specialized prompt */
  prompt: string;
  /** Agent's primary focus area */
  focusArea: string;
  /** Tasks this agent should handle */
  taskTypes: string[];
}

/**
 * Get the Deep Learning Expert configuration with recommended settings.
 */
export function getDeepLearningExpertConfig(): SystemPromptConfig {
  // Import dynamically to avoid circular dependency issues
  const { DEEP_LEARNING_EXPERT_SYSTEM_PROMPT } = require('./deep-learning-expert-prompt.js');

  return {
    systemPrompt: DEEP_LEARNING_EXPERT_SYSTEM_PROMPT,
    personaName: 'Dr. Neural',
    expertiseDomains: [
      'Neural Network Architectures',
      'Transformers',
      'Generative Models',
      'Reinforcement Learning',
      'Optimization Theory',
      'Distributed Training',
    ],
    recommendedTemperature: 0.7,
    recommendedMaxTokens: 8192,
  };
}

/**
 * Get all specialized DL agent configurations.
 */
export function getSpecializedAgentConfigs(): AgentPromptConfig[] {
  // Import dynamically to avoid circular dependency issues
  const {
    ARCHITECTURE_DESIGN_AGENT_PROMPT,
    TRAINING_OPTIMIZATION_AGENT_PROMPT,
    DEBUGGING_PROFILING_AGENT_PROMPT,
    LITERATURE_REVIEW_AGENT_PROMPT,
    EXPERIMENT_DESIGN_AGENT_PROMPT,
    MODEL_EVALUATION_AGENT_PROMPT,
  } = require('./dl-specialized-agents-prompts.js');

  return [
    {
      id: 'architecture-design',
      name: 'Architecture Design Agent',
      prompt: ARCHITECTURE_DESIGN_AGENT_PROMPT,
      focusArea: 'Neural network architecture design and analysis',
      taskTypes: [
        'design_architecture',
        'analyze_architecture',
        'suggest_modifications',
        'capacity_planning',
        'component_selection',
      ],
    },
    {
      id: 'training-optimization',
      name: 'Training Optimization Agent',
      prompt: TRAINING_OPTIMIZATION_AGENT_PROMPT,
      focusArea: 'Training process optimization and efficiency',
      taskTypes: [
        'optimizer_selection',
        'learning_rate_tuning',
        'regularization_setup',
        'distributed_training',
        'mixed_precision',
      ],
    },
    {
      id: 'debugging-profiling',
      name: 'Debugging & Profiling Agent',
      prompt: DEBUGGING_PROFILING_AGENT_PROMPT,
      focusArea: 'Diagnosing and resolving training issues',
      taskTypes: [
        'debug_nan_loss',
        'fix_oom_error',
        'profile_performance',
        'analyze_gradients',
        'memory_optimization',
      ],
    },
    {
      id: 'literature-review',
      name: 'Literature Review Agent',
      prompt: LITERATURE_REVIEW_AGENT_PROMPT,
      focusArea: 'Research paper analysis and synthesis',
      taskTypes: [
        'summarize_paper',
        'compare_methods',
        'identify_trends',
        'find_related_work',
        'critical_analysis',
      ],
    },
    {
      id: 'experiment-design',
      name: 'Experiment Design Agent',
      prompt: EXPERIMENT_DESIGN_AGENT_PROMPT,
      focusArea: 'Designing rigorous ML experiments',
      taskTypes: [
        'design_ablation',
        'plan_experiment',
        'statistical_analysis',
        'resource_estimation',
        'hypothesis_testing',
      ],
    },
    {
      id: 'model-evaluation',
      name: 'Model Evaluation Agent',
      prompt: MODEL_EVALUATION_AGENT_PROMPT,
      focusArea: 'Model performance evaluation and analysis',
      taskTypes: [
        'compute_metrics',
        'error_analysis',
        'fairness_evaluation',
        'robustness_testing',
        'calibration_analysis',
      ],
    },
  ];
}

/**
 * Route a task to the appropriate specialized agent.
 *
 * @param taskDescription - Description of the task
 * @returns The most appropriate agent configuration, or null if no match
 */
export function routeToAgent(taskDescription: string): AgentPromptConfig | null {
  const agents = getSpecializedAgentConfigs();
  const taskLower = taskDescription.toLowerCase();

  // Simple keyword matching for routing
  const routingKeywords: Record<string, string[]> = {
    'architecture-design': [
      'architecture', 'design', 'layer', 'block', 'module', 'structure',
      'transformer', 'cnn', 'rnn', 'attention', 'capacity',
    ],
    'training-optimization': [
      'train', 'optimizer', 'learning rate', 'lr', 'schedule', 'regularization',
      'dropout', 'weight decay', 'distributed', 'fsdp', 'ddp',
    ],
    'debugging-profiling': [
      'debug', 'error', 'nan', 'inf', 'oom', 'memory', 'profile', 'slow',
      'gradient', 'vanishing', 'exploding', 'crash',
    ],
    'literature-review': [
      'paper', 'research', 'survey', 'review', 'compare', 'literature',
      'arxiv', 'publication', 'study', 'method',
    ],
    'experiment-design': [
      'experiment', 'ablation', 'hypothesis', 'test', 'compare',
      'statistical', 'significance', 'baseline', 'sweep',
    ],
    'model-evaluation': [
      'evaluate', 'metric', 'accuracy', 'loss', 'performance', 'benchmark',
      'fairness', 'robustness', 'calibration', 'error analysis',
    ],
  };

  // Score each agent based on keyword matches
  let bestAgent: AgentPromptConfig | null = null;
  let bestScore = 0;

  for (const agent of agents) {
    const keywords = routingKeywords[agent.id] || [];
    const score = keywords.filter(keyword => taskLower.includes(keyword)).length;

    if (score > bestScore) {
      bestScore = score;
      bestAgent = agent;
    }
  }

  return bestScore > 0 ? bestAgent : null;
}
