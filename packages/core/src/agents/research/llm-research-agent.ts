/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { z } from 'zod';
import type { AgentDefinition } from '../types.js';

/**
 * Input schema for LLM Research Agent.
 */
const LLMResearchInputSchema = z.object({
  /** Research objective or question */
  objective: z.string().describe('The research objective or question to investigate'),

  /** Specific LLM topic (e.g., "transformer architecture", "prompt engineering", "RLHF") */
  topic: z.string().optional().describe('Specific LLM topic to focus on'),

  /** Papers or resources to analyze */
  papers: z.array(z.string()).optional().describe('List of paper paths or URLs to analyze'),

  /** Research depth */
  depth: z.enum(['quick', 'thorough', 'comprehensive']).default('thorough')
    .describe('Research depth: quick (survey), thorough (detailed), comprehensive (in-depth)'),

  /** Focus areas */
  focus: z.array(z.enum([
    'architecture',
    'training',
    'inference',
    'optimization',
    'evaluation',
    'applications',
    'theory',
    'implementation'
  ])).optional().describe('Specific focus areas for research'),

  /** Include implementation */
  includeImplementation: z.boolean().default(false)
    .describe('Whether to provide implementation guidance'),
});

/**
 * Output schema for LLM Research Agent.
 */
const LLMResearchOutputSchema = z.object({
  /** Research summary */
  Summary: z.string().describe('Executive summary of research findings'),

  /** Key findings */
  KeyFindings: z.array(z.object({
    title: z.string(),
    description: z.string(),
    significance: z.enum(['critical', 'high', 'medium', 'low']),
    category: z.string(), // e.g., "architecture", "training", etc.
  })).describe('Most important findings from the research'),

  /** Technical analysis */
  TechnicalAnalysis: z.object({
    architecture: z.string().optional(),
    mathematicalFoundations: z.string().optional(),
    trainingMethodology: z.string().optional(),
    optimizationTechniques: z.string().optional(),
    performanceCharacteristics: z.string().optional(),
  }).describe('Detailed technical analysis'),

  /** State-of-the-art comparison */
  SOTAComparison: z.array(z.object({
    model: z.string(),
    strengths: z.array(z.string()),
    weaknesses: z.array(z.string()),
    performance: z.string(),
  })).optional().describe('Comparison with state-of-the-art models'),

  /** Implementation insights */
  ImplementationInsights: z.object({
    codeStructure: z.string().optional(),
    keyComponents: z.array(z.string()).optional(),
    dependencies: z.array(z.string()).optional(),
    implementationChallenges: z.array(z.string()).optional(),
    bestPractices: z.array(z.string()).optional(),
  }).optional().describe('Implementation guidance and insights'),

  /** Experiments to run */
  SuggestedExperiments: z.array(z.object({
    title: z.string(),
    objective: z.string(),
    methodology: z.string(),
    expectedOutcomes: z.string(),
    priority: z.enum(['high', 'medium', 'low']),
  })).describe('Suggested experiments based on research'),

  /** Mathematical formulations */
  MathematicalFormulations: z.array(z.object({
    concept: z.string(),
    formula: z.string(),
    explanation: z.string(),
  })).optional().describe('Key mathematical formulations'),

  /** Resources */
  Resources: z.object({
    papers: z.array(z.object({
      title: z.string(),
      authors: z.string().optional(),
      year: z.string().optional(),
      url: z.string().optional(),
      relevance: z.string(),
    })),
    codebases: z.array(z.string()).optional(),
    datasets: z.array(z.string()).optional(),
  }).describe('Relevant resources for further study'),

  /** Future directions */
  FutureDirections: z.array(z.string())
    .describe('Potential future research directions'),

  /** Implementation roadmap */
  ImplementationRoadmap: z.array(z.object({
    phase: z.string(),
    tasks: z.array(z.string()),
    estimatedTime: z.string(),
    dependencies: z.array(z.string()),
  })).optional().describe('Step-by-step implementation roadmap'),
});

/**
 * System prompt for LLM Research Agent.
 */
const SYSTEM_PROMPT = `You are an expert LLM researcher with deep knowledge of:
- Transformer architectures and their variants (GPT, BERT, T5, etc.)
- Training methodologies (pre-training, fine-tuning, RLHF, DPO)
- Attention mechanisms and efficiency improvements
- Tokenization and embedding strategies
- Optimization techniques (AdamW, learning rate schedules, gradient accumulation)
- Evaluation metrics and benchmarks
- Prompt engineering and in-context learning
- Model compression and quantization
- Inference optimization and serving
- Latest research trends and SOTA models

Your role is to conduct thorough research on LLM topics and provide expert-level insights.

Research Methodology:
1. **Literature Review**: Analyze relevant papers, focusing on methodology, results, and innovations
2. **Technical Analysis**: Deep dive into architecture, training, and optimization details
3. **Mathematical Understanding**: Explain key formulations and theoretical foundations
4. **Implementation Insights**: Provide practical guidance for implementation
5. **Comparative Analysis**: Compare with existing approaches and SOTA models
6. **Experimental Design**: Suggest meaningful experiments to validate findings
7. **Future Directions**: Identify open problems and potential improvements

Output Requirements:
- Be technically precise and mathematically rigorous
- Cite specific papers, models, or techniques when relevant
- Provide actionable insights for implementation
- Include code snippets or pseudocode when helpful
- Suggest concrete experiments with clear objectives
- Explain complex concepts clearly but don't oversimplify
- Focus on practical applicability alongside theoretical understanding

When analyzing papers:
1. Extract key contributions and novelty
2. Understand the methodology in detail
3. Identify limitations and potential improvements
4. Consider computational requirements
5. Evaluate reproducibility and practical applicability

Use available tools to:
- Read papers and documentation
- Search for related work
- Analyze existing implementations
- Gather performance benchmarks
- Find relevant datasets`;

/**
 * LLM Research Agent - Expert-level LLM algorithm research.
 *
 * This agent specializes in:
 * - Deep analysis of LLM architectures and techniques
 * - Paper analysis and literature review
 * - Mathematical foundations and theory
 * - Implementation guidance
 * - Experimental design
 * - Performance optimization
 * - SOTA comparisons
 */
export const LLMResearchAgent: AgentDefinition<typeof LLMResearchOutputSchema> = {
  name: 'llm_research_agent',
  description: 'Expert LLM researcher for deep analysis of language models and techniques',
  version: '1.0.0',
  system_prompt: SYSTEM_PROMPT,
  input_schema: LLMResearchInputSchema,
  output_schema: LLMResearchOutputSchema,
  max_turns: 30,
  max_time_minutes: 30,
  thinking: {
    type: 'enabled',
    budget_tokens: 10000,
  },
  tool_config: {
    allowed_tools: [
      'read_file',
      'read_many_files',
      'grep',
      'glob',
      'web_fetch',
      'web_search',
    ],
    parallel_tool_calls: true,
  },
  model: 'gemini-2.0-flash-thinking-exp-01-21',
};
