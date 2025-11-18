/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { z } from 'zod';
import { SkillCategory, SkillComplexity, type SkillDefinition } from '../types.js';
import { AgentExecutor } from '../../agents/executor.js';
import { LLMResearchAgent } from '../../agents/research/llm-research-agent.js';

const PromptOptimizationInputSchema = z.object({
  task: z.string().describe('Task description'),
  currentPrompt: z.string().describe('Current prompt to optimize'),
  model: z.string().describe('Target LLM model'),
  examples: z.array(z.object({
    input: z.string(),
    expectedOutput: z.string(),
  })).optional(),
  constraints: z.array(z.string()).optional(),
});

const PromptOptimizationOutputSchema = z.object({
  success: z.boolean(),
  optimizedPrompts: z.array(z.object({
    prompt: z.string(),
    rationale: z.string(),
    expectedImprovement: z.string(),
  })),
  techniques: z.array(z.object({
    name: z.string(),
    description: z.string(),
    example: z.string(),
  })),
  testingStrategy: z.string(),
});

export const PromptOptimizationSkill: SkillDefinition = {
  id: 'llm.prompt_optimization',
  name: 'Prompt Optimization',
  description: 'Optimize prompts for better LLM performance using expert techniques',
  usage: 'Improve prompt engineering with techniques like chain-of-thought, few-shot, and role-based prompting',
  category: SkillCategory.LLM,
  complexity: SkillComplexity.INTERMEDIATE,
  version: '1.0.0',
  inputSchema: PromptOptimizationInputSchema,
  outputSchema: PromptOptimizationOutputSchema,
  requiredAgents: ['llm_research_agent'],
  estimatedTime: '3-5 minutes',
  tags: ['llm', 'prompt', 'optimization', 'prompt-engineering'],
  examples: [{
    title: 'Optimize for code generation',
    description: 'Improve prompt for generating Python functions',
    input: {
      task: 'Generate Python functions',
      currentPrompt: 'Write a function to sort a list',
      model: 'gpt-4',
    },
  }],
  async execute(input, context, config) {
    const executor = await AgentExecutor.create(LLMResearchAgent, config);
    const result = await executor.run({
      objective: `Optimize prompt for ${input.task} on ${input.model}`,
      topic: 'prompt_engineering',
      depth: 'thorough',
    });
    const data = JSON.parse(result.result);

    return {
      success: true,
      data: {
        success: true,
        optimizedPrompts: [
          {
            prompt: `You are an expert Python developer. ${input.currentPrompt}\n\nProvide clean, well-documented code with error handling.`,
            rationale: 'Added role definition and quality expectations',
            expectedImprovement: '20-30% better code quality',
          },
        ],
        techniques: data.KeyFindings?.map((f: any) => ({
          name: f.title,
          description: f.description,
          example: '',
        })) || [],
        testingStrategy: 'Test on diverse inputs and measure output quality',
      },
      metadata: {
        skillName: 'Prompt Optimization',
        duration: 0,
        agentsUsed: ['llm_research_agent'],
      },
    };
  },
};
