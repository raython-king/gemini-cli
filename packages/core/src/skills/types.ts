/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { z } from 'zod';
import type { AgentDefinition } from '../agents/types.js';

/**
 * Skill category for organization.
 */
export enum SkillCategory {
  LLM = 'llm',
  CV = 'cv',
  EXPERIMENT = 'experiment',
  OPTIMIZATION = 'optimization',
  ANALYSIS = 'analysis',
  DEPLOYMENT = 'deployment',
  GENERAL = 'general',
}

/**
 * Skill complexity level.
 */
export enum SkillComplexity {
  BASIC = 'basic',
  INTERMEDIATE = 'intermediate',
  ADVANCED = 'advanced',
  EXPERT = 'expert',
}

/**
 * Skill execution context.
 */
export interface SkillContext {
  /** Working directory */
  workingDir: string;
  /** Project context (if available) */
  projectInfo?: {
    name?: string;
    type?: string;
    languages?: string[];
  };
  /** Available resources */
  resources?: {
    gpu?: boolean;
    cpuCores?: number;
    memoryGB?: number;
  };
  /** User preferences */
  preferences?: Record<string, any>;
}

/**
 * Skill execution result.
 */
export interface SkillResult<T = any> {
  /** Whether execution succeeded */
  success: boolean;
  /** Result data */
  data?: T;
  /** Error message if failed */
  error?: string;
  /** Execution metadata */
  metadata: {
    skillName: string;
    duration: number;
    agentsUsed: string[];
    tokensUsed?: number;
  };
  /** Artifacts produced (file paths) */
  artifacts?: string[];
  /** Follow-up suggestions */
  suggestions?: string[];
}

/**
 * Skill definition.
 */
export interface SkillDefinition<TInput = any, TOutput = any> {
  /** Unique skill identifier */
  id: string;

  /** Display name */
  name: string;

  /** Short description */
  description: string;

  /** Detailed usage instructions */
  usage: string;

  /** Category */
  category: SkillCategory;

  /** Complexity level */
  complexity: SkillComplexity;

  /** Version */
  version: string;

  /** Author/maintainer */
  author?: string;

  /** Input schema */
  inputSchema: z.ZodSchema<TInput>;

  /** Output schema */
  outputSchema: z.ZodSchema<TOutput>;

  /** Required agents */
  requiredAgents: string[];

  /** Optional agents (for enhanced functionality) */
  optionalAgents?: string[];

  /** Prerequisites (other skills or conditions) */
  prerequisites?: string[];

  /** Estimated execution time */
  estimatedTime?: string;

  /** Tags for searchability */
  tags: string[];

  /** Examples */
  examples: Array<{
    title: string;
    description: string;
    input: TInput;
    expectedOutput?: string;
  }>;

  /** Skill execution function */
  execute: (
    input: TInput,
    context: SkillContext,
    config: any,
  ) => Promise<SkillResult<TOutput>>;
}

/**
 * Skill metadata for registry.
 */
export interface SkillMetadata {
  id: string;
  name: string;
  description: string;
  category: SkillCategory;
  complexity: SkillComplexity;
  version: string;
  tags: string[];
  requiredAgents: string[];
}

/**
 * Skill search filters.
 */
export interface SkillSearchFilters {
  category?: SkillCategory;
  complexity?: SkillComplexity;
  tags?: string[];
  query?: string;
}

/**
 * Skill execution options.
 */
export interface SkillExecutionOptions {
  /** Verbose output */
  verbose?: boolean;
  /** Dry run (don't execute, just validate) */
  dryRun?: boolean;
  /** Timeout in milliseconds */
  timeout?: number;
  /** Progress callback */
  onProgress?: (progress: SkillProgress) => void;
}

/**
 * Skill execution progress.
 */
export interface SkillProgress {
  stage: string;
  progress: number; // 0-100
  message: string;
  currentAgent?: string;
}

/**
 * Skill chain - sequence of skills to execute.
 */
export interface SkillChain {
  /** Chain ID */
  id: string;
  /** Chain name */
  name: string;
  /** Description */
  description: string;
  /** Skills in order */
  skills: Array<{
    skillId: string;
    inputs: any;
    /** Map output to next skill's input */
    outputMapping?: Record<string, string>;
  }>;
}
